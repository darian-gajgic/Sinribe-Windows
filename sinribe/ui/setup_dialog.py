"""In-app setup: shows what is missing and installs it.

This is the second of the two setup surfaces. The first — tools/setup_wizard.py — is written in
tkinter because it has to run before any virtualenv exists, when PySide6 is not installed yet.
Once .venv is there, the app itself can offer the same thing in its own window, which is what
this is: the same `setup_core` engine behind a Qt front end, reachable from the "Setup" button
and opened automatically when a launch finds something missing.

The installation itself runs on a worker thread. It downloads gigabytes and takes minutes; doing
it on the GUI thread would freeze the window and Windows would paint "Not responding" over it.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .. import setup_core, winenv
from ..config import (
    LOCAL_MODELS, WHISPER_MODELS, hf_token, load_config, model_available, save_config,
)

TOKEN_HELP = (
    "Speaker names come from pyannote, whose model is free but gated. One-time setup:\n"
    "  1.  Create an account at huggingface.co\n"
    "  2.  Open huggingface.co/pyannote/speaker-diarization-community-1 and accept the "
    "conditions\n"
    "  3.  Create a READ token at huggingface.co/settings/tokens and paste it below\n"
    "Without a token Sinribe still transcribes — the text just has no speaker labels."
)


class InstallWorker(QObject):
    """Runs setup_core.install off the GUI thread and turns its callbacks into signals."""
    log = Signal(str)
    step = Signal(int, int, str)
    frac = Signal(float, str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, opts: setup_core.Options, only: list[str] | None = None):
        super().__init__()
        self.opts = opts
        # Only the steps the plan actually listed, so the "[2/5]" counter matches the "5 steps
        # to run" the user was shown. Every step re-checks anyway, so this is honesty rather
        # than an optimisation.
        self.only = only
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        progress = setup_core.Progress(
            log=self.log.emit,
            step=lambda i, n, t: self.step.emit(i, n, t),
            frac=lambda f, d: self.frac.emit(f, d),
            should_cancel=lambda: self._cancel,
        )
        try:
            setup_core.install(self.opts, progress, only=self.only)
        except setup_core.Cancelled:
            self.failed.emit("Setup was cancelled. Nothing that finished has been undone — "
                             "running it again picks up where it stopped.")
        except Exception as e:  # noqa: BLE001 - every failure belongs in the window, not a crash
            self.failed.emit(str(e))
        else:
            self.finished.emit()


class SetupDialog(QDialog):
    """Component list, options, install button, log."""

    def __init__(self, cfg: dict | None = None, blockers: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg if cfg is not None else load_config()
        self.setWindowTitle("Sinribe — setup")
        self.setMinimumSize(720, 620)
        self._thread: QThread | None = None
        self._worker: InstallWorker | None = None
        self._step: tuple[int, int] = (1, 1)
        self._gpu = winenv.gpu_info()

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QLabel("Set up Sinribe")
        title.setObjectName("H1")
        root.addWidget(title)

        intro = QLabel(
            "Everything Sinribe needs is downloaded and installed into this folder — no "
            "administrator rights, no system-wide changes, nothing added to your PATH. "
            "Removing the folder removes the program."
            if not blockers else
            "Sinribe is missing some of what it needs:\n  • " + "\n  • ".join(blockers))
        intro.setObjectName("Subtle")
        intro.setWordWrap(True)
        root.addWidget(intro)

        gpu_lbl = QLabel(self._gpu.summary())
        gpu_lbl.setObjectName("Mono")
        gpu_lbl.setWordWrap(True)
        root.addWidget(gpu_lbl)

        root.addWidget(self._build_options())
        root.addWidget(self._build_components())

        self.detail = QLabel("")
        self.detail.setObjectName("Subtle")
        root.addWidget(self.detail)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        root.addWidget(self.bar)

        row = QHBoxLayout()
        self.install_btn = QPushButton("Install what's missing")
        self.install_btn.setObjectName("Primary")
        self.install_btn.clicked.connect(self._start)
        row.addWidget(self.install_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        row.addWidget(self.close_btn)
        row.addStretch(1)
        self.log_toggle = QPushButton("Show details ▾")
        self.log_toggle.setObjectName("Ghost")
        self.log_toggle.clicked.connect(self._toggle_log)
        row.addWidget(self.log_toggle)
        root.addLayout(row)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("Log")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFixedHeight(190)
        self.log_view.hide()
        root.addWidget(self.log_view)

        self._refresh()

    # ---------------------------------------------------------------- widgets
    def _build_options(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        g = QGridLayout(card)
        g.setContentsMargins(16, 14, 16, 14)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(9)

        head = QLabel("Options")
        head.setObjectName("H2")
        g.addWidget(head, 0, 0, 1, 3)

        g.addWidget(QLabel("Speech model"), 1, 0)
        self.model_cb = QComboBox()
        for name, _repo, gb, note in WHISPER_MODELS:
            self.model_cb.addItem(f"{name} — {gb:.1f} GB · {note}", name)
        want = setup_core.downloadable_model(self.cfg)
        # A CPU-only machine defaulting to large-v3 is a trap: it works, at roughly half
        # realtime, so a one-hour recording takes two hours. Suggest something usable instead.
        if not winenv.cuda_supported(self._gpu) and want == "large-v3":
            want = "small"
        idx = self.model_cb.findData(want)
        self.model_cb.setCurrentIndex(max(0, idx))
        self.model_cb.currentIndexChanged.connect(self._refresh)
        g.addWidget(self.model_cb, 1, 1, 1, 2)

        g.addWidget(QLabel("Run on"), 2, 0)
        self.device_cb = QComboBox()
        self.device_cb.addItem("Automatic", "auto")
        self.device_cb.addItem("Graphics card (NVIDIA, much faster)", "gpu")
        self.device_cb.addItem("Processor only (works anywhere, slow)", "cpu")
        self.device_cb.currentIndexChanged.connect(self._refresh)
        g.addWidget(self.device_cb, 2, 1, 1, 2)

        self.diar_cb = QCheckBox("Separate and label speakers")
        self.diar_cb.setChecked(bool(self.cfg.get("diarize", True)))
        self.diar_cb.setToolTip("Adds PyTorch and pyannote (about 3 GB) and needs a free "
                                "HuggingFace token.")
        self.diar_cb.stateChanged.connect(self._refresh)
        g.addWidget(self.diar_cb, 3, 0, 1, 3)

        self.german_cb = QCheckBox("Add the German speech models (optional, ~5 GB)")
        self.german_cb.setChecked(
            all(model_available(n) for n in LOCAL_MODELS))
        self.german_cb.setToolTip(
            "Whisper fine-tuned on German alone. Downloaded and converted here, once, which "
            "takes a few minutes of processor time.\n\n"
            "Not automatically better: on a hard far-field German interview plain large-v3 "
            "still measured more accurate. They are what the fastest quality setting runs, and "
            "the middle settings vote one of them against large-v3.")
        self.german_cb.stateChanged.connect(self._refresh)
        g.addWidget(self.german_cb, 4, 0, 1, 3)

        self.token_help = QLabel(TOKEN_HELP)
        self.token_help.setObjectName("Subtle")
        self.token_help.setWordWrap(True)
        g.addWidget(self.token_help, 5, 0, 1, 3)

        g.addWidget(QLabel("HF token"), 6, 0)
        self.token_edit = QLineEdit(hf_token() or "")
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("hf_…")
        # editingFinished, not textChanged: a re-check walks the model cache and asks uv where
        # Python is, which is far too much work to do on every keystroke.
        self.token_edit.editingFinished.connect(self._refresh)
        g.addWidget(self.token_edit, 6, 1, 1, 2)
        g.setColumnStretch(1, 1)
        return card

    def _build_components(self) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(6)
        head = QLabel("Status")
        head.setObjectName("H2")
        v.addWidget(head)
        self.comp_box = QVBoxLayout()
        self.comp_box.setSpacing(4)
        v.addLayout(self.comp_box)
        self.total_lbl = QLabel("")
        self.total_lbl.setObjectName("Subtle")
        v.addWidget(self.total_lbl)
        return card

    # ---------------------------------------------------------------- state
    def _options(self) -> setup_core.Options:
        dev = self.device_cb.currentData()
        return setup_core.Options(
            model=str(self.model_cb.currentData()),
            gpu=True if dev == "gpu" else (False if dev == "cpu" else None),
            diarize=self.diar_cb.isChecked(),
            hf_token=self.token_edit.text().strip(),
            shortcuts=True,
            german_models=self.german_cb.isChecked(),
        )

    def _refresh(self) -> None:
        """Re-run detection and repaint the component list."""
        opts = self._options()
        want_diar = opts.diarize
        self.token_help.setVisible(want_diar)
        self.token_edit.setVisible(want_diar)

        while self.comp_box.count():
            item = self.comp_box.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        components = setup_core.detect(opts)
        for c in components:
            row = QLabel(f"{'✓' if c.ok else '•'}  {c.title} — {c.detail}")
            row.setWordWrap(True)
            row.setObjectName("Subtle" if c.ok else "Mono")
            self.comp_box.addWidget(row)

        keys, gb = setup_core.plan(opts)
        if keys:
            self.total_lbl.setText(f"{len(keys)} step(s) to run · about {gb:.1f} GB to download")
            self.install_btn.setEnabled(True)
            self.install_btn.setText("Install what's missing")
        else:
            self.total_lbl.setText("Everything is installed.")
            self.install_btn.setText("Re-check")
            self.install_btn.setEnabled(True)

    def _toggle_log(self) -> None:
        vis = self.log_view.isVisible()
        self.log_view.setVisible(not vis)
        self.log_toggle.setText("Show details ▾" if vis else "Hide details ▴")

    # ---------------------------------------------------------------- install
    def _start(self) -> None:
        opts = self._options()
        keys, gb = setup_core.plan(opts)
        if not keys:
            self._refresh()
            return
        if opts.diarize and not (opts.hf_token or hf_token()):
            QMessageBox.warning(
                self, "Sinribe — setup",
                "Speaker separation needs a HuggingFace token.\n\n" + TOKEN_HELP)
            return
        if gb > 1.0:
            answer = QMessageBox.question(
                self, "Sinribe — setup",
                f"This will download about {gb:.1f} GB and can take a while on a slow "
                f"connection.\n\nStart now?")
            if answer != QMessageBox.StandardButton.Yes:
                return

        # Remember the choices now, so a crash mid-install does not lose them and so the app
        # runs with the same model that is about to be downloaded.
        setup_core.remember_model(self.cfg, opts.model)
        self.cfg["diarize"] = opts.diarize
        dev = self.device_cb.currentData()
        self.cfg["asr_device"] = "cpu" if dev == "cpu" else ("cuda" if dev == "gpu" else "auto")
        save_config(self.cfg)

        self.install_btn.setEnabled(False)
        self.close_btn.setText("Cancel")
        self.log_view.show()
        self.log_toggle.setText("Hide details ▴")
        self.bar.setValue(0)

        self._thread = QThread()
        self._worker = InstallWorker(opts, only=keys + ["shortcuts"])
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._on_log)
        self._worker.step.connect(self._on_step)
        self._worker.frac.connect(self._on_frac)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        for sig in (self._worker.finished, self._worker.failed):
            sig.connect(self._thread.quit)
        self._thread.start()

    def _on_log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def _on_step(self, i: int, n: int, title: str) -> None:
        self._step = (i, n)
        self.detail.setText(f"[{i}/{n}] {title}")
        self.bar.setValue(int(1000 * (i - 1) / max(1, n)))

    def _on_frac(self, f: float, detail: str) -> None:
        i, n = self._step
        self.bar.setValue(int(1000 * ((i - 1) + max(0.0, min(1.0, f))) / max(1, n)))
        if detail:
            self.detail.setText(f"[{i}/{n}] {detail}")

    def _on_finished(self) -> None:
        self.bar.setValue(1000)
        self.detail.setText("Setup finished.")
        self.close_btn.setText("Close")
        self._refresh()
        QMessageBox.information(self, "Sinribe", "Setup finished — Sinribe is ready to use.")
        self.accept()

    def _on_failed(self, msg: str) -> None:
        self.detail.setText("Setup did not finish.")
        self.install_btn.setEnabled(True)
        self.close_btn.setText("Close")
        self.log_view.show()
        self._on_log(f"ERROR: {msg}")
        self._refresh()
        QMessageBox.critical(self, "Sinribe — setup", msg)

    # ---------------------------------------------------------------- closing
    def reject(self) -> None:
        if self._thread and self._thread.isRunning() and self._worker:
            answer = QMessageBox.question(
                self, "Sinribe — setup",
                "Stop the installation?\n\nWhatever has already been downloaded is kept, so "
                "starting again continues from there.")
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(10000)
        super().reject()
