#!/usr/bin/env python3
"""Sinribe first-run setup window.

Written in tkinter, and that is the whole point: this runs on a machine where nothing has been
installed yet, under the private Python that tools\\bootstrap.ps1 just fetched. PySide6 — which
the app itself uses — is one of the things this window is about to install, so it cannot be used
to draw it.

All the actual work lives in sinribe/setup_core.py, shared with the in-app setup dialog, so the
two can never disagree about what "installed" means.

    python tools\\setup_wizard.py            open the window
    python tools\\setup_wizard.py --console  no window; plain console install

If tkinter is somehow unavailable, the console path is taken automatically rather than failing.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sinribe import setup_core, winenv                                       # noqa: E402
from sinribe.config import (  # noqa: E402
    LOCAL_MODELS, WHISPER_MODELS, hf_token, load_config, model_available, save_config,
)

# The app's palette, hand-mapped to tk's flat colour options.
BG = "#0c0d11"
CARD = "#191c24"
FG = "#f2f4f9"
SUBTLE = "#96a0b4"
ACCENT = "#5b93ff"
BORDER = "#39415a"
LOGBG = "#0a0b0f"

TOKEN_HELP = (
    "Speaker names come from pyannote, whose model is free but gated:\n"
    "  1.  Create an account at huggingface.co\n"
    "  2.  Open huggingface.co/pyannote/speaker-diarization-community-1 and accept the "
    "conditions\n"
    "  3.  Make a READ token at huggingface.co/settings/tokens, and paste it below\n"
    "Without a token Sinribe still transcribes — the text just has no speaker labels."
)


def run_console() -> int:
    """Fallback path: no window, same engine."""
    return setup_core.main(["--install"])


class Wizard:
    def __init__(self, tk_mod, ttk_mod):
        self.tk = tk_mod
        self.ttk = ttk_mod
        self.cfg = load_config()
        self.gpu = winenv.gpu_info()
        self.queue: queue.Queue = queue.Queue()
        self.thread: threading.Thread | None = None
        self.cancel_flag = False
        self.done = False
        self.step = (1, 1)

        self.root = tk_mod.Tk()
        self.root.title("Sinribe — setup")
        self.root.configure(bg=BG)
        self.root.geometry("780x680")
        self.root.minsize(700, 600)
        icon = ROOT / "assets" / "sinribe.ico"
        if icon.is_file():
            try:
                self.root.iconbitmap(str(icon))
            except Exception:  # noqa: BLE001 - a missing icon must not stop setup
                pass

        self._style()
        self._build()
        self.refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._drain)

    # ------------------------------------------------------------------ chrome
    def _style(self) -> None:
        st = self.ttk.Style()
        try:
            st.theme_use("clam")   # the only bundled theme that honours custom colours
        except Exception:  # noqa: BLE001
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=CARD)
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD, relief="flat")
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Card.TLabel", background=CARD, foreground=FG)
        st.configure("Subtle.TLabel", background=BG, foreground=SUBTLE)
        st.configure("CardSubtle.TLabel", background=CARD, foreground=SUBTLE)
        st.configure("H1.TLabel", background=BG, foreground=FG, font=("Segoe UI", 17, "bold"))
        st.configure("H2.TLabel", background=CARD, foreground="#93b8ff",
                     font=("Segoe UI", 11, "bold"))
        st.configure("TCheckbutton", background=CARD, foreground=FG)
        st.map("TCheckbutton", background=[("active", CARD)])
        st.configure("TButton", background="#242835", foreground=FG, borderwidth=1,
                     focusthickness=0, padding=(14, 7))
        st.map("TButton", background=[("active", "#2e3342")])
        st.configure("Primary.TButton", background=ACCENT, foreground="#06101f",
                     font=("Segoe UI", 10, "bold"), padding=(18, 8))
        st.map("Primary.TButton", background=[("active", "#93b8ff"), ("disabled", "#2b3547")])
        st.configure("TCombobox", fieldbackground="#12141b", background="#242835",
                     foreground=FG, arrowcolor=SUBTLE)
        st.configure("TEntry", fieldbackground="#12141b", foreground=FG,
                     insertcolor=FG, bordercolor=BORDER)
        st.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#12141b",
                     bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)

    def _card(self, parent, title: str):
        f = self.ttk.Frame(parent, style="Card.TFrame", padding=14)
        f.pack(fill="x", pady=(0, 12))
        self.ttk.Label(f, text=title, style="H2.TLabel").pack(anchor="w", pady=(0, 8))
        return f

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Set up Sinribe", style="H1.TLabel").pack(anchor="w")
        ttk.Label(outer, style="Subtle.TLabel", wraplength=720, justify="left",
                  text=("Everything is downloaded into this folder — no administrator rights, "
                        "no system-wide changes, nothing added to your PATH. Deleting the "
                        "folder removes the program.")).pack(anchor="w", pady=(2, 4))
        ttk.Label(outer, text=self.gpu.summary(), style="Subtle.TLabel",
                  wraplength=720, justify="left").pack(anchor="w", pady=(0, 12))

        # ---- options
        opts = self._card(outer, "Options")
        grid = ttk.Frame(opts, style="Card.TFrame")
        grid.pack(fill="x")

        ttk.Label(grid, text="Speech model", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", pady=4)
        self.model_var = tk.StringVar()
        self.model_choices = [f"{n} — {gb:.1f} GB · {note}" for n, _r, gb, note in WHISPER_MODELS]
        self.model_names = [n for n, _r, _gb, _note in WHISPER_MODELS]
        want = setup_core.downloadable_model(self.cfg)
        # large-v3 on a CPU-only machine runs at about half realtime — an hour of audio takes
        # two. Steer that case to a model the machine can actually keep up with.
        if not winenv.cuda_supported(self.gpu) and want == "large-v3":
            want = "small"
        self.model_var.set(self.model_choices[self.model_names.index(want)]
                           if want in self.model_names else self.model_choices[0])
        self.model_cb = ttk.Combobox(grid, textvariable=self.model_var, state="readonly",
                                     values=self.model_choices, width=60)
        self.model_cb.grid(row=0, column=1, sticky="ew", pady=4, padx=(10, 0))
        self.model_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())

        ttk.Label(grid, text="Run on", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=4)
        self.device_var = tk.StringVar(value="Automatic")
        self.device_cb = ttk.Combobox(
            grid, textvariable=self.device_var, state="readonly", width=60,
            values=["Automatic", "Graphics card (NVIDIA, much faster)",
                    "Processor only (works anywhere, slow)"])
        self.device_cb.grid(row=1, column=1, sticky="ew", pady=4, padx=(10, 0))
        self.device_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        grid.columnconfigure(1, weight=1)

        self.diar_var = tk.BooleanVar(value=bool(self.cfg.get("diarize", True)))
        ttk.Checkbutton(opts, text="Separate and label speakers (adds about 3 GB)",
                        variable=self.diar_var,
                        command=self.refresh).pack(anchor="w", pady=(8, 4))

        # Off by default. It is another ~5 GB and a few minutes of conversion, and it only helps
        # German recordings — on a hard far-field German interview plain large-v3 still measured
        # more accurate. Offered here because the fastest quality setting runs one of them.
        self.german_var = tk.BooleanVar(
            value=all(model_available(n) for n in LOCAL_MODELS))
        ttk.Checkbutton(opts, text="Add the German speech models (optional, about 5 GB)",
                        variable=self.german_var,
                        command=self.refresh).pack(anchor="w", pady=(0, 4))

        self.token_frame = ttk.Frame(opts, style="Card.TFrame")
        self.token_frame.pack(fill="x")
        ttk.Label(self.token_frame, text=TOKEN_HELP, style="CardSubtle.TLabel",
                  wraplength=700, justify="left").pack(anchor="w", pady=(0, 6))
        row = ttk.Frame(self.token_frame, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="HF token", style="Card.TLabel").pack(side="left")
        self.token_var = tk.StringVar(value=hf_token() or "")
        self.token_entry = ttk.Entry(row, textvariable=self.token_var, show="•")
        self.token_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
        # On leaving the field, not on every keystroke: a re-check walks the model cache and
        # asks uv where Python is, which is far too much work per character typed.
        self.token_entry.bind("<FocusOut>", lambda _e: self.refresh())
        self.token_entry.bind("<Return>", lambda _e: self.refresh())

        # ---- status
        status = self._card(outer, "Status")
        self.status_body = ttk.Frame(status, style="Card.TFrame")
        self.status_body.pack(fill="x")
        self.total_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.total_var,
                  style="CardSubtle.TLabel").pack(anchor="w", pady=(8, 0))

        # ---- progress + buttons
        self.detail_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.detail_var, style="Subtle.TLabel").pack(anchor="w")
        self.bar = ttk.Progressbar(outer, maximum=1000, style="Horizontal.TProgressbar")
        self.bar.pack(fill="x", pady=(4, 10))

        btns = ttk.Frame(outer)
        btns.pack(fill="x")
        self.install_btn = ttk.Button(btns, text="Install what's missing",
                                      style="Primary.TButton", command=self.start)
        self.install_btn.pack(side="left")
        self.close_btn = ttk.Button(btns, text="Close", command=self._on_close)
        self.close_btn.pack(side="left", padx=(8, 0))
        self.launch_btn = ttk.Button(btns, text="Start Sinribe", command=self.launch)
        self.log_btn = ttk.Button(btns, text="Show details", command=self.toggle_log)
        self.log_btn.pack(side="right")

        self.log_frame = ttk.Frame(outer)
        self.log = tk.Text(self.log_frame, height=11, bg=LOGBG, fg=SUBTLE, insertbackground=FG,
                           relief="flat", wrap="none", font=("Consolas", 9))
        scroll = ttk.Scrollbar(self.log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.log_visible = False

    # ------------------------------------------------------------------ state
    def options(self) -> setup_core.Options:
        dev = self.device_var.get()
        gpu = None
        if dev.startswith("Graphics"):
            gpu = True
        elif dev.startswith("Processor"):
            gpu = False
        name = self.model_names[self.model_choices.index(self.model_var.get())]
        return setup_core.Options(
            model=name, gpu=gpu, diarize=bool(self.diar_var.get()),
            hf_token=self.token_var.get().strip(), shortcuts=True,
            german_models=bool(self.german_var.get()))

    def refresh(self) -> None:
        opts = self.options()
        if opts.diarize:
            self.token_frame.pack(fill="x")
        else:
            self.token_frame.pack_forget()

        for w in self.status_body.winfo_children():
            w.destroy()
        for c in setup_core.detect(opts):
            mark = "✓" if c.ok else "•"
            self.ttk.Label(self.status_body, text=f"{mark}  {c.title} — {c.detail}",
                           style="CardSubtle.TLabel" if c.ok else "Card.TLabel",
                           wraplength=680, justify="left").pack(anchor="w")

        keys, gb = setup_core.plan(opts)
        if keys:
            self.total_var.set(f"{len(keys)} step(s) to run · about {gb:.1f} GB to download")
            self.install_btn.configure(text="Install what's missing")
        else:
            self.total_var.set("Everything is installed.")
            self.install_btn.configure(text="Re-check")

    def toggle_log(self) -> None:
        if self.log_visible:
            self.log_frame.pack_forget()
            self.log_btn.configure(text="Show details")
        else:
            self.log_frame.pack(fill="both", expand=True, pady=(10, 0))
            self.log_btn.configure(text="Hide details")
        self.log_visible = not self.log_visible

    def say(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    # ------------------------------------------------------------------ install
    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        opts = self.options()
        keys, gb = setup_core.plan(opts)
        if not keys:
            self.refresh()
            return

        from tkinter import messagebox

        if opts.diarize and not (opts.hf_token or hf_token()):
            messagebox.showwarning("Sinribe — setup",
                                   "Speaker separation needs a HuggingFace token.\n\n"
                                   + TOKEN_HELP)
            return
        if gb > 1.0 and not messagebox.askyesno(
                "Sinribe — setup",
                f"This downloads about {gb:.1f} GB and can take a while on a slow "
                f"connection.\n\nStart now?"):
            return

        setup_core.remember_model(self.cfg, opts.model)
        self.cfg["diarize"] = opts.diarize
        dev = self.device_var.get()
        self.cfg["asr_device"] = ("cpu" if dev.startswith("Processor")
                                 else "cuda" if dev.startswith("Graphics") else "auto")
        save_config(self.cfg)

        if not self.log_visible:
            self.toggle_log()
        self.install_btn.state(["disabled"])
        self.close_btn.configure(text="Cancel")
        self.cancel_flag = False

        progress = setup_core.Progress(
            log=lambda m: self.queue.put(("log", m)),
            step=lambda i, n, t: self.queue.put(("step", (i, n, t))),
            frac=lambda f, d: self.queue.put(("frac", (f, d))),
            should_cancel=lambda: self.cancel_flag,
        )

        # Only the steps the plan listed, so the "[2/5]" counter matches the announced total.
        only = keys + ["shortcuts"]

        def work() -> None:
            try:
                setup_core.install(opts, progress, only=only)
            except setup_core.Cancelled:
                self.queue.put(("failed", "Setup was cancelled. Anything already downloaded "
                                          "is kept — running setup again continues from there."))
            except Exception as e:  # noqa: BLE001 - show it, never crash the window
                self.queue.put(("failed", str(e)))
            else:
                self.queue.put(("done", ""))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        """Pump worker messages onto the GUI thread. tkinter is not thread-safe, so the worker
        only ever puts tuples in a queue and this — running on the main loop — draws them."""
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.say(payload)
                elif kind == "step":
                    i, n, title = payload
                    self.step = (i, n)
                    self.detail_var.set(f"[{i}/{n}] {title}")
                    self.bar["value"] = 1000 * (i - 1) / max(1, n)
                elif kind == "frac":
                    f, detail = payload
                    i, n = self.step
                    self.bar["value"] = 1000 * ((i - 1) + max(0.0, min(1.0, f))) / max(1, n)
                    if detail:
                        self.detail_var.set(f"[{i}/{n}] {detail}")
                elif kind == "done":
                    self._finish(True, "")
                elif kind == "failed":
                    self._finish(False, payload)
        except queue.Empty:
            pass
        self.root.after(120, self._drain)

    def _finish(self, ok: bool, msg: str) -> None:
        from tkinter import messagebox

        self.close_btn.configure(text="Close")
        self.install_btn.state(["!disabled"])
        self.refresh()
        if ok:
            self.done = True
            self.bar["value"] = 1000
            self.detail_var.set("Setup finished.")
            self.launch_btn.pack(side="left", padx=(8, 0))
            self.say("--- Sinribe is ready.")
            messagebox.showinfo("Sinribe", "Setup finished — Sinribe is ready to use.")
        else:
            self.detail_var.set("Setup did not finish.")
            self.say(f"ERROR: {msg}")
            messagebox.showerror("Sinribe — setup", msg)

    def launch(self) -> None:
        pyw = winenv.venv_python(winenv.ASR_VENV, windowed=True)
        exe = pyw if pyw.is_file() else winenv.venv_python(winenv.ASR_VENV)
        if not exe.is_file():
            return
        subprocess.Popen([str(exe), "-m", "sinribe"], cwd=str(ROOT), **winenv.no_window())
        self.root.destroy()

    def _on_close(self) -> None:
        if self.thread and self.thread.is_alive():
            from tkinter import messagebox

            if not messagebox.askyesno("Sinribe — setup",
                                       "Stop the installation?\n\nWhat has already been "
                                       "downloaded is kept, so starting again continues "
                                       "from there."):
                return
            self.cancel_flag = True
            self.thread.join(timeout=15)
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--console" in argv:
        return run_console()
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter is not available in this Python — falling back to the console installer.")
        return run_console()
    return Wizard(tk, ttk).run()


if __name__ == "__main__":
    sys.exit(main())
