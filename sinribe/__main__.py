"""Sinribe entry point.

    Sinribe.bat                        launch the window
    Sinribe.bat AUDIO [-o DIR]         transcribe a local file, headless
    Sinribe.bat URL   [-o DIR]         download a podcast/video first, then transcribe it
    Sinribe.bat --setup                open the setup window (install or repair)
    Sinribe.bat --check                report what is installed, and exit

Every launch starts by asking whether the machine can actually run a job. If something is
missing — no FFmpeg, no model, no speaker-separation environment — the setup window opens
instead of the app failing halfway through someone's four-hour recording.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import load_config, save_config


def _ensure_streams() -> None:
    """Give the process real stdout/stderr objects when it has none.

    pythonw.exe — the console-less interpreter the window is launched with — sets sys.stdout and
    sys.stderr to None. print() tolerates that (it silently does nothing), but any library that
    reaches for sys.stderr.write, and the interpreter's own unhandled-exception path, raise
    instead. A crash whose traceback cannot be printed is a window that vanishes with no
    explanation, so point the streams at the null device before anything else runs.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity.

    Without this Windows groups the window under "pythonw.exe" and shows the interpreter's icon
    instead of Sinribe's, because that is genuinely the executable that is running.
    """
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sinribe.App")
    except Exception:  # noqa: BLE001 - cosmetic only, and absent on every non-Windows box
        pass


def _app_icon():
    from PySide6.QtGui import QIcon

    from .winenv import ASSETS_DIR

    for name in ("sinribe.ico", "sinribe.png"):
        p = ASSETS_DIR / name
        if p.is_file():
            return QIcon(str(p))
    return QIcon()


def _cli(args: argparse.Namespace) -> int:
    from .pipeline import Cancelled, JobSpec, Runner, StageError
    from .setup_core import launch_blockers

    cfg = load_config()
    if args.model:
        cfg["asr_model"] = args.model
    if args.quality:
        cfg["speed_target"] = args.quality
    if args.hotwords:
        cfg["hotwords"] = args.hotwords
    if args.language:
        cfg["language"] = args.language
    if args.speakers:
        cfg["speaker_mode"] = "exact"
        cfg["num_speakers"] = args.speakers
    if args.enrich:
        cfg["llm_enrich"] = True
    if args.no_enrich:
        cfg["llm_enrich"] = False
    if args.no_diarize:
        cfg["diarize"] = False

    blockers = launch_blockers(cfg)
    if blockers:
        print("Sinribe is not ready to run:", file=sys.stderr)
        for b in blockers:
            print(f"  - {b}", file=sys.stderr)
        print("\nRun  Sinribe.bat --setup  to install what is missing.", file=sys.stderr)
        return 3

    from .fetch import is_url

    out_dir = Path(args.output or cfg["output_dir"]).expanduser()
    if is_url(args.audio):
        spec = JobSpec(input_path=None, output_dir=out_dir, cfg=cfg, url=args.audio.strip())
    else:
        src = Path(args.audio).expanduser().resolve()
        if not src.is_file():
            print(f"ERROR: no such file: {src}", file=sys.stderr)
            return 2
        spec = JobSpec(input_path=src, output_dir=out_dir, cfg=cfg)

    state = {"phase": ""}

    def on_progress(frac: float, phase: str, detail: str) -> None:
        if phase != state["phase"]:
            state["phase"] = phase
            print(f"\n>> {phase}", flush=True)
        bar = int(frac * 40)
        sys.stdout.write(f"\r   [{'#' * bar}{'.' * (40 - bar)}] {frac * 100:5.1f}%  {detail}   ")
        sys.stdout.flush()

    def on_log(msg: str) -> None:
        sys.stdout.write(f"\r   {msg}\n")
        sys.stdout.flush()

    runner = Runner(spec, on_progress=on_progress, on_log=on_log)
    try:
        result = runner.run()
    except Cancelled:
        print("\ncancelled")
        return 130
    except (StageError, Exception) as e:  # noqa: BLE001
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("\n")
    for p in result["written"]:
        print(f"   wrote {p}")
    for name, s in sorted(result["stats"].items(), key=lambda kv: -kv[1]["seconds"]):
        print(f"   {name}: {s['seconds']:.1f}s ({s['share'] * 100:.0f}%)")
    return 0


def _gui(force_setup: bool = False) -> int:
    from PySide6.QtWidgets import QApplication

    from .setup_core import launch_blockers
    from .theme import apply_theme
    from .ui.setup_dialog import SetupDialog

    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("Sinribe")
    app.setApplicationDisplayName("Sinribe")
    app.setWindowIcon(_app_icon())
    apply_theme(app)

    cfg = load_config()
    blockers = [] if force_setup else launch_blockers(cfg)
    if blockers or force_setup:
        dlg = SetupDialog(cfg, blockers=blockers)
        dlg.exec()
        cfg = load_config()
        if launch_blockers(cfg):
            # Still not usable: they closed setup without finishing. Opening the main window
            # now would only offer a Start button that cannot work.
            return 0

    from .ui.main_window import MainWindow

    win = MainWindow(cfg)
    win.show()
    rc = app.exec()
    save_config(win.cfg)
    return rc


def main() -> int:
    _ensure_streams()
    ap = argparse.ArgumentParser(prog="Sinribe", description="Offline transcription with "
                                                             "speaker diarization")
    ap.add_argument("--cli", action="store_true", help="run headless instead of the window")
    ap.add_argument("audio", nargs="?",
                    help="audio/video file, or a YouTube/podcast URL (implies --cli)")
    ap.add_argument("-o", "--output", help="output directory")
    ap.add_argument("-m", "--model",
                    help="whisper model, or 'auto' to let --quality choose "
                         "(large-v3, medium, small, tiny, large-v3-german, "
                         "large-v3-turbo-german, …)")
    ap.add_argument("-q", "--quality", type=int, metavar="RTF",
                    help="target realtime factor: 1 is slowest and most accurate, 84 fastest")
    ap.add_argument("--hotwords", help="comma-separated names and jargon to expect")
    ap.add_argument("-l", "--language", help="language code, or 'auto'")
    ap.add_argument("-s", "--speakers", type=int, help="exact number of speakers")
    ap.add_argument("--enrich", action="store_true", help="force LLM chapters + summary on")
    ap.add_argument("--no-enrich", action="store_true", help="force LLM enrichment off")
    ap.add_argument("--no-diarize", action="store_true",
                    help="skip speaker separation for this run")
    ap.add_argument("--setup", action="store_true", help="open the setup window")
    ap.add_argument("--check", action="store_true",
                    help="report what is installed, and exit")
    args = ap.parse_args()

    if args.check:
        from .setup_core import main as setup_main

        return setup_main(["--check"])
    if args.setup:
        return _gui(force_setup=True)
    if args.audio:
        return _cli(args)
    if args.cli:
        ap.error("--cli requires an audio file")
    return _gui()


if __name__ == "__main__":
    sys.exit(main())
