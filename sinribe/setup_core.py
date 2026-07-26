"""Detect what Sinribe needs, and install whatever is missing.

This is the engine behind every setup surface: the tkinter first-run wizard (which runs before
any virtualenv exists), the in-app "Repair install" dialog, and `Sinribe.bat --setup`. There is
exactly one implementation of "is it installed" and "install it" so the three can never drift
apart and disagree about the state of the machine.

**Standard library only.** It has to be: on a fresh Windows box this module runs under a bare
interpreter with nothing installed, and the whole point is to be the thing that installs the
rest. Downloads use urllib, archives are unpacked with zipfile, and every third-party import
happens inside a subprocess running in a virtualenv this module built.

Nothing here needs administrator rights, touches the registry, or installs a driver.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import winenv
from .config import (
    DIAR_SUPPORT_REPOS, WHISPER_REPOS, WHISPER_SIZES_GB, hf_token, load_state, save_hf_token,
    save_state,
)
from .winenv import (
    APP_ROOT, ASR_VENV, DIAR_VENV, FFMPEG_DIR, IS_WINDOWS, PYTHON_VERSION, TOOLS_DIR, UV_EXE,
)

__all__ = ["Component", "Options", "Progress", "detect", "install", "launch_blockers", "plan",
           "main"]

# --------------------------------------------------------------------------- sources
#: uv is used for one reason the alternatives cannot match: it can install a private, known-good
#: CPython for us. That removes the worst part of "make this run on any Windows machine" —
#: whether the box has a Python at all, and whether it is a version every wheel here supports.
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

#: FFmpeg 7.1, *shared* build. Shared rather than static because two different things need it:
#: ffmpeg.exe/ffprobe.exe for decoding, and the avcodec/avformat/avutil DLLs that torchcodec
#: (imported by pyannote 4) loads at import time. Pinned to the 7.1 branch rather than master:
#: torchcodec ships loaders for FFmpeg 4-8, and a master build that rolls to 9 would break the
#: diarization venv on a machine that had been working.
FFMPEG_URLS = [
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl-shared.zip",
]

REQ_ASR = APP_ROOT / "requirements-asr.txt"
REQ_CUDA = APP_ROOT / "requirements-cuda.txt"
REQ_DIAR = APP_ROOT / "requirements-diar.txt"

#: Rough download volume per step, in GB, for the "this will fetch about N GB" line. Wheels are
#: measured from PyPI; the CUDA pair is the dominant term and the reason the estimate is worth
#: showing before someone starts this on a hotel wifi.
SIZE_GB = {
    "uv": 0.03,
    "python": 0.03,
    "ffmpeg": 0.08,
    "asr_venv": 0.45,
    "asr_cuda": 1.30,
    "diar_venv_cpu": 0.60,
    "diar_venv_cuda": 3.20,
    "diar_model": 0.10,
}


class SetupError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


# --------------------------------------------------------------------------- data types
@dataclass
class Options:
    """What to install. `gpu=None` means "decide from the hardware"."""
    model: str = "large-v3"
    gpu: bool | None = None
    diarize: bool = True
    hf_token: str = ""
    shortcuts: bool = True

    def resolve_gpu(self, gpu_info: winenv.GpuInfo | None = None) -> bool:
        if self.gpu is not None:
            return self.gpu
        return winenv.cuda_supported(gpu_info if gpu_info is not None else winenv.gpu_info())


@dataclass
class Component:
    """One installable piece, and whether it is currently present."""
    key: str
    title: str
    ok: bool
    detail: str
    required: bool = True
    size_gb: float = 0.0

    @property
    def status(self) -> str:
        return "ok" if self.ok else ("missing" if self.required else "optional")


def _noop(*_a, **_k) -> None:
    return None


def _never() -> bool:
    return False


@dataclass
class Progress:
    """Callbacks the caller supplies; all optional.

    `log` gets a line of output, `step` the (index, count, title) of the current step, `frac` a
    0..1 fraction with a human detail string for the step in flight.
    """
    log: Callable[[str], None] = _noop
    step: Callable[[int, int, str], None] = _noop
    frac: Callable[[float, str], None] = _noop
    should_cancel: Callable[[], bool] = _never

    def check(self) -> None:
        if self.should_cancel():
            raise Cancelled("cancelled")


# --------------------------------------------------------------------------- small helpers
def _human(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _pkg_present(site_packages: Path, *names: str) -> bool:
    """Is every one of `names` importable from `site_packages`?

    Answered from the filesystem rather than by starting an interpreter: this runs on every
    launch, and spawning Python twice to answer a question `Path.exists` can answer costs a
    visible pause on a cold disk. A non-empty package directory counts, because namespace
    packages (pyannote.audio) and C extensions have no __init__.py.
    """
    for name in names:
        pkg = site_packages / name
        if pkg.is_dir() and any(pkg.iterdir()):
            continue
        if (site_packages / f"{name}.py").exists():
            continue
        return False
    return True


def _run(cmd: list[str], progress: Progress, env: dict | None = None,
         cwd: Path | None = None, what: str = "") -> None:
    """Run a command, streaming its output into the log. Raises SetupError on failure.

    Output is streamed rather than captured because these commands take minutes: a wizard that
    shows nothing while uv downloads 3 GB of torch is indistinguishable from one that has hung.
    """
    progress.check()
    progress.log(f"$ {' '.join(str(c) for c in cmd)}")
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd or APP_ROOT), env=env, text=True, bufsize=1,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            errors="replace", **winenv.no_window())
    except OSError as e:
        raise SetupError(f"could not start {cmd[0]}: {e}") from e

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                progress.log(line)
            if progress.should_cancel():
                proc.kill()
                proc.wait(timeout=15)
                raise Cancelled("cancelled")
        rc = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)
    if rc != 0:
        raise SetupError(f"{what or cmd[0]} failed (exit code {rc}) — see the log above")


def download(url: str, dest: Path, progress: Progress, what: str = "") -> Path:
    """Download `url` to `dest` with progress, resuming a partial file if one is there.

    Resume matters more than it looks: these are 70 MB to 3 GB downloads, and a wizard that
    restarts a 3 GB fetch from zero because a laptop slept is a wizard people give up on.
    """
    progress.check()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "Sinribe-Setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    mode = "ab" if have else "wb"

    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if have and e.code in (416, 200):          # server ignored/rejected the range
            part.unlink(missing_ok=True)
            return download(url, dest, progress, what)
        raise SetupError(f"download failed ({e.code}) for {url}") from e
    except (urllib.error.URLError, OSError) as e:
        raise SetupError(f"could not reach {url}: {e}") from e

    with resp:
        if have and resp.status == 200:            # no range support: start over
            have, mode = 0, "wb"
        total = int(resp.headers.get("Content-Length") or 0) + have
        got = have
        t0 = time.time()
        last = 0.0
        progress.log(f"downloading {what or url.rsplit('/', 1)[-1]}"
                     + (f" ({_human(total)})" if total else ""))
        with open(part, mode) as fh:
            while True:
                if progress.should_cancel():
                    raise Cancelled("cancelled")
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last >= 0.3:
                    last = now
                    speed = got - have
                    rate = speed / max(0.001, now - t0)
                    detail = f"{_human(got)}" + (f" of {_human(total)}" if total else "")
                    detail += f" at {rate / 1e6:.1f} MB/s"
                    progress.frac((got / total) if total else 0.0, detail)

    if total and got < total:
        raise SetupError(f"download of {what or url} ended early ({_human(got)} of "
                         f"{_human(total)}) — check the connection and run setup again")
    part.replace(dest)
    progress.frac(1.0, _human(got))
    return dest


def _extract(zip_path: Path, dest: Path, members: Callable[[str], str | None],
             progress: Progress) -> int:
    """Extract entries a zip, renaming through `members` (return None to skip an entry)."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for i, name in enumerate(names):
            progress.check()
            out_rel = members(name)
            if out_rel is None:
                continue
            # Never trust a path out of an archive: a crafted entry containing .. would
            # otherwise write outside the destination directory.
            out = (dest / out_rel).resolve()
            if not out.is_relative_to(dest.resolve()):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
            progress.frac((i + 1) / max(1, len(names)), out.name)
    return n


# --------------------------------------------------------------------------- detection
def detect(opts: Options | None = None) -> list[Component]:
    """Full inventory of what is and is not installed on this machine."""
    opts = opts or Options()
    gpu = winenv.gpu_info()
    want_gpu = opts.resolve_gpu(gpu)
    asr_sp = winenv.venv_site_packages(ASR_VENV)
    diar_sp = winenv.venv_site_packages(DIAR_VENV)
    out: list[Component] = []

    out.append(Component(
        "uv", "Package manager (uv)", UV_EXE.is_file(),
        str(UV_EXE) if UV_EXE.is_file() else "will be downloaded (25 MB)",
        size_gb=SIZE_GB["uv"]))

    py = _managed_python()
    out.append(Component(
        "python", f"Python {PYTHON_VERSION}", bool(py),
        str(py) if py else "will be installed privately, just for Sinribe",
        size_gb=SIZE_GB["python"]))

    out.append(Component(
        "ffmpeg", "FFmpeg (audio decoding)", winenv.have_ffmpeg(),
        winenv.ffmpeg_exe() if winenv.have_ffmpeg() else "will be downloaded (76 MB)",
        size_gb=SIZE_GB["ffmpeg"]))

    asr_ok = (winenv.venv_python(ASR_VENV).is_file()
              and _pkg_present(asr_sp, "faster_whisper", "ctranslate2", "PySide6", "yt_dlp"))
    asr_cuda = bool(winenv.cuda_dll_dirs(asr_sp))
    if not asr_ok:
        detail = "transcription engine + interface"
    elif asr_cuda:
        detail = "installed · GPU (CUDA 12)"
    elif want_gpu:
        # The environment is there but was built for CPU — usually a machine that was set up
        # before its graphics card was usable, or a driver that has since been updated.
        detail = "installed, but without the CUDA runtime — add it to use the graphics card"
    else:
        detail = "installed · CPU only"
    out.append(Component(
        "asr_venv", "Transcription environment", asr_ok and (asr_cuda or not want_gpu), detail,
        size_gb=(SIZE_GB["asr_cuda"] if asr_ok else
                 SIZE_GB["asr_venv"] + (SIZE_GB["asr_cuda"] if want_gpu else 0.0))))

    repo = WHISPER_REPOS.get(opts.model, "")
    have_model = bool(repo) and winenv.model_is_cached(repo)
    out.append(Component(
        "whisper_model", f"Speech model ({opts.model})", have_model,
        "downloaded" if have_model else f"{WHISPER_SIZES_GB.get(opts.model, 1.0):.1f} GB download",
        size_gb=0.0 if have_model else WHISPER_SIZES_GB.get(opts.model, 1.0)))

    diar_ok = (winenv.venv_python(DIAR_VENV).is_file()
               and _pkg_present(diar_sp, "torch", "pyannote", "torchcodec"))
    out.append(Component(
        "diar_venv", "Speaker-separation environment", diar_ok or not opts.diarize,
        "installed" if diar_ok else ("pyannote + PyTorch" if opts.diarize else "not needed"),
        required=opts.diarize,
        size_gb=(SIZE_GB["diar_venv_cuda"] if want_gpu else SIZE_GB["diar_venv_cpu"])
        if opts.diarize and not diar_ok else 0.0))

    tok = (opts.hf_token or hf_token() or "").strip()
    out.append(Component(
        "hf_token", "HuggingFace token (for speaker names)", bool(tok) or not opts.diarize,
        "saved" if tok else "needed once, to download the speaker model",
        required=opts.diarize))

    diar_repos = DIAR_SUPPORT_REPOS.get(_default_pipeline(), [])
    have_diar_model = bool(diar_repos) and all(winenv.model_is_cached(r) for r in diar_repos)
    out.append(Component(
        "diar_model", "Speaker-separation model", have_diar_model or not opts.diarize,
        "downloaded" if have_diar_model else "downloaded once, then works offline",
        required=opts.diarize,
        size_gb=0.0 if have_diar_model else SIZE_GB["diar_model"]))

    return out


def _default_pipeline() -> str:
    from .config import DIAR_PIPELINES, load_config
    return str(load_config().get("diar_pipeline") or DIAR_PIPELINES[0])


def missing(components: Iterable[Component]) -> list[Component]:
    return [c for c in components if not c.ok and c.required]


def is_ready(opts: Options | None = None) -> bool:
    """True when a full install is in place."""
    return not missing(detect(opts))


def launch_blockers(cfg: dict | None = None) -> list[str]:
    """Reasons the app cannot transcribe right now, in plain language. Empty means go.

    Deliberately cheaper than `detect()`: filesystem checks only, no nvidia-smi and no uv, so
    it can run on the startup path of every launch without adding a perceptible delay. It also
    asks a narrower question — "can a job run" rather than "is the install pristine" — so a
    missing shortcut or an absent uv.exe does not stop someone from working.
    """
    from .config import load_config
    cfg = cfg if cfg is not None else load_config()
    out: list[str] = []

    if not winenv.venv_python(ASR_VENV).is_file():
        out.append("the transcription engine is not installed")
    elif not _pkg_present(winenv.venv_site_packages(ASR_VENV), "faster_whisper", "ctranslate2"):
        out.append("the transcription engine is incomplete")

    if not winenv.have_ffmpeg():
        out.append("FFmpeg is missing (needed to read audio and video files)")

    model = str(cfg.get("asr_model", "large-v3"))
    repo = WHISPER_REPOS.get(model)
    if repo and not winenv.model_is_cached(repo):
        out.append(f"the {model} speech model has not been downloaded")

    if cfg.get("diarize", True):
        if not winenv.venv_python(DIAR_VENV).is_file():
            out.append("the speaker-separation environment is not installed")
        elif not _pkg_present(winenv.venv_site_packages(DIAR_VENV), "torch", "pyannote"):
            out.append("the speaker-separation environment is incomplete")
        elif not hf_token():
            out.append("no HuggingFace token is saved (needed for the speaker model)")
    return out


def plan(opts: Options) -> tuple[list[str], float]:
    """(step keys still to run, estimated download GB) for the current machine."""
    comps = {c.key: c for c in detect(opts)}
    keys = [key for key, _, _ in STEPS
            if key in comps and not comps[key].ok and comps[key].required]
    return keys, sum(comps[k].size_gb for k in keys)


# --------------------------------------------------------------------------- install steps
_PYTHON_CACHE: Path | None = None


def _managed_python() -> Path | None:
    """Path to the Python that virtualenvs are built from, or None if it is not installed yet.

    Prefers a uv-managed 3.12 (private to Sinribe, identical on every machine) and falls back to
    the interpreter currently running if it happens to be the right version — which is the case
    when the wizard was started by the bootstrap script.

    Cached, because `uv python find` is a process launch and detection runs whenever the setup
    window repaints. `ensure_python` clears the cache after installing one.
    """
    global _PYTHON_CACHE
    if _PYTHON_CACHE is not None:
        return _PYTHON_CACHE
    _PYTHON_CACHE = _find_python()
    return _PYTHON_CACHE


def _find_python() -> Path | None:
    if UV_EXE.is_file():
        try:
            out = winenv.run([str(UV_EXE), "python", "find", PYTHON_VERSION], timeout=60)
            if out.returncode == 0:
                p = Path((out.stdout or "").strip().splitlines()[0].strip())
                if p.is_file():
                    return p
        except (OSError, IndexError, subprocess.SubprocessError):
            pass
    cur = Path(sys.executable)
    if cur.is_file() and f"{sys.version_info.major}.{sys.version_info.minor}" == PYTHON_VERSION:
        return cur
    return None


def ensure_uv(opts: Options, progress: Progress) -> None:
    if UV_EXE.is_file():
        progress.log(f"uv already present at {UV_EXE}")
        return
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TOOLS_DIR / "uv.zip"
    download(UV_URL, zip_path, progress, what="uv")
    got = _extract(zip_path, TOOLS_DIR,
                   lambda n: Path(n).name if Path(n).name in ("uv.exe", "uvx.exe", "uv", "uvx")
                   else None, progress)
    zip_path.unlink(missing_ok=True)
    if not UV_EXE.is_file():
        raise SetupError(f"uv.exe was not found in the downloaded archive ({got} files extracted)")
    if not IS_WINDOWS:
        UV_EXE.chmod(0o755)
    progress.log(f"uv installed at {UV_EXE}")


def ensure_python(opts: Options, progress: Progress) -> None:
    global _PYTHON_CACHE
    if _managed_python():
        progress.log(f"Python {PYTHON_VERSION} already available")
        return
    _run([str(UV_EXE), "python", "install", PYTHON_VERSION], progress,
         what=f"installing Python {PYTHON_VERSION}")
    _PYTHON_CACHE = None      # it exists now; the cached "missing" answer is stale
    if not _managed_python():
        raise SetupError(f"Python {PYTHON_VERSION} still not found after installing it")


def ensure_ffmpeg(opts: Options, progress: Progress) -> None:
    if winenv.have_ffmpeg():
        progress.log(f"ffmpeg already present: {winenv.ffmpeg_exe()}")
        return
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = FFMPEG_DIR / "ffmpeg.zip"
    last: Exception | None = None
    for url in FFMPEG_URLS:
        try:
            download(url, zip_path, progress, what="FFmpeg")
            break
        except SetupError as e:          # try the next mirror before giving up
            last = e
            progress.log(f"  {e}")
            zip_path.unlink(missing_ok=True)
    else:
        raise SetupError(f"could not download FFmpeg: {last}")

    # Take the whole bin/ directory: the .exe files decode audio, and the av*.dll files next to
    # them are what torchcodec (imported by pyannote) loads at import time.
    def member(name: str) -> str | None:
        parts = name.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[-2] == "bin" and parts[-1]:
            return f"bin/{parts[-1]}"
        return None

    n = _extract(zip_path, FFMPEG_DIR, member, progress)
    zip_path.unlink(missing_ok=True)
    if not winenv.have_ffmpeg():
        raise SetupError(f"FFmpeg archive unpacked ({n} files) but ffmpeg/ffprobe are missing")
    for exe in ("ffmpeg", "ffprobe"):
        p = FFMPEG_DIR / "bin" / (f"{exe}.exe" if IS_WINDOWS else exe)
        if p.is_file() and not IS_WINDOWS:
            p.chmod(0o755)
    progress.log(f"FFmpeg installed at {FFMPEG_DIR / 'bin'} ({n} files)")


def _uv_pip(args: list[str], venv: Path, progress: Progress, what: str) -> None:
    _run([str(UV_EXE), "pip", "install", "--python", str(winenv.venv_python(venv)), *args],
         progress, what=what)


def ensure_asr_venv(opts: Options, progress: Progress) -> None:
    """Build .venv: faster-whisper + ctranslate2 + PySide6 + yt-dlp, and the CUDA 12 runtime.

    The NVIDIA wheels are a separate requirements file rather than a conditional line in the
    main one because they are 1.3 GB that a CPU-only machine must not download.
    """
    py = winenv.venv_python(ASR_VENV)
    if not py.is_file():
        _run([str(UV_EXE), "venv", "--python", PYTHON_VERSION, str(ASR_VENV)], progress,
             what="creating the transcription environment")
    _uv_pip(["-r", str(REQ_ASR)], ASR_VENV, progress, "installing the transcription engine")

    gpu = winenv.gpu_info()
    if opts.resolve_gpu(gpu):
        progress.log(f"GPU: {gpu.summary()}")
        _uv_pip(["-r", str(REQ_CUDA)], ASR_VENV, progress,
                "installing the CUDA runtime (cuBLAS + cuDNN, ~1.3 GB)")
        dirs = winenv.cuda_dll_dirs(winenv.venv_site_packages(ASR_VENV))
        if not dirs:
            raise SetupError("the CUDA wheels installed but their DLL directories are missing")
        progress.log(f"CUDA libraries: {', '.join(str(d) for d in dirs)}")
    else:
        progress.log("no usable NVIDIA GPU — installing the CPU-only transcription stack")


def ensure_diar_venv(opts: Options, progress: Progress) -> None:
    """Build .venv-diar: PyTorch + pyannote, in their own environment.

    Two virtualenvs is not tidiness, it is a hard requirement inherited from the Linux build:
    ctranslate2 is a CUDA 12 binary and torch is (usually) a CUDA 13 one, and they cannot share
    a process. Keeping them apart also means the diarization process *exits* between stages,
    which is the only reliable way to hand a GPU's memory back — an in-process `del model` does
    not return the CUDA context.
    """
    if not opts.diarize:
        progress.log("speaker separation disabled — skipping")
        return
    py = winenv.venv_python(DIAR_VENV)
    if not py.is_file():
        _run([str(UV_EXE), "venv", "--python", PYTHON_VERSION, str(DIAR_VENV)], progress,
             what="creating the speaker-separation environment")

    gpu = winenv.gpu_info()
    index, specs, is_cuda, reason = winenv.torch_target(gpu)
    if opts.gpu is False and is_cuda:
        index, specs, is_cuda, reason = winenv.CPU_INDEX, winenv.TORCH_TRIPLE, False, "forced CPU"
    progress.log(f"PyTorch: {reason} ({index.rsplit('/', 1)[-1]})")
    if gpu.present and not is_cuda:
        progress.log(f"  note: {gpu.name} was detected but {reason}")

    # torch, torchaudio and torchcodec come as a set from one index; everything else from PyPI.
    _uv_pip(["--index-url", index, *specs], DIAR_VENV, progress,
            "installing PyTorch (this is the big one)")
    _uv_pip(["-r", str(REQ_DIAR)], DIAR_VENV, progress, "installing pyannote")


def _model_script(kind: str) -> str:
    """Source of the throwaway script that pulls one model, run inside the relevant venv.

    Written to a temp file and executed rather than piped through stdin so that a traceback
    carries a real file and line number into the log.
    """
    if kind == "whisper":
        return (
            "import sys\n"
            "from huggingface_hub import snapshot_download\n"
            "repo = sys.argv[1]\n"
            "p = snapshot_download(repo)\n"
            "print('cached at', p)\n"
        )
    return (
        "import sys\n"
        "from pyannote.audio import Pipeline\n"
        "name, tok = sys.argv[1], sys.argv[2]\n"
        "p = Pipeline.from_pretrained(name, token=tok)\n"
        "if p is None:\n"
        "    print('GATED', file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "print('ok', name)\n"
    )


def _model_env(venv: Path) -> dict:
    """Environment for a download subprocess: online, into Sinribe's own model cache."""
    env = dict(os.environ)
    env["HF_HOME"] = str(winenv.hf_home())
    env["HF_HUB_OFFLINE"] = "0"
    # Without Developer Mode a Windows user cannot create symlinks, and huggingface_hub warns
    # about it on every single file. It falls back to real files, which is what we want anyway.
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if venv == DIAR_VENV:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def ensure_whisper_model(opts: Options, progress: Progress) -> None:
    repo = WHISPER_REPOS.get(opts.model)
    if not repo:
        raise SetupError(f"unknown speech model {opts.model!r}")
    if winenv.model_is_cached(repo):
        progress.log(f"{opts.model} already downloaded")
        return
    script = TOOLS_DIR / "_fetch_whisper.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_model_script("whisper"), encoding="utf-8")
    progress.log(f"downloading {repo} (~{WHISPER_SIZES_GB.get(opts.model, 1.0):.1f} GB) into "
                 f"{winenv.hf_home()}")
    try:
        _run([str(winenv.venv_python(ASR_VENV)), str(script), repo], progress,
             env=_model_env(ASR_VENV), what=f"downloading {opts.model}")
    finally:
        script.unlink(missing_ok=True)
    if not winenv.model_is_cached(repo):
        raise SetupError(f"{repo} did not appear in the cache after downloading")


def ensure_diar_model(opts: Options, progress: Progress) -> None:
    """Download the pyannote pipeline. Needs the HuggingFace token — the repos are gated."""
    if not opts.diarize:
        return
    token = (opts.hf_token or hf_token() or "").strip()
    if not token:
        raise SetupError(
            "a HuggingFace token is needed to download the speaker-separation model.\n"
            "  1. create a free account at https://huggingface.co/join\n"
            "  2. open https://huggingface.co/pyannote/speaker-diarization-community-1 and "
            "accept the conditions\n"
            "  3. create a READ token at https://huggingface.co/settings/tokens\n"
            "  4. paste it into the setup window and run setup again")
    if opts.hf_token:
        save_hf_token(opts.hf_token)

    pipeline = _default_pipeline()
    script = TOOLS_DIR / "_fetch_diar.py"
    script.write_text(_model_script("diar"), encoding="utf-8")
    progress.log(f"downloading {pipeline}")
    try:
        _run([str(winenv.venv_python(DIAR_VENV)), str(script), pipeline, token], progress,
             env=_model_env(DIAR_VENV), what="downloading the speaker model")
    except SetupError as e:
        raise SetupError(
            f"{e}\n\nThe usual cause is that this token's account has not accepted the "
            f"conditions at https://huggingface.co/{pipeline} — open that page while signed in, "
            f"accept, then run setup again.") from e
    finally:
        script.unlink(missing_ok=True)


def ensure_shortcuts(opts: Options, progress: Progress) -> None:
    """Desktop + Start Menu shortcuts, via the bundled PowerShell script."""
    if not opts.shortcuts:
        return
    if not IS_WINDOWS:
        progress.log("not Windows — skipping shortcuts")
        return
    ps1 = TOOLS_DIR / "make_shortcuts.ps1"
    if not ps1.is_file():
        progress.log(f"WARNING: {ps1} is missing; skipping shortcuts")
        return
    try:
        _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
             progress, what="creating shortcuts")
    except SetupError as e:
        # A missing shortcut is cosmetic; refusing to finish an otherwise complete install
        # because of it would be worse than the problem.
        progress.log(f"WARNING: could not create shortcuts ({e})")


STEPS: list[tuple[str, str, Callable[[Options, Progress], None]]] = [
    ("uv", "Package manager", ensure_uv),
    ("python", f"Python {PYTHON_VERSION}", ensure_python),
    ("ffmpeg", "FFmpeg", ensure_ffmpeg),
    ("asr_venv", "Transcription engine", ensure_asr_venv),
    ("whisper_model", "Speech model", ensure_whisper_model),
    ("diar_venv", "Speaker separation", ensure_diar_venv),
    ("diar_model", "Speaker model", ensure_diar_model),
    ("shortcuts", "Shortcuts", ensure_shortcuts),
]


def install(opts: Options, progress: Progress | None = None,
            only: Iterable[str] | None = None) -> None:
    """Install everything missing. Idempotent: every step re-checks and returns early.

    Raises SetupError with a message meant to be shown verbatim to the person at the keyboard,
    or Cancelled if they pressed the button.
    """
    progress = progress or Progress()
    keep = set(only) if only is not None else None
    wanted = [s for s in STEPS if keep is None or s[0] in keep]

    if opts.hf_token:
        save_hf_token(opts.hf_token)

    total = len(wanted)
    t0 = time.time()
    for i, (_key, title, fn) in enumerate(wanted, start=1):
        progress.check()
        progress.step(i, total, title)
        progress.log(f"--- [{i}/{total}] {title}")
        fn(opts, progress)

    state = load_state()
    state.update({
        "installed_at": time.strftime("%Y-%m-%d %H:%M"),
        "model": opts.model,
        "gpu": opts.resolve_gpu(),
        "diarize": opts.diarize,
        "elapsed_s": round(time.time() - t0, 1),
        "app_root": str(APP_ROOT),
    })
    save_state(state)
    progress.log(f"--- done in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- CLI
def _console_progress() -> Progress:
    state = {"line": ""}

    def log(msg: str) -> None:
        if state["line"]:
            sys.stdout.write("\n")
            state["line"] = ""
        print(msg, flush=True)

    def frac(f: float, detail: str) -> None:
        bar = int(f * 30)
        state["line"] = f"\r   [{'#' * bar}{'.' * (30 - bar)}] {f * 100:5.1f}%  {detail}   "
        sys.stdout.write(state["line"])
        sys.stdout.flush()

    def step(i: int, n: int, title: str) -> None:
        log(f"\n==> [{i}/{n}] {title}")

    return Progress(log=log, step=step, frac=frac)


def main(argv: list[str] | None = None) -> int:
    """`python -m sinribe.setup_core --check | --install [options]`"""
    import argparse

    ap = argparse.ArgumentParser(prog="sinribe-setup",
                                 description="Check for, and install, everything Sinribe needs")
    ap.add_argument("--check", action="store_true", help="report what is installed and exit")
    ap.add_argument("--install", action="store_true", help="install whatever is missing")
    ap.add_argument("--json", action="store_true", help="machine-readable output for --check")
    ap.add_argument("--model", default=None, help=f"speech model ({', '.join(WHISPER_REPOS)})")
    ap.add_argument("--cpu", action="store_true", help="force the CPU-only stack")
    ap.add_argument("--gpu", action="store_true", help="force the CUDA stack")
    ap.add_argument("--no-diarize", action="store_true",
                    help="skip speaker separation (no HuggingFace token needed)")
    ap.add_argument("--token", default="", help="HuggingFace token, saved for later runs")
    ap.add_argument("--no-shortcuts", action="store_true")
    args = ap.parse_args(argv)

    from .config import load_config
    cfg = load_config()
    opts = Options(
        model=args.model or str(cfg.get("asr_model", "large-v3")),
        gpu=True if args.gpu else (False if args.cpu else None),
        diarize=not args.no_diarize and bool(cfg.get("diarize", True)),
        hf_token=args.token,
        shortcuts=not args.no_shortcuts,
    )

    if args.check or not args.install:
        comps = detect(opts)
        if args.json:
            print(json.dumps([c.__dict__ for c in comps], indent=2))
        else:
            gpu = winenv.gpu_info()
            print(f"Sinribe setup — {APP_ROOT}")
            print(f"  {gpu.summary()}")
            for c in comps:
                mark = "OK  " if c.ok else ("MISS" if c.required else "----")
                print(f"  [{mark}] {c.title:38s} {c.detail}")
            keys, gb = plan(opts)
            print(f"\n  {len(keys)} step(s) to run, about {gb:.1f} GB to download"
                  if keys else "\n  Everything is installed.")
        return 0 if is_ready(opts) else 1

    progress = _console_progress()
    try:
        install(opts, progress)
    except Cancelled:
        print("\ncancelled")
        return 130
    except SetupError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
