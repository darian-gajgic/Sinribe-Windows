"""Windows platform layer: where things live, how to spawn them, how CUDA DLLs are found.

Every OS-specific decision in Sinribe-Windows is made here so the rest of the package stays the
same code as the Linux original. Three of them are load-bearing and worth stating up front:

1. **CUDA DLLs are found per-process, not per-environment.** On Linux the launcher exports
   LD_LIBRARY_PATH and the child inherits it. Windows has no equivalent that works: since
   Python 3.8, PATH is deliberately excluded from the search used to resolve an extension
   module's dependent DLLs. The only reliable mechanism is `os.add_dll_directory()` called
   *inside* the process that will load the extension — which is why `enable_cuda_dlls()` is
   called by the ASR worker itself rather than being set up by the parent.

2. **Every subprocess must be spawned with CREATE_NO_WINDOW.** The GUI runs under `pythonw.exe`,
   which has no console; without the flag each ffmpeg/ffprobe/nvidia-smi/worker call allocates a
   fresh console window that flashes on screen. A four-hour job spawns enough of them to matter.

3. **ffmpeg is not on a Windows box by default.** It is downloaded into `tools/ffmpeg/bin/` at
   setup time, and every call site resolves it through `ffmpeg_exe()` / `ffprobe_exe()` instead
   of trusting PATH — while still preferring a system copy if the user already has one.

The module imports cleanly on Linux (nothing here is imported lazily or conditionally), so the
test suite runs on either platform; the POSIX branches exist for that reason, not for shipping.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = os.name == "nt"

#: Repository root — the directory holding Sinribe.bat, sinribe/, tools/.
APP_ROOT = Path(__file__).resolve().parent.parent

TOOLS_DIR = APP_ROOT / "tools"
ASSETS_DIR = APP_ROOT / "assets"
ASR_VENV = APP_ROOT / ".venv"
DIAR_VENV = APP_ROOT / ".venv-diar"
FFMPEG_DIR = TOOLS_DIR / "ffmpeg"
UV_EXE = TOOLS_DIR / ("uv.exe" if IS_WINDOWS else "uv")

#: Python 3.12 everywhere: it is the newest version with a win_amd64 wheel for every pin in
#: requirements-asr.txt (ctranslate2, onnxruntime, av) *and* a CUDA torch build on the PyTorch
#: index. 3.13+ loses one or the other depending on the week.
PYTHON_VERSION = "3.12"

# CREATE_NO_WINDOW. Defined by value rather than read from subprocess so this module imports on
# Linux, where the constant does not exist.
CREATE_NO_WINDOW = 0x08000000


# --------------------------------------------------------------------------- process spawning
def no_window() -> dict:
    """Extra subprocess kwargs that keep a child from opening a console window.

    Returns {} on POSIX. Merge into every Popen/run call; see the module docstring for why.
    """
    if not IS_WINDOWS:
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": CREATE_NO_WINDOW, "startupinfo": si}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """subprocess.run with the console-window suppression already applied."""
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("check", False)
    return subprocess.run(cmd, **{**no_window(), **kw})


# --------------------------------------------------------------------------- venv layout
def venv_python(venv: Path, windowed: bool = False) -> Path:
    """Interpreter inside a virtualenv.

    `windowed=True` asks for pythonw.exe — the console-less interpreter the GUI is launched
    with. It only exists on Windows; elsewhere the normal interpreter is returned.
    """
    if IS_WINDOWS:
        return venv / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
    return venv / "bin" / "python"


def venv_site_packages(venv: Path) -> Path:
    """site-packages of a virtualenv, without importing or executing it.

    Windows keeps it at a fixed path; POSIX puts the Python version in the path, so glob for it
    and fall back to the expected name when the venv does not exist yet (detection code asks
    about venvs before they are built).
    """
    if IS_WINDOWS:
        return venv / "Lib" / "site-packages"
    found = sorted((venv / "lib").glob("python3.*"))
    base = found[-1] if found else venv / "lib" / f"python{PYTHON_VERSION}"
    return base / "site-packages"


# --------------------------------------------------------------------------- CUDA DLLs
#: pip's NVIDIA wheels lay their binaries out differently per platform: `nvidia/<pkg>/bin` holds
#: the .dll on Windows, `nvidia/<pkg>/lib` the .so on Linux. Both are listed so a mis-set path
#: is impossible rather than merely unlikely.
_CUDA_WHEEL_DIRS = (
    ("cublas", "bin"), ("cublas", "lib"),
    ("cudnn", "bin"), ("cudnn", "lib"),
    ("cuda_nvrtc", "bin"), ("cuda_nvrtc", "lib"),
    ("cuda_runtime", "bin"), ("cuda_runtime", "lib"),
)


def cuda_dll_dirs(site_packages: Path | None = None) -> list[Path]:
    """Directories inside the ASR venv that hold the pip-installed CUDA runtime libraries.

    Only directories that actually exist are returned, so a CPU-only install yields [].
    """
    sp = Path(site_packages) if site_packages else venv_site_packages(ASR_VENV)
    out: list[Path] = []
    for pkg, sub in _CUDA_WHEEL_DIRS:
        d = sp / "nvidia" / pkg / sub
        if d.is_dir() and d not in out:
            out.append(d)
    return out


def enable_cuda_dlls(site_packages: Path | None = None) -> list[Path]:
    """Make the pip CUDA wheels loadable *by this process*. Returns the directories added.

    ctranslate2 does not look for its CUDA dependencies anywhere useful on its own: the model
    object constructs fine and then dies on the first encode() with "Library cublas64_12.dll is
    not found". On Linux the fix is LD_LIBRARY_PATH in the launcher. On Windows an inherited
    environment variable cannot do it — CPython 3.8+ resolves an extension module's dependent
    DLLs with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which covers System32 and directories registered
    through os.add_dll_directory(), and pointedly not PATH.

    PATH is still prepended as a second line of defence: a library that loads its dependency
    itself via ctypes/LoadLibraryA (rather than through an import table) does use PATH, and the
    two mechanisms do not conflict.

    Call this before importing ctranslate2 / faster_whisper. It is a no-op on POSIX, where the
    launcher's LD_LIBRARY_PATH has already done the job.
    """
    dirs = cuda_dll_dirs(site_packages)
    if not dirs:
        return []
    if IS_WINDOWS and hasattr(os, "add_dll_directory"):
        for d in dirs:
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
        os.environ["PATH"] = os.pathsep.join([str(d) for d in dirs] + [os.environ.get("PATH", "")])
    return dirs


# --------------------------------------------------------------------------- user directories
def _env_dir(var: str, fallback: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else fallback


def appdata_dir() -> Path:
    """Roaming application data: settings and the HuggingFace token live here.

    The POSIX branch deliberately says `sinribe-windows`, not `sinribe`: this port and the Linux
    original are separate programs with separate pins, and running the test suite here must not
    read — let alone overwrite — a working Linux install's settings.
    """
    if IS_WINDOWS:
        return _env_dir("APPDATA", Path.home() / "AppData" / "Roaming") / "Sinribe"
    return Path.home() / ".config" / "sinribe-windows"


def localdata_dir() -> Path:
    """Machine-local application data: caches and downloaded models. Never roams — this
    directory reaches several gigabytes and a roaming profile would try to sync all of it."""
    if IS_WINDOWS:
        return _env_dir("LOCALAPPDATA", Path.home() / "AppData" / "Local") / "Sinribe"
    return Path.home() / ".cache" / "sinribe-windows"


def hf_home() -> Path:
    """Where model weights are cached.

    An existing HF_HOME wins, so a machine that already has a HuggingFace cache (and possibly
    the 3 GB whisper model in it) is reused rather than re-downloaded. Otherwise everything
    Sinribe pulls lands under its own local-data directory, which keeps "uninstall" to deleting
    two folders instead of hunting through %USERPROFILE%\\.cache.
    """
    raw = os.environ.get("HF_HOME")
    if raw:
        return Path(raw)
    return localdata_dir() / "models"


def hf_hub_dir() -> Path:
    """The `hub/` subdirectory of the HuggingFace cache, where `models--org--name` dirs sit."""
    return hf_home() / "hub"


def model_cache_dir(repo_id: str) -> Path:
    """Cache directory HuggingFace would use for `repo_id`, e.g. models--Systran--x."""
    return hf_hub_dir() / ("models--" + repo_id.replace("/", "--"))


def model_is_cached(repo_id: str) -> bool:
    """True if `repo_id` has at least one complete snapshot on disk.

    Checks for a non-empty snapshot directory rather than just the top-level folder: an
    interrupted download leaves the folder and a `blobs/*.incomplete` behind, and treating that
    as "installed" would send the user straight into a confusing runtime failure instead of a
    resumable download.
    """
    snaps = model_cache_dir(repo_id) / "snapshots"
    if not snaps.is_dir():
        return False
    for snap in snaps.iterdir():
        if snap.is_dir() and any(snap.iterdir()):
            return True
    return False


# --------------------------------------------------------------------------- ffmpeg
def _which(name: str) -> Path | None:
    from shutil import which
    found = which(name)
    return Path(found) if found else None


def _bundled(name: str) -> Path | None:
    exe = f"{name}.exe" if IS_WINDOWS else name
    for cand in (FFMPEG_DIR / "bin" / exe, FFMPEG_DIR / exe):
        if cand.is_file():
            return cand
    return None


def ffmpeg_exe() -> str:
    """Path to ffmpeg: the copy Sinribe downloaded, else one already on PATH, else the bare
    name so the resulting error message says "ffmpeg" and not a path that never existed."""
    return str(_bundled("ffmpeg") or _which("ffmpeg") or "ffmpeg")


def ffprobe_exe() -> str:
    return str(_bundled("ffprobe") or _which("ffprobe") or "ffprobe")


def have_ffmpeg() -> bool:
    return bool(_bundled("ffmpeg") and _bundled("ffprobe")) or bool(
        _which("ffmpeg") and _which("ffprobe"))


def ffmpeg_location() -> str | None:
    """Directory to hand yt-dlp as --ffmpeg-location, or None to let it search PATH."""
    exe = _bundled("ffmpeg")
    return str(exe.parent) if exe else None


def enable_ffmpeg_dlls() -> Path | None:
    """Make the bundled FFmpeg *shared libraries* loadable by this process.

    This is not about running ffmpeg.exe — it is about `import pyannote.audio`, which imports
    torchcodec, whose extension dlopens avcodec/avformat/avutil at import time and raises if it
    cannot find any supported FFmpeg. That is why setup downloads the *shared* FFmpeg build
    (avcodec-61.dll and friends next to ffmpeg.exe) rather than the smaller static one.

    Returns the directory that was registered, or None if no bundled FFmpeg is present — in
    which case a system-wide FFmpeg on PATH may still satisfy torchcodec.
    """
    exe = _bundled("ffmpeg")
    if exe is None:
        return None
    d = exe.parent
    if IS_WINDOWS and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(d))
        except OSError:
            return None
    os.environ["PATH"] = os.pathsep.join([str(d), os.environ.get("PATH", "")])
    return d


# --------------------------------------------------------------------------- GPU
@dataclass
class GpuInfo:
    """What nvidia-smi says about the best GPU in the machine. `present` False means CPU-only."""
    present: bool = False
    name: str = ""
    vram_mib: int = 0
    driver: str = ""
    compute_cap: float = 0.0

    @property
    def driver_major(self) -> int:
        try:
            return int(str(self.driver).split(".")[0])
        except (ValueError, IndexError):
            return 0

    def summary(self) -> str:
        if not self.present:
            return "No NVIDIA GPU detected — Sinribe will run on the CPU (much slower)"
        return (f"{self.name} · {self.vram_mib / 1024:.0f} GB VRAM · driver {self.driver} "
                f"· compute {self.compute_cap:.1f}")


def nvidia_smi() -> str | None:
    """Locate nvidia-smi. The driver installs it into System32 (so PATH normally finds it), but
    a machine with a stale PATH still has it at the documented location."""
    found = _which("nvidia-smi")
    if found:
        return str(found)
    if IS_WINDOWS:
        for cand in (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        ):
            if cand.is_file():
                return str(cand)
    return None


_GPU_CACHE: GpuInfo | None = None


def gpu_info(refresh: bool = False) -> GpuInfo:
    """Probe the GPU. Any failure means "assume no GPU" — every caller degrades to CPU, which is
    slow but correct, whereas assuming CUDA on a machine without it fails at the first encode.

    Cached: the answer cannot change while the process runs, and the setup window re-checks its
    component list on every option change — without the cache that is an nvidia-smi launch per
    keystroke, which is visible as lag.
    """
    global _GPU_CACHE
    if _GPU_CACHE is not None and not refresh:
        return _GPU_CACHE
    _GPU_CACHE = _probe_gpu()
    return _GPU_CACHE


def _probe_gpu() -> GpuInfo:
    exe = nvidia_smi()
    if not exe:
        return GpuInfo()
    try:
        out = run([exe, "--query-gpu=name,memory.total,driver_version,compute_cap",
                   "--format=csv,noheader,nounits"], timeout=15)
    except (OSError, subprocess.SubprocessError):
        return GpuInfo()
    if out.returncode != 0 or not (out.stdout or "").strip():
        return GpuInfo()

    best = GpuInfo()
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            vram = int(float(parts[1]))
        except ValueError:
            vram = 0
        try:
            cc = float(parts[3])
        except ValueError:
            cc = 0.0
        cand = GpuInfo(present=True, name=parts[0], vram_mib=vram, driver=parts[2],
                       compute_cap=cc)
        # Multi-GPU laptops report the discrete card and (rarely) an integrated one; the card
        # with the most VRAM is the one CUDA will pick as device 0 for our workloads.
        if cand.vram_mib >= best.vram_mib:
            best = cand
    return best


# --------------------------------------------------------------------------- torch selection
#: torch, torchaudio and torchcodec are installed as one triple from one PyTorch index, because
#: all three load into the same process and torchcodec links against libtorch's ABI. Only
#: combinations that the index actually publishes together for win_amd64/cp312 appear here —
#: notably cu128 is absent: it carries no torchcodec at all, and pyannote 4 imports torchcodec
#: unconditionally, so a cu128 environment cannot even start.
#:
#: torchaudio's version does not track torch's any more (it stopped at 2.11.0 when audio I/O
#: moved to torchcodec); that mismatch is expected, not a mistake.
TORCH_TRIPLE = "torch==2.13.0 torchaudio==2.11.0 torchcodec==0.15.0".split()

CU130 = "https://download.pytorch.org/whl/cu130"
CU126 = "https://download.pytorch.org/whl/cu126"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def torch_target(gpu: GpuInfo) -> tuple[str, list[str], bool, str]:
    """Pick (index_url, package specs, is_cuda, reason) for the diarization venv.

    * CUDA 13 dropped Maxwell, Pascal and Volta, so cu130 needs compute >= 7.5, and its wheels
      want an r580+ driver.
    * cu126 covers Maxwell through Hopper on any r525+ driver.
    * A Blackwell card (compute >= 10.0) is deliberately NOT sent to cu126: that build has no
      sm_120 kernels, and relying on PTX JIT to cover the gap is exactly the kind of "works on
      the bench, fails eight minutes into a four-hour job" behaviour this pipeline avoids. Such
      a machine gets the CPU build plus a message saying a driver update fixes it.

    Anything unrecognised lands on the CPU build. CPU diarization is several times slower than
    GPU — annoying, but it finishes, which a wheel built for the wrong architecture does not.
    """
    if not gpu.present:
        return CPU_INDEX, TORCH_TRIPLE, False, "no NVIDIA GPU detected"
    if gpu.compute_cap >= 7.5 and gpu.driver_major >= 580:
        return CU130, TORCH_TRIPLE, True, "CUDA 13 build"
    if 5.0 <= gpu.compute_cap < 10.0 and gpu.driver_major >= 525:
        return CU126, TORCH_TRIPLE, True, "CUDA 12.6 build"
    if gpu.compute_cap >= 10.0:
        return (CPU_INDEX, TORCH_TRIPLE, False,
                f"driver {gpu.driver} is too old for this GPU — update to 580 or newer for "
                f"GPU speed")
    if gpu.driver_major < 525:
        return (CPU_INDEX, TORCH_TRIPLE, False,
                f"driver {gpu.driver} predates CUDA 12 — update the NVIDIA driver for GPU speed")
    return CPU_INDEX, TORCH_TRIPLE, False, f"GPU compute capability {gpu.compute_cap} is too old"


def cuda_supported(gpu: GpuInfo) -> bool:
    """Can ctranslate2's CUDA 12 build run here?

    ctranslate2 4.8 ships kernels from Maxwell up and, being a CUDA 12 binary, needs an r525+
    driver. This is a separate question from `torch_target` — the two stacks have different
    floors, and on an old-driver machine the ASR half can still be on the GPU when the
    diarization half is not.
    """
    return gpu.present and gpu.compute_cap >= 5.0 and gpu.driver_major >= 525


# --------------------------------------------------------------------------- misc
def is_windowed_launcher() -> bool:
    """True when running under pythonw.exe, i.e. with no console attached.

    Anything that wants to report progress to a human has to know this: under pythonw there is
    nowhere for stdout to go, so the answer belongs in a window or a log file.
    """
    return IS_WINDOWS and Path(sys.executable).name.lower() == "pythonw.exe"
