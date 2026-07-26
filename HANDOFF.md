# HANDOFF — read this first

Orientation for a session that does not share this one's chat history. `README.md` is the
user-facing doc; this is the "what is actually proven, and what would bite you" companion.

This repository is the **Windows port** of `~/Sinribe` (Linux, private repo
`darian-gajgic/Sinribe`). The pipeline, merge logic, renderers and interface are the same code.
What changed is everything about how the program finds, installs and loads its dependencies —
which on Windows is most of the work.

## State: proven vs unproven

| | Status |
|---|---|
| Ports cleanly, imports on a bare interpreter | ✅ |
| 145 tests, ruff clean (`F,E,W,B`, 100 cols) | ✅ run on Linux |
| Component detection, install planning, download-size estimates | ✅ unit-tested |
| GPU → PyTorch build matrix | ✅ unit-tested against the published wheel index |
| Every pinned wheel exists for win_amd64/cp312 | ✅ verified against PyPI and download.pytorch.org |
| FFmpeg/uv/model download URLs resolve | ✅ checked, 2026-07-26 |
| **Anything actually executed on Windows** | ❌ **never** — no Windows machine was available |
| A real transcription end to end on Windows | ❌ |
| The wizard's appearance under tkinter on Windows | ❌ |
| Desktop/Start Menu shortcut creation | ❌ |

Treat the first Windows run as the acceptance test. `Sinribe.bat --check` is the cheapest first
move; it exercises detection without downloading anything.

## The three constraints everything else follows from

**1. CUDA DLLs are found per-process, not per-environment.** On Linux the launcher exports
`LD_LIBRARY_PATH` and children inherit it. Windows has no equivalent that works: since Python
3.8, PATH is deliberately excluded from the search that resolves an extension module's dependent
DLLs. The only mechanism is `os.add_dll_directory()` called *inside* the process that will load
the extension — hence `winenv.enable_cuda_dlls()`, called by `asr_worker` itself before it
imports `faster_whisper`. Get it wrong and the model constructs perfectly, then dies on the
first `encode()` with `Library cublas64_12.dll is not found`. (`_asr_env` also puts the
directories on PATH; that is belt-and-braces for libraries that dlopen their own dependencies,
not the primary mechanism.)

**2. pyannote 4 imports torchcodec, and torchcodec needs FFmpeg's shared libraries.** The
torchcodec wheel is 3 MB and ships `libtorchcodec_core{4..8}.dll`, which dlopen avcodec /
avformat / avutil at import time and raise if none is found. Windows has no system FFmpeg. That
is why setup downloads the **shared** FFmpeg build (`av*.dll` next to `ffmpeg.exe`), not the
smaller static one, and why `diar_worker` calls `winenv.enable_ffmpeg_dlls()` before importing
pyannote. Sinribe never decodes anything with torchcodec — the worker is handed an
already-decoded waveform — but the import still has to succeed.

**3. Two virtualenvs, same as Linux, for the same reason.** `ctranslate2` is a CUDA 12 binary;
torch is (usually) a CUDA 13 one. They cannot share a process. The split also means the
diarization process *exits* between stages, which is the only reliable way to hand a GPU's
memory back — deleting the model in-process does not return the CUDA context.

## Version pins, and why these

Checked against PyPI and `download.pytorch.org` on 2026-07-26; every one has a
win_amd64 wheel for cp312.

- **Python 3.12** — the newest version with a wheel for *every* pin here. 3.13+ loses one or
  another depending on the week.
- **torch 2.13.0 + torchaudio 2.11.0 + torchcodec 0.15.0**, always installed together from one
  index. torchaudio's version stopped tracking torch's when audio I/O moved to torchcodec; that
  mismatch is expected, not a mistake.
- **cu128 is deliberately absent** from `winenv.TORCH_TARGETS`. That index carries no torchcodec
  at all, so a cu128 environment cannot even import pyannote. Only cu130, cu126 and cpu publish
  the whole trio.
- **A Blackwell card on a pre-580 driver gets the CPU build**, not cu126. cu126 has no sm_120
  kernels, and hoping PTX JIT covers the gap is exactly the failure mode this pipeline is built
  to avoid — it surfaces eight minutes into a four-hour job. The reason string tells the user a
  driver update fixes it.
- **FFmpeg pinned to the n7.1 branch**, not `master-latest`. torchcodec ships loaders for FFmpeg
  4–8; a master build rolling to 9 would break a machine that had been working.

## Layout

```
Sinribe.bat              double-click target: launches the app, or bootstraps setup
Sinribe.vbs              same thing without the console flash (shortcuts point here)
Setup.bat                open setup at any time
tools/
  bootstrap.ps1          pre-Python: fetch uv -> install CPython 3.12 -> open the wizard
  setup_wizard.py        first-run window (tkinter, because PySide6 is not installed yet)
  make_shortcuts.ps1     Desktop + Start Menu .lnk, per-user, no admin
  make_icon.py           regenerates assets/sinribe.ico (stdlib zlib+struct, no Pillow)
sinribe/
  winenv.py              ← every OS-specific decision lives here
  setup_core.py          ← detect + install engine, stdlib-only, shared by both setup surfaces
  pipeline.py            orchestrator: stages, progress, checkpoints, cancel, VRAM tiers
  merge.py               word -> speaker alignment   ← accuracy lives here
  ui/setup_dialog.py     the in-app setup window (Qt front end, same engine)
  workers/asr_worker.py  faster-whisper subprocess   (.venv,      CUDA 12)
  workers/diar_worker.py pyannote subprocess         (.venv-diar, CUDA 13 or CPU)
```

Two setup surfaces, one engine: `tools/setup_wizard.py` (tkinter, runs before any venv exists)
and `sinribe/ui/setup_dialog.py` (Qt, runs from inside the app). Both call `setup_core`, so they
cannot disagree about what "installed" means. Fix detection bugs there, never in a front end.

## Divergences from the Linux build

Beyond the platform layer, deliberately:

- **Diarization is optional.** Windows users may not want to create a HuggingFace account. With
  it off, the stage is skipped and `merge()` attributes everything to one speaker — that path
  already existed for single-speaker files. `config["diarize"]` drives it.
- **Models are downloaded, not assumed.** The Linux box already had `large-v3` in its HF cache.
  Here the model is a setup step, and the pipeline can still fetch one on demand if the user
  switches models later (`_transcribe` decides, `job["offline"]` carries it).
- **`asr_device` config key** (`auto|cuda|cpu`) and `_start_tier()`, which sends a job to the CPU
  when the CUDA runtime was never installed. Without that check `free_vram_mib()` reports plenty
  of memory on a CPU-only install and the worker dies on a missing DLL.
- **Model cache under `%LOCALAPPDATA%\Sinribe\models`** unless `HF_HOME` is already set, so
  uninstalling is two folders rather than a hunt.
- **Three small fixes worth porting back to Linux:** re-export derives its filename stem from
  `markdown_path` rather than `Path(source).stem` (which is wrong for a URL source); the speaker
  preview player and the "Transcribed X" label use `local_file`/`title` instead of `source`, so
  both work for downloaded episodes.

## STOP-don't-fix rules

- **Do not** "simplify" the two virtualenvs into one, or move `enable_cuda_dlls()` out of the
  worker into the parent process. Both are load-bearing; see the constraints above.
- **Do not** switch the FFmpeg download to a static build because it is smaller. torchcodec
  needs the DLLs.
- **Do not** pin `yt-dlp`. Sites change delivery constantly and a pinned copy silently rots. If
  a link stops working, update it first, then debug.
- `pyannote`'s batch sizes look like they default to 1 in the constructor signature — they do
  not; `community-1`'s config sets both to 32. Don't "fix" that.
- A cancelled stage must **never** write a checkpoint. An early bug in the Linux build did, and
  a resume then silently produced a transcript with zero speaker separation. Guarded in
  `_run_worker` and in both stages.

## Working on it from Linux

The package imports and the whole suite runs here; only the app itself needs Windows.

```bash
python3 -m pytest tests/ -q                                    # 145 tests, no GPU, no network
python3 -m ruff check sinribe/ tests/ tools/ --select F,E,W,B --line-length 100
python3 -m sinribe.setup_core --check                          # detection against this machine
python3 tools/make_icon.py                                     # regenerate the icon
```

`winenv` keeps POSIX branches purely so this works. Note that `appdata_dir()` returns
`~/.config/sinribe-windows` on Linux, deliberately not `~/.config/sinribe` — running these tests
must not read or overwrite a working install of the original.

## First moves on a real Windows machine

1. `Sinribe.bat --check` — detection only, no downloads. Confirms paths and GPU probing.
2. Full setup with `--model tiny` and speakers off. Smallest possible install; proves uv, the
   private Python, FFmpeg, the venv and the launcher.
3. Transcribe a two-minute file. Proves ctranslate2 found its CUDA DLLs (the most likely failure
   point on this platform).
4. Add the HuggingFace token and speaker separation. Proves the torchcodec/FFmpeg import chain
   (the second most likely failure point).
5. Only then a long file, and record the realtime factor — the README currently quotes the Linux
   number and says so.
