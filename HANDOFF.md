# HANDOFF — read this first

Orientation for a session that does not share this one's chat history. `README.md` is the
user-facing doc; this is the "what is actually proven, and what would bite you" companion.

This repository is the **Windows port** of `~/Sinribe` (Linux, private repo
`darian-gajgic/Sinribe`). The pipeline, merge logic, renderers and interface are the same code.
What changed is everything about how the program finds, installs and loads its dependencies —
which on Windows is most of the work.

**Last synced with the Linux tree 2026-08-07**, bringing across the whole accuracy effort: the
quality slider (`presets.py`), multi-pass voting (`rover.py`), the voice-print speaker refiner
(`workers/refine_worker.py`), the "Worth a listen" review section, the collapsed-decode guard, and
the `sinribe.eval` scorer. The two repositories are forks, not branches, and neither has the other
as a remote — every sync is done by hand, so when a bug is fixed in one, check whether it still
exists in the other.

## State: proven vs unproven

| | Status |
|---|---|
| Ports cleanly, imports on a bare interpreter | ✅ |
| 306 tests, ruff clean (`F,E,W,B`, 100 cols) | ✅ run on Linux |
| Component detection, install planning, download-size estimates | ✅ unit-tested |
| GPU → PyTorch build matrix | ✅ unit-tested against the published wheel index |
| Both setup windows construct and drive headlessly | ✅ Qt offscreen + tkinter withdrawn |
| Main window builds, slider sweeps every rung, options round-trip | ✅ Qt offscreen |
| `sinribe.eval` scores a transcript end to end | ✅ on this tree, from Linux |
| Every pinned wheel exists for win_amd64/cp312 | ✅ verified against PyPI and download.pytorch.org |
| FFmpeg/uv/model download URLs resolve | ✅ checked, 2026-07-26 |
| Word error rates in `presets.py` | ✅ measured — **on the Linux box**, and they carry over |
| Realtime factors in `presets.py` | ⚠️ measured on the Linux box only; treat as estimates |
| **Anything actually executed on Windows** | ❌ **never** — no Windows machine was available |
| A real transcription end to end on Windows | ❌ |
| The German-model conversion step | ❌ never run on any platform from this tree |
| The wizard's appearance under tkinter on Windows | ❌ |
| Desktop/Start Menu shortcut creation | ❌ |

Treat the first Windows run as the acceptance test. `Sinribe.bat --check` is the cheapest first
move; it exercises detection without downloading anything.

**On the numbers.** The error rates were produced by decoding fixed weights with fixed settings
and are deterministic — they are properties of the models, not of the machine, and they transfer.
The *timings* do not. They came off one RTX 5070 Ti on one day, and even there the same settings
measured 50× and 20× depending on what else was resident on the GPU. Nothing in this repository
has ever been timed on Windows. `config.observed_rtf` exists for exactly this reason: after the
first finished job at a given rung, the slider stops quoting the shipped figure and quotes the
user's own machine. Do not "correct" the shipped numbers from a single Windows run — feed the
sweep a reference and measure properly, or leave them.

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
SinribeEval.bat          the scorer / sweep runner (runs in .venv, needs no extra plumbing)
tools/
  bootstrap.ps1          pre-Python: fetch uv -> install CPython 3.12 -> open the wizard
  setup_wizard.py        first-run window (tkinter, because PySide6 is not installed yet)
  make_shortcuts.ps1     Desktop + Start Menu .lnk, per-user, no admin
  make_icon.py           regenerates assets/sinribe.ico (stdlib zlib+struct, no Pillow)
  _ct2conv/              THROWAWAY venv for the German conversion; deleted after use
sinribe/
  winenv.py              ← every OS-specific decision lives here
  setup_core.py          ← detect + install engine, stdlib-only, shared by both setup surfaces
  presets.py             the quality ladder: each rung's decode settings + its measured WER
  rover.py               word-by-word majority vote across several decodes
  pipeline.py            orchestrator: stages, progress, checkpoints, cancel, VRAM tiers
  merge.py               word -> speaker alignment   ← accuracy lives here
  eval/                  accuracy measurement — nothing here runs during a normal transcription
    reference.py           load .docx/.odt/.md/.srt, incl. multi-part restarting timestamps
    normalize.py           German-aware tokenisation; spelling variants must not count as errors
    align.py               difflib anchors + edit-distance DP with free compound merges
    score.py               WER, per speaker, over time, speaker attribution, confusion pairs
    sweep.py               decode N ways, score each, one table
    ceiling.py             how much a perfect chooser would win — the "keep tuning?" answer
  ui/setup_dialog.py     the in-app setup window (Qt front end, same engine)
  workers/asr_worker.py  faster-whisper subprocess   (.venv,      CUDA 12)
  workers/diar_worker.py pyannote subprocess         (.venv-diar, CUDA 13 or CPU)
  workers/refine_worker.py per-sentence voice-print check (.venv-diar, embedding model only)
```

Two setup surfaces, one engine: `tools/setup_wizard.py` (tkinter, runs before any venv exists)
and `sinribe/ui/setup_dialog.py` (Qt, runs from inside the app). Both call `setup_core`, so they
cannot disagree about what "installed" means. Fix detection bugs there, never in a front end.
`setup_core.remember_model()` exists for the same reason — both front ends have to make the same
decision about when a download choice becomes a pinned preference.

## Accuracy: what is measured, and how to measure it

`sinribe/eval/` scores a transcript against a human reference. Use it before and after **any**
change to decode settings — the Linux tree has twice adopted a change that sounded obviously right
and measurably made accuracy worse.

```bat
SinribeEval.bat reference.docx transcript.md
SinribeEval.bat --sweep variants.json audio.wav reference.docx -o results.json
```

The benchmark is a 65-minute Bavarian-accented German interview recorded on a phone across a room.
It lives on the Linux box (`~/Sinribe-Transcripts/Kev/`) and is **not in this repository** — the
figures below are inherited measurements, not something this tree can reproduce on its own. To
re-validate anything here you need that file, or your own reference recording.

**Baseline, `large-v3` sequential, beam 5:** WER **21.3 %**, 87.0 % of words correct, speaker
attribution **98.9 %**. The ladder in `presets.py` runs from 23.1 % at 85× realtime down to 20.4 %
at 11×, and the ★ sits at the bend.

### Things that sounded right and measured worse

Recorded so they are not re-tried. Each cost a sweep run to disprove.

| Change | Expectation | Measured |
|---|---|---|
| `condition_on_previous_text=True` | Whisper's LM context fixes function-word errors | **21.3 % → 24.4 %.** Deletions 395 → 749, attribution 98.9 % → 94.2 % |
| `large-v3-german` (primeline fine-tune) | German-only training beats multilingual on German | **21.3 % → 24.8 %.** More substitutions *and* more insertions |
| `large-v3-german` + context | Both fixes together | **99.2 % — the decode collapsed** |
| `hotwords` with the interview's proper nouns | The documented fix for mangled names | **21.3 % → 22.3 %.** "Dieter" recovered; *Fujitsu* still wrong |
| Wider beam (8) with `patience=2` | More search on hard audio | 21.7 %, and slower |
| Beam 20 with `patience=2` | Much more search | 21.6 % at **5.6× the compute** |
| More VAD padding (700 ms / 300 ms silence) | Catch the quiet run-ins | 21.7 % |
| `vad_filter=False` | Stop the VAD trimming speech | 22.7 % |
| `audio_filter="clean"` (highpass + level) | Help the voice further from the phone | 21.5 % — a wash |
| `audio_filter="denoise"` (+ FFT denoise) | Same, harder | 22.3 % — denoising removes signal whisper uses |
| An `initial_prompt` full of German fillers | Push whisper toward verbatim output | Worse. Whisper normalises disfluencies by design |

The lesson underneath most of that row set: **beam search explores the model's own hypothesis
space.** It helps when the right answer was in the model's head and the search missed it. Here the
errors are acoustic — the distribution is itself wrong about whether it heard *das* or *es* — so
searching a wrong distribution harder returns the same wrong answer at five times the price. Every
"spend more compute" knob that failed, failed for that reason.

What *does* convert compute into accuracy is **voting across several genuinely different decodes**
(`rover.py`): 21.3 % for one pass, 20.4 % for three. The single biggest step is adding a different
*model*, not re-chunking the same one. That is the whole design of the slow half of the slider.

The floor is **8.7 % of words that no configuration tried gets right**. Accept it and spend the
effort on the review workflow instead — `SinribeEval.bat --ceiling` re-derives that number.

### `large-v3-german` returns an empty transcript when given hotwords

Not a subtle degradation — zero words, for the whole recording, with a hotword string as short as
a single letter. Reproduced on two different files. `large-v3-turbo-german` from the same publisher
is fine, so this is that fine-tune having lost the ability to condition on a prompt prefix, not a
conversion fault. `config.MODELS_WITHOUT_HOTWORDS` records it: the pipeline drops the hotwords and
logs loudly rather than failing the job, and the interface disables the field when that model is
selected. If a future model shows the same behaviour, add it to that set.

### Whisper can collapse silently — there is a guard

`large-v3-german` with context conditioning once returned **156 words for a 65-minute interview**
and exited 0. `pipeline.speech_coverage()` now compares diarized speech time against the time
actually spanned by words: healthy runs score 100 %, that one scored 0.8 %. Below 50 % the job logs
a warning. The threshold is deliberately far from the healthy band — this is a "something is
catastrophically wrong" detector, not a quality metric.

## Divergences from the Linux build

Beyond the platform layer, deliberately:

- **Diarization is optional.** Windows users may not want to create a HuggingFace account. With
  it off, the stage is skipped and `merge()` attributes everything to one speaker — that path
  already existed for single-speaker files. `config["diarize"]` drives it, and it also drops the
  refine stage, since there are no speakers to build voice prints for.
- **Models are downloaded, not assumed.** The Linux box already had `large-v3` in its HF cache.
  Here the model is a setup step, and the pipeline can still fetch one on demand if the user
  switches models later (`_transcribe_once` decides, `job["offline"]` carries it).
- **`resolve_model()` returns a full repository id**, not a bare size name. faster-whisper would
  map `large-v3` itself, but then setup and the worker would each be trusting their own mapping;
  with `HF_HUB_OFFLINE=1` a divergence between the two is a job that fails after decoding the
  audio. `config.WHISPER_REPOS` is the single answer to "which repository is that".
- **The German models are converted here, by setup**, not by a shell script the user runs.
  `setup_core.ensure_german_models` builds a throwaway venv with CPU torch, runs
  `ct2-transformers-converter`, and deletes the venv again. They are optional and off by default.
  This is the one step in the file that has never been executed anywhere.
- **`asr_device` config key** (`auto|cuda|cpu`) and `_start_tier()`, which sends a job to the CPU
  when the CUDA runtime was never installed. Without that check `free_vram_mib()` reports plenty
  of memory on a CPU-only install and the worker dies on a missing DLL.
- **Model cache under `%LOCALAPPDATA%\Sinribe\models`** unless `HF_HOME` is already set, so
  uninstalling is two folders rather than a hunt. The CTranslate2 conversions sit beside it in
  `ct2-models\`, deliberately *outside* the HuggingFace cache: they are not HuggingFace artefacts,
  they take minutes to rebuild, and a cache clean aimed at the hub must not sweep them away.
- **`config._migrate` is narrower than the Linux one.** This port never shipped the `asr_mode`
  key, so the only ambiguous value it can find on disk is an `asr_model` of `large-v3` — the old
  default, which records no preference and is therefore turned back into `auto`.
- **Three small fixes worth porting back to Linux:** re-export derives its filename stem from
  `markdown_path` rather than `Path(source).stem` (which is wrong for a URL source); the speaker
  preview player and the "Transcribed X" label use `local_file`/`title` instead of `source`, so
  both work for downloaded episodes.

## Design notes worth knowing before changing anything

- **Attribution is sentence-level, decided from word timings.** Whisper *segments* straddle speaker
  changes, so they cannot be the unit. Individual *words* are too small in the other direction: one
  200 ms token against a boundary that is routinely 300 ms late tears a clause in half. Each word
  is assigned by maximal overlap, then every **sentence** takes its duration-weighted majority
  (`merge.vote_sentences`). `sentence_atomic: false` restores word-level.
- **Diarization is a prior, not an oracle.** The refine stage builds one voice print per speaker
  from the diarization's long *isolated* turns, embeds each sentence's own audio, and overrules the
  label when the cosine gap to the runner-up clears `refine_margin`. This is the only thing that
  repairs a span the diarizer labelled outright wrong — regrouping cannot, because by then the
  wrong label is the only evidence left. Measured margins are bimodal (confident corrections 0.3+,
  coin flips under 0.1), which is why 0.15 is safe.
- The refine stage loads **only** the embedding half of the pipeline and is non-fatal by
  construction: any failure logs and leaves the plain diarization result in place. On Windows that
  matters more than on Linux — a machine that never installed the speaker-separation environment
  reaches this stage and must lose a few seconds, not the transcript.
- **`refine_worker` calls `enable_ffmpeg_dlls()` before importing pyannote**, for the same reason
  `diar_worker` does. It reads its audio with soundfile and never touches torchcodec, but pyannote
  imports torchcodec unconditionally and the import has to succeed.
- **Decode settings live in `presets.py`, not scattered through the config.** The quality slider
  picks a rung; `presets.decode_settings(cfg)` resolves it, and `decode_overrides` lets the sweep
  drive one knob without inventing a config key per experiment.
- **`asr_mode: accurate` decodes sequentially instead of batching.** Batching splits at VAD
  boundaries and decodes the pieces independently, which is where edge words vanish: batched
  21.6 % WER at 50×, sequential 21.3 % at 13×. One thing that turned out **not** to be true:
  batching was assumed to loosen word timestamps enough to hurt attribution. It does not — 99.0 %
  batched against 98.9 % sequential. Don't avoid batching for that reason.
- pyannote 4 returns a `DiarizeOutput`, not an `Annotation`. Use
  `.exclusive_speaker_diarization` — no overlapping turns, built for exactly this case.

## STOP-don't-fix rules

- **Do not** "simplify" the two virtualenvs into one, or move `enable_cuda_dlls()` out of the
  worker into the parent process. Both are load-bearing; see the constraints above.
- **Do not** install `transformers` or PyTorch into `.venv` to avoid the throwaway conversion
  environment. That is exactly the CUDA 12 / CUDA 13 collision the two-environment split exists to
  prevent, and it would break transcription to save a few hundred MB of temporary download.
- **Do not** switch the FFmpeg download to a static build because it is smaller. torchcodec
  needs the DLLs.
- **Do not** pin `yt-dlp`. Sites change delivery constantly and a pinned copy silently rots. If
  a link stops working, update it first, then debug.
- **Do not** re-enable `condition_on_previous_text`, an `initial_prompt`, or a wider beam without
  re-running the sweep. All three are in the measured-worse table above.
- **Do not** print a realtime factor as if it had been measured on Windows. Nothing here has.
  `config.observed_rtf` is how a real number gets into the interface.
- `pyannote`'s batch sizes look like they default to 1 in the constructor signature — they do
  not; `community-1`'s config sets both to 32. Don't "fix" that.
- A cancelled stage must **never** write a checkpoint. An early bug in the Linux build did, and
  a resume then silently produced a transcript with zero speaker separation. Guarded in
  `_run_worker` and in every stage.
- The ASR checkpoint key must contain **every** setting that can change the words. A knob passed
  to the worker but left out of `_transcribe_once`'s `settings` dict makes a re-run silently reuse
  the previous result — during a sweep that reads as "the setting does nothing" rather than as a
  bug. `tests/test_pipeline.py::TestCheckpointShape` guards it.

## Working on it from Linux

The package imports and the whole suite runs here; only the app itself needs Windows.

```bash
python3 -m pytest tests/ -q                                    # 306 tests, no GPU, no network
python3 -m ruff check sinribe/ tests/ tools/ --select F,E,W,B --line-length 100
python3 -m sinribe.setup_core --check                          # detection against this machine
python3 tools/make_icon.py                                     # regenerate the icon
```

Both windows can also be built headlessly from here, which is how they were checked:

```bash
QT_QPA_PLATFORM=offscreen python3 -c "
from PySide6.QtWidgets import QApplication
from sinribe.config import load_config
from sinribe.ui.main_window import MainWindow
app = QApplication([]); MainWindow(load_config()).show(); print('ok')"
```

(The tkinter wizard needs a real display; construct `Wizard(tk, ttk)` and call `root.withdraw()`
before `refresh()` so it never appears on the user's desktop.)

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
5. Transcribe the same file again with the quality slider at a voting rung. Proves the multi-pass
   path, the per-pass checkpoints, and that the refine worker loads its embedding model.
6. Only then a long file, and record the realtime factor. The README quotes the Linux number and
   says so; the app will learn the real one by itself after this run.
7. Last, and only if German is in scope: tick **Add the German speech models** and let the
   conversion run. It is the one step nobody has ever executed. Expect it to be where a problem
   is, not to be surprised by one.
