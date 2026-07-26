# Sinribe for Windows

Turns a recording — a lecture, an interview, a podcast — into a readable transcript with each
person labelled:

```
**[00:04:12] Person 1**
So the question I keep coming back to is whether any of this actually scales.

**[00:04:31] Person 2**
It scales. The part that doesn't is the review process around it.
```

Everything runs on your own machine. Audio is never uploaded anywhere. The only time Sinribe
touches the network is when you paste a link for it to download, and the one-time setup that
fetches the program's own components.

Windows port of the Linux original. Same pipeline, same output; the differences are all in how
it installs itself, which is the part Windows makes interesting.

---

## Install

1. Put the `Sinribe-Windows` folder wherever you like — Desktop, Documents, another drive.
   Somewhere you can write to; it does not have to be Program Files.
2. Double-click **`Sinribe.bat`**.
3. A window opens, tells you what is missing, and installs it when you press the button.

That is the whole procedure. There is no installer to run as administrator, nothing is written
to the registry, your PATH is not modified, and any Python you already have is left alone.
Sinribe fetches its own private copy of everything it needs into its own folder.

**Uninstall** = delete the folder, plus `%APPDATA%\Sinribe` (settings) and
`%LOCALAPPDATA%\Sinribe` (models and cache) if you want the disk space back.

### What it downloads, and how much

| | Size | Why |
|---|---:|---|
| uv + Python 3.12 | ~60 MB | A private interpreter, so no existing Python is touched |
| FFmpeg 7.1 | ~76 MB | Reads audio out of any file format |
| Transcription engine | ~450 MB | faster-whisper, ctranslate2, the interface |
| CUDA runtime | ~1.3 GB | Only on machines with an NVIDIA graphics card |
| Speech model | 0.08–3.1 GB | Your choice — see below |
| Speaker separation | ~3.2 GB | PyTorch + pyannote. Optional |

A full GPU install with the best model is around **8 GB**. Setup tells you the number before it
starts, and can be stopped and resumed — anything already downloaded is kept.

### Choosing a speech model

| Model | Size | Good for |
|---|---:|---|
| `large-v3` | 3.1 GB | Best accuracy. Wants an NVIDIA card |
| `distil-large-v3` | 1.5 GB | Nearly as good, about twice as fast. English only |
| `medium` | 1.5 GB | A sensible middle on a mid-range card |
| `small` | 0.5 GB | The right default without a GPU |
| `tiny` | 0.08 GB | Draft quality; useful to test that everything works |

Setup picks `large-v3` when it sees a usable graphics card and `small` when it does not. You can
change it later from the **Setup** button in the app.

### Speaker names need a free HuggingFace token

The speaker-separation model (pyannote) is free but gated — its authors want you to accept their
conditions before downloading it. One time, five minutes:

1. Create an account at <https://huggingface.co/join>
2. Open <https://huggingface.co/pyannote/speaker-diarization-community-1> and accept the
   conditions
3. Create a **read** token at <https://huggingface.co/settings/tokens>
4. Paste it into the setup window

Skip it if you would rather not: untick **Separate and label speakers** and Sinribe still
transcribes — the text just arrives without speaker labels. You can add the token later.

---

## Using it

Double-click `Sinribe.bat` (or the desktop shortcut setup creates).

- **Choose a file**, or drag one onto the window. Any format FFmpeg reads: mp3, m4a, wav, flac,
  opus, mp4, mkv, mov…
- **Or paste a link** to a YouTube video or podcast episode. *Check* looks up the title and
  length without downloading anything yet.
- Press **Start transcription** and leave it. Progress, an estimated finish time and a cancel
  button are all there while it runs.
- When it finishes you get a **Speakers** panel: play a sample of each person, type their real
  name, and press **Re-export**. That rewrites the files instantly — nothing is transcribed
  twice.

### What you get

For `interview.m4a` in your output folder:

| File | |
|---|---|
| `interview.md` | The transcript: metadata, talk-time table, timestamped turns |
| `interview.sinribe.json` | Word-level data. Lets you rename speakers and re-export later |
| `interview.srt` / `.vtt` | Subtitles |

### From the command line

```bat
Sinribe.bat recording.m4a -o "D:\Transcripts"
Sinribe.bat "https://www.youtube.com/watch?v=..." -o "D:\Transcripts"
Sinribe.bat recording.mp3 --speakers 2 --language de
Sinribe.bat --check                     :: what is installed
Sinribe.bat --setup                     :: open setup again
```

Run with arguments and it stays on the console so you can watch the progress bar. Run it with
none and it opens the window.

---

## How fast is it?

On the Linux machine this was ported from — an RTX 5070 Ti laptop — a four-hour recording with
three speakers takes **11 minutes**, about 22× realtime. Windows numbers on comparable hardware
should be in the same range and have not been measured; see `HANDOFF.md`, which is explicit
about what has and has not been verified.

Without an NVIDIA card everything still works, on the processor, several times slower than
realtime with `large-v3`. Choose `small` and it is roughly realtime — an hour of audio in about
an hour.

Sinribe adapts to whatever memory the graphics card has free, dropping to a smaller batch size
or precision rather than failing, and falling back to the processor if it must. If another
program is holding the card (a game, a local LLM), a banner in the window says so.

---

## Practical notes

**Interruptions are cheap.** Each stage checkpoints into `%LOCALAPPDATA%\Sinribe\jobs`. Cancel a
three-hour job at 80% and start it again later: it resumes rather than restarts. Downloads
resume too, including during setup.

**Speaker count.** Auto-detection is good but not psychic. If you know there are exactly two
people, say so under Options — it measurably improves attribution.

**Chapters and a summary.** If you happen to run [Ollama](https://ollama.com), tick *LLM
chapters + summary* and Sinribe asks your local model for section titles and an overview. Off by
default, entirely local, and skipped silently if Ollama is not there.

**Where things live**

| | |
|---|---|
| Settings, HuggingFace token | `%APPDATA%\Sinribe` |
| Models, checkpoints, downloads | `%LOCALAPPDATA%\Sinribe` |
| Python, FFmpeg, uv | inside the Sinribe folder (`.venv`, `.venv-diar`, `tools\`) |
| Transcripts | `Documents\Sinribe Transcripts` by default |

**If a YouTube link stops working**, yt-dlp needs updating — sites change constantly:

```bat
tools\uv.exe pip install --python .venv\Scripts\python.exe -U yt-dlp
```

**Unattended install**, for setting up several machines:

```bat
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1 -Unattended -Model small -Cpu -NoDiarize
```

---

## Requirements

- Windows 10 or 11, 64-bit
- ~10 GB of free disk for a full install
- 8 GB RAM (16 GB comfortable for long recordings)
- Optional: an NVIDIA graphics card with a recent driver. Setup detects it and installs the
  right build; without one, everything falls back to the processor

## Licences

Sinribe installs components with their own licences — FFmpeg (GPL build), faster-whisper (MIT),
Whisper models (MIT), PyTorch (BSD), pyannote (MIT, gated), PySide6 (LGPL), yt-dlp (Unlicense).
They are downloaded from their own publishers at setup time, not redistributed here.
