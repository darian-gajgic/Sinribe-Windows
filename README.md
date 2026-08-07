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
| German models | ~5 GB | Two German-only models. Optional, off by default |

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

At the end of the transcript, unless you turn it off, comes a **Worth a listen** section: the
passages the recogniser was least sure of, with timestamps and the doubtful words in bold.

```markdown
## Worth a listen

- **[00:14:52]** Person 2: schon ein elementares Element. **Bei einer** Nasenatmung zum Beispiel
- **[00:31:07]** Person 1: wenn das in eurem **Ruf steht.**
```

Fifty-seven percent of the words it flags are genuinely wrong, so on a difficult recording this is
where an hour of checking gets you the most back — considerably more than any decode setting will.

### From the command line

```bat
Sinribe.bat recording.m4a -o "D:\Transcripts"
Sinribe.bat "https://www.youtube.com/watch?v=..." -o "D:\Transcripts"
Sinribe.bat recording.mp3 --speakers 2 --language de
Sinribe.bat recording.mp3 -q 5                     :: slower, more accurate
Sinribe.bat recording.mp3 --hotwords "Fujitsu, VR-Bank, Dr. Meier"
Sinribe.bat --check                                :: what is installed
Sinribe.bat --setup                                :: open setup again
```

Run with arguments and it stays on the console so you can watch the progress bar. Run it with
none and it opens the window.

---

## Options

| Option | Default | Notes |
|---|---|---|
| Quality | 10× realtime ★ | The slider *is* the speed, 84× down to 1×. Each position shows its measured error rate |
| Model | `auto` | The quality setting picks one. Override with `large-v3`, `small`, a German model… |
| Names & terms | empty | Proper nouns to expect. Helps with names the model nearly gets; see below |
| Language | Auto-detect | Or pin it, which is slightly faster and safer on quiet openings |
| Separate and label speakers | on | Off skips the whole diarization half — no token needed |
| Speakers | Auto-detect | Or set exactly N, or a min–max range |
| Voice-print speaker check | on | Re-checks each sentence against each voice. Costs a few seconds |
| Flag uncertain passages | on | Appends the spots worth checking by ear, with timestamps |
| Sidecar `.json` | on | Needed for instant re-export |
| `.srt` / `.vtt` | on | Speaker-prefixed cues |
| LLM chapters + summary | off | Uses your local Ollama (`:11434`); still offline |

Settings live in `%APPDATA%\Sinribe\config.json`.

### What the quality slider is, and what it is not

Its value is the target realtime factor, so "20×" means an hour of audio in three minutes. Each
position carries the error rate it actually scored on a hard German interview, and the estimated
time for the file you have loaded — so it reads *"20.4× realtime · 21.3% errors · ≈ 3m 13s for
this file"* rather than *"high quality"*.

| Slider | Passes | Speed | Errors | |
|---|---|---|---|---|
| 84× | 1 | 85× | 23.1 % | turbo German model, batched — needs the optional German models |
| 50× | 1 | 51× | 21.6 % | `large-v3`, batched |
| 20× | 1 | 20× | 21.3 % | `large-v3`, one pass in order |
| 14× | 2 | 15× | 20.8 % | + a batched pass, voted |
| **10× ★** | 3 | 11× | **20.4 %** | + a second *model*, voted |
| 5× | 5 | 5.4× | 20.1 % | five passes, two models |
| 2× | 7 | 2.7× | 20.1 % | seven passes |
| 1× | 9 | 1.6× | 20.1 % | nine passes, incl. a very wide beam |

**Read the speeds as estimates, not promises.** They were measured on the Linux original, on one
RTX 5070 Ti, and have never been timed on Windows. The error rates carry over — same weights, same
settings, same deterministic result — the timings do not. After your first job at a given setting
the app replaces the shipped figure with what your machine actually did.

Without the optional German models the top rung is unavailable and the voting rungs simply run one
pass fewer; the interface says which is which.

### How the slow half works, and why the fast half doesn't just do it

Below 20× the audio is transcribed **several times with different settings, and every word is
decided by majority vote** (`sinribe/rover.py`). Passes that chunk the audio differently mishear
*different* words, so where two agree against a third the majority is usually right. That takes
21.3 % down to 20.1 %.

The single biggest step is adding a **different model** (20.8 % → 20.4 %), not re-running the same
one with different chunking. Two models trained differently make different mistakes; one model
re-segmented mostly makes the same ones.

What does *not* work is simply searching harder. A beam of 20 costs 5.6× the compute and returns
21.6 % — worse than the default. Beam search explores one model's own hypotheses, so it only
rescues an answer the model already had; here the model's probabilities are themselves wrong about
whether it heard *das* or *es*, and searching a wrong distribution harder returns the same wrong
answer at five times the price.

Returns flatten at five passes and turn back up past nine, as weaker decodes start outvoting
better ones. The slowest positions are kept so you can confirm that on your own recordings; the
★ sits where the curve bends.

### Names & terms — worth trying, not a cure

Whisper mangles proper nouns it has no reason to predict. In one interview *Fujitsu* came back as
"fiuzi", "jitze" and "future service" — never once correctly — while *VR-Bank* became "Feuerbank".
Listing the expected names biases the recogniser's vocabulary:

```
Fujitsu, Siemens, VR-Bank, Ramada, Dr. Mühlbauer
```

Measured on that interview it is a weak lever. A name the recogniser was already nearly getting
("Dieter", right once out of three) came good; *Fujitsu* and *VR-Bank* did not move at all, and
overall WER rose slightly, 21.3 % → 22.3 %. A vocabulary hint is no match for a word the model
confidently mis-hears. Try it, then check whether it earned its place — `SinribeEval.bat` will say.

### The German models are optional, and not automatically better

`large-v3-german` and `large-v3-turbo-german` are fine-tuned on German alone, where vanilla
`large-v3` is multilingual. That sounds like a straight upgrade and is not one: on a hard
far-field Bavarian interview the German fine-tune scored **24.8 % WER against large-v3's 21.3 %**,
with more substitutions *and* more invented words. Its published numbers come from read speech,
and the narrower training appears to cost large-v3's robustness on spontaneous, distant, accented
audio.

They are worth trying on clean, close-miked German — and `SinribeEval.bat` will tell you, which is
the point. Tick **Add the German speech models** in the setup window, or:

```bat
.venv\Scripts\python.exe -m sinribe.setup_core --install --german-models
```

The upstream models ship in transformers format, so this downloads them and converts them to
CTranslate2, which takes a few minutes of processor time. The conversion runs in a throwaway
environment on purpose — see `HANDOFF.md` for why nothing may install PyTorch into `.venv`.

---

## Measuring accuracy

Transcription settings are easy to argue about and easy to measure, so Sinribe ships the scorer:

```bat
SinribeEval.bat reference.docx transcript.md
```

It aligns the two word by word and reports WER, per-speaker accuracy, how the error rate moves
over the recording, speaker-attribution accuracy, and the most frequent confusions. The reference
can be `.docx`, `.odt`, `.md`, `.txt`, `.srt` or `.vtt`; timestamps that restart part-way through
(a document stitched from two recordings) are handled, and `(unverständlich)` markers are dropped
rather than counted against the recogniser. German spelling variants — `softskills`/`soft skills`,
`gernhabt`/`gern habt` — score as correct, because they are.

To compare settings instead of grading one run:

```bat
SinribeEval.bat --sweep variants.json audio.wav reference.docx -o results.json
```

Each variant re-decodes the audio and is scored; the decoded audio and the diarization are shared
between variants, so only the recognition pass is repeated.

And to answer "is more tuning worth it?" rather than "which of these is best?":

```bat
SinribeEval.bat --ceiling "%LOCALAPPDATA%\Sinribe\sweep" reference.docx
```

That scores every transcript it finds and reports what a *perfect* choice between them would
score. The gap to the best single configuration is the entire prize available to any cleverer
scheme; the words none of them get right are the floor. On the benchmark interview the floor is
8.7 % of all words — fast, broad-dialect, far-field speech that no setting reaches — which is how
two-model consensus decoding got cancelled before it was built.

---

## How fast is it?

On the Linux machine this was ported from — an RTX 5070 Ti laptop — a four-hour recording with
three speakers takes **11 minutes**, about 22× realtime at the fast end of the slider. Windows
numbers on comparable hardware should be in the same range and **have not been measured**; see
`HANDOFF.md`, which is explicit about what has and has not been verified.

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

**Speaker labels get a second opinion.** After transcribing, Sinribe builds a voice print for each
speaker from the diarizer's long, unambiguous turns, then re-checks every sentence against them and
overrules the diarizer where the acoustic match is decisive. It is the only fix for a short answer
that got swallowed into the previous question, and it costs a few seconds. Untick *Voice-print
speaker check* to skip it.

**Chapters and a summary.** If you happen to run [Ollama](https://ollama.com), tick *LLM
chapters + summary* and Sinribe asks your local model for section titles and an overview. Off by
default, entirely local, and skipped silently if Ollama is not there.

**Where things live**

| | |
|---|---|
| Settings, HuggingFace token | `%APPDATA%\Sinribe` |
| Models, checkpoints, downloads | `%LOCALAPPDATA%\Sinribe` |
| Converted German models | `%LOCALAPPDATA%\Sinribe\ct2-models` |
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
