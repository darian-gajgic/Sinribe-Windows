#!/usr/bin/env python3
"""Diarization worker — runs under .venv-diar (CUDA 13 / torch stack).

This MUST be a separate process in a separate venv: torch 2.13 from PyPI is a cu13 build shipping
libcublas.so.13, while ctranslate2 in the ASR venv hard-requires libcublas.so.12. They cannot
coexist. Running the two GPU stages as sequential processes also caps peak VRAM at
max(diarize, transcribe) instead of the sum, which is what makes this fit on a 12 GB card that
already hosts ollama and the wisprflow daemon.

The pipeline is fed a PRE-DECODED waveform tensor rather than a path, which skips torchcodec /
container handling entirely and reuses the same 16 kHz WAV the ASR worker reads.

Protocol: one JSON job object on stdin, newline-delimited JSON events on stdout, log on stderr.
"""

from __future__ import annotations

import json
import os
import sys
import time

_OOM_MARKERS = ("out of memory", "cuda error", "cublas_status_alloc_failed")

# The only diarization pipeline setup downloads into the offline model cache.
DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"


def emit(**kw) -> None:
    sys.stdout.write(json.dumps(kw, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    sys.stderr.write(f"[diar] {msg}\n")
    sys.stderr.flush()


def is_oom(exc: BaseException) -> bool:
    return any(m in str(exc).lower() for m in _OOM_MARKERS)


class ProgressHook:
    """pyannote calls this as it walks each internal step.

    Signature is (step_name, step_artifact, file=..., total=..., completed=...). We turn the
    per-step counters into one monotonic 0..1 fraction across the known step sequence so the GUI
    bar never jumps backwards when pyannote moves from segmentation to embedding to clustering.
    """

    # Rough share of wall-clock each stage takes; only their relative size matters.
    WEIGHTS = {"segmentation": 0.45, "speaker_counting": 0.05, "embeddings": 0.45,
               "discrete_diarization": 0.05}

    def __init__(self) -> None:
        self._order: list[str] = []
        self._done: dict[str, float] = {}
        self._last_emit = 0.0

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if step_name not in self._order:
            self._order.append(step_name)
            log(f"step: {step_name}")
        frac = 1.0
        if total:
            try:
                frac = max(0.0, min(1.0, float(completed or 0) / float(total)))
            except (TypeError, ValueError, ZeroDivisionError):
                frac = 0.0
        self._done[step_name] = frac

        known = sum(self.WEIGHTS.values())
        overall = sum(self.WEIGHTS.get(k, 0.05) * v for k, v in self._done.items())
        overall = max(0.0, min(1.0, overall / known if known else 0.0))

        now = time.time()
        if now - self._last_emit >= 0.4 or overall >= 1.0:
            self._last_emit = now
            emit(ev="progress", done=overall, total=1.0, step=step_name)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> int:
    raw = sys.stdin.readline()
    if not raw.strip():
        emit(ev="error", msg="no job received on stdin")
        return 2
    job = json.loads(raw)

    wav = job["wav"]
    pipeline_name = job.get("pipeline") or DEFAULT_PIPELINE
    token = job.get("hf_token") or None
    device_pref = job.get("device", "cuda")

    speaker_mode = job.get("speaker_mode", "auto")
    diar_kwargs: dict = {}
    if speaker_mode == "exact":
        diar_kwargs["num_speakers"] = int(job.get("num_speakers", 2))
    elif speaker_mode == "range":
        diar_kwargs["min_speakers"] = int(job.get("min_speakers", 1))
        diar_kwargs["max_speakers"] = int(job.get("max_speakers", 6))

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # pyannote is chatty on import and some of it goes to stdout, which would corrupt our
    # JSON stream. Silence the known offenders before importing. os.devnull rather than
    # "/dev/null": on Windows that path does not exist and pyannote-database would try to parse
    # it as a YAML file.
    os.environ.setdefault("PYANNOTE_DATABASE_CONFIG", os.devnull)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # MUST run before pyannote is imported. pyannote.audio 4 imports torchcodec, whose extension
    # loads FFmpeg's avcodec/avformat/avutil at import time and raises if it finds none. Windows
    # has no system FFmpeg, so point the loader at the copy setup downloaded. Sinribe never
    # decodes anything with torchcodec — this worker is handed an already-decoded waveform — but
    # the import still has to succeed.
    from sinribe.winenv import enable_ffmpeg_dlls

    ff = enable_ffmpeg_dlls()
    if ff:
        log(f"ffmpeg libraries: {ff}")

    t0 = time.time()
    import torch
    from pyannote.audio import Pipeline

    # pyannote disables TF32 on import for bit-reproducibility and then tells you to turn it
    # back on. For diarization the accuracy difference is immaterial while the speedup on the
    # conv/matmul-heavy segmentation and embedding passes is not — and multi-hour files are
    # exactly where that matters. See pyannote-audio issue #1370.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    try:
        pipeline = Pipeline.from_pretrained(pipeline_name, token=token)
    except Exception as e:  # noqa: BLE001
        # We run with HF_HUB_OFFLINE=1, so a pipeline that was never downloaded — or one whose
        # gated conditions were never accepted, which is why it was never downloaded — surfaces
        # as a cache miss rather than as a 403. Say which of those it is instead of leaving a
        # message about an internet connection this app deliberately does not use.
        if "LocalEntryNotFound" in type(e).__name__ or "cannot find the requested files" in str(e):
            emit(ev="error", kind="load",
                 msg=f"{pipeline_name} is not in the local model cache, and Sinribe runs fully "
                     f"offline. Choose {DEFAULT_PIPELINE} under Diarizer — that is the pipeline "
                     f"setup downloads. (A different pipeline, such as "
                     f"pyannote/speaker-diarization-3.1, also needs its conditions accepted on "
                     f"huggingface.co with this token's account before it can be fetched.)")
            return 1
        emit(ev="error", kind="load",
             msg=f"could not load {pipeline_name}: {type(e).__name__}: {e}")
        return 1
    if pipeline is None:
        emit(ev="error", kind="load",
             msg=f"{pipeline_name} returned None — the HF token is missing or the "
                 f"repository licence has not been accepted for this account")
        return 1

    device = "cpu"
    if device_pref == "cuda" and torch.cuda.is_available():
        device = "cuda"
    try:
        pipeline.to(torch.device(device))
    except Exception as e:  # noqa: BLE001
        log(f"could not move pipeline to {device} ({e}); staying on CPU")
        device = "cpu"

    log(f"pipeline ready on {device} in {time.time() - t0:.1f}s")
    emit(ev="ready", device=device, pipeline=pipeline_name)

    try:
        import soundfile as sf

        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (channel, time)
        log(f"loaded {waveform.shape[1] / sr:.1f}s of audio at {sr} Hz")

        with ProgressHook() as hook:
            output = pipeline({"waveform": waveform, "sample_rate": sr},
                              hook=hook, **diar_kwargs)

        # pyannote 4.x returns a DiarizeOutput; 3.x returned a bare Annotation.
        # Prefer `exclusive_speaker_diarization` — pyannote builds it specifically for downstream
        # transcription because it contains no overlapping turns, so every word maps to exactly
        # one speaker instead of being split arbitrarily between two.
        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", None)
        if annotation is None:
            annotation = output  # pyannote 3.x

        turns = []
        for segment, _, label in annotation.itertracks(yield_label=True):
            turns.append({"start": round(float(segment.start), 3),
                          "end": round(float(segment.end), 3),
                          "speaker": str(label)})
        turns.sort(key=lambda t: (t["start"], t["end"]))
        speakers = sorted({t["speaker"] for t in turns})

        elapsed = time.time() - t0
        log(f"done: {len(turns)} turns, {len(speakers)} speakers in {elapsed:.1f}s")
        emit(ev="result", turns=turns, speakers=speakers)
        emit(ev="done", elapsed=round(elapsed, 2), turns=len(turns), speakers=len(speakers))
        return 0
    except Exception as e:  # noqa: BLE001
        emit(ev="error", kind="oom" if is_oom(e) else "runtime",
             msg=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
