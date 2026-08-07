"""Settings persistence for Sinribe.

Whitelist-filtered load + atomic write: unknown keys in the on-disk file are dropped, a corrupt
file degrades to defaults with a warning instead of crashing, and writes go through a temp file
+ os.replace.

Windows note: settings and the HuggingFace token live in %APPDATA%\\Sinribe (roaming, tiny),
while caches and model weights go to %LOCALAPPDATA%\\Sinribe (several GB — a roaming profile
would try to synchronise all of it). Both are resolved in winenv so this module stays portable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .winenv import IS_WINDOWS, appdata_dir, localdata_dir, model_is_cached

CONFIG_DIR = appdata_dir()
CONFIG_PATH = CONFIG_DIR / "config.json"
TOKEN_PATH = CONFIG_DIR / "hf_token"
CACHE_DIR = localdata_dir()
JOBS_DIR = CACHE_DIR / "jobs"
DOWNLOADS_DIR = CACHE_DIR / "downloads"
STATE_PATH = CONFIG_DIR / "setup-state.json"

#: Whisper models offered in the UI, with the HuggingFace repository each one resolves to and
#: its download size (read from the HF API, not estimated). Unlike the Linux build — where these
#: were already in the box's cache — nothing is present on a fresh Windows machine, so the size
#: is part of the choice and is shown in the setup wizard.
WHISPER_MODELS: list[tuple[str, str, float, str]] = [
    ("large-v3", "Systran/faster-whisper-large-v3", 3.09,
     "Best accuracy. Wants a GPU to be quick."),
    ("distil-large-v3", "Systran/faster-distil-whisper-large-v3", 1.52,
     "Near large-v3 quality at roughly twice the speed. English only."),
    ("medium", "Systran/faster-whisper-medium", 1.53, "Good balance on a mid-range GPU."),
    ("small", "Systran/faster-whisper-small", 0.49, "Fast. The sensible choice for CPU-only."),
    ("tiny", "Systran/faster-whisper-tiny", 0.08, "Draft quality. Useful for a quick test."),
]

WHISPER_REPOS = {name: repo for name, repo, _, _ in WHISPER_MODELS}
WHISPER_SIZES_GB = {name: gb for name, _, gb, _ in WHISPER_MODELS}

#: CTranslate2 conversions of models that have no ready-made faster-whisper build. Deliberately
#: NOT under winenv.hf_home() — which is also %LOCALAPPDATA%\\Sinribe\\models — because these are
#: not HuggingFace artefacts. They are converted on this machine, take minutes to redo, and a
#: cache clean aimed at the HF hub must not sweep them away with it.
MODELS_DIR = CACHE_DIR / "ct2-models"

#: German fine-tunes of whisper. Vanilla large-v3 is multilingual and treats German as one
#: language among ninety-nine; these were trained on German alone, which is what an
#: accented room recording needs. The upstream repos ship transformers weights, not CTranslate2
#: ones, so setup converts them locally — see tools/convert_german_models.py. Values are
#: (destination directory, source repository, approximate download GB).
LOCAL_MODELS: dict[str, tuple[str, str, float]] = {
    "large-v3-german": ("whisper-large-v3-german-ct2",
                        "primeline/whisper-large-v3-german", 3.1),
    "large-v3-turbo-german": ("whisper-large-v3-turbo-german-ct2",
                              "primeline/whisper-large-v3-turbo-german", 1.6),
}

LOCAL_MODEL_REPOS = {name: repo for name, (_, repo, _) in LOCAL_MODELS.items()}
LOCAL_MODEL_SIZES_GB = {name: gb for name, (_, _, gb) in LOCAL_MODELS.items()}

#: Models that cannot be given a hotword prompt. `large-v3-german` returns an EMPTY transcript
#: for the whole recording when one is supplied — measured on two different files, with hotword
#: strings as short as a single letter, while the same model with no hotwords transcribes
#: normally. Its tokenizer and vocabulary match the base model exactly, so this is the fine-tune
#: itself having lost the ability to condition on a prefix, not a conversion fault.
#: `large-v3-turbo-german` from the same publisher is unaffected.
MODELS_WITHOUT_HOTWORDS = frozenset({"large-v3-german"})

# "auto" lets the quality setting choose; everything else is an explicit override. The German
# fine-tunes are listed even before they are converted — the interface greys out what is not
# installed, which is more use than hiding the fact that they exist.
ASR_MODELS = ["auto"] + [m[0] for m in WHISPER_MODELS] + list(LOCAL_MODELS)


def supports_hotwords(name: str) -> bool:
    return name not in MODELS_WITHOUT_HOTWORDS


def model_dir(name: str) -> Path | None:
    """Where a locally converted model lives, or None if `name` is not one of ours."""
    entry = LOCAL_MODELS.get(name)
    return (MODELS_DIR / entry[0]) if entry else None


def model_available(name: str) -> bool:
    """Is this model ready to use right now?

    A converted model has to be on disk. A HuggingFace one does not: the pipeline is allowed to
    fetch a whisper model it has never seen, once, so "not cached" is a download and not a
    blocker. Anything unrecognised is assumed usable — faster-whisper accepts repo names this
    build has never heard of.
    """
    d = model_dir(name)
    return (d / "model.bin").exists() if d else True


def resolve_model(name: str) -> str:
    """Model name as faster-whisper wants it — a directory for our conversions, else the name.

    `WhisperModel` treats an existing directory as the model and anything else as a repository
    to fetch, so this is the whole of the local-model plumbing.
    """
    d = model_dir(name)
    return str(d) if d else WHISPER_REPOS.get(name, name)


#: pyannote's pipelines are gated: the account behind the HuggingFace token must accept each
#: repository's conditions before it can be downloaded. community-1 is the current one.
DIAR_PIPELINES = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]

#: Repositories each pipeline pulls in behind the scenes. Listed so setup can tell "model
#: present" from "model missing" by looking at the cache, without importing torch.
DIAR_SUPPORT_REPOS = {
    "pyannote/speaker-diarization-community-1": ["pyannote/speaker-diarization-community-1"],
    "pyannote/speaker-diarization-3.1": ["pyannote/speaker-diarization-3.1",
                                         "pyannote/segmentation-3.0"],
}


def diar_available(name: str) -> bool:
    """Is this diarization pipeline actually in the local cache?

    Workers run with `HF_HUB_OFFLINE=1`, so a pipeline that was never downloaded cannot be
    fetched at job time — it fails several minutes in, after the audio has been decoded. Both
    entries above are offered in the interface, but `speaker-diarization-3.1` is gated behind a
    separate licence acceptance and is routinely absent; selecting it used to be a working
    choice right up until the job died. Checked so the interface can grey it out instead.
    """
    repos = DIAR_SUPPORT_REPOS.get(name) or [name]
    return all(model_is_cached(r) for r in repos)

# Subset of whisper's languages worth surfacing in the UI; "auto" lets whisper detect.
LANGUAGES = [
    ("auto", "Auto-detect"),
    ("de", "German"),
    ("en", "English"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("it", "Italian"),
    ("nl", "Dutch"),
    ("pt", "Portuguese"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("tr", "Turkish"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
]


def default_output_dir() -> Path:
    """Documents\\Sinribe Transcripts, following a redirected Documents folder if there is one.

    OneDrive commonly moves Documents out of %USERPROFILE%. Writing to the literal path would
    then create a second folder that never appears where the user looks for their files.
    """
    if IS_WINDOWS:
        local_docs = Path.home() / "Documents"
        candidates = []
        one = os.environ.get("OneDrive") or os.environ.get("OneDriveConsumer")
        if local_docs.is_dir():
            candidates.append(local_docs)
        elif one:
            candidates.append(Path(one) / "Documents")
        for c in candidates:
            if c.is_dir():
                return c / "Sinribe Transcripts"
        return Path.home() / "Sinribe Transcripts"
    return Path.home() / "Sinribe-Transcripts"


DEFAULTS: dict = {
    # paths
    "output_dir": str(default_output_dir()),
    "last_input_dir": str(Path.home()),
    # ASR
    # "auto" lets the accuracy slider pick the model, which is what makes the slider a single
    # honest control instead of a setting you have to pair correctly with the dropdown.
    "asr_model": "auto",
    "language": "auto",
    "vad_filter": True,
    # Force a device instead of picking one from free VRAM: auto | cuda | cpu.
    "asr_device": "auto",
    # The accuracy slider, as a target realtime factor: lower is slower and more accurate.
    # Everything it implies (model, beam width, VAD padding, audio conditioning) lives in
    # presets.py next to the WER each rung actually measured. The default is the rung with the
    # best measured accuracy-per-minute, not the slowest one on offer.
    "speed_target": 10,
    # Realtime factors actually observed on THIS machine, keyed by rung, updated after every
    # successful job. The shipped figures in presets.py came off one benchmark run on the Linux
    # box and proved unreproducible even there — the same settings measured 50x and 20x on
    # different days, because the GPU is shared with whatever else is resident. Rather than
    # assert a number the user can catch being wrong, the estimate corrects itself from their
    # own runs. On Windows this matters more, not less: none of the shipped figures were
    # measured on this platform at all.
    "observed_rtf": {},
    # Names and jargon to bias the recogniser toward, comma-separated. Whisper reliably mangles
    # proper nouns it has no reason to expect — "Fujitsu" came out as *fiuzi*, *jitze* and
    # *future service* in one interview — and this is the supported fix.
    "hotwords": "",
    # diarization
    "diarize": True,
    "diar_pipeline": "pyannote/speaker-diarization-community-1",
    "speaker_mode": "auto",  # auto | exact | range
    "num_speakers": 2,
    "min_speakers": 1,
    "max_speakers": 6,
    # Re-check each sentence against the speakers' voice prints and correct the diarization
    # where the acoustic match is decisive. Costs a few seconds; see workers/refine_worker.py.
    "refine_speakers": True,
    # Cosine gap to the runner-up needed to overrule diarization. Below ~0.1 the two voices are
    # indistinguishable on that span and the diarization's wider context is the better guess.
    "refine_margin": 0.15,
    # A span shorter than this is too little voice to identify; backchannels ("Okay.", "Right.")
    # stay with whatever the diarization said. Measured margins on real audio are bimodal —
    # confident corrections sit at 0.3+, coin flips under 0.1 — so 0.15 falls in the empty gap.
    "refine_min_seconds": 0.6,
    # Decide attribution one sentence at a time rather than one word at a time.
    "sentence_atomic": True,
    # merge / formatting
    "turn_gap_s": 1.5,
    "max_turn_chars": 1200,
    "flicker_min_words": 3,
    "orphan_word_window_s": 2.0,
    # outputs
    # Append the list of passages the recogniser was least sure of. Cheap, and it turns
    # proofreading a transcript from "re-listen to all of it" into "check these forty spots".
    "review_section": True,
    "write_json": True,
    "write_srt": True,
    "write_vtt": True,
    # optional LLM enrichment (Ollama, offline). 11434 is Ollama's default port on Windows.
    "llm_enrich": False,
    "llm_url": "http://127.0.0.1:11434",
    "llm_model": "gemma3:4b",
    "llm_timeout": 180,
    # URL download (yt-dlp)
    "last_url": "",
    # Browser to lift cookies from for age-restricted or members-only media, e.g. "firefox"
    # or "chrome". Empty means no cookies, which is fine for ordinary public podcasts.
    "yt_cookies_from_browser": "",
    "keep_downloads": True,
    # housekeeping
    "keep_decoded_wav": False,
    "cache_days": 14,
    "download_cache_days": 14,
}


def _migrate(user: dict, cfg: dict) -> None:
    """Bring a pre-slider config forward.

    Only ever runs against a config written by an older build, and only fills in what that build
    had no way to record. Anything the user actually chose is left alone.
    """
    if "speed_target" not in user and user.get("asr_model") == "large-v3":
        # large-v3 was the default AND what the setup wizard offered first, so a saved
        # "large-v3" records no preference — it is just what the app came with. Treated as a pin
        # it would quietly override the quality slider's model choice and hand an existing user
        # none of the accuracy work. A genuine preference can still be set explicitly now that
        # there is a choice to make.
        cfg["asr_model"] = "auto"


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
            _migrate(user, cfg)
        except Exception as e:  # noqa: BLE001 - a broken config must never block startup
            print(f"[sinribe] WARNING: could not read {CONFIG_PATH}: {e!r}; using defaults",
                  flush=True)
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    payload = {k: cfg[k] for k in DEFAULTS if k in cfg}
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)


def hf_token() -> str | None:
    """The HuggingFace token, from the environment or the file the setup wizard wrote."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    try:
        tok = TOKEN_PATH.read_text(encoding="utf-8").strip()
        return tok or None
    except OSError:
        return None


def save_hf_token(token: str) -> Path:
    """Persist the HuggingFace token.

    Windows has no chmod worth relying on, so the protection is location rather than mode: the
    file sits in the user's own %APPDATA%, which other standard users cannot read. It is never
    written into the repository, and never into a transcript or a log.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(token.strip(), encoding="utf-8")
    if not IS_WINDOWS:
        os.chmod(TOKEN_PATH, 0o600)
    return TOKEN_PATH


def load_state() -> dict:
    """Setup bookkeeping: what the installer did, and with which options."""
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATE_PATH)
