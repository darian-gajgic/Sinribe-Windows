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

from .winenv import IS_WINDOWS, appdata_dir, localdata_dir

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

ASR_MODELS = [m[0] for m in WHISPER_MODELS]
WHISPER_REPOS = {name: repo for name, repo, _, _ in WHISPER_MODELS}
WHISPER_SIZES_GB = {name: gb for name, _, gb, _ in WHISPER_MODELS}

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
    "asr_model": "large-v3",
    "language": "auto",
    "beam_size": 5,
    "vad_filter": True,
    # Force a device instead of picking one from free VRAM: auto | cuda | cpu.
    "asr_device": "auto",
    # diarization
    "diarize": True,
    "diar_pipeline": "pyannote/speaker-diarization-community-1",
    "speaker_mode": "auto",  # auto | exact | range
    "num_speakers": 2,
    "min_speakers": 1,
    "max_speakers": 6,
    # merge / formatting
    "turn_gap_s": 1.5,
    "max_turn_chars": 1200,
    "flicker_min_words": 3,
    "orphan_word_window_s": 2.0,
    # outputs
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


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
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
