"""Tests for the detect-and-install engine.

Nothing here touches the network or installs anything: the download and subprocess layers are
replaced, and what is checked is the decision-making — what counts as installed, what a given
set of options implies, and that the archive extractor cannot be talked into writing outside its
destination.
"""

from __future__ import annotations

import zipfile

import pytest

from sinribe import config, setup_core, winenv
from sinribe.setup_core import Component, Options, Progress


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Point every path setup_core touches at a temporary directory."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setattr(winenv, "APP_ROOT", tmp_path)
    monkeypatch.setattr(winenv, "TOOLS_DIR", tmp_path / "tools")
    monkeypatch.setattr(winenv, "FFMPEG_DIR", tmp_path / "tools" / "ffmpeg")
    monkeypatch.setattr(winenv, "ASR_VENV", tmp_path / ".venv")
    monkeypatch.setattr(winenv, "DIAR_VENV", tmp_path / ".venv-diar")
    monkeypatch.setattr(setup_core, "ASR_VENV", tmp_path / ".venv")
    monkeypatch.setattr(setup_core, "DIAR_VENV", tmp_path / ".venv-diar")
    monkeypatch.setattr(setup_core, "UV_EXE", tmp_path / "tools" / "uv.exe")
    monkeypatch.setattr(setup_core, "_managed_python", lambda: None)
    monkeypatch.setattr(winenv, "gpu_info", lambda: winenv.GpuInfo())
    # The locally converted models live outside the HuggingFace cache, so HF_HOME does not
    # cover them; redirect them too or a test would see whatever this machine happens to have.
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "ct2-models")
    monkeypatch.setattr(setup_core, "MODELS_DIR", tmp_path / "ct2-models")
    return tmp_path


def _make_venv(root, name: str, packages: list[str]) -> None:
    """Fake an installed virtualenv convincingly enough for detection."""
    venv = root / name
    py = winenv.venv_python(venv)
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    sp = winenv.venv_site_packages(venv)
    for pkg in packages:
        d = sp / pkg
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("")


def _cache_model(repo: str) -> None:
    snap = winenv.model_cache_dir(repo) / "snapshots" / "v1"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "model.bin").write_bytes(b"x")


class TestPackageDetection:
    def test_directory_package_is_found(self, tmp_path):
        (tmp_path / "torch").mkdir()
        (tmp_path / "torch" / "__init__.py").write_text("")
        assert setup_core._pkg_present(tmp_path, "torch")

    def test_single_module_is_found(self, tmp_path):
        (tmp_path / "six.py").write_text("")
        assert setup_core._pkg_present(tmp_path, "six")

    def test_namespace_package_without_init_is_found(self, tmp_path):
        # pyannote.audio installs as a namespace package: a directory with no __init__.py.
        (tmp_path / "pyannote" / "audio").mkdir(parents=True)
        (tmp_path / "pyannote" / "audio" / "core.py").write_text("")
        assert setup_core._pkg_present(tmp_path, "pyannote")

    def test_empty_directory_does_not_count(self, tmp_path):
        (tmp_path / "torch").mkdir()
        assert not setup_core._pkg_present(tmp_path, "torch")

    def test_all_names_must_be_present(self, tmp_path):
        (tmp_path / "torch").mkdir()
        (tmp_path / "torch" / "__init__.py").write_text("")
        assert not setup_core._pkg_present(tmp_path, "torch", "pyannote")


class TestDetect:
    def test_a_bare_machine_reports_everything_missing(self, sandbox):
        comps = {c.key: c for c in setup_core.detect(Options())}
        assert not comps["uv"].ok
        assert not comps["asr_venv"].ok
        assert not comps["whisper_model"].ok
        assert not comps["diar_venv"].ok

    def test_installed_pieces_are_recognised(self, sandbox, monkeypatch):
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2", "PySide6", "yt_dlp"])
        _cache_model("Systran/faster-whisper-small")
        comps = {c.key: c for c in setup_core.detect(Options(model="small"))}
        assert comps["asr_venv"].ok
        assert comps["whisper_model"].ok

    def test_switching_model_invalidates_the_model_check(self, sandbox):
        _cache_model("Systran/faster-whisper-small")
        assert {c.key: c for c in setup_core.detect(Options(model="small"))}["whisper_model"].ok
        assert not {c.key: c
                    for c in setup_core.detect(Options(model="large-v3"))}["whisper_model"].ok

    def test_disabling_diarization_drops_its_requirements(self, sandbox):
        comps = {c.key: c for c in setup_core.detect(Options(diarize=False))}
        for key in ("diar_venv", "diar_model", "hf_token"):
            assert not comps[key].required
            assert comps[key].ok      # "not needed" counts as satisfied

    def test_gpu_machine_wants_the_cuda_runtime_in_the_asr_venv(self, sandbox, monkeypatch):
        # A .venv built before the graphics card existed has no nvidia/* directories, so a
        # GPU install must still report the transcription environment as incomplete.
        monkeypatch.setattr(winenv, "gpu_info",
                            lambda: winenv.GpuInfo(present=True, name="RTX", vram_mib=8192,
                                                   driver="580.1", compute_cap=8.9))
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2", "PySide6", "yt_dlp"])
        comps = {c.key: c for c in setup_core.detect(Options(gpu=True))}
        assert not comps["asr_venv"].ok
        assert "CPU only" not in comps["asr_venv"].detail

    def test_cpu_machine_is_content_without_cuda(self, sandbox):
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2", "PySide6", "yt_dlp"])
        comps = {c.key: c for c in setup_core.detect(Options(gpu=False))}
        assert comps["asr_venv"].ok
        assert "CPU only" in comps["asr_venv"].detail


class TestPlan:
    def test_plan_lists_missing_steps_and_a_download_estimate(self, sandbox):
        keys, gb = setup_core.plan(Options(model="small", diarize=False))
        assert "uv" in keys and "asr_venv" in keys and "whisper_model" in keys
        assert "diar_venv" not in keys
        assert gb > 0.4

    def test_plan_is_empty_once_everything_is_present(self, sandbox, monkeypatch):
        monkeypatch.setattr(setup_core, "detect",
                            lambda opts=None: [Component("x", "X", True, "", True, 0.0)])
        keys, gb = setup_core.plan(Options())
        assert keys == []
        assert gb == 0.0

    def test_a_bigger_model_means_a_bigger_download(self, sandbox):
        _, small = setup_core.plan(Options(model="small", diarize=False))
        _, large = setup_core.plan(Options(model="large-v3", diarize=False))
        assert large > small


class TestLaunchBlockers:
    def test_nothing_installed_blocks_with_readable_reasons(self, sandbox):
        blockers = setup_core.launch_blockers({"asr_model": "small", "diarize": False})
        assert blockers
        assert all(isinstance(b, str) and b[0].islower() for b in blockers)

    def test_a_complete_cpu_install_does_not_block(self, sandbox, monkeypatch):
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2"])
        _cache_model("Systran/faster-whisper-small")
        monkeypatch.setattr(winenv, "have_ffmpeg", lambda: True)
        assert setup_core.launch_blockers({"asr_model": "small", "diarize": False}) == []

    def test_diarization_without_a_token_blocks(self, sandbox, monkeypatch):
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2"])
        _make_venv(sandbox, ".venv-diar", ["torch", "pyannote"])
        _cache_model("Systran/faster-whisper-small")
        monkeypatch.setattr(winenv, "have_ffmpeg", lambda: True)
        monkeypatch.setattr(setup_core, "hf_token", lambda: None)
        blockers = setup_core.launch_blockers({"asr_model": "small", "diarize": True})
        assert any("token" in b for b in blockers)

    def test_it_asks_a_narrower_question_than_detect(self, sandbox, monkeypatch):
        # A missing uv.exe or shortcut means the install is not pristine, but it does not stop
        # anyone from transcribing — launch must not be blocked by it.
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2"])
        _cache_model("Systran/faster-whisper-small")
        monkeypatch.setattr(winenv, "have_ffmpeg", lambda: True)
        cfg = {"asr_model": "small", "diarize": False}
        assert setup_core.launch_blockers(cfg) == []
        assert not setup_core.is_ready(Options(model="small", diarize=False))


class TestInstallSequencing:
    """install() is a filtered walk over STEPS; ordering and idempotence matter."""

    @pytest.fixture
    def recorder(self, monkeypatch):
        calls: list[str] = []

        def make(key):
            return lambda opts, progress: calls.append(key)

        monkeypatch.setattr(setup_core, "STEPS",
                            [(k, k.title(), make(k)) for k in
                             ("uv", "python", "ffmpeg", "asr_venv", "whisper_model",
                              "diar_venv", "diar_model", "german_models", "shortcuts")])
        monkeypatch.setattr(setup_core, "save_state", lambda state: None)
        monkeypatch.setattr(setup_core, "load_state", dict)
        monkeypatch.setattr(setup_core, "save_hf_token", lambda tok: None)
        return calls

    def test_everything_runs_in_dependency_order_by_default(self, recorder):
        setup_core.install(Options())
        assert recorder == ["uv", "python", "ffmpeg", "asr_venv", "whisper_model",
                            "diar_venv", "diar_model", "german_models", "shortcuts"]

    def test_only_restricts_the_walk_but_keeps_the_order(self, recorder):
        setup_core.install(Options(), only=["shortcuts", "asr_venv", "uv"])
        assert recorder == ["uv", "asr_venv", "shortcuts"]

    def test_the_step_counter_matches_what_was_asked_for(self, recorder):
        seen: list[tuple[int, int, str]] = []
        setup_core.install(Options(), Progress(step=lambda i, n, t: seen.append((i, n, t))),
                           only=["uv", "ffmpeg"])
        assert [(i, n) for i, n, _ in seen] == [(1, 2), (2, 2)]

    def test_cancelling_stops_between_steps(self, recorder):
        with pytest.raises(setup_core.Cancelled):
            setup_core.install(Options(), Progress(should_cancel=lambda: True))
        assert recorder == []


class TestExtract:
    def test_selected_members_are_renamed_on_the_way_out(self, tmp_path):
        src = tmp_path / "a.zip"
        with zipfile.ZipFile(src, "w") as zf:
            zf.writestr("ffmpeg-n7.1/bin/ffmpeg.exe", "binary")
            zf.writestr("ffmpeg-n7.1/bin/avcodec-61.dll", "lib")
            zf.writestr("ffmpeg-n7.1/doc/readme.txt", "docs")

        def member(name: str) -> str | None:
            parts = name.split("/")
            return f"bin/{parts[-1]}" if len(parts) >= 2 and parts[-2] == "bin" else None

        out = tmp_path / "out"
        n = setup_core._extract(src, out, member, Progress())
        assert n == 2
        assert (out / "bin" / "ffmpeg.exe").is_file()
        assert (out / "bin" / "avcodec-61.dll").is_file()
        assert not (out / "doc").exists()

    def test_a_traversal_entry_cannot_escape_the_destination(self, tmp_path):
        src = tmp_path / "evil.zip"
        with zipfile.ZipFile(src, "w") as zf:
            zf.writestr("ok.txt", "fine")
        out = tmp_path / "out"
        # The renamer is what decides the output path, so this is the shape of a bad one.
        setup_core._extract(src, out, lambda n: "../../escaped.txt", Progress())
        assert not (tmp_path.parent / "escaped.txt").exists()
        assert not (tmp_path / "escaped.txt").exists()


class TestOptions:
    def test_explicit_choice_beats_hardware(self):
        strong = winenv.GpuInfo(present=True, name="RTX", vram_mib=12288, driver="580.1",
                                compute_cap=8.9)
        assert Options(gpu=False).resolve_gpu(strong) is False
        assert Options(gpu=True).resolve_gpu(winenv.GpuInfo()) is True

    def test_auto_follows_the_hardware(self):
        strong = winenv.GpuInfo(present=True, name="RTX", vram_mib=12288, driver="580.1",
                                compute_cap=8.9)
        assert Options().resolve_gpu(strong) is True
        assert Options().resolve_gpu(winenv.GpuInfo()) is False


class TestProgress:
    def test_defaults_swallow_everything(self):
        p = Progress()
        p.log("x")
        p.step(1, 2, "y")
        p.frac(0.5, "z")
        p.check()          # must not raise

    def test_check_raises_once_cancelled(self):
        p = Progress(should_cancel=lambda: True)
        with pytest.raises(setup_core.Cancelled):
            p.check()


def _convert_model(name: str) -> None:
    """Fake a finished CTranslate2 conversion."""
    d = config.model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.bin").write_bytes(b"x")
    (d / "tokenizer.json").write_text("{}")


class TestModelResolution:
    """"auto" is a delegation to the quality slider, not a model. Everything that asks whether
    a model is installed has to resolve it first, or a converted model is never checked at all.
    """

    def test_auto_resolves_to_the_rungs_model(self):
        assert setup_core.resolved_model({"asr_model": "auto", "speed_target": 10}) == "large-v3"

    def test_auto_can_resolve_to_a_converted_model(self):
        # The fastest rung runs a German fine-tune, which is not a HuggingFace download.
        model = setup_core.resolved_model({"asr_model": "auto", "speed_target": 84})
        assert model in config.LOCAL_MODELS

    def test_a_pin_wins(self):
        assert setup_core.resolved_model({"asr_model": "small", "speed_target": 10}) == "small"

    def test_setup_never_tries_to_download_a_converted_model(self):
        # Options.model names one HuggingFace repository to fetch. A German fine-tune has none,
        # and every voting rung needs a plain whisper model underneath it anyway.
        cfg = {"asr_model": "large-v3-turbo-german", "speed_target": 10}
        assert setup_core.downloadable_model(cfg) == "large-v3"
        assert setup_core.downloadable_model({"asr_model": "auto", "speed_target": 84}) \
            == "large-v3"

    def test_setup_does_not_pin_a_user_who_left_it_on_auto(self):
        cfg = {"asr_model": "auto", "speed_target": 10}
        setup_core.remember_model(cfg, "large-v3")
        assert cfg["asr_model"] == "auto"

    def test_setup_does_pin_a_deliberate_change(self):
        cfg = {"asr_model": "auto", "speed_target": 10}
        setup_core.remember_model(cfg, "small")
        assert cfg["asr_model"] == "small"


class TestGermanModels:
    """Optional, converted locally, and — unlike a whisper model — not fetchable at job time."""

    def test_absent_by_default_and_not_required(self, sandbox):
        comps = {c.key: c for c in setup_core.detect(Options())}
        assert not comps["german_models"].ok
        assert not comps["german_models"].required

    def test_asking_for_them_makes_them_a_planned_step(self, sandbox):
        keys, _gb = setup_core.plan(Options(german_models=True, diarize=False))
        assert "german_models" in keys

    def test_not_asking_for_them_leaves_them_out_of_the_plan(self, sandbox):
        keys, _gb = setup_core.plan(Options(diarize=False))
        assert "german_models" not in keys

    def test_a_finished_conversion_is_recognised(self, sandbox):
        for name in config.LOCAL_MODELS:
            _convert_model(name)
        comps = {c.key: c for c in setup_core.detect(Options(german_models=True))}
        assert comps["german_models"].ok
        assert setup_core.plan(Options(german_models=True, diarize=False))[1] < 4.0

    def test_a_half_finished_conversion_does_not_count(self, sandbox):
        # An interrupted convert leaves the directory behind. Treating that as installed sends
        # the next job into faster-whisper with no weights to load.
        d = config.model_dir("large-v3-german")
        d.mkdir(parents=True)
        assert not config.model_available("large-v3-german")

    def test_installing_them_is_a_no_op_when_they_are_there(self, sandbox, monkeypatch):
        for name in config.LOCAL_MODELS:
            _convert_model(name)
        monkeypatch.setattr(setup_core, "_build_ct2_venv",
                            lambda *a: pytest.fail("should not have built a venv"))
        setup_core.ensure_german_models(Options(german_models=True), Progress())

    def test_they_are_skipped_entirely_when_not_asked_for(self, sandbox, monkeypatch):
        monkeypatch.setattr(setup_core, "_build_ct2_venv",
                            lambda *a: pytest.fail("should not have built a venv"))
        setup_core.ensure_german_models(Options(german_models=False), Progress())

    def test_a_missing_conversion_does_not_lock_the_window(self, sandbox, monkeypatch):
        # Blocking launch would refuse to open the main window, and the quality slider — the only
        # control that can undo this — lives inside it. The window checks on Start instead.
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2"])
        _cache_model("Systran/faster-whisper-large-v3")
        monkeypatch.setattr(winenv, "have_ffmpeg", lambda: True)
        cfg = {"asr_model": "large-v3-turbo-german", "diarize": False}
        assert not config.model_available("large-v3-turbo-german")
        assert setup_core.launch_blockers(cfg) == []

    def test_a_missing_whisper_download_still_blocks(self, sandbox, monkeypatch):
        # The other half of the same decision: a HuggingFace model is fetched by the job itself,
        # but only if it was ever asked for — an install that never downloaded one is not ready.
        _make_venv(sandbox, ".venv", ["faster_whisper", "ctranslate2"])
        monkeypatch.setattr(winenv, "have_ffmpeg", lambda: True)
        blockers = setup_core.launch_blockers({"asr_model": "large-v3", "diarize": False})
        assert any("has not been downloaded" in b for b in blockers)
