"""Tests for the Windows platform layer.

These run on Linux (where the suite is developed) as well as on Windows, so they exercise the
decisions rather than the platform: which PyTorch build a given GPU should get, where a
virtualenv's interpreter lives, whether a model counts as cached, how ffmpeg is resolved. The
handful of assertions that only make sense on one platform are guarded by IS_WINDOWS.
"""

from __future__ import annotations

import os

import pytest

from sinribe import winenv
from sinribe.winenv import CPU_INDEX, CU126, CU130, GpuInfo, IS_WINDOWS


def gpu(cc: float, driver: str, vram: int = 8192, name: str = "Test GPU") -> GpuInfo:
    return GpuInfo(present=True, name=name, vram_mib=vram, driver=driver, compute_cap=cc)


class TestTorchSelection:
    def test_modern_card_and_driver_gets_cuda_13(self):
        index, specs, is_cuda, _ = winenv.torch_target(gpu(12.0, "580.88"))
        assert index == CU130
        assert is_cuda
        assert any(s.startswith("torch==") for s in specs)

    def test_ampere_on_an_older_driver_falls_back_to_cuda_126(self):
        index, _, is_cuda, _ = winenv.torch_target(gpu(8.6, "537.13"))
        assert index == CU126
        assert is_cuda

    def test_blackwell_on_an_old_driver_uses_cpu_rather_than_the_wrong_kernels(self):
        # cu126 has no sm_120 kernels. Sending a Blackwell card there would either fail
        # outright or limp along on JIT'd PTX; the honest answer is CPU plus a hint.
        index, _, is_cuda, reason = winenv.torch_target(gpu(12.0, "550.01"))
        assert index == CPU_INDEX
        assert not is_cuda
        assert "driver" in reason.lower()

    def test_cuda_13_is_not_offered_to_pascal(self):
        # CUDA 13 dropped Maxwell, Pascal and Volta.
        index, _, is_cuda, _ = winenv.torch_target(gpu(6.1, "580.88"))
        assert index == CU126
        assert is_cuda

    def test_no_gpu_means_cpu(self):
        index, _, is_cuda, reason = winenv.torch_target(GpuInfo())
        assert index == CPU_INDEX
        assert not is_cuda
        assert "no nvidia" in reason.lower()

    def test_ancient_driver_means_cpu(self):
        index, _, is_cuda, reason = winenv.torch_target(gpu(7.5, "470.82"))
        assert index == CPU_INDEX
        assert not is_cuda
        assert "driver" in reason.lower()

    def test_every_target_pins_the_whole_torch_trio(self):
        # torch, torchaudio and torchcodec load into one process and torchcodec links against
        # libtorch's ABI, so they must always be installed together from the same index.
        for g in (gpu(12.0, "580.88"), gpu(8.6, "537.13"), GpuInfo()):
            _, specs, _, _ = winenv.torch_target(g)
            names = {s.split("==")[0] for s in specs}
            assert names == {"torch", "torchaudio", "torchcodec"}

    def test_unparseable_driver_is_treated_as_old(self):
        assert gpu(8.6, "unknown").driver_major == 0
        _, _, is_cuda, _ = winenv.torch_target(gpu(8.6, "unknown"))
        assert not is_cuda


class TestCtranslate2Support:
    def test_needs_cuda_12_era_driver(self):
        assert winenv.cuda_supported(gpu(8.6, "537.13"))
        assert not winenv.cuda_supported(gpu(8.6, "470.82"))

    def test_blackwell_is_supported_even_where_torch_falls_back(self):
        # The two stacks have different floors: ctranslate2 is a CUDA 12 binary and runs on an
        # r525+ driver, while torch's Blackwell build wants r580.
        g = gpu(12.0, "550.01")
        assert winenv.cuda_supported(g)
        assert not winenv.torch_target(g)[2]

    def test_no_gpu_is_not_supported(self):
        assert not winenv.cuda_supported(GpuInfo())


class TestVenvLayout:
    def test_interpreter_path_matches_the_platform(self):
        p = winenv.venv_python(winenv.ASR_VENV)
        assert p.name == ("python.exe" if IS_WINDOWS else "python")
        assert p.parent.name == ("Scripts" if IS_WINDOWS else "bin")

    def test_windowed_interpreter_is_pythonw_on_windows(self):
        p = winenv.venv_python(winenv.ASR_VENV, windowed=True)
        assert p.name == ("pythonw.exe" if IS_WINDOWS else "python")

    def test_site_packages_is_under_the_venv(self, tmp_path):
        sp = winenv.venv_site_packages(tmp_path / "v")
        assert sp.name == "site-packages"
        assert str(sp).startswith(str(tmp_path))

    def test_the_two_venvs_are_distinct(self):
        assert winenv.ASR_VENV != winenv.DIAR_VENV


class TestCudaDllDirs:
    def test_only_existing_directories_are_returned(self, tmp_path):
        (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
        (tmp_path / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
        dirs = winenv.cuda_dll_dirs(tmp_path)
        assert len(dirs) == 2
        assert all(d.is_dir() for d in dirs)

    def test_cpu_only_install_reports_nothing(self, tmp_path):
        assert winenv.cuda_dll_dirs(tmp_path) == []

    def test_enable_is_safe_when_there_is_nothing_to_enable(self, tmp_path):
        assert winenv.enable_cuda_dlls(tmp_path) == []


class TestModelCache:
    def test_repo_id_maps_to_the_huggingface_layout(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        d = winenv.model_cache_dir("Systran/faster-whisper-large-v3")
        assert d.name == "models--Systran--faster-whisper-large-v3"
        assert d.parent == tmp_path / "hub"

    def test_absent_model_is_not_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert not winenv.model_is_cached("Systran/faster-whisper-large-v3")

    def test_complete_snapshot_counts_as_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        snap = winenv.model_cache_dir("a/b") / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.bin").write_bytes(b"x")
        assert winenv.model_is_cached("a/b")

    def test_interrupted_download_does_not_count(self, monkeypatch, tmp_path):
        # An aborted fetch leaves the directory tree and a blobs/*.incomplete behind. Calling
        # that "installed" sends the user into a confusing runtime failure instead of a resume.
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        root = winenv.model_cache_dir("a/b")
        (root / "snapshots" / "abc123").mkdir(parents=True)
        (root / "blobs").mkdir(parents=True)
        (root / "blobs" / "deadbeef.incomplete").write_bytes(b"partial")
        assert not winenv.model_is_cached("a/b")

    def test_explicit_hf_home_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
        assert winenv.hf_home() == tmp_path / "elsewhere"

    def test_default_hf_home_lives_under_the_app_data_dir(self, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        assert winenv.hf_home().parent == winenv.localdata_dir()


class TestUserDirectories:
    def test_settings_and_cache_are_separate(self):
        assert winenv.appdata_dir() != winenv.localdata_dir()

    @pytest.mark.skipif(IS_WINDOWS, reason="POSIX-only isolation check")
    def test_posix_paths_do_not_collide_with_the_linux_build(self):
        # Running this port's tests on Linux must not read or overwrite a working install of
        # the original Sinribe.
        assert winenv.appdata_dir().name == "sinribe-windows"
        assert winenv.localdata_dir().name == "sinribe-windows"

    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only path check")
    def test_windows_uses_appdata(self, monkeypatch, tmp_path):
        monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
        assert winenv.appdata_dir() == tmp_path / "Roaming" / "Sinribe"
        assert winenv.localdata_dir() == tmp_path / "Local" / "Sinribe"


class TestFfmpegResolution:
    def test_bundled_copy_wins_over_path(self, monkeypatch, tmp_path):
        exe = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / exe).write_bytes(b"")
        monkeypatch.setattr(winenv, "FFMPEG_DIR", tmp_path)
        assert winenv.ffmpeg_exe() == str(bin_dir / exe)
        assert winenv.ffmpeg_location() == str(bin_dir)

    def test_falls_back_to_a_bare_name_so_errors_are_readable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(winenv, "FFMPEG_DIR", tmp_path / "nothing")
        monkeypatch.setattr(winenv, "_which", lambda name: None)
        assert winenv.ffmpeg_exe() == "ffmpeg"
        assert winenv.ffprobe_exe() == "ffprobe"
        assert winenv.ffmpeg_location() is None
        assert not winenv.have_ffmpeg()


class TestProcessSpawning:
    def test_no_window_is_a_noop_off_windows(self):
        kw = winenv.no_window()
        if IS_WINDOWS:
            # CREATE_NO_WINDOW; without it every ffmpeg call flashes a console under pythonw.
            assert kw["creationflags"] & winenv.CREATE_NO_WINDOW
            assert "startupinfo" in kw
        else:
            assert kw == {}

    def test_run_returns_a_completed_process(self):
        out = winenv.run([os.sys.executable, "-c", "print('hi')"], timeout=60)
        assert out.returncode == 0
        assert "hi" in out.stdout
