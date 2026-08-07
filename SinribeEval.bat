@echo off
rem ============================================================================
rem  Score a transcript against a reference, or sweep decode settings.
rem
rem    SinribeEval.bat REFERENCE.docx OUTPUT.md
rem        Word error rate, deletions, insertions and speaker attribution for a
rem        transcript Sinribe already wrote.
rem
rem    SinribeEval.bat --sweep variants.json AUDIO.wav REFERENCE.docx -o out.json
rem        Decode the same audio once per variant and put the scores side by
rem        side. This is how presets.py was calibrated.
rem
rem  A sweep RE-DECODES the audio, so it runs in the transcription environment
rem  and needs the CUDA runtime the same way a normal job does. The worker
rem  registers its own DLL directories (see winenv.enable_cuda_dlls), so there
rem  is nothing to set up here.
rem
rem  Two rules the calibration learned the hard way, both in HANDOFF.md: run
rem  nothing else on the GPU, and interleave variants rather than running them
rem  in blocks. Error rates are reproducible; wall-clock timings are not.
rem ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Sinribe is not installed yet -- run Setup.bat first. 1>&2
    exit /b 1
)

if "%HF_HUB_OFFLINE%"=="" set "HF_HUB_OFFLINE=1"
if "%TOKENIZERS_PARALLELISM%"=="" set "TOKENIZERS_PARALLELISM=false"
set "PYTHONIOENCODING=utf-8"

"%PY%" -m sinribe.eval %*
exit /b !errorlevel!
