@echo off
rem ============================================================================
rem  Open Sinribe's setup: install missing pieces, change the speech model,
rem  add a HuggingFace token, or repair a broken install.
rem
rem  Safe to run at any time. Every step checks before it acts, so running it
rem  twice does nothing the second time.
rem ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if exist "%PYW%" (
    start "" "%PYW%" -m sinribe --setup
    exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap.ps1"
exit /b !errorlevel!
