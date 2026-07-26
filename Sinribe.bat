@echo off
rem ============================================================================
rem  Sinribe launcher.
rem
rem  Double-click this (or the desktop shortcut) to run Sinribe. It works out
rem  what state the machine is in and does the right thing:
rem
rem    * installed already  -> starts the app with pythonw.exe, which has no
rem                            console, so no black window appears behind it
rem    * not installed yet  -> hands over to tools\bootstrap.ps1, which fetches
rem                            a private Python and opens the setup wizard
rem    * given arguments    -> runs on the console instead, so you can see the
rem                            progress bar:  Sinribe.bat recording.mp3 -o C:\Out
rem
rem  Nothing here needs administrator rights.
rem ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYW=%~dp0.venv\Scripts\pythonw.exe"
set "PY=%~dp0.venv\Scripts\python.exe"

rem --setup and --check are asked for by name and want the window, not a console.
if /i "%~1"=="--setup" goto windowed
if "%~1"=="" goto windowed

rem Anything else is a headless run: keep the console so output is visible.
if exist "%PY%" (
    "%PY%" -m sinribe %*
    exit /b !errorlevel!
)
goto bootstrap

:windowed
if exist "%PYW%" (
    start "" "%PYW%" -m sinribe %*
    exit /b 0
)

:bootstrap
rem First run, or a wiped .venv. bootstrap.ps1 needs no Python of its own.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap.ps1"
exit /b !errorlevel!
