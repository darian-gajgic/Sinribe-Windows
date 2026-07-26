# =============================================================================
#  Sinribe bootstrap — the part that runs before Sinribe can run anything.
#
#  A fresh Windows machine may have no Python at all, or one too old (or too
#  new) for the wheels Sinribe pins. Rather than asking anyone to install one,
#  this fetches uv — a single 25 MB executable — and lets it install a private
#  CPython 3.12 next to the user's profile. Nothing is registered, no PATH is
#  modified, no administrator rights are needed, and an existing Python
#  installation is left completely alone.
#
#  Then it hands over to tools\setup_wizard.py, which is written in tkinter
#  precisely because at that moment nothing else is installed yet.
#
#  Run it directly at any time:
#      powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
# =============================================================================
#Requires -Version 5.1
[CmdletBinding()]
param(
    # Skip the wizard and install with defaults — for unattended setups.
    [switch] $Unattended,
    [string] $Model = "",
    [string] $HfToken = "",
    [switch] $Cpu,
    [switch] $NoDiarize
)

$ErrorActionPreference = 'Stop'
$root    = Split-Path -Parent $PSScriptRoot
$tools   = Join-Path $root 'tools'
$uvExe   = Join-Path $tools 'uv.exe'
$pyVer   = '3.12'
$uvUrl   = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'

function Write-Head($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Write-Note($text) { Write-Host "    $text" -ForegroundColor DarkGray }

Write-Host ""
Write-Host "  Sinribe — first-run setup" -ForegroundColor White
Write-Host "  $root"
Write-Note "This window shows the groundwork; the setup window opens in a moment."

# TLS 1.2 is not the default on older PowerShell hosts, and github.com refuses
# anything less — without this the download fails with a bare "could not create
# SSL/TLS secure channel".
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

# ---------------------------------------------------------------- uv
if (-not (Test-Path $uvExe)) {
    Write-Head "Downloading uv (25 MB)"
    $tmp = Join-Path $env:TEMP ("sinribe-uv-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    $zip = Join-Path $tmp 'uv.zip'
    try {
        # ProgressPreference off: the progress bar makes Invoke-WebRequest an
        # order of magnitude slower on large files in Windows PowerShell.
        $oldProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $uvUrl -OutFile $zip -UseBasicParsing
        $ProgressPreference = $oldProgress

        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        $found = Get-ChildItem -Path $tmp -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $found) { throw "uv.exe was not in the archive" }
        New-Item -ItemType Directory -Path $tools -Force | Out-Null
        Copy-Item $found.FullName $uvExe -Force
        $uvx = Get-ChildItem -Path $tmp -Recurse -Filter 'uvx.exe' | Select-Object -First 1
        if ($uvx) { Copy-Item $uvx.FullName (Join-Path $tools 'uvx.exe') -Force }
    } catch {
        Write-Host "`nERROR: could not download uv: $_" -ForegroundColor Red
        Write-Host "Check the internet connection (and any proxy) and run Setup.bat again."
        Read-Host "`nPress Enter to close"
        exit 1
    } finally {
        Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Note "uv installed at $uvExe"
} else {
    Write-Note "uv already present"
}

# ---------------------------------------------------------------- Python
Write-Head "Checking for Python $pyVer"
$py = & $uvExe python find $pyVer 2>$null | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or -not $py -or -not (Test-Path $py)) {
    Write-Note "not found — installing a private copy (about 30 MB)"
    & $uvExe python install $pyVer
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nERROR: could not install Python $pyVer." -ForegroundColor Red
        Read-Host "`nPress Enter to close"
        exit 1
    }
    $py = & $uvExe python find $pyVer | Select-Object -First 1
}
if (-not $py -or -not (Test-Path $py)) {
    Write-Host "`nERROR: Python $pyVer is still not available." -ForegroundColor Red
    Read-Host "`nPress Enter to close"
    exit 1
}
Write-Note "using $py"

# ---------------------------------------------------------------- hand over
$wizard = Join-Path $tools 'setup_wizard.py'
if ($Unattended) {
    Write-Head "Installing (unattended)"
    $cliArgs = @('-m', 'sinribe.setup_core', '--install')
    if ($Model)     { $cliArgs += @('--model', $Model) }
    if ($HfToken)   { $cliArgs += @('--token', $HfToken) }
    if ($Cpu)       { $cliArgs += '--cpu' }
    if ($NoDiarize) { $cliArgs += '--no-diarize' }
    Push-Location $root
    try { & $py @cliArgs } finally { Pop-Location }
    exit $LASTEXITCODE
}

Write-Head "Opening the setup window"
Push-Location $root
try {
    & $py $wizard
    $rc = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($rc -ne 0) {
    Write-Host "`nSetup exited with code $rc." -ForegroundColor Yellow
    Write-Host "You can run Setup.bat again at any time — finished steps are not repeated."
    Read-Host "`nPress Enter to close"
}
exit $rc
