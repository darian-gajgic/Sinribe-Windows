# =============================================================================
#  Create the Desktop and Start Menu shortcuts.
#
#  Both are per-user (they land in the current profile), so no administrator
#  rights are involved and nothing is written outside the user's own folders.
#  Run it again after moving the Sinribe folder to repoint them.
#
#      powershell -ExecutionPolicy Bypass -File tools\make_shortcuts.ps1
#      powershell -ExecutionPolicy Bypass -File tools\make_shortcuts.ps1 -Remove
# =============================================================================
#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch] $Remove,
    [switch] $NoDesktop
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$icon = Join-Path $root 'assets\sinribe.ico'

$desktop   = [Environment]::GetFolderPath('Desktop')
$startMenu = Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs'
$links = @(
    (Join-Path $desktop   'Sinribe.lnk'),
    (Join-Path $startMenu 'Sinribe.lnk')
)
if ($NoDesktop) { $links = @($links[1]) }

if ($Remove) {
    foreach ($l in $links) {
        if (Test-Path $l) { Remove-Item $l -Force; Write-Host "removed $l" }
    }
    exit 0
}

# Prefer the VBScript wrapper: it starts the launcher hidden, so no console
# window flashes on the way to the app. Windows Script Host can be disabled by
# policy, and VBScript is on its way out of Windows, so fall back to the batch
# file — which works everywhere, it just blinks once.
$wsh    = Join-Path $env:SystemRoot 'System32\wscript.exe'
$vbs    = Join-Path $root 'Sinribe.vbs'
$target = Join-Path $root 'Sinribe.bat'
if ((Test-Path $wsh) -and (Test-Path $vbs)) { $target = $vbs }

$shell = New-Object -ComObject WScript.Shell
foreach ($linkPath in $links) {
    $dir = Split-Path -Parent $linkPath
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $sc = $shell.CreateShortcut($linkPath)
    $sc.TargetPath       = $target
    $sc.WorkingDirectory = $root
    $sc.Description      = 'Offline transcription with speaker labels'
    if (Test-Path $icon) { $sc.IconLocation = $icon }
    $sc.Save()
    Write-Host "created $linkPath"
}
