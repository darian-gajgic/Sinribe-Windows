' ===========================================================================
'  Sinribe — silent launcher.
'
'  Sinribe.bat is a batch file, so Windows flashes a console window for a
'  fraction of a second before the app appears. This wrapper starts the same
'  batch file hidden, which is what the desktop and Start Menu shortcuts point
'  at when Windows Script Host is available (make_shortcuts.ps1 falls back to
'  Sinribe.bat when it is not).
'
'  It is a convenience only — Sinribe.bat does the actual work and can always
'  be used directly.
' ===========================================================================
Option Explicit

Dim fso, shell, here, target, showWindow

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

here = fso.GetParentFolderName(WScript.ScriptFullName)
target = here & "\Sinribe.bat"

If Not fso.FileExists(target) Then
    MsgBox "Sinribe.bat was not found next to this file." & vbCrLf & _
           "Keep the Sinribe folder together, or run Sinribe.bat directly.", _
           vbExclamation, "Sinribe"
    WScript.Quit 1
End If

' Already installed: launch hidden (the app opens its own window).
' Not installed yet: show the console, because setup prints progress there
' before the wizard window appears.
If fso.FileExists(here & "\.venv\Scripts\pythonw.exe") Then
    showWindow = 0
Else
    showWindow = 1
End If

shell.Run """" & target & """", showWindow, False
