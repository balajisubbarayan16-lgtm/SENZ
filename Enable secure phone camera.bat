@echo off
echo Make sure SPONGE is running and Tailscale is installed and signed in on this PC.
echo The phone must also have Tailscale installed and signed in to the same account.
echo.
set "TS_EXE="
where tailscale.exe >nul 2>nul && set "TS_EXE=tailscale.exe"
if not defined TS_EXE if exist "%ProgramFiles%\Tailscale\tailscale.exe" set "TS_EXE=%ProgramFiles%\Tailscale\tailscale.exe"
if not defined TS_EXE if exist "%ProgramFiles(x86)%\Tailscale IPN\tailscale.exe" set "TS_EXE=%ProgramFiles(x86)%\Tailscale IPN\tailscale.exe"
if not defined TS_EXE if exist "%ProgramFiles%\Tailscale IPN\tailscale.exe" set "TS_EXE=%ProgramFiles%\Tailscale IPN\tailscale.exe"
if not defined TS_EXE if exist "%LocalAppData%\Tailscale\tailscale.exe" set "TS_EXE=%LocalAppData%\Tailscale\tailscale.exe"
if not defined TS_EXE (
  echo Windows could not find the Tailscale command. Please send me a screenshot of this window.
  pause
  exit /b 1
)
"%TS_EXE%" serve --bg 8765
if errorlevel 1 (
  echo Tailscale could not enable HTTPS. Please send me a screenshot of this window.
  pause
  exit /b 1
)
echo.
echo Secure link status:
"%TS_EXE%" serve status
echo.
echo Open the HTTPS address shown above on your phone while Tailscale is connected.
pause
