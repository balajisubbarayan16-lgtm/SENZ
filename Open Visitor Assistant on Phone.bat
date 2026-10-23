@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project virtual environment was not found in .venv.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" web_app.py
pause
