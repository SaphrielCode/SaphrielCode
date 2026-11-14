@echo off
cd /d "%~dp0"
if not exist ".venv-data\Scripts\python.exe" (
  echo The local Python environment was not found. See ChatInstructions.txt.
  pause
  exit /b 1
)
".venv-data\Scripts\python.exe" tools\restore_published_artifacts.py
if errorlevel 1 (
  pause
  exit /b 1
)
".venv-data\Scripts\python.exe" tools\chat_server.py --open-browser
if errorlevel 1 pause
