@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow the Windows setup in README.md first.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "YOLO_AUTOINSTALL=false"
set "YOLO_OFFLINE=true"
".venv\Scripts\python.exe" -m vision_servo_tracker.desktop --source http://127.0.0.1:8080/stream %*
set "TRACK_EXIT=%ERRORLEVEL%"
if not "%TRACK_EXIT%"=="0" pause
exit /b %TRACK_EXIT%
