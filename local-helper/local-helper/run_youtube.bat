@echo off
setlocal
cd /d "%~dp0.."
if "%~1"=="" (
  echo Usage: local-helper\run_youtube.bat "https://www.youtube.com/watch?v=..."
  exit /b 2
)
py -3.11 -m pip install --upgrade "yt-dlp[default]" requests
py -3.11 local-helper\skimreel_helper.py %*
