@echo off
title TradeSense v7 Local Terminal
cd /d "%~dp0"
set PYTHONPATH=%~dp0
call "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\activate.bat"
python run.py
pause
