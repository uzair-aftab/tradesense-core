@echo off
title TradeSense v7 Public Gateway Server
cd /d "%~dp0"
set PYTHONPATH=%~dp0
call "C:\Python Virtual Environments\Personal Projects\TradeSense\Scripts\activate.bat"
python start_public_server.py
pause
