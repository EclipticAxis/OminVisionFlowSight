@echo off
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" main.py
if errorlevel 1 pause
