@echo off
cd /d "%~dp0"
set "PYEXE=%~dp0venv\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [start.bat] Bundled venv not found: %PYEXE%
    echo [start.bat] Fresh clone? Create it and install dependencies first:
    echo [start.bat]   python -m venv venv
    echo [start.bat]   venv\Scripts\python.exe -m pip install -r requirements.txt
    echo [start.bat] Falling back to "python" on PATH...
    set "PYEXE=python"
)
"%PYEXE%" main.py
if errorlevel 1 pause
