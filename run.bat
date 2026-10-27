@echo off
REM ==============================================================================
REM C12 Aerial Thermal Human Detection Ground Station (Windows Launcher)
REM ==============================================================================

cd /d "%~dp0"

REM Activate virtual environment if present
if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
) else if exist "%~dp0c12_thermal_detection\.venv\Scripts\activate.bat" (
    call "%~dp0c12_thermal_detection\.venv\Scripts\activate.bat"
)

REM Set PYTHONPATH
set PYTHONPATH=%~dp0;%~dp0c12_thermal_detection;%PYTHONPATH%

python main.py %*

