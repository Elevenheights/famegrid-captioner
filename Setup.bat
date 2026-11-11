@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.10 or newer with Tcl/Tk and add Python to PATH.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
if not exist ".env" copy /y ".env.example" ".env" >nul
echo Setup complete. Edit .env to add your API key, then run Start Famegrid Captioner.bat.
pause
exit /b 0
:failed
echo Setup failed. Review the error above.
pause
exit /b 1
