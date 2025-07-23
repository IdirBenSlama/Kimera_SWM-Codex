@echo off
REM KIMERA Windows Startup Script
REM =============================

echo.
echo ============================================================
echo                    KIMERA SWM SYSTEM
echo            Semantic Web Mind - Alpha Prototype
echo ============================================================
echo.

REM Set UTF-8 encoding
chcp 65001 > nul

REM Set environment variables
set PYTHONIOENCODING=utf-8

REM Load DATABASE_URL from .env if it exists
if exist .env (
    echo Loading environment from .env file...
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%b"=="" set %%a=%%b
    )
)

REM Verify PostgreSQL connection
echo Checking PostgreSQL connection...
python -c "import os; print(f'DATABASE_URL: {os.environ.get(\"DATABASE_URL\", \"NOT SET\")}')"

REM Check Python version
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

REM Display options
echo Select startup mode:
echo.
echo 1. Normal Startup (Recommended)
echo 2. System Repair
echo 3. Aerospace Startup (Full diagnostics)
echo 4. Exit
echo.

choice /c 1234 /n /m "Enter your choice (1-4): "

if errorlevel 4 exit /b 0
if errorlevel 3 goto aerospace
if errorlevel 2 goto repair
if errorlevel 1 goto normal

:normal
echo.
echo Starting KIMERA in normal mode...
python kimera_windows_startup.py
goto end

:repair
echo.
echo Running system repair...
python kimera_system_repair_windows.py
echo.
echo Repair complete. Press any key to start KIMERA...
pause > nul
python kimera_windows_startup.py
goto end

:aerospace
echo.
echo Starting KIMERA with aerospace-grade diagnostics...
python kimera_aerospace_startup.py
goto end

:end
echo.
echo KIMERA shutdown complete.
pause