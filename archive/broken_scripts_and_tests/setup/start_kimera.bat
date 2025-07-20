@echo off
REM ====================================================================
REM 🚀 KIMERA SYSTEM LAUNCHER - WINDOWS BATCH FILE
REM ====================================================================
REM This batch file makes it super easy to start KIMERA on Windows
REM Just double-click this file or run it from Command Prompt
REM ====================================================================

title KIMERA System Launcher

echo.
echo ================================================================================
echo 🚀 KIMERA SYSTEM LAUNCHER - WINDOWS EDITION
echo    Kinetic Intelligence for Multidimensional Analysis
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.10+ first.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if we're in the right directory
if not exist "backend" (
    if not exist "requirements.txt" (
        echo.
        echo ❌ KIMERA project files not found!
        echo 💡 Make sure you're running this from the KIMERA project directory.
        echo    Look for a directory containing: backend/, requirements.txt, README.md
        echo.
        pause
        exit /b 1
    )
)

echo ✅ KIMERA project files found

REM Ask user what they want to do
echo.
echo What would you like to do?
echo.
echo [1] Start KIMERA (normal mode)
echo [2] Start KIMERA (development mode with auto-reload)
echo [3] First time setup (create venv, install dependencies)
echo [4] Check system status
echo [5] Show help
echo [6] Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto start_normal
if "%choice%"=="2" goto start_dev
if "%choice%"=="3" goto setup
if "%choice%"=="4" goto check_status
if "%choice%"=="5" goto show_help
if "%choice%"=="6" goto exit
echo Invalid choice. Please try again.
pause
goto :eof

:start_normal
echo.
echo 🚀 Starting KIMERA in normal mode...
python run_kimera.py
goto end

:start_dev
echo.
echo 🚀 Starting KIMERA in development mode...
python run_kimera.py --dev
goto end

:setup
echo.
echo 🔧 Setting up KIMERA environment...
python run_kimera.py --setup
echo.
echo ✅ Setup complete! You can now start KIMERA normally.
pause
goto end

:check_status
echo.
echo 🔍 Checking KIMERA system status...
python run_kimera.py --help
pause
goto end

:show_help
echo.
echo 📚 KIMERA HELP INFORMATION
echo ========================
echo.
echo This batch file provides an easy way to start KIMERA on Windows.
echo.
echo What each option does:
echo   [1] Normal mode     - Starts KIMERA server for regular use
echo   [2] Development mode - Starts with auto-reload for development
echo   [3] First time setup - Creates virtual environment and installs dependencies
echo   [4] Check status     - Shows system information and help
echo   [5] Show help        - Shows this help information
echo.
echo KIMERA will be available at: http://localhost:8001
echo API documentation at: http://localhost:8001/docs
echo.
echo For troubleshooting:
echo   - Make sure Python 3.10+ is installed
echo   - Run option [3] for first time setup
echo   - Check that port 8001 is not already in use
echo.
pause
goto end

:exit
echo.
echo 👋 Goodbye!
goto end

:end
echo.
echo ================================================================================
echo 🎯 KIMERA LAUNCHER COMPLETE
echo ================================================================================
pause 