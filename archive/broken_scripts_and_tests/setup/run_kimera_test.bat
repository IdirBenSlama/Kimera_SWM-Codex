@echo off
cls
echo ================================================================================
echo                    KIMERA MIRROR PORTAL INTEGRATION TEST
echo ================================================================================
echo.
echo This will verify the Mirror Portal integration and provide instructions
echo for running the full KIMERA system.
echo.
echo Press any key to start...
pause >nul

cls
python test_kimera_live.py

echo.
echo ================================================================================
echo                              TEST COMPLETE
echo ================================================================================
echo.
echo To start KIMERA server with Mirror Portal:
echo   python start_kimera_server.py
echo.
echo Then open: http://localhost:8001/docs
echo.
pause