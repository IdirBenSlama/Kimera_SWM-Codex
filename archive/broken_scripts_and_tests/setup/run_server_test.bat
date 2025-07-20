@echo off
echo ================================================================================
echo                    KIMERA SERVER STARTUP TEST
echo ================================================================================
echo.
echo Starting KIMERA server with Mirror Portal integration...
echo This will:
echo   1. Start the KIMERA server on port 8001
echo   2. Test API endpoints
echo   3. Verify Mirror Portal functionality
echo.
echo Press Ctrl+C to stop the server when done.
echo.

set PYTHONPATH=%cd%
python test_server_startup.py

echo.
echo Server test completed.
pause