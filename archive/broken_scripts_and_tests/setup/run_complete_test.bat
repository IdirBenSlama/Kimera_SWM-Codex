@echo off
echo ================================================================================
echo                    KIMERA MIRROR PORTAL COMPLETE TEST
echo ================================================================================
echo.

echo [1/3] Running direct Mirror Portal test...
echo --------------------------------------------------------------------------------
python test_kimera_portal.py
echo.

echo [2/3] Test results:
echo --------------------------------------------------------------------------------
dir /b kimera_portal_test_results_*.txt 2>nul
if exist kimera_portal_test_results_*.txt (
    echo Results files created successfully
) else (
    echo No result files found
)
echo.

echo [3/3] To start KIMERA server with Mirror Portal:
echo --------------------------------------------------------------------------------
echo Run: python start_kimera_server.py
echo Then visit: http://localhost:8001/docs
echo.

echo ================================================================================
echo                           TEST COMPLETE
echo ================================================================================
pause