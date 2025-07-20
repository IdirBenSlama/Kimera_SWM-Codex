@echo off
echo Running Mirror Portal Integration Verification...
echo.
set PYTHONPATH=%cd%
python verify_integration.py
echo.
echo Verification complete. Check integration_verification_*.txt for details.
pause