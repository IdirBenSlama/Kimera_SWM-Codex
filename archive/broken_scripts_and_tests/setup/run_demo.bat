@echo off
echo Running Mirror Portal Demonstration...
echo.
python demonstrate_mirror_portal.py > mirror_portal_output.txt 2>&1
type mirror_portal_output.txt
echo.
echo Output saved to mirror_portal_output.txt
pause