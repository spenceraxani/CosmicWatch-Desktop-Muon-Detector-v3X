@echo off
REM CosmicWatch GUI — Dependency Installer for Windows
REM Double-click this file, or run it from Command Prompt / PowerShell

echo ============================================
echo   CosmicWatch GUI — Installing dependencies
echo ============================================
echo.

python -m pip install PyQt5 pyqtgraph numpy scipy matplotlib pyserial numpy-stl Pillow PyOpenGL

echo.
echo Installing PyOpenGL_accelerate (optional, safe to skip if it fails)...
python -m pip install PyOpenGL_accelerate
if errorlevel 1 echo   Skipped PyOpenGL_accelerate (not required).

echo.
echo ============================================
echo   Done! Run the GUI with:
echo     python GUI.py
echo ============================================
echo.
pause
