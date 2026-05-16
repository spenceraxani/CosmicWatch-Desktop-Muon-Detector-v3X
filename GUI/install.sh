#!/bin/bash
# CosmicWatch GUI — Dependency Installer
# Installs all required Python packages for GUI.py
# Works on macOS, Linux, and Windows (Git Bash / WSL)

echo "============================================"
echo "  CosmicWatch GUI — Installing dependencies"
echo "============================================"

# Use pip3 if available, fall back to pip
if command -v pip3 &>/dev/null; then
    PIP=pip3
elif command -v pip &>/dev/null; then
    PIP=pip
else
    echo "ERROR: pip not found. Please install Python first."
    echo "  macOS:   brew install python"
    echo "  Windows: https://www.python.org/downloads/"
    exit 1
fi

echo "Using: $($PIP --version)"
echo ""

$PIP install \
    PyQt5 \
    pyqtgraph \
    numpy \
    scipy \
    matplotlib \
    pyserial \
    numpy-stl \
    Pillow \
    PyOpenGL

# PyOpenGL_accelerate is optional — speeds up rendering but can fail on Windows
echo ""
echo "Installing PyOpenGL_accelerate (optional, safe to skip if it fails)..."
$PIP install PyOpenGL_accelerate || echo "  Skipped PyOpenGL_accelerate (not required)."

echo ""
echo "============================================"
echo "  Done! Run the GUI with:"
echo "    python GUI.py"
echo "============================================"
