#!/bin/bash
# setup.command — First-run setup for Plottle (macOS)
# Double-click this file in Finder to create the virtual environment
# and install all dependencies.

cd "$(dirname "$0")"

echo "=== Plottle Setup ==="
echo

# ── Check Python 3 is available ──────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 was not found."
    echo "Please install Python 3.9 or later from https://www.python.org/downloads/"
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

# ── Check the Python version (pyproject.toml: >=3.9, !=3.9.7) ─────────────────
if ! python3 -c 'import sys; v = sys.version_info[:3]; sys.exit(0 if v >= (3, 9) and v != (3, 9, 7) else 1)'; then
    echo "ERROR: Plottle needs Python 3.9 or later (but not 3.9.7)."
    echo "Found: $(python3 --version 2>&1)"
    echo "Install a newer Python from https://www.python.org/downloads/"
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

# ── Create virtual environment (skip if already exists) ──────────────────────
if [ -d ".venv" ]; then
    echo "Virtual environment already exists -- skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Could not create virtual environment."
        echo "Make sure Python 3.9 or later is installed."
        echo
        read -r -p "Press Enter to close..."
        exit 1
    fi
    echo "Done."
fi

echo
echo "Installing dependencies (this may take a few minutes on first run)..."
echo

if ! source .venv/bin/activate; then
    echo "ERROR: Could not activate .venv. Delete the .venv folder and run setup again."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi
pip install -e ".[formats,nist]"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Dependency installation failed."
    echo "Check the output above for details."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

echo
echo "============================================================"
echo " Setup complete!"
echo " Double-click launch.command to start Plottle."
echo "============================================================"
echo
read -r -p "Press Enter to close..."
