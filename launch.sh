#!/bin/bash
# launch.sh — Start EaglePlot (Linux)
# Run from a terminal after running ./setup.sh once:  ./launch.sh

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run ./setup.sh first."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

source .venv/bin/activate
exec streamlit run eagleplot/Home.py
