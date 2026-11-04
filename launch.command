#!/bin/bash
# launch.command — Start Plottle (macOS)
# Double-click this file in Finder after running setup.command once.

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Please run setup.command first."
    echo
    read -r -p "Press Enter to close..."
    exit 1
fi

source .venv/bin/activate
streamlit run plottle/Home.py
read -r -p "Press Enter to close..."
