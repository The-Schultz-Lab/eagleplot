@echo off
cd /d "%~dp0"

if not exist .venv (
    echo ERROR: Virtual environment not found.
    echo Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
streamlit run eagleplot/Home.py
pause
