
@echo off
cd /d "%~dp0"

set "USERPROFILE=%CD%"
set "HOME=%CD%"
set "STREAMLIT_CONFIG_DIR=%CD%\.streamlit"

start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"

venv\Scripts\python.exe -m streamlit run app.py

pause
