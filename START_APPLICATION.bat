@echo off
title Internal Automation System
color 0A

echo ====================================================
echo          INTERNAL AUTOMATION SYSTEM
echo ====================================================
echo.
echo [*] Initializing application...
echo.

REM Create necessary folders if they don't exist
if not exist "uploads" mkdir uploads
if not exist "merged_pdfs" mkdir merged_pdfs
if not exist "templates" mkdir templates
if not exist "static" mkdir static

echo [*] Starting server...
echo.

REM Start the application and open browser after 3 seconds
start "" http://localhost:5000
timeout /t 3 /nobreak >nul

python main.py

echo.
echo ====================================================
echo Application stopped.
echo ====================================================
pause