@echo off
rem ===== STEP 2: chat with the compressed model (artifact in results\) =====
rem Auto-finds the newest field_* artifact. Options: --temperature 0.8 --max-new 400
cd /d "%~dp0"
python hf_chat.py %*
echo.
pause
