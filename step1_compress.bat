@echo off
rem ===== STEP 1: compress OLMoE with the field engine (one command) =====
rem Downloads Q4_K_M GGUF (~4.4 GB) into hf_cache\, replaces experts with the
rem field (rank 32), saves a normal HF model into results\field_..._r32 and
rem verifies it. Extra options: --low-mem  --threads 4  --cleanup
cd /d "%~dp0"
python hf_pipeline.py %*
echo.
pause
