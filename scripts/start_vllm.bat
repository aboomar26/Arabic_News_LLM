@echo off
title Arabic News NLP - Launcher
color 0A

echo.
echo  ================================
echo   Arabic News NLP - Starting...
echo  ================================
echo.

:: Activate conda env
call conda activate nlpnews

:: Move to project root
cd /d "%~dp0.."

:: Step 1: Start vLLM in a new window
echo [1/2] Starting vLLM server...
start "vLLM Server" cmd /k "conda activate nlpnews && cd /d "%~dp0.." && vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype=half --gpu-memory-utilization 0.8 --max_lora_rank 64 --enable-lora --lora-modules news-lora=./model"

:: Wait for vLLM to load
echo Waiting for vLLM to start (90 seconds)...
timeout /t 90 /nobreak

:: Step 2: Start FastAPI in a new window
echo [2/2] Starting FastAPI server...
start "FastAPI Server" cmd /k "conda activate nlpnews && cd /d "%~dp0.." && uvicorn app.main:app --port 8080"

:: Wait for FastAPI to start
timeout /t 5 /nobreak

:: Step 3: Open frontend in browser
echo Opening frontend...
start http://localhost:3000

:: Step 4: Start frontend server in a new window
start "Frontend Server" cmd /k "cd /d "%~dp0.."\frontend && python -m http.server 3000"

echo.
echo  ================================
echo   All services started!
echo.
echo   vLLM     → localhost:8000
echo   FastAPI  → localhost:8080
echo   Frontend → localhost:3000
echo  ================================
echo.
pause