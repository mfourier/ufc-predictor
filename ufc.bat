@echo off
set IMAGE_NAME=ufc-cli

echo 🥋 Starting UFC Fight Predictor CLI...

:: Check if image exists
docker image inspect %IMAGE_NAME% >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Docker image not found. Building it now...
    docker build -t %IMAGE_NAME% .
)

echo 🎮 Launching the CLI app...
docker run -it %IMAGE_NAME%

pause
