#!/bin/bash

IMAGE_NAME="ufc-cli"

# Usa sudo si docker no tiene permisos
DOCKER_CMD="docker"
if ! docker info > /dev/null 2>&1; then
  DOCKER_CMD="sudo docker"
fi

echo "🥋 Starting UFC Fight Predictor CLI..."

if ! $DOCKER_CMD image inspect $IMAGE_NAME >/dev/null 2>&1; then
  echo "📦 Docker image not found. Building it now..."
  $DOCKER_CMD build -t $IMAGE_NAME .
fi

echo "🎮 Launching the CLI app..."
$DOCKER_CMD run -it $IMAGE_NAME
