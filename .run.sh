#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- GPU Detection ---
if lspci | grep -iq 'vga.*nvidia'; then
  echo "✅ NVIDIA GPU detected."
  export BASE_IMAGE_VAR="paperspace/gradient-base:pt211-tf215-cudatk120-py311-20240202"
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.nvidia.yml"
elif lspci | grep -iq 'vga.*amd'; then
  echo "✅ AMD GPU detected."
  export BASE_IMAGE_VAR="rocm/tensorflow:rocm6.0-tf2.15-python3.11"
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.amd.yml"
else
  echo "⚠️ No GPU NVIDIA or AMD detected. Starting on CPU mode."
  export BASE_IMAGE_VAR="python:3.11-slim"
  COMPOSE_FILES="-f docker-compose.yml"
fi

# --- Conditional Build Flag ---
# Initialize BUILD_FLAG as an empty string
BUILD_FLAG=""
# Check if the first argument passed to the script is "--build"
if [ "$1" == "--build" ]; then
  echo "🛠️  Build flag detected. Forcing image rebuild."
  BUILD_FLAG="--build"
fi

echo "📖 Using base image: ${BASE_IMAGE_VAR}"
echo "🚀 Starting Ai Delver Intelligence server..."

# The BUILD_FLAG variable will expand to "--build" or to nothing
docker-compose ${COMPOSE_FILES} up ${BUILD_FLAG} --remove-orphans