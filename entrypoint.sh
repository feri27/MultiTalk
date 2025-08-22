#!/bin/bash
set -e

echo "Checking and downloading models if not exist..."

MODELS=(
  "Wan-AI/Wan2.1-I2V-14B-480P"
  "TencentGameMate/chinese-wav2vec2-base"
  "hexgrad/Kokoro-82M"
  "MeiGen-AI/MeiGen-MultiTalk"
)

for model in "${MODELS[@]}"; do
  target="./weights/$(basename "$model")"
  if [ ! -d "$target" ]; then
    echo "Downloading $model..."
    huggingface-cli download "$model" --local-dir "$target" --resume-download
  else
    echo "$model already exists, skipping..."
  fi
done

# jalankan app
exec "$@"
