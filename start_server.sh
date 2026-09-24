#!/bin/bash
# Start persistent llama-server (Vulkan GPU) for the voice assistant.
set -euo pipefail

MODEL="$HOME/models/qwen2.5-coder-7b/qwen2.5-coder-7b-instruct-q6_k.gguf"
PORT=8080

if pgrep -f "llama-server.*$MODEL" > /dev/null; then
    echo "llama-server already running on port $PORT"
    exit 0
fi

nohup taskset -c 8-15 llama-server -m "$MODEL" -ngl 99 -t 8 -c 4096 --host 127.0.0.1 --port "$PORT" \
    >> ~/.local/opt/llama.cpp/server.log 2>&1 &

echo "llama-server starting on port $PORT (model loads ~5-10s)..."