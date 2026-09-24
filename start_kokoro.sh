#!/bin/bash
# Start persistent Kokoro TTS server for the voice assistant.
set -euo pipefail

if pgrep -f "[k]okoro_server.py" > /dev/null; then
    echo "kokoro server already running on port 8081"
    exit 0
fi

export HF_HUB_OFFLINE=1
nohup taskset -c 0-7 /home/yassine/.local/venvs/kokoro/bin/python \
    /home/yassine/code/speak/kokoro_server.py \
    >> ~/.local/opt/llama.cpp/kokoro.log 2>&1 &

echo "kokoro server starting on port 8081 (loads model ~10s)..."