#!/bin/bash
echo "Running setup_env.sh..."
pip install "torchao>=0.16.0"

# Build and prepare Pantograph REPL binary
echo "=== [5/5] Aligning Pantograph native REPL binary ==="
mkdir -p /tmp/PyPantograph/pantograph
if [ -f "/usr/local/lib/python3.13/dist-packages/pantograph/pantograph-repl" ]; then
    cp /usr/local/lib/python3.13/dist-packages/pantograph/pantograph-repl /tmp/PyPantograph/pantograph/pantograph-repl
    chmod +x /tmp/PyPantograph/pantograph/pantograph-repl
    echo "Pantograph native binary aligned successfully from dist-packages!"
else
    echo "Error: Compiled binary not found in python distribution packages!"
    exit 1
fi
