#!/bin/bash
set -e

echo "=== [1/5] System packages & compilers ==="
apt-get update -qq
apt-get install -y -qq build-essential python3-dev patchelf curl git > /dev/null
pip install "torchao>=0.16.0"  --upgrade -q pip maturin

echo "=== [2/5] Rust Toolchain ==="
if [ ! -d "$HOME/.cargo/bin" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"
echo "Rust compiler: $(rustc --version)"

echo "=== [3/5] Elan & Lean 4 ==="
if [ ! -d "$HOME/.elan/bin" ]; then
    curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
fi
export PATH="$HOME/.elan/bin:$PATH"
echo "Elan: $(elan --version)"

echo "=== [4/5] Python packages (LeanDojo & PyTorch) ==="
pip install "torchao>=0.16.0"  -q lean-dojo-v2 transformers peft accelerate bitsandbytes nest_asyncio
pip install "torchao>=0.16.0"  -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "=== [5/5] PyPantograph ==="
if [ ! -d "/tmp/PyPantograph" ]; then
    git clone https://github.com/stanford-centaur/PyPantograph /tmp/PyPantograph
fi
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
pip install "torchao>=0.16.0"  -q git+https://github.com/stanford-centaur/PyPantograph

python3 -c "import pantograph; print('[OK] Pantograph successfully imported!')"

echo "=================================================="
echo "=== Complete environment setup successful! ==="
echo "=================================================="

pip install "torchao>=0.16.0"  "torchao>=0.16.0"












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
