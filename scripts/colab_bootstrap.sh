#!/usr/bin/env bash
# ==============================================================================
# Google Colab Bootstrap for UOR-R4 Training (Supports GPU & CPU)
# ==============================================================================
set -euo pipefail

echo "==> [1/4] Inspecting Compute Environment (GPU / CPU)..."
USE_CUDA=0
CAP=""

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    RAW_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')

    case "$GPU_NAME" in
        *A100*) CAP=80 ;;
        *H100*|*H200*) CAP=90 ;;
        *L4*) CAP=89 ;;
        *T4*) CAP=75 ;;
        *V100*) CAP=70 ;;
        *) CAP=$RAW_CAP ;;
    esac

    echo "Detected GPU: $GPU_NAME (CUDA Compute Capability: $CAP)"
    if [ "${CAP:-0}" -ge 80 ]; then
        export CUDA_COMPUTE_CAP=$CAP
        USE_CUDA=1
    else
        echo "Note: candle-kernels 0.9.2 bf16 WMMA requires sm_80+ (Ampere/L4/A100). Falling back to CPU build on $GPU_NAME (sm_$CAP)..."
        USE_CUDA=0
    fi
else
    echo "No NVIDIA GPU detected. Proceeding with CPU acceleration..."
fi

echo "==> [2/4] Installing / Verifying Rust toolchain..."
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed: $(cargo --version)"
fi
export PATH="$HOME/.cargo/bin:$PATH"

echo "==> [3/4] Installing Python requirements..."
pip install --quiet --upgrade pip
pip install --quiet torch safetensors tokenizers huggingface_hub

echo "==> [4/4] Building geometric-stack training binary..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UOR_R4_DIR="$REPO_ROOT/vendor/uor-r4"

if [ ! -d "$UOR_R4_DIR" ]; then
    echo "Cloning vendor/uor-r4 from https://github.com/UOR-Foundation/uor-r4..."
    mkdir -p "$(dirname "$UOR_R4_DIR")"
    git clone https://github.com/UOR-Foundation/uor-r4 "$UOR_R4_DIR"
fi

if [ ! -d "$REPO_ROOT/vendor/LeanDojo-v2" ] && [ ! -d "$REPO_ROOT/vendor/leandojo" ]; then
    echo "Cloning LeanDojo-v2 from https://github.com/lean-dojo/LeanDojo-v2.git..."
    git clone https://github.com/lean-dojo/LeanDojo-v2.git "$REPO_ROOT/vendor/LeanDojo-v2"
fi

cd "$UOR_R4_DIR"
git checkout bbbdd362 || true
if [ -f "$REPO_ROOT/scripts/patches/uor_r4_train_mask.patch" ]; then
    git apply --check "$REPO_ROOT/scripts/patches/uor_r4_train_mask.patch" 2>/dev/null && \
        git apply "$REPO_ROOT/scripts/patches/uor_r4_train_mask.patch" || true
fi
if [ "$USE_CUDA" -eq 1 ]; then
    echo "Compiling with CUDA support (Compute Capability: $CAP)..."
    CARGO_INCREMENTAL=0 CUDA_COMPUTE_CAP=$CAP cargo build --release \
        -p uor-r4-training \
        --features cuda \
        --example geometric-stack
else
    echo "Compiling with CPU support..."
    cargo build --release \
        -p uor-r4-training \
        --example geometric-stack
fi

BINARY_PATH="$UOR_R4_DIR/target/release/examples/geometric-stack"
if [ -f "$BINARY_PATH" ]; then
    echo "==> SUCCESS: Built geometric-stack at $BINARY_PATH"
else
    echo "ERROR: Binary not found at $BINARY_PATH"
    exit 1
fi
