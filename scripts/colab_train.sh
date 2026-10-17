#!/usr/bin/env bash
# ==============================================================================
# Google Colab Training Launcher for R4 Models (Supports CPU & GPU)
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_BIN="$REPO_ROOT/vendor/uor-r4/target/release/examples/geometric-stack"

if [ ! -f "$GS_BIN" ]; then
    echo "ERROR: Training binary not found at $GS_BIN. Run scripts/colab_bootstrap.sh first."
    exit 1
fi

# Ensure Rust/Cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Detect device (cuda if GPU available, else cpu)
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    DEVICE="${DEVICE:-cuda}"
    TF32_FLAG="tf32=true"
else
    DEVICE="cpu"
    TF32_FLAG=""
fi

# Configuration: default to Mathlib 4 dataset
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/mathlib4}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.u16}"
VALID_FILE="${VALID_FILE:-$DATA_DIR/valid.u16}"
TOKENIZER="${TOKENIZER:-$DATA_DIR/tokenizer.json}"

# Checkpoint output path
if [ -d "/content/drive/MyDrive" ]; then
    DEFAULT_OUT="/content/drive/MyDrive/r4_checkpoints/run_$(date +%Y%m%d_%H%M%S)"
else
    DEFAULT_OUT="$REPO_ROOT/reports/r4_colab_run_$(date +%Y%m%d_%H%M%S)"
fi
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT}"

# Parent directory can exist, but target output directory must not exist prior to training
mkdir -p "$(dirname "$OUT_DIR")"
if [ -d "$OUT_DIR" ]; then
    echo "WARNING: $OUT_DIR exists. Removing to allow clean creation by geometric-stack..."
    rm -rf "$OUT_DIR"
fi

# Model hyperparameters
MODEL_SIZE="${MODEL_SIZE:-8m}"

if [ "$MODEL_SIZE" = "8m" ]; then
    WIDTH=288
    HEADS=6
    LAYERS=6
    PATTERN="rrarra"
    CONTEXT="${CONTEXT:-256}"
    BATCH="${BATCH:-16}"
    LR="${LR:-0.004}"
    STEPS="${STEPS:-7324}"
elif [ "$MODEL_SIZE" = "29m" ]; then
    WIDTH=576
    HEADS=8
    LAYERS=10
    PATTERN="rrarrarrar"
    CONTEXT="${CONTEXT:-384}"
    BATCH="${BATCH:-16}"
    LR="${LR:-0.0005}"
    STEPS="${STEPS:-70609}"
elif [ "$MODEL_SIZE" = "214m" ]; then
    WIDTH=1536
    HEADS=24
    LAYERS=16
    PATTERN="rrarrarrarrarrar"
    CONTEXT="${CONTEXT:-384}"
    BATCH="${BATCH:-16}"
    LR="${LR:-0.0005}"
    STEPS="${STEPS:-70609}"
else
    echo "Unknown MODEL_SIZE: $MODEL_SIZE (options: 8m, 29m, 214m)"
    exit 1
fi

EVAL_EVERY="${EVAL_EVERY:-250}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
READ_METRIC="${READ_METRIC:-l2}" # lorentz or l2
RESUME="${RESUME:-}" # Set to path of OLD_ROOT/checkpoint to resume

# Validate required data files exist
for f in "$TRAIN_FILE" "$VALID_FILE" "$TOKENIZER"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required input file: $f"
        echo "Please ensure data is prepared by running scripts/prepare_mathlib_for_r4.py first."
        exit 1
    fi
done

RESUME_ARG=()
if [ -n "$RESUME" ]; then
    echo "Resuming from checkpoint: $RESUME"
    RESUME_ARG=(resume="$RESUME")
fi

echo "=========================================================="
echo " Starting UOR-R4 Training on Google Colab"
echo " Device: $DEVICE"
echo " Model: $MODEL_SIZE (Width=$WIDTH, Heads=$HEADS, Layers=$LAYERS, Context=$CONTEXT)"
echo " Pattern: $PATTERN"
echo " Steps: $STEPS | Batch: $BATCH | LR: $LR"
echo " Output directory: $OUT_DIR"
echo "=========================================================="

ARGS=(
    train="$TRAIN_FILE"
    valid="$VALID_FILE"
    tokenizer="$TOKENIZER"
    out="$OUT_DIR"
    arch=geometric
    width="$WIDTH"
    heads="$HEADS"
    layers="$LAYERS"
    pattern="$PATTERN"
    context="$CONTEXT"
    read="$READ_METRIC"
    rotation=true
    steps="$STEPS"
    batch="$BATCH"
    lr="$LR"
    warmup=200
    min_lr=0.1
    weight_decay=0.1
    clip=1.0
    eval_every="$EVAL_EVERY"
    eval_windows=64
    final_windows=256
    checkpoint_every="$CHECKPOINT_EVERY"
    device="$DEVICE"
)

if [ "$DEVICE" = "cuda" ]; then
    ARGS+=(tf32=true data_parallel=1)
fi

"$GS_BIN" train \
    "${ARGS[@]}" \
    ${RESUME_ARG[@]+"${RESUME_ARG[@]}"}

echo "==> Training finished successfully. Artifacts saved in: $OUT_DIR"
