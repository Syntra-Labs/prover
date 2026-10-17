#!/usr/bin/env bash
# ==============================================================================
# Train R4 Geometric Model on Mathlib 4 Theorem Steps
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_BIN="$REPO_ROOT/vendor/uor-r4/target/release/examples/geometric-stack"

if [ ! -f "$GS_BIN" ]; then
    echo "Building geometric-stack binary..."
    cargo build --release -p uor-r4-training --example geometric-stack --manifest-path "$REPO_ROOT/vendor/uor-r4/Cargo.toml"
fi

DATA_DIR="$REPO_ROOT/data/mathlib4"
TRAIN_FILE="$DATA_DIR/train.u16"
VALID_FILE="$DATA_DIR/valid.u16"
TOKENIZER="$DATA_DIR/tokenizer.json"

for f in "$TRAIN_FILE" "$VALID_FILE" "$TOKENIZER"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f. Run scripts/prepare_mathlib_for_r4.py first."
        exit 1
    fi
done

OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/r4_mathlib_$(date +%Y%m%d_%H%M%S)}"
# Binary creates out dir exclusively; ensure it doesn't pre-exist
if [ -d "$OUT_DIR" ]; then
    echo "WARNING: $OUT_DIR exists; removing..."
    rm -rf "$OUT_DIR"
fi

STEPS="${STEPS:-500}"
BATCH="${BATCH:-8}"
CONTEXT="${CONTEXT:-128}"
WIDTH="${WIDTH:-128}"
HEADS="${HEADS:-4}"
LAYERS="${LAYERS:-4}"
PATTERN="${PATTERN:-rrar}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cpu}"

echo "=========================================================="
echo " Training R4 on Mathlib 4 Theorem Proof Data"
echo " Pattern: $PATTERN | Width: $WIDTH | Heads: $HEADS | Layers: $LAYERS"
echo " Context: $CONTEXT | Steps: $STEPS | Batch: $BATCH"
echo " Device: $DEVICE"
echo " Output: $OUT_DIR"
echo "=========================================================="

"$GS_BIN" train \
    train="$TRAIN_FILE" \
    valid="$VALID_FILE" \
    tokenizer="$TOKENIZER" \
    out="$OUT_DIR" \
    arch=geometric \
    width="$WIDTH" \
    heads="$HEADS" \
    layers="$LAYERS" \
    pattern="$PATTERN" \
    context="$CONTEXT" \
    read=l2 \
    rotation=true \
    steps="$STEPS" \
    batch="$BATCH" \
    lr="$LR" \
    warmup=50 \
    min_lr=0.1 \
    weight_decay=0.1 \
    clip=1.0 \
    eval_every=100 \
    eval_windows=16 \
    final_windows=64 \
    checkpoint_every=250 \
    sample_tokens=0 \
    device="$DEVICE" \
    2>&1 | tee "$OUT_DIR/training.log"

echo "==> Training complete! Checkpoint saved in: $OUT_DIR/model"
