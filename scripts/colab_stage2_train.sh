#!/usr/bin/env bash
# ==============================================================================
# Google Colab Stage 2 Dialogue-Train Launcher (Response-Masked SFT)
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_BIN="$REPO_ROOT/vendor/uor-r4/target/release/examples/geometric-stack"
DATA_DIR="$REPO_ROOT/data/mathlib4"

# 1. Prepare .uort files and manifests if missing
python3 "$REPO_ROOT/scripts/prepare_dialogue_uort.py" "$DATA_DIR"

# 2. Stage 1 Init Model
INIT_MODEL="${INIT_MODEL:-$REPO_ROOT/reports/r4_colab_run_20261017_053354/model}"

# 3. Output directory
OUT_DIR="$REPO_ROOT/reports/r4_stage2_dialogue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUT_DIR")"

echo "=========================================================="
echo " Starting UOR-R4 Stage 2 Fine-Tuning on Goal->Tactic Pairs"
echo " Initialized from: $INIT_MODEL"
echo " Output:           $OUT_DIR"
echo "=========================================================="

"$GS_BIN" train \
    init="$INIT_MODEL" \
    train="$DATA_DIR/train.u16" \
    valid="$DATA_DIR/valid.u16" \
    tokenizer="$DATA_DIR/tokenizer.json" \
    out="$OUT_DIR" \
    steps="${STEPS:-1000}" \
    batch="${BATCH:-8}" \
    lr="${LR:-0.001}" \
    warmup=100 \
    min_lr=0.1 \
    weight_decay=0.1 \
    eval_every=200 \
    checkpoint_every=200 \
    context=256

echo "==> Stage 2 Fine-tuning finished successfully. Artifacts saved in: $OUT_DIR"
