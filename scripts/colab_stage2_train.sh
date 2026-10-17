#!/usr/bin/env bash
# ==============================================================================
# Google Colab Stage 2 Response-Masked SFT Launcher (Native weighted_loss)
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_BIN="$REPO_ROOT/vendor/uor-r4/target/release/examples/geometric-stack"
DATA_DIR="$REPO_ROOT/data/mathlib4"

# 1. Stage 1 Init Model
INIT_MODEL="${INIT_MODEL:-$REPO_ROOT/reports/r4_colab_run_20261017_053354/model}"

# 2. Output directory
OUT_DIR="$REPO_ROOT/reports/r4_stage2_masked_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUT_DIR")"

echo "=========================================================="
echo " Starting UOR-R4 Stage 2 Masked SFT on Goal->Tactic Pairs"
echo " Initialized from: $INIT_MODEL"
echo " Output:           $OUT_DIR"
echo "=========================================================="

"$GS_BIN" train \
    out="$OUT_DIR" \
    tokenizer="$DATA_DIR/tokenizer.json" \
    train="$DATA_DIR/train.u16" \
    train_mask="$DATA_DIR/train.mask" \
    valid="$DATA_DIR/valid.u16" \
    init="$INIT_MODEL" \
    device="${DEVICE:-cuda}" \
    steps="${STEPS:-1000}" \
    batch="${BATCH:-8}" \
    lr="${LR:-0.001}" \
    warmup=100 \
    min_lr=0.1 \
    weight_decay=0.1 \
    eval_every=200 \
    checkpoint_every=200

echo "==> Stage 2 Masked SFT finished successfully. Artifacts saved in: $OUT_DIR"
bash "$REPO_ROOT/scripts/persist_checkpoint_cloud.sh" "$OUT_DIR" "stage2_masked_sft"
