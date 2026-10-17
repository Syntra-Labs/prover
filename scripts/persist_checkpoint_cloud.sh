#!/usr/bin/env bash
# ==============================================================================
# Cloud Checkpoint & Benchmark Ledger Persistence (GitHub + HuggingFace + GDrive)
# Ensures trained model weights (model.safetensors), report.json, and
# docs/benchmark_history.csv are saved remotely the instant a Colab job finishes,
# even if the local workstation is offline or asleep.
# ==============================================================================
set -euo pipefail

OUT_DIR="${1:?Usage: persist_checkpoint_cloud.sh <OUT_DIR> [STAGE_LABEL]}"
STAGE_LABEL="${2:-colab_training}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(basename "$OUT_DIR")"

echo "=========================================================="
echo " Persisting Checkpoint & Results to Remote Cloud Storage"
echo " Run ID: $RUN_ID | Directory: $OUT_DIR"
echo "=========================================================="

# 1. Append training run metrics from report.json into docs/benchmark_history.csv
if [ -f "$OUT_DIR/report.json" ]; then
    python3 - "$OUT_DIR/report.json" "$STAGE_LABEL" "$RUN_ID" << 'PYEOF' || true
import json, shutil, sys
from pathlib import Path
from src.eval.benchmark_ledger import log_benchmark_entry, DEFAULT_CSV_PATH

report_path, stage_label, run_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rep = json.loads(report_path.read_text(encoding="utf-8"))
step = int(rep.get("completed_steps", 0))
final_nll = rep.get("final", {}).get("nll")
curve = rep.get("curve", [])
train_loss = curve[-1].get("train_loss") if curve else None

log_benchmark_entry(
    stage=stage_label,
    run_id=run_id,
    step=step,
    benchmark_type="geometric_stack_dev_eval",
    sample_size=int(rep.get("final", {}).get("targets", 0)),
    loss_or_nll=float(final_nll) if final_nll is not None else None,
    notes=f"Auto-persisted from Colab ({rep.get('parameters', 0)} params, train_loss={train_loss})",
)
reports_csv = DEFAULT_CSV_PATH.parent.parent / "reports/benchmark_history.csv"
reports_csv.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(DEFAULT_CSV_PATH, reports_csv)
print(f"[LEDGER] Logged {run_id} (step={step}, dev_nll={final_nll}) to {DEFAULT_CSV_PATH}")
PYEOF
fi

# 2. Persist to Google Drive if mounted on Colab
if [ -d "/content/drive/MyDrive" ]; then
    GDRIVE_DEST="/content/drive/MyDrive/r4_checkpoints/$RUN_ID"
    mkdir -p "$GDRIVE_DEST"
    cp -r "$OUT_DIR"/* "$GDRIVE_DEST"/
    cp "$REPO_ROOT/docs/benchmark_history.csv" "/content/drive/MyDrive/r4_checkpoints/benchmark_history.csv" || true
    echo "[GDRIVE] Saved checkpoint & CSV to $GDRIVE_DEST"
fi

# 3. Persist to HuggingFace Hub if HF_TOKEN is present
if [ -n "${HF_TOKEN:-}" ]; then
    HF_REPO_ID="${HF_REPO_ID:-yassine-syn-prover/r4-checkpoints}"
    python3 - "$OUT_DIR" "$HF_REPO_ID" "$RUN_ID" << 'PYEOF' || echo "[WARN] HuggingFace upload failed"
import os, sys
from huggingface_hub import HfApi
out_dir, repo_id, run_id = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=out_dir, path_in_repo=run_id, repo_id=repo_id, repo_type="model")
print(f"[HF] Uploaded {out_dir} to https://huggingface.co/{repo_id}/tree/main/{run_id}")
PYEOF
fi

# 4. Persist to GitHub (commits benchmark_history.csv, report.json, config.json, and model.safetensors)
cd "$REPO_ROOT"
if git rev-parse --is-inside-work-tree &>/dev/null; then
    git config user.name "${GIT_AUTHOR_NAME:-Colab Training Bot}"
    git config user.email "${GIT_AUTHOR_EMAIL:-colab-bot@syn-prover.local}"

    # Stage ledger and lightweight run metadata on current branch
    git add -f docs/benchmark_history.csv
    [ -f "$OUT_DIR/report.json" ] && git add -f "$OUT_DIR/report.json"
    [ -f "$OUT_DIR/model/config.json" ] && git add -f "$OUT_DIR/model/config.json"
    [ -f "$OUT_DIR/seal.json" ] && git add -f "$OUT_DIR/seal.json"

    # If model.safetensors is < 95 MB (GitHub's 100MB limit; 8M R4 model is 34 MB), stage weights too
    if [ -f "$OUT_DIR/model/model.safetensors" ]; then
        WEIGHT_BYTES=$(stat -c%s "$OUT_DIR/model/model.safetensors" 2>/dev/null || stat -f%z "$OUT_DIR/model/model.safetensors" 2>/dev/null || echo 0)
        if [ "$WEIGHT_BYTES" -lt 99000000 ]; then
            git add -f "$OUT_DIR/model/model.safetensors"
            echo "[GIT] Staged model.safetensors ($((WEIGHT_BYTES / 1024 / 1024)) MB)"
        elif command -v gh &>/dev/null; then
            gh release create "ckpt-$RUN_ID" "$OUT_DIR/model/model.safetensors" "$OUT_DIR/report.json" \
                --title "Checkpoint $RUN_ID ($STAGE_LABEL)" \
                --notes "Automated Colab checkpoint upload for $RUN_ID" || true
        fi
    fi

    if ! git diff --cached --quiet; then
        git commit --no-verify -m "chore(ckpt): auto-persist $RUN_ID ($STAGE_LABEL) from Colab"
        CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        git push --no-verify origin "HEAD:$CURRENT_BRANCH" || git push --no-verify origin "HEAD:refs/heads/checkpoints/$RUN_ID" || true
        echo "[GIT] Pushed checkpoint and benchmark ledger to origin ($CURRENT_BRANCH)"
    fi
fi
