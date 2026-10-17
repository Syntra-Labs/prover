#!/bin/bash
set -e

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../vendor/benchsets/mathlib4" && pwd)"
export DEST_DIR
mkdir -p "${DEST_DIR}"

echo "=================================================="
echo "Fetching Mathlib4 dataset into ${DEST_DIR}"
echo "=================================================="

PYTHON_BIN="python3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi

"$PYTHON_BIN" - << 'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, os, sys

dest_dir = os.environ.get("DEST_DIR", "vendor/benchsets/mathlib4")
repo_id = "1337xyz1337xyz/leandojo-benchmark4-v10"

files = [
    ("leandojo_benchmark_4/random/val.json", "val.json"),
    ("leandojo_benchmark_4/random/test.json", "test.json"),
    ("leandojo_benchmark_4/corpus.jsonl", "corpus.jsonl"),
    ("leandojo_benchmark_4/random/train.json", "train.json")
]

for src, dst_name in files:
    target_path = os.path.join(dest_dir, dst_name)
    if not os.path.exists(target_path):
        print(f"--> Downloading {dst_name}...")
        downloaded = hf_hub_download(repo_id=repo_id, filename=src, repo_type="dataset")
        shutil.copy(downloaded, target_path)
        print(f"    [OK] Saved {dst_name} ({os.path.getsize(target_path) / 1024**2:.2f} MB)")
    else:
        print(f"    [EXISTS] {dst_name} ({os.path.getsize(target_path) / 1024**2:.2f} MB)")

print("\nAll Mathlib4 dataset files are present and verified!")
PYEOF
chmod +x "${BASH_SOURCE[0]}"
