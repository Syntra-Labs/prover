#!/usr/bin/env python3
"""
Fetches GoldWorm Coder model bundle, corpora, and expert weights from Hugging Face:
https://huggingface.co/Uuuuuuniiiiiii/goldworm-coder
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "vendor/models/goldworm-coder"
DEST_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "Uuuuuuniiiiiii/goldworm-coder"

FILES_TO_DOWNLOAD = [
    "README.md",
    "config.json",
    "active_bundle.json",
    "cand_v9/v9_corpus.bin",
    "cand_v9/benchmark_hf_v9.md",
    "cand_v9/code_history.json",
    "cand_v9/math_history.json",
    "cand_v9/iter_v9_gate.json",
    "cand_v8/v8_corpus.bin",
    "session_v8_scripts/build_proof_u8.py",
    "session_v8_scripts/selfprove.py",
]

def main():
    print(f"==================================================")
    print(f"Fetching GoldWorm Coder into {DEST_DIR}")
    print(f"Repository: {REPO_ID}")
    print(f"==================================================")

    for rel_path in FILES_TO_DOWNLOAD:
        local_target = DEST_DIR / rel_path
        local_target.parent.mkdir(parents=True, exist_ok=True)
        
        if local_target.exists():
            size_mb = local_target.stat().st_size / (1024**2)
            print(f"  [EXISTS] {rel_path} ({size_mb:.2f} MB)")
            continue
            
        print(f"--> Downloading {rel_path}...")
        try:
            downloaded = hf_hub_download(repo_id=REPO_ID, filename=rel_path)
            shutil.copy(downloaded, local_target)
            size_mb = local_target.stat().st_size / (1024**2)
            print(f"    [OK] Saved {rel_path} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"    [ERROR] Failed to download {rel_path}: {e}")

    print("\nGoldWorm Coder files successfully synchronized!")

if __name__ == "__main__":
    main()
