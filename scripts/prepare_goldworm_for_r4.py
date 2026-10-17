#!/usr/bin/env python3
"""
Prepares the GoldWorm Coder dataset for R4 Geometric Training.
1. Downloads cand_v9/v9_corpus.bin or expert corpora from Hugging Face:
   Uuuuuuniiiiiii/goldworm-coder
2. Tokenizes the byte stream with the R4 BPE tokenizer.
3. Exports data/goldworm/train.u16 and data/goldworm/valid.u16 for training.
"""

import os
import sys
import struct
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDWORM_DIR = REPO_ROOT / "vendor/models/goldworm-coder"
OUTPUT_DIR = REPO_ROOT / "data/goldworm"
MATHLIB_TOK = REPO_ROOT / "data/mathlib4/tokenizer.json"

REPO_ID = "Uuuuuuniiiiiii/goldworm-coder"

def ensure_goldworm_corpus() -> Path:
    corpus_path = GOLDWORM_DIR / "cand_v9/v9_corpus.bin"
    if corpus_path.exists():
        print(f"Using cached corpus at {corpus_path}")
        return corpus_path
        
    print(f"Downloading v9_corpus.bin (190MB) from {REPO_ID}...")
    GOLDWORM_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=REPO_ID, filename="cand_v9/v9_corpus.bin")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(downloaded, corpus_path)
    print(f"Saved to {corpus_path}")
    return corpus_path

def main():
    corpus_file = ensure_goldworm_corpus()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer: prefer Mathlib's tokenizer, or fall back to training a new one
    if MATHLIB_TOK.exists():
        print(f"Loading existing tokenizer from {MATHLIB_TOK}")
        tokenizer = Tokenizer.from_file(str(MATHLIB_TOK))
    else:
        print("ERROR: Tokenizer not found. Run scripts/prepare_mathlib_for_r4.py first.")
        sys.exit(1)
        
    print(f"Reading corpus {corpus_file.name} ({corpus_file.stat().st_size / 1024**2:.2f} MB)...")
    # Read the raw corpus (byte stream)
    raw_bytes = corpus_file.read_bytes()
    total_len = len(raw_bytes)
    
    # 95% train, 5% valid split
    split_idx = int(total_len * 0.95)
    train_raw = raw_bytes[:split_idx]
    valid_raw = raw_bytes[split_idx:]
    
    # Process in chunks and decode text safely
    CHUNK_SIZE = 1024 * 1024 # 1MB text chunks
    for name, raw in [("valid", valid_raw), ("train", train_raw)]:
        print(f"Encoding {name} ({len(raw) / 1024**2:.2f} MB raw bytes)...")
        tokens = []
        for i in range(0, len(raw), CHUNK_SIZE):
            chunk = raw[i:i + CHUNK_SIZE]
            text = chunk.decode("utf-8", errors="replace")
            encoded = tokenizer.encode(text)
            tokens.extend(encoded.ids)
            
        out_path = OUTPUT_DIR / f"{name}.u16"
        with open(out_path, "wb") as f:
            f.write(struct.pack(f"<{len(tokens)}H", *[min(t, 4095) for t in tokens]))
            
        print(f"  [OK] Exported {name}: {len(tokens):,} tokens -> {out_path} ({out_path.stat().st_size / 1024**2:.2f} MB)")

    # Copy tokenizer into data/goldworm/ for convenience
    import shutil
    shutil.copy(MATHLIB_TOK, OUTPUT_DIR / "tokenizer.json")
    
    print(f"\n[DONE] Goldworm Coder R4 dataset ready in: {OUTPUT_DIR}")
    for f in OUTPUT_DIR.iterdir():
        print(f"  {f.name} ({f.stat().st_size / 1024**2:.2f} MB)")

if __name__ == "__main__":
    main()
