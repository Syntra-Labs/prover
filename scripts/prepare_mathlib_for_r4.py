#!/usr/bin/env python3
"""
Prepares Lean 4 Mathlib dataset (extracted via LeanDojo) for R4 Geometric Training.
1. Optionally trains/exports a canonical 4096-vocab ByteLevel BPE tokenizer for Mathlib.
2. Extracts theorem goal states and traced tactics from train.json / val.json.
3. Encodes and exports:
   - train.u16 & valid.u16 (for base language modeling)
   - tokens.u16 & response_mask.u8 & manifest.json (for goal->tactic SFT dialogue training)
"""

import os
import sys
import json
import struct
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
except ImportError:
    print("ERROR: 'tokenizers' library required. Run: pip install tokenizers")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BENCHSET_DIR = REPO_ROOT / "vendor/benchsets/mathlib4"
OUTPUT_DIR = REPO_ROOT / "data/mathlib4"

def train_or_load_tokenizer(samples: List[str], target_vocab_size: int = 4096) -> Tokenizer:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = OUTPUT_DIR / "tokenizer.json"
    
    if tok_path.exists():
        print(f"Loading existing tokenizer from {tok_path}")
        return Tokenizer.from_file(str(tok_path))
    
    print(f"Training new ByteLevel BPE tokenizer (vocab_size={target_vocab_size}) on Mathlib text...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = ["<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=target_vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train_from_iterator(samples, trainer=trainer)
    tokenizer.save(str(tok_path))
    print(f"Tokenizer saved to {tok_path}")
    return tokenizer

from src.ir.canonicalizer import canonicalize, invert_sigma
from src.ir.lemma_table import GLOBAL_LEMMA_TABLE

def extract_state_tactic_pairs(json_path: Path, max_pairs: int = 0) -> List[Tuple[str, str]]:
    if not json_path.exists():
        print(f"Warning: {json_path} not found.")
        return []
        
    print(f"Reading {json_path.name}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pairs = []
    for item in data:
        tactics = item.get("traced_tactics", [])
        for step in tactics:
            sb = step.get("state_before", "").strip()
            tac = step.get("tactic", "").strip()
            if sb and tac:
                canon_sb, canon_tac, _ = canonicalize(sb, tac, lemma_table=GLOBAL_LEMMA_TABLE)
                pairs.append((canon_sb, canon_tac))
                if max_pairs > 0 and len(pairs) >= max_pairs:
                    print(f"  Extracted {len(pairs)} (state, tactic) pairs from {json_path.name} (capped at {max_pairs})")
                    return pairs
    print(f"  Extracted {len(pairs)} (state, tactic) pairs from {json_path.name}")
    return pairs

def main():
    train_json = BENCHSET_DIR / "train.json"
    val_json = BENCHSET_DIR / "val.json"
    max_train = int(os.environ.get("MAX_TRAIN_PAIRS", "0"))
    
    if not val_json.exists():
        print(f"ERROR: {val_json} not found. Please run scripts/fetch_mathlib4_dataset.sh first.")
        sys.exit(1)
        
    GLOBAL_LEMMA_TABLE.name_to_id.clear()
    GLOBAL_LEMMA_TABLE.id_to_name.clear()
    use_full = os.environ.get("USE_FULL_TRAIN", "0") == "1"
    val_pairs = extract_state_tactic_pairs(val_json)
    train_pairs = (
        extract_state_tactic_pairs(train_json, max_pairs=max_train)
        if (train_json.exists() and (max_train > 0 or use_full))
        else []
    )
    
    if not train_pairs:
        print("Using validation set as train & val sample since full train.json is not processed yet.")
        split_idx = int(len(val_pairs) * 0.8)
        train_pairs = val_pairs[:split_idx]
        val_pairs = val_pairs[split_idx:]

    # Save global lemma lookup table (L0, L1, ... <-> Mathlib lemma names)
    lemma_table_path = GLOBAL_LEMMA_TABLE.save(OUTPUT_DIR / "lemma_table.json")
    print(f"Saved {len(GLOBAL_LEMMA_TABLE):,} unique Mathlib lemmas -> {lemma_table_path}")

    # Collect sample text for tokenizer training
    sample_corpus = []
    for sb, tac in (train_pairs[:5000] + val_pairs[:1000]):
        sample_corpus.append(f"Goal:\n{sb}\n\nTactic:\n{tac}\n")
        
    tokenizer = train_or_load_tokenizer(sample_corpus, target_vocab_size=4096)
    
    # Process and encode
    for name, pairs in [("train", train_pairs), ("valid", val_pairs)]:
        tokens_list = []
        mask_list = []
        
        for sb, tac in pairs:
            prompt_str = f"Goal:\n{sb}\n\nTactic:\n"
            tac_str = f"{tac}\n"
            eos_str = "<|eos|>\n"
            
            p_ids = tokenizer.encode(prompt_str).ids
            t_ids = tokenizer.encode(tac_str).ids
            e_ids = tokenizer.encode(eos_str).ids
            
            tokens_list.extend(p_ids + t_ids + e_ids)
            # 2 = Goal start boundary (weight 0), 0 = prompt/eos (weight 0), 1 = canonical tactic (weight 1)
            p_mask = [2] + [0] * (len(p_ids) - 1) if p_ids else []
            mask_list.extend(p_mask + [1] * len(t_ids) + [0] * len(e_ids))
            
        # Write .u16 tokens
        u16_path = OUTPUT_DIR / f"{name}.u16"
        with open(u16_path, "wb") as f:
            f.write(struct.pack(f"<{len(tokens_list)}H", *[min(tid, 4095) for tid in tokens_list]))
            
        # Write response mask (1 byte per token: 2=goal_start, 0=prompt/eos, 1=tactic)
        mask_path = OUTPUT_DIR / f"{name}.mask"
        with open(mask_path, "wb") as f:
            f.write(bytes(mask_list))
            
        print(f"Exported {name}: {len(tokens_list):,} tokens -> {u16_path} ({u16_path.stat().st_size / 1024**2:.2f} MB)")

    print(f"\n[DONE] Mathlib4 R4 dataset ready in: {OUTPUT_DIR}")
    print(f"Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024**2:.2f} MB)")

if __name__ == "__main__":
    main()
