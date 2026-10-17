#!/usr/bin/env python3
"""
Prepares .uort binary files and manifests for geometric-stack dialogue-train.
Converts raw .u16 token streams into UORT 64-byte header binary stores.
"""

import struct
import json
from pathlib import Path

# FIXME(stage2-response-mask | tier:A | depends_on:[gate-r4-token] | blocks:[policy-r4, param-tactic-synthesis]):
# Description: Encode (Goal, Tactic) episodes under uor-r4.literal-role-dialogue/1 contract (<|bos|>User: <goal>\nAssistant: <tactic><|eos|>) with mask=0 on prompt and mask=1 on tactic.
# Criterion: assert: uses_dialogue_train_subcommand and episode_starts_with_bos and assistant_marker_precedes_mask_one
def convert_u16_to_uort(u16_path: Path, uort_path: Path, manifest_path: Path, vocab_size: int = 4096):
    file_bytes = u16_path.stat().st_size
    total_tokens = file_bytes // 2
    print(f"Converting {u16_path.name} -> {uort_path.name} ({total_tokens:,} tokens)...")

    # Construct 64-byte UORT header
    # 0..4: b"UORT"
    # 4..8: version 1 (u32 LE)
    # 8..16: total_tokens (u64 LE)
    # 16..20: vocab_size (u32 LE)
    # 20..64: 44 bytes zeros
    header = struct.pack("<4sI Q I 44x", b"UORT", 1, total_tokens, vocab_size)
    assert len(header) == 64

    with open(uort_path, "wb") as out_f, open(u16_path, "rb") as in_f:
        out_f.write(header)
        # Stream copy payload in 1MB chunks
        while chunk := in_f.read(1024 * 1024):
            out_f.write(chunk)

    # Write manifest expected by DialogueSplit::load
    manifest = {
        "drops": {
            "special_token_occurrences": 0
        },
        "files": [
            {
                "label": "mathlib4_theorems",
                "tokens": total_tokens,
                "special_token_occurrences": 0
            }
        ]
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [OK] Created {uort_path.name} ({uort_path.stat().st_size:,} bytes)")
    print(f"  [OK] Created {manifest_path.name}")

def main():
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/mathlib4")
    
    convert_u16_to_uort(
        data_dir / "train.u16",
        data_dir / "train.uort",
        data_dir / "train_manifest.json"
    )
    convert_u16_to_uort(
        data_dir / "valid.u16",
        data_dir / "valid.uort",
        data_dir / "valid_manifest.json"
    )
    print("\n[COMPLETE] All dialogue-train stores and manifests prepared.")

if __name__ == "__main__":
    main()
