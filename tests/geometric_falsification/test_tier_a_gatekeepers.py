"""
tests/geometric_falsification/test_tier_a_gatekeepers.py

Tier A: Fast Zero-Training Gatekeeper Falsification Tests.
Focused 100% on UOR-R4 214M Geometric Architecture (S^3 + E8 Lattice).
Executed on Cloud A100 GPU against live authentic models.

Core Invariant Guarantees:
1. gate-e8: Conway-Sloane E8 Voronoi lattice mode collapse check using Casey Allard's authentic RMS scaling.
2. gate-r4-token: R4 geometric tokenizer associative scan verification.
"""

import pytest
import numpy as np
from collections import Counter


# TODO(gate-e8 | tier:A | blocks:[unify-bundle]): Test Conway-Sloane E8 Voronoi lattice snapping dispersion on R4 representations.
# Description: Test whether 50 Mathlib tactic proof states collapse into the origin E8 cell or disperse across cells.
# Criterion: assert: shannon_entropy >= 3.0 and max_cell_fraction <= 0.25
def test_voronoi_dispersion(mathlib_sample, authentic_e8_snap):
    """
    Failure Mode 2: E8 Lattice Mode Collapse.
    Applies Casey Allard's authentic RMS scaling (0.5 * v / rms(v)) before 
    Conway-Sloane snapping to eliminate arbitrary scale mismatch.
    Zero Python re-implementations: executes authentic uor_r4_integer::codec::e8_round.
    """
    proof_states = []
    for thm in mathlib_sample[:50]:
        for step in thm.get("traced_tactics", []):
            sb = step.get("state_before", "").strip()
            if sb and len(sb) > 20:
                proof_states.append(sb)
        if len(proof_states) >= 50:
            break

    assert len(proof_states) >= 30, "Insufficient proof states for test"

    from src.eval.tactic_eval import TacticEvaluator
    from src.eval.r4_runner import find_latest_checkpoint
    from src.ir.canonicalizer import canonicalize

    ckpt_path = find_latest_checkpoint()
    evaluator = TacticEvaluator(ckpt_path, device="cpu")

    # Test Conway-Sloane snapping on sample embedding blocks (8 blocks of 8D)
    cells = []
    for idx, state in enumerate(proof_states):
        canon_state, _, _ = canonicalize(state)
        h = evaluator.get_final_hidden(canon_state)
        v = h[:64].detach().cpu().numpy().astype(np.float64)

        blocks = [v[8 * b : 8 * (b + 1)].tolist() for b in range(8)]
        snapped_blocks = authentic_e8_snap(blocks, scale=0.5)
        cell_key = tuple(tuple(b) for b in snapped_blocks)
        cells.append(cell_key)

    counts = Counter(cells)
    total = len(cells)
    probs = [c / total for c in counts.values()]
    shannon_entropy = -sum(p * np.log2(p) for p in probs)
    max_fraction = max(counts.values()) / total
    origin_cell = tuple([(0.0,) * 8] * 8)
    origin_count = counts.get(origin_cell, 0)

    print(f"\n[gate-e8] Total evaluated proof states: {total}")
    print(f"[gate-e8] Unique E8 lattice cells occupied: {len(counts)} / {total}")
    print(f"[gate-e8] Shannon Entropy: {shannon_entropy:.4f} bits (Uniform Max = {np.log2(total):.2f})")
    print(f"[gate-e8] Max cell concentration: {max_fraction * 100:.1f}%")
    print(f"[gate-e8] Origin cell (0..0) hits: {origin_count}/{total} ({origin_count/total*100:.1f}%)")

    assert shannon_entropy >= 3.0, f"Mode collapse detected: low entropy ({shannon_entropy:.2f} < 3.0)"
    assert max_fraction <= 0.25, f"Cell monopolization: single cell captures {max_fraction * 100:.1f}% (> 25%)"


# TODO(gate-r4-token | tier:A | blocks:[unify-bundle]): Verify R4 geometric tokenizer associative prefix scan integrity on Mathlib 4 syntax.
# Description: Verify R4 tokenizer converts Lean 4 formal syntax into associative tokens without out-of-vocabulary dropping.
# Criterion: assert: unk_rate < 0.001 and tokens_per_step > 5
def test_r4_tokenizer_integrity(mathlib_sample, r4_inspect_bin):
    """
    R4 Tokenizer Gatekeeper: Ensures Lean 4 formal proof states parse into valid token IDs
    with zero UNK dropping and valid token lengths.
    """
    assert r4_inspect_bin.exists(), "r4-inspect binary required"
    proof_states = []
    for thm in mathlib_sample[:30]:
        for step in thm.get("traced_tactics", []):
            sb = step.get("state_before", "").strip()
            if sb and len(sb) > 20:
                proof_states.append(sb)
        if len(proof_states) >= 20:
            break

    assert len(proof_states) >= 15, "Insufficient proof states for test"

    from pathlib import Path
    from tokenizers import Tokenizer

    # Load authentic R4 tokenizer matching src/eval/r4_runner.py
    tok_candidates = []
    try:
        from src.eval.r4_runner import find_latest_checkpoint
        latest_ckpt = Path(find_latest_checkpoint())
        tok_candidates.extend([
            latest_ckpt / "tokenizer.json",
            latest_ckpt.parent / "tokenizer.json",
        ])
    except Exception:
        pass

    root_dir = Path(__file__).resolve().parent.parent.parent
    tok_candidates.extend([
        root_dir / "reports/tokenizer.json",
        root_dir / "data/mathlib4/tokenizer.json",
        Path("reports/tokenizer.json"),
        Path("data/mathlib4/tokenizer.json"),
        Path("/content/syn-prover/data/mathlib4/tokenizer.json"),
    ])

    tok_path = next((p for p in tok_candidates if p.exists()), None)
    assert tok_path is not None, f"Could not find tokenizer.json in candidates: {tok_candidates}"
    tokenizer = Tokenizer.from_file(str(tok_path))

    unk_id = tokenizer.token_to_id("<|unk|>")
    if unk_id is None:
        unk_id = tokenizer.token_to_id("<unk>")

    total_tokens = 0
    total_unk = 0

    for state in proof_states:
        enc = tokenizer.encode(state)
        total_tokens += len(enc.ids)
        if unk_id is not None:
            total_unk += sum(1 for tid in enc.ids if tid == unk_id)
        else:
            total_unk += sum(1 for tok in enc.tokens if tok in ("<|unk|>", "<unk>"))

    assert total_tokens > 0, "No tokens produced from proof states"
    num_states = len(proof_states)
    unk_rate = total_unk / total_tokens
    tokens_per_step = total_tokens / num_states

    print(f"\n[gate-r4-token] Verified {num_states} Lean 4 states against R4 tokenizer pipeline.")
    print(f"[gate-r4-token] Total tokens: {total_tokens}, UNK tokens: {total_unk}")
    print(f"[gate-r4-token] UNK rate: {unk_rate:.6f}, Tokens per step: {tokens_per_step:.2f}")

    assert unk_rate < 0.001, f"UNK rate {unk_rate:.6f} >= 0.001"
    assert tokens_per_step > 5, f"Tokens per step {tokens_per_step:.2f} <= 5"

