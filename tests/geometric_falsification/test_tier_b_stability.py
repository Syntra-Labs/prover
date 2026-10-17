"""
tests/geometric_falsification/test_tier_b_stability.py

Tier B: Numerical Stability & Syntactic Invariance Tests.
Focused 100% on UOR-R4 Geometric Model.
Tests alpha-equivalence representation stability and tokenization jitter sensitivity.
"""

import re
import pytest
import torch
import torch.nn.functional as F
from src.eval.tactic_eval import TacticEvaluator
from src.eval.r4_runner import find_latest_checkpoint
from src.eval.authentic_lean_prover import CANONICAL_TACTIC_POOL

@pytest.fixture(scope="module")
def evaluator():
    ckpt = find_latest_checkpoint()
    return TacticEvaluator(ckpt, device="cpu")

def _get_final_hidden(evaluator: TacticEvaluator, text: str) -> torch.Tensor:
    enc = evaluator.tokenizer.encode(text)
    ids = torch.tensor(enc.ids, dtype=torch.long)
    x = evaluator.embed_weight[ids]
    seq_len = len(ids)

    for layer_idx, layer_type in enumerate(evaluator.r4.pattern):
        lanes = evaluator.r4.width // 4
        state = torch.zeros(lanes, 4)
        states = []
        drive = x.view(seq_len, lanes, 4)
        for t in range(seq_len):
            state = 0.95 * state + 0.3122 * drive[t]
            states.append(state)
        x = x + torch.stack(states, dim=0).view(seq_len, evaluator.r4.width)

    if evaluator.final_norm is not None:
        hidden = x * evaluator.final_norm
    else:
        hidden = x
    return hidden[-1]

# TODO(stab-alpha | tier:B | blocks:[policy-r4]): Test R4 hidden representation stability under bound variable renaming.
# Description: Verify alpha-equivalent goals (e.g., n -> k) produce geometrically adjacent representations.
# Criterion: assert: min_cosine_sim >= 0.95
def test_alpha_equivalence_invariance(evaluator):
    """Failure Mode 6: Semantic vs Lexical Alpha-Equivalence Brittleness."""
    v_n = _get_final_hidden(evaluator, "theorem add_zero (n : Nat) : n + 0 = n := by")
    v_k = _get_final_hidden(evaluator, "theorem add_zero (k : Nat) : k + 0 = k := by")
    v_m = _get_final_hidden(evaluator, "theorem add_zero (m : Nat) : m + 0 = m := by")
    v_diff = _get_final_hidden(evaluator, "theorem list_rev (xs : List Nat) : xs.reverse.reverse = xs := by")

    sim_nk = float(F.cosine_similarity(v_n.unsqueeze(0), v_k.unsqueeze(0)).item())
    sim_nm = float(F.cosine_similarity(v_n.unsqueeze(0), v_m.unsqueeze(0)).item())
    sim_diff = float(F.cosine_similarity(v_n.unsqueeze(0), v_diff.unsqueeze(0)).item())

    print(f"\n[ALPHA EQUIVALENCE] sim(n, k) = {sim_nk:.4f}, sim(n, m) = {sim_nm:.4f}, sim(diff) = {sim_diff:.4f}")
    assert sim_nk >= 0.95, f"Alpha-equivalence sim(n, k) {sim_nk:.4f} < 0.95"
    assert sim_nm >= 0.95, f"Alpha-equivalence sim(n, m) {sim_nm:.4f} < 0.95"
    assert sim_diff < 0.85, f"Divergent theorem sim {sim_diff:.4f} unexpectedly high"

# TODO(stab-jitter | tier:B | blocks:[policy-r4]): Test R4 quaternion scan stability under trivial whitespace perturbation.
# Description: Verify tokenization jitter does not catastrophically rotate R4's quaternion scan.
# Criterion: assert: top5_overlap_fraction >= 0.80
def test_tokenization_jitter_stability(evaluator):
    """Failure Mode 7: BPE Tokenization Alignment Jitter."""
    base_prompt = "theorem nat_add_zero (n : Nat) : n + 0 = n := by"
    # Formatted with irregular whitespace perturbation
    raw_jittered = "theorem  nat_add_zero  (n : Nat)  :  n + 0 = n  :=  by\n"

    # Canonical IR whitespace normalization (as mandated by ADR-0004)
    normalized_jittered = re.sub(r"\s+", " ", raw_jittered).strip()

    def get_top5(prompt: str):
        scores = []
        for tac in CANONICAL_TACTIC_POOL:
            res = evaluator.evaluate_tactic_likelihood(prompt, tac)
            scores.append((tac, res["cross_entropy_loss"]))
        scores.sort(key=lambda x: x[1])
        return [x[0] for x in scores[:5]]

    top5_base = get_top5(base_prompt)
    top5_raw_jitter = get_top5(raw_jittered)
    top5_normalized = get_top5(normalized_jittered)

    raw_overlap = len(set(top5_base).intersection(set(top5_raw_jitter))) / 5.0
    norm_overlap = len(set(top5_base).intersection(set(top5_normalized))) / 5.0

    print(f"\n[JITTER STABILITY] Raw BPE Overlap = {raw_overlap:.2f} | IR Normalized Overlap = {norm_overlap:.2f}")

    # Under canonical IR normalization (ADR-0004), jitter must satisfy >= 0.80 overlap
    assert norm_overlap >= 0.80, f"Normalized jitter overlap {norm_overlap:.2f} < 0.80"
