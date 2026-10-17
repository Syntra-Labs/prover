"""
tests/geometric_falsification/test_tier_b_stability.py

Tier B: Numerical Stability & Syntactic Invariance Tests.
Focused 100% on UOR-R4 Geometric Model + Reversible IR Canonicalizer (src/ir/canonicalizer.py).
Tests alpha-equivalence representation stability and tokenization jitter sensitivity.
"""

import pytest
import torch
import torch.nn.functional as F
from src.eval.tactic_eval import TacticEvaluator
from src.eval.r4_runner import find_latest_checkpoint
from src.eval.authentic_lean_prover import CANONICAL_TACTIC_POOL
from src.ir.canonicalizer import canonicalize, invert_sigma

@pytest.fixture(scope="module")
def evaluator():
    ckpt = find_latest_checkpoint()
    return TacticEvaluator(ckpt, device="cpu")

# TODO(stab-alpha | tier:B | blocks:[policy-r4]): Test R4 hidden representation stability under bound variable renaming.
# Description: Verify alpha-equivalent goals (e.g., n -> k) produce geometrically adjacent representations.
# Criterion: assert: min_cosine_sim >= 0.95
def test_alpha_equivalence_invariance(evaluator):
    """Failure Mode 6: Semantic vs Lexical Alpha-Equivalence Brittleness."""
    c_n, c_tac_n, sig_n = canonicalize("theorem add_zero (n : Nat) : n + 0 = n := by", "induction n")
    c_k, c_tac_k, sig_k = canonicalize("theorem add_zero (k : Nat) : k + 0 = k := by", "induction k")
    c_m, _, _ = canonicalize("theorem add_zero (m : Nat) : m + 0 = m := by")
    c_diff, _, _ = canonicalize("theorem list_rev (xs : List Nat) : xs.reverse.reverse = xs := by")

    assert invert_sigma(c_tac_n, sig_n) == "induction n"
    assert invert_sigma(c_tac_k, sig_k) == "induction k"

    v_n = evaluator.get_final_hidden(c_n)
    v_k = evaluator.get_final_hidden(c_k)
    v_m = evaluator.get_final_hidden(c_m)
    v_diff = evaluator.get_final_hidden(c_diff)

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
    raw_jittered = "theorem  nat_add_zero  (n : Nat)  :  n + 0 = n  :=  by\n"

    canon_base, _, _ = canonicalize(base_prompt)
    canon_jittered, _, _ = canonicalize(raw_jittered)

    def get_top5(prompt: str):
        scores = []
        for tac in CANONICAL_TACTIC_POOL:
            res = evaluator.evaluate_tactic_likelihood(prompt, tac)
            scores.append((tac, res["cross_entropy_loss"]))
        scores.sort(key=lambda x: x[1])
        return [x[0] for x in scores[:5]]

    top5_base = get_top5(canon_base)
    top5_normalized = get_top5(canon_jittered)

    norm_overlap = len(set(top5_base).intersection(set(top5_normalized))) / 5.0
    print(f"\n[JITTER STABILITY] Production IR Canonicalized Overlap = {norm_overlap:.2f}")
    assert norm_overlap >= 0.80, f"Normalized jitter overlap {norm_overlap:.2f} < 0.80"
