"""
tests/geometric_falsification/test_tier_a_gatekeepers.py

Tier A: Fast Zero-Training Gatekeeper Falsification Tests.
Executed on Cloud A100 GPU against live authentic models.
"""

import random
import pytest
import numpy as np
from scipy import stats
from collections import Counter


def d8_round(y: np.ndarray) -> np.ndarray:
    """D8 root lattice rounding (Conway & Sloane, 1982)."""
    f = np.round(y)
    if int(np.sum(f)) % 2 != 0:
        err = np.abs(y - f)
        worst = np.argmax(err)
        f[worst] += 1.0 if y[worst] > f[worst] else -1.0
    return f


def e8_snap(y: np.ndarray) -> np.ndarray:
    """
    Casey Allard's authentic E8 snapping function.
    Direct port of e8_snap from uor-r4-graph-certify/tests/e8_membership_ab.rs.
    """
    a = d8_round(y)
    b0 = d8_round(y - 0.5)
    b = b0 + 0.5
    da = np.sum((y - a) ** 2)
    db = np.sum((y - b) ** 2)
    return a if da <= db else b


# TODO(gate-ood | tier:A | blocks:[unify-bundle]): Compare GFN trajectory norms on Mathlib 4 vs scrambled character noise.
# Description: Test whether DepthMuun Shakespeare checkpoint distinguishes formal Lean 4 syntax from random noise.
# Criterion: assert: ks_pvalue < 0.01
def test_ood_noise_discrimination(mathlib_sample, gfn_evaluator):
    """
    Failure Mode 4: Out-Of-Domain Noise Indifference.
    Hypothesis: Authentic GFN dynamics produce statistically distinct trajectory 
    norms on valid mathematical statements vs character-permuted scrambled text.
    """
    # 1. Extract 30 valid Mathlib 4 proof states
    proof_states = []
    for thm in mathlib_sample[:30]:
        for step in thm.get("traced_tactics", []):
            sb = step.get("state_before", "").strip()
            if sb and len(sb) > 20:
                proof_states.append(sb)
                break
        if len(proof_states) >= 20:
            break

    assert len(proof_states) >= 10, "Insufficient proof states for test"

    # 2. Run authentic GFN forward pass on real proof states
    real_norms = []
    for state in proof_states:
        res = gfn_evaluator.forward_state(state)
        real_norms.append(res["emitted_embeddings_norm"])

    # 3. Create scrambled noise strings (same length and character distribution)
    noise_norms = []
    for state in proof_states:
        scrambled = "".join(random.sample(state, len(state)))
        res = gfn_evaluator.forward_state(scrambled)
        noise_norms.append(res["emitted_embeddings_norm"])

    # 4. Kolmogorov-Smirnov test to verify statistical separation
    ks_stat, ks_pvalue = stats.ks_2samp(real_norms, noise_norms)

    print(f"\n[gate-ood] Real Mean Norm: {np.mean(real_norms):.4f} +/- {np.std(real_norms):.4f}")
    print(f"[gate-ood] Noise Mean Norm: {np.mean(noise_norms):.4f} +/- {np.std(noise_norms):.4f}")
    print(f"[gate-ood] KS Statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4e}")

    # Machine-verifiable criterion
    assert ks_pvalue < 0.01, f"GFN failed to distinguish Lean syntax from noise (p={ks_pvalue:.4e})"


# TODO(gate-perm | tier:A | blocks:[unify-bundle]): Test GFN Torus sensitivity to hypothesis premise order.
# Description: Test whether GFN's velocity coupling overcomes Abelian commutativity to detect premise permutation.
# Criterion: assert: cosine_distance > 0.05
def test_torus_permutation_order(mathlib_sample, gfn_evaluator):
    """
    Failure Mode 3: Torus Commutativity Blindness.
    Hypothesis: GFN's velocity and Christoffel coupling allow it to distinguish 
    premise order in multi-hypothesis proof states despite Torus commutativity.
    """
    # Create synthetic proof states with permuted hypotheses:
    # State A: "h1 : A, h2 : B ⊢ C"
    # State B: "h2 : B, h1 : A ⊢ C"
    test_pairs = [
        (
            "h1 : x > 0\nh2 : y < 10\n⊢ x + y > 0",
            "h2 : y < 10\nh1 : x > 0\n⊢ x + y > 0"
        ),
        (
            "α : Type\ninst : Group α\nx : α\n⊢ x * 1 = x",
            "x : α\ninst : Group α\nα : Type\n⊢ x * 1 = x"
        ),
        (
            "h_even : Even n\nh_prime : Nat.Prime n\n⊢ n = 2",
            "h_prime : Nat.Prime n\nh_even : Even n\n⊢ n = 2"
        )
    ]

    distances = []
    for state_a, state_b in test_pairs:
        res_a = gfn_evaluator.forward_state(state_a)
        res_b = gfn_evaluator.forward_state(state_b)
        
        va = np.array(res_a["raw_res"]["final_state"].squeeze(0).cpu())
        vb = np.array(res_b["raw_res"]["final_state"].squeeze(0).cpu())
        
        # Cosine distance: 1 - cosine_similarity
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a > 1e-7 and norm_b > 1e-7:
            cos_sim = np.dot(va, vb) / (norm_a * norm_b)
            cos_dist = float(1.0 - cos_sim)
        else:
            cos_dist = 0.0
            
        distances.append(cos_dist)

    mean_dist = float(np.mean(distances))
    print(f"\n[gate-perm] Pairwise Cosine Distances: {distances}")
    print(f"[gate-perm] Mean Permutation Cosine Distance: {mean_dist:.6f}")

    # Machine-verifiable criterion
    assert mean_dist > 0.05, f"GFN showed order-blindness under premise permutation (dist={mean_dist:.6f})"


# TODO(gate-e8 | tier:A | blocks:[unify-bundle]): Test Conway-Sloane E8 Voronoi lattice snapping dispersion.
# Description: Test whether 50 Mathlib tactic proof states collapse into the origin E8 cell or disperse across cells.
# Criterion: assert: shannon_entropy >= 3.0 and max_cell_fraction <= 0.25
def test_voronoi_dispersion(mathlib_sample, gfn_evaluator):
    """
    Failure Mode 2: E8 Lattice Mode Collapse.
    Hypothesis: Snapping continuous 64-dim manifold vectors into E8 root lattice 
    coordinates yields non-trivial entropy across distinct proof states.
    """
    proof_states = []
    for thm in mathlib_sample[:50]:
        for step in thm.get("traced_tactics", []):
            sb = step.get("state_before", "").strip()
            if sb and len(sb) > 15:
                proof_states.append(sb)
        if len(proof_states) >= 40:
            break

    assert len(proof_states) >= 20, "Insufficient proof states for test"

    cells = []
    for state in proof_states:
        res = gfn_evaluator.forward_state(state)
        v = np.array(res["raw_res"]["final_state"].squeeze(0).cpu(), dtype=np.float64) # 64-dim
        
        # Snap 8 blocks of 8D vectors to E8 lattice points
        snapped_blocks = []
        for b in range(8):
            blk = v[8 * b : 8 * (b + 1)]
            snapped = e8_snap(blk)
            snapped_blocks.append(tuple(snapped.tolist()))
            
        cell_key = tuple(snapped_blocks)
        cells.append(cell_key)

    # Compute cell distribution statistics
    counts = Counter(cells)
    total = len(cells)
    probs = [c / total for c in counts.values()]
    shannon_entropy = -sum(p * np.log2(p) for p in probs)
    max_fraction = max(counts.values()) / total
    origin_cell = tuple([(0.0,) * 8] * 8)
    origin_count = counts.get(origin_cell, 0)

    print(f"\n[gate-e8] Total states: {total} | Unique E8 lattice cells: {len(counts)}")
    print(f"[gate-e8] Shannon Entropy: {shannon_entropy:.4f} bits")
    print(f"[gate-e8] Max cell concentration: {max_fraction * 100:.1f}%")
    print(f"[gate-e8] Origin cell (0..0) hits: {origin_count}/{total} ({origin_count/total*100:.1f}%)")

    # Machine-verifiable criterion
    assert shannon_entropy >= 3.0, f"Mode collapse detected: low entropy ({shannon_entropy:.2f} < 3.0)"
    assert max_fraction <= 0.25, f"Cell monopolization: single cell captures {max_fraction * 100:.1f}% (> 25%)"
