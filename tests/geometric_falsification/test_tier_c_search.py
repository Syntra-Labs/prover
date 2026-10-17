"""
tests/geometric_falsification/test_tier_c_search.py

Tier C: Multi-Step Proof Trajectory & Search Policy Calibration Tests.
Verifies:
1. Total geometric action is bounded and stationary on real Mathlib 4 proof paths vs divergent paths.
2. Inapplicable dead tactics (e.g. ring on propositional logic) are strongly suppressed.
"""

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from src.eval.tactic_eval import TacticEvaluator
from src.eval.r4_runner import find_latest_checkpoint
from src.ir.canonicalizer import canonicalize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

@pytest.fixture(scope="module")
def evaluator():
    ckpt = find_latest_checkpoint()
    return TacticEvaluator(ckpt, device="cpu")

# TODO(search-action | tier:C | depends_on:[unify-bundle] | blocks:[macro-decider]):
# Description: Verify total geometric action is finite and stationary on complete Mathlib 4 proof paths.
# Criterion: assert: valid_action < divergent_action and not is_inf(valid_action)
def test_path_action_boundedness(evaluator):
    """Failure Mode 1: Action Non-Boundedness along Real Mathlib 4 Proof Trajectories."""
    val_sample = json.loads((REPO_ROOT / "vendor/benchsets/mathlib4/val_sample.json").read_text(encoding="utf-8"))
    multi_step_thm = next(t for t in val_sample if len(t.get("traced_tactics", [])) >= 5)
    valid_states = [s["state_before"] for s in multi_step_thm["traced_tactics"][:5]]

    # Divergent sequence: states drawn across distinct unrelated theorems
    unrelated_states = [
        t["traced_tactics"][0]["state_before"]
        for t in val_sample
        if t.get("traced_tactics")
    ][:5]

    def trajectory_action(states):
        vecs = []
        for st in states:
            canon_st, _, _ = canonicalize(st)
            h = evaluator.get_final_hidden(canon_st)
            vecs.append(F.normalize(h, dim=0))
        return sum(
            1.0 - float(torch.dot(vecs[i], vecs[i - 1]).item())
            for i in range(1, len(vecs))
        )

    valid_action = trajectory_action(valid_states)
    divergent_action = trajectory_action(unrelated_states)

    print(f"\n[ACTION BOUNDEDNESS] Valid Mathlib Path Action = {valid_action:.4f} | Divergent Path Action = {divergent_action:.4f}")
    assert not np.isinf(valid_action)
    assert not np.isnan(valid_action)
    assert valid_action < divergent_action, f"Valid action {valid_action:.4f} not less than divergent {divergent_action:.4f}"

# TODO(search-dead-tactic | tier:C | depends_on:[policy-r4] | blocks:[macro-decider]):
# Description: Verify policy suppresses trivial no-op tactics (simp/ring self-loops) when inapplicable.
# Criterion: assert: noop_tactic_prob < 0.10 on non-applicable states
def test_dead_tactic_suppression(evaluator):
    """Failure Mode 8: Dead Tactic Attractor in Search Tree."""
    goal = "Goal:\np : Prop\nh : p\n⊢ p\n\nTactic:\n"
    tactics = ["omega", "simp", "exact h", "ring"]

    losses = []
    for t in tactics:
        res = evaluator.evaluate_tactic_likelihood(goal, t)
        losses.append(res["cross_entropy_loss"])

    probs = F.softmax(-torch.tensor(losses), dim=0)
    prob_dict = {t: probs[i].item() for i, t in enumerate(tactics)}

    ring_prob = prob_dict["ring"]
    print(f"\n[DEAD TACTIC SUPPRESSION] Probability of inapplicable 'ring' on propositional goal: {ring_prob:.6f}")
    assert ring_prob < 0.10, f"Inapplicable tactic 'ring' probability {ring_prob:.4f} >= 0.10"
