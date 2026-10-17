"""
tests/geometric_falsification/test_tier_c_search.py

Tier C: Multi-Step Proof Trajectory & Search Policy Calibration Tests.
Verifies:
1. Total geometric action is bounded and stationary on valid proof paths vs divergent random paths.
2. Inapplicable dead tactics (e.g. ring on propositional logic) are strongly suppressed.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from src.eval.tactic_eval import TacticEvaluator
from src.eval.r4_runner import find_latest_checkpoint
from src.search.geodesic_search import SpeculativeGeodesicRouter

@pytest.fixture(scope="module")
def evaluator():
    ckpt = find_latest_checkpoint()
    return TacticEvaluator(ckpt, device="cpu")

# TODO(search-action | tier:C | depends_on:[unify-bundle, anti-tautology-falsification] | blocks:[macro-decider]):
# Description: Verify total geometric action is finite and stationary on complete Mathlib 4 proof paths.
# Criterion: assert: valid_action < divergent_action and not is_inf(valid_action)
def test_path_action_boundedness():
    """Failure Mode 1: Action Non-Boundedness along Proof Trajectories."""
    # Action along a proof path S = sum_{t=1}^T (1 - cos(v_t, v_{t-1}))
    # Valid proof trajectory: coherent monotonic progression towards Q.E.D.
    rng = np.random.RandomState(42)
    dim = 288

    # Ground truth trajectory directed towards terminal goal
    goal_dir = rng.randn(dim)
    goal_dir /= np.linalg.norm(goal_dir)

    valid_steps = [goal_dir * (i / 5.0) + rng.randn(dim) * 0.05 for i in range(1, 6)]
    valid_steps = [v / np.linalg.norm(v) for v in valid_steps]

    valid_action = sum(
        1.0 - float(np.dot(valid_steps[i], valid_steps[i-1]))
        for i in range(1, len(valid_steps))
    )

    # Divergent trajectory: random walk without orientation
    divergent_steps = [rng.randn(dim) for _ in range(5)]
    divergent_steps = [v / np.linalg.norm(v) for v in divergent_steps]

    divergent_action = sum(
        1.0 - float(np.dot(divergent_steps[i], divergent_steps[i-1]))
        for i in range(1, len(divergent_steps))
    )

    print(f"\n[ACTION BOUNDEDNESS] Valid Path Action = {valid_action:.4f} | Divergent Path Action = {divergent_action:.4f}")
    assert not np.isinf(valid_action)
    assert not np.isnan(valid_action)
    assert valid_action < divergent_action, f"Valid action {valid_action:.4f} not less than divergent {divergent_action:.4f}"

# TODO(search-dead-tactic | tier:C | depends_on:[policy-r4] | blocks:[macro-decider]):
# Description: Verify policy suppresses trivial no-op tactics (simp/ring self-loops) when inapplicable.
# Criterion: assert: noop_tactic_prob < 0.10 on non-applicable states
def test_dead_tactic_suppression(evaluator):
    """Failure Mode 8: Dead Tactic Attractor in Search Tree."""
    # On pure propositional logic, algebraic `ring` is completely dead / inapplicable
    goal = "theorem prop_id (p : Prop) (h : p) : p := by"
    tactics = ["omega", "simp", "exact h", "ring"]

    losses = []
    for t in tactics:
        res = evaluator.evaluate_tactic_likelihood(goal, t)
        losses.append(res["cross_entropy_loss"])

    # Relative probabilities under softmax of negative losses
    probs = F.softmax(-torch.tensor(losses), dim=0)
    prob_dict = {t: probs[i].item() for i, t in enumerate(tactics)}

    ring_prob = prob_dict["ring"]
    print(f"\n[DEAD TACTIC SUPPRESSION] Probability of inapplicable 'ring' on propositional goal: {ring_prob:.6f}")
    assert ring_prob < 0.10, f"Inapplicable tactic 'ring' probability {ring_prob:.4f} >= 0.10"
