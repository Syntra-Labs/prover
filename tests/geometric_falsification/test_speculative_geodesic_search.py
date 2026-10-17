"""
Falsification and verification test suite for the Speculative Adaptive-Margin Geodesic Prover.
Verifies:
1. Dynamic margin gating collapses to Top-1 when Δ cos >= tau.
2. Angular ambiguity triggers Top-2 speculative queue when Δ cos < tau.
3. Speculative short-circuiting skips candidate #2 on kernel success, reducing Lean REPL overhead.
4. Fallback recovers from candidate #1 rejection (angular shadowing).
5. 100% pass rate across randomized proof search trees.
"""

import pytest
import numpy as np
from src.search.geodesic_search import (
    SpeculativeGeodesicRouter,
    SpeculativeGeodesicProver,
    KernelStatus,
    SearchStatus,
)

def test_margin_gating_collapse_to_top1():
    router = SpeculativeGeodesicRouter(margin_threshold=0.05)
    tangent = np.array([1.0, 0.0, 0.0])
    actions = {
        "exact": np.array([0.99, 0.01, 0.0]),
        "simp": np.array([0.50, 0.50, 0.0]),
        "omega": np.array([0.10, 0.90, 0.0]),
    }
    ranked = router.rank_candidates(tangent, actions)
    queue, margin, is_ambiguous = router.select_speculative_queue(ranked)

    assert not is_ambiguous
    assert len(queue) == 1
    assert queue[0].name == "exact"
    assert margin > 0.05

def test_margin_gating_triggers_top2():
    router = SpeculativeGeodesicRouter(margin_threshold=0.05)
    tangent = np.array([1.0, 0.0, 0.0])
    # Distractor is very close in angle (difference < 0.05)
    actions = {
        "distractor_simp": np.array([0.985, 0.05, 0.0]),
        "ground_truth_exact": np.array([0.980, 0.06, 0.0]),
        "omega": np.array([0.20, 0.80, 0.0]),
    }
    ranked = router.rank_candidates(tangent, actions)
    queue, margin, is_ambiguous = router.select_speculative_queue(ranked)

    assert is_ambiguous
    assert len(queue) == 2
    assert queue[0].name == "distractor_simp"
    assert queue[1].name == "ground_truth_exact"
    assert margin < 0.05

def test_speculative_short_circuiting_saves_repl_call():
    router = SpeculativeGeodesicRouter(margin_threshold=0.10) # wide threshold to force queue
    prover = SpeculativeGeodesicProver(router=router, max_steps=10)

    actions = {
        "candidate_1": np.array([0.98, 0.02]),
        "candidate_2": np.array([0.95, 0.05]),
    }

    # Candidate 1 closes goal immediately
    kernel_calls = []
    def mock_kernel(goal, tactic):
        kernel_calls.append(tactic)
        if tactic == "candidate_1":
            return KernelStatus.GoalsClosed, []
        return KernelStatus.KernelError, []

    policy_fn = lambda g: np.array([1.0, 0.0])
    result = prover.search(
        initial_goal="⊢ P",
        policy_fn=policy_fn,
        action_dict=actions,
        kernel_eval_fn=mock_kernel
    )

    assert result.status == SearchStatus.SuccessProven
    assert result.proof_steps == ["candidate_1"]
    assert result.kernel_calls == 1
    assert result.repl_calls_saved == 1
    assert kernel_calls == ["candidate_1"] # candidate_2 was NEVER called!

def test_speculative_fallback_on_shadowed_distractor():
    router = SpeculativeGeodesicRouter(margin_threshold=0.05)
    prover = SpeculativeGeodesicProver(router=router, max_steps=10)

    # Candidate 1 is collinear distractor, but invalid in Lean.
    # Candidate 2 is valid ground truth.
    actions = {
        "shadow_distractor": np.array([0.995, 0.05]),
        "valid_lemma": np.array([0.985, 0.06]),
    }

    kernel_calls = []
    def mock_kernel(goal, tactic):
        kernel_calls.append(tactic)
        if tactic == "shadow_distractor":
            return KernelStatus.KernelError, []
        elif tactic == "valid_lemma":
            return KernelStatus.GoalsClosed, []
        return KernelStatus.KernelError, []

    policy_fn = lambda g: np.array([1.0, 0.0])
    result = prover.search(
        initial_goal="⊢ P",
        policy_fn=policy_fn,
        action_dict=actions,
        kernel_eval_fn=mock_kernel
    )

    assert result.status == SearchStatus.SuccessProven
    assert result.proof_steps == ["valid_lemma"]
    assert result.kernel_calls == 2
    assert result.repl_calls_saved == 0
    assert kernel_calls == ["shadow_distractor", "valid_lemma"]

def test_100_random_trees_speculative_savings():
    """
    Simulates 100 random multi-step proof trees.
    Verifies 100% path discovery and counts saved Lean REPL round-trips.
    """
    router = SpeculativeGeodesicRouter(margin_threshold=0.05)
    prover = SpeculativeGeodesicProver(router=router, max_steps=30)

    rng = np.random.RandomState(42)
    success_count = 0
    total_kernel_calls = 0
    total_saved_calls = 0

    for tree_idx in range(100):
        # Create a 3-step proof problem with 10 distractor tactics per step
        target_tactics = [f"step_{i}_tactic" for i in range(3)]
        target_directions = [rng.randn(16) for _ in range(3)]
        target_directions = [v / np.linalg.norm(v) for v in target_directions]

        # Action dictionary with true tactics and random distractors
        step_idx = [0] # mutable state tracking
        action_dict = {}
        for i, name in enumerate(target_tactics):
            action_dict[name] = target_directions[i]
        for d in range(20):
            rand_v = rng.randn(16)
            action_dict[f"distractor_{d}"] = rand_v / np.linalg.norm(rand_v)

        def mock_policy(goal_str: str) -> np.ndarray:
            curr = step_idx[0]
            # Policy predicts direction with slight noise
            noise = rng.randn(16) * 0.02
            pred = target_directions[curr] + noise
            return pred / np.linalg.norm(pred)

        def mock_kernel(goal_str: str, tac: str):
            curr = step_idx[0]
            expected = target_tactics[curr]
            if tac == expected:
                if curr == 2:
                    return KernelStatus.GoalsClosed, []
                else:
                    step_idx[0] += 1
                    return KernelStatus.SubgoalsOpen, [f"goal_step_{curr+1}"]
            return KernelStatus.KernelError, []

        res = prover.search(
            initial_goal="goal_step_0",
            policy_fn=mock_policy,
            action_dict=action_dict,
            kernel_eval_fn=mock_kernel
        )

        if res.status == SearchStatus.SuccessProven:
            success_count += 1
            total_kernel_calls += res.kernel_calls
            total_saved_calls += res.repl_calls_saved

    assert success_count == 100, f"Expected 100% success rate, got {success_count}/100"
    print(f"\n[PASS] Speculative Prover: 100/100 trees proven.")
    print(f"Total Kernel Calls: {total_kernel_calls} | Saved REPL Round-Trips: {total_saved_calls}")
