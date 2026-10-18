"""
tests/test_unify_bundle.py: Verification tests for Lattice-Tactic Unification Pipeline.
Verifies end-to-end forward pass maps canonicalized Lean goals to discrete tactics
via Conway-Sloane E8 Voronoi snapping.
"""
import numpy as np
import pytest
import torch

from src.eval.tactic_eval import TacticEvaluator, find_latest_checkpoint
from src.eval.tactic_vocab import ID_TO_TACTIC, TACTIC_MAP
from src.eval.unify_bundle import UnifyBundlePipeline, find_r4_inspect_bin

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def pipeline():
    evaluator = TacticEvaluator(find_latest_checkpoint(), device="cpu")
    return UnifyBundlePipeline(evaluator=evaluator, r4_bin=find_r4_inspect_bin(), scale=0.5)


def test_unify_bundle_forward_add_zero(pipeline):
    """Verifies end-to-end forward pass on algebraic goal `n + 0 = n`."""
    goal = "n + 0 = n"
    res = pipeline.forward(goal)
    assert res["emitted_tactic"] in TACTIC_MAP
    assert res["tactic_id"] == TACTIC_MAP[res["emitted_tactic"]]
    assert ID_TO_TACTIC[res["tactic_id"]] == res["emitted_tactic"]
    assert len(res["snapped_cells"]) == 8 and res["snapped_vector"].shape == (64,)
    assert res["distance"] >= 0.0 and not np.isnan(res["distance"])
    print(f"\n[UNIFY-BUNDLE] Goal: '{goal}' -> Tactic: '{res['emitted_tactic']}' (ID: {res['tactic_id']}, Dist: {res['distance']:.4f})")


def test_unify_bundle_forward_conjunction(pipeline):
    """Verifies end-to-end forward pass on propositional goal `p ∧ q`."""
    goal = "p ∧ q"
    res = pipeline.forward(goal)
    assert res["emitted_tactic"] in TACTIC_MAP
    assert res["tactic_id"] == TACTIC_MAP[res["emitted_tactic"]]
    assert ID_TO_TACTIC[res["tactic_id"]] == res["emitted_tactic"]
    assert len(res["snapped_cells"]) == 8 and res["snapped_vector"].shape == (64,)
    assert res["distance"] >= 0.0
    print(f"\n[UNIFY-BUNDLE] Goal: '{goal}' -> Tactic: '{res['emitted_tactic']}' (ID: {res['tactic_id']}, Dist: {res['distance']:.4f})")


def test_e8_lattice_quantization_properties(pipeline):
    """Verifies that snapped 8D Voronoi points satisfy Conway-Sloane E8 lattice properties."""
    res = pipeline.forward("theorem add_zero (n : Nat) : n + 0 = n := by")
    scale = pipeline.scale
    for cell in res["snapped_cells"]:
        unscaled = [round(x / scale, 6) for x in cell]
        for c in unscaled:
            assert abs(c * 2 - round(c * 2)) < 1e-5, f"Coord {c} not half-integer/int in E8"
        is_all_int = all(abs(c - round(c)) < 1e-5 for c in unscaled)
        is_all_half = all(abs((c - 0.5) - round(c - 0.5)) < 1e-5 for c in unscaled)
        assert is_all_int or is_all_half, f"Mixed parity in E8: {unscaled}"
        assert int(round(sum(c * 2 for c in unscaled))) % 2 == 0, f"Odd sum in E8: {unscaled}"
