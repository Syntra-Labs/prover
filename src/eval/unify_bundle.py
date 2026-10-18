"""
src/eval/unify_bundle.py: Lattice-Tactic Unification Forward Pipeline.
Connects R4 geometric representations and Conway-Sloane E8 lattice discrete
Voronoi cell snapping to Lean 4 discrete tactic emission.
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.eval.tactic_eval import TacticEvaluator, find_latest_checkpoint
from src.eval.tactic_vocab import ID_TO_TACTIC, TACTIC_MAP
from src.ir.canonicalizer import canonicalize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_r4_inspect_bin() -> Path:
    """Locates authentic compiled r4-inspect binary."""
    candidates = [
        REPO_ROOT / "vendor/uor-r4/target/debug/r4-inspect",
        REPO_ROOT / "vendor/uor-r4/target/release/r4-inspect",
        Path("/content/prover/vendor/uor-r4/target/debug/r4-inspect"),
        Path("/content/prover/vendor/uor-r4/target/release/r4-inspect"),
    ]
    r4_bin = next((p for p in candidates if p.exists() and os.access(p, os.X_OK)), None)
    if r4_bin is None:
        raise FileNotFoundError(f"r4-inspect binary not found in: {candidates}")
    return r4_bin


def authentic_e8_snap(
    blocks_8d: Union[List[List[float]], List[float]],
    scale: float = 0.5,
    r4_bin: Optional[Path] = None,
) -> Any:
    """Calls Conway-Sloane E8 lattice snapping via compiled r4-inspect binary."""
    if r4_bin is None:
        r4_bin = find_r4_inspect_bin()

    is_single = isinstance(blocks_8d, (list, tuple)) and len(blocks_8d) == 8 and not isinstance(blocks_8d[0], (list, tuple))
    payload = [blocks_8d] if is_single else blocks_8d
    blocks_json = json.dumps([list(float(x) for x in b) for b in payload])
    proc = subprocess.run(
        [str(r4_bin), "snap", "--scale", str(scale)],
        input=blocks_json, text=True, capture_output=True, check=True,
    )
    res = json.loads(proc.stdout)
    snapped = [tuple(b) for b in res["snapped"]]
    return snapped[0] if is_single else snapped


def embed_goal_64d(evaluator: TacticEvaluator, canon_goal: str) -> np.ndarray:
    """Embeds canonicalized Lean goal using R4 recurrence state to 64D vector."""
    prompt = canon_goal if "Goal:" in canon_goal else f"Goal:\n{canon_goal}\n\nTactic:\n"
    h = evaluator.get_final_hidden(prompt)
    v = h[:64].detach().cpu().numpy().astype(np.float64)
    return np.pad(v, (0, 64 - len(v))) if len(v) < 64 else v


class TacticCodebook:
    """Maintains 64D E8 Voronoi centroids for discrete tactics in TACTIC_MAP."""

    def __init__(self, centroids: Optional[Dict[str, np.ndarray]] = None):
        self.centroids: Dict[str, np.ndarray] = centroids or {}

    @classmethod
    def build_from_evaluator(
        cls, evaluator: TacticEvaluator, scale: float = 0.5, r4_bin: Optional[Path] = None
    ) -> "TacticCodebook":
        tactics = [k for k in TACTIC_MAP.keys() if k != "unknown"]
        all_blocks = []
        for tac in tactics:
            h = evaluator.get_final_hidden(f"Tactic:\n{tac}")
            v = h[:64].detach().cpu().numpy().astype(np.float64)
            v = np.pad(v, (0, 64 - len(v))) if len(v) < 64 else v
            for b in range(8):
                all_blocks.append(v[8 * b : 8 * (b + 1)].tolist())

        snapped_all = authentic_e8_snap(all_blocks, scale=scale, r4_bin=r4_bin)
        centroids = {
            tac: np.array(snapped_all[i * 8 : (i + 1) * 8], dtype=np.float64).flatten()
            for i, tac in enumerate(tactics)
        }
        return cls(centroids=centroids)

    def nearest_tactic(self, snapped_vec: np.ndarray) -> Tuple[str, int, float, List[Tuple[str, int, float]]]:
        scored = [
            (tac, TACTIC_MAP.get(tac, TACTIC_MAP["unknown"]), float(np.linalg.norm(snapped_vec - c)))
            for tac, c in self.centroids.items()
        ]
        scored.sort(key=lambda x: x[2])
        best = scored[0] if scored else ("unknown", TACTIC_MAP["unknown"], 0.0)
        return best[0], best[1], best[2], scored


class UnifyBundlePipeline:
    """Forward pipeline mapping Lean goals to discrete tactics via E8 snapping."""

    def __init__(
        self,
        evaluator: Optional[TacticEvaluator] = None,
        checkpoint_path: Optional[str] = None,
        r4_bin: Optional[Path] = None,
        scale: float = 0.5,
        codebook: Optional[TacticCodebook] = None,
        device: str = "cpu",
    ):
        torch.set_num_threads(2)
        self.device = device
        self.scale = scale
        self.r4_bin = r4_bin or find_r4_inspect_bin()
        self.evaluator = evaluator or TacticEvaluator(checkpoint_path or find_latest_checkpoint(), device=device)
        self.codebook = codebook or TacticCodebook.build_from_evaluator(self.evaluator, scale=scale, r4_bin=self.r4_bin)

    def forward(self, goal_state: str) -> Dict[str, Any]:
        canon_goal, _, sigma = canonicalize(goal_state)
        v = embed_goal_64d(self.evaluator, canon_goal)
        blocks = [v[8 * b : 8 * (b + 1)].tolist() for b in range(8)]
        snapped_blocks = authentic_e8_snap(blocks, scale=self.scale, r4_bin=self.r4_bin)
        snapped_vec = np.array(snapped_blocks, dtype=np.float64).flatten()
        best_tac, best_id, dist, ranked = self.codebook.nearest_tactic(snapped_vec)
        return {
            "goal": goal_state,
            "canonical_goal": canon_goal,
            "sigma": sigma,
            "snapped_cells": [tuple(b) for b in snapped_blocks],
            "snapped_vector": snapped_vec,
            "emitted_tactic": best_tac,
            "tactic_id": best_id,
            "distance": dist,
            "ranked_tactics": ranked,
        }


def forward_unify_bundle(goal_state: str, pipeline: Optional[UnifyBundlePipeline] = None, **kwargs) -> Dict[str, Any]:
    return (pipeline or UnifyBundlePipeline(**kwargs)).forward(goal_state)
