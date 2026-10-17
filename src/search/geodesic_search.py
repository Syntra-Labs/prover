"""
Speculative Adaptive-Margin Geodesic Search Engine.
Implements the continuous-to-discrete geodesic router with:
1. Dynamic Angular Margin Gating (Δ cos >= tau).
2. Speculative Execution with Kernel Short-Circuiting.
3. Strict alignment with specs/src/ProverEngine.lex.tex state invariants.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np

class KernelStatus(Enum):
    GoalsClosed = "GoalsClosed"
    SubgoalsOpen = "SubgoalsOpen"
    KernelError = "KernelError"
    KernelTimeout = "KernelTimeout"

class SearchStatus(Enum):
    Searching = "Searching"
    SuccessProven = "SuccessProven"
    BudgetExhausted = "BudgetExhausted"
    SearchFailed = "SearchFailed"

@dataclass
class TacticCandidate:
    name: str
    vector: np.ndarray
    score: float = 0.0

@dataclass
class SearchNode:
    state_id: str
    goal_str: str
    depth: int = 0
    parent: Optional['SearchNode'] = None
    applied_tactic: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)

@dataclass
class GeodesicSearchResult:
    status: SearchStatus
    proof_steps: List[str]
    total_steps: int
    kernel_calls: int
    repl_calls_saved: int
    depth_reached: int

class SpeculativeGeodesicRouter:
    """
    Continuous geodesic router that projects continuous tangent trajectories
    onto discrete action candidates with speculative short-circuiting.
    """
    def __init__(self, margin_threshold: float = 0.05, max_beam_width: int = 2):
        self.margin_threshold = margin_threshold
        self.max_beam_width = max_beam_width

    # FIXME(param-tactic-synthesis | tier:B | blocks:[macro-decider]):
    # Description: Factorize tactic emission into (1) geodesic macro-tactic head routing and (2) contextual argument/premise generation without ground-truth candidate pool leakage.
    # Criterion: assert: zero_ground_truth_candidate_injection and emits_full_parameterized_tactics
    def rank_candidates(
        self,
        tangent_vector: np.ndarray,
        action_dict: Dict[str, np.ndarray]
    ) -> List[TacticCandidate]:
        """
        Rank discrete tactics by cosine similarity to the continuous tangent vector.
        """
        norm_v = np.linalg.norm(tangent_vector)
        if norm_v < 1e-9:
            raise ValueError("Zero tangent vector cannot determine geodesic direction.")
        unit_v = tangent_vector / norm_v

        candidates = []
        for name, vec in action_dict.items():
            norm_w = np.linalg.norm(vec)
            if norm_w < 1e-9:
                sim = 0.0
            else:
                sim = float(np.dot(unit_v, vec / norm_w))
            candidates.append(TacticCandidate(name=name, vector=vec, score=sim))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def select_speculative_queue(
        self,
        ranked: List[TacticCandidate]
    ) -> Tuple[List[TacticCandidate], float, bool]:
        """
        Applies dynamic margin gating:
        Δ cos = cos(v, e_(1)) - cos(v, e_(2))
        If Δ cos >= tau -> Top-1 only (decisive, b=1)
        If Δ cos < tau -> Top-2 speculative queue (ambiguous / potential shadow, b=2)
        """
        if not ranked:
            return [], 0.0, False
        if len(ranked) == 1:
            return [ranked[0]], 1.0, False

        margin = ranked[0].score - ranked[1].score
        if margin >= self.margin_threshold or self.max_beam_width < 2:
            return [ranked[0]], margin, False
        else:
            return [ranked[0], ranked[1]], margin, True

class SpeculativeGeodesicProver:
    """
    Executes proof search using the Speculative Geodesic Router against Lean kernel evaluation.
    """
    def __init__(
        self,
        router: SpeculativeGeodesicRouter,
        max_steps: int = 100,
        max_depth: int = 20
    ):
        self.router = router
        self.max_steps = max_steps
        self.max_depth = max_depth

    def search(
        self,
        initial_goal: str,
        policy_fn: Callable[[str], np.ndarray],
        action_dict: Dict[str, np.ndarray],
        kernel_eval_fn: Callable[[str, str], Tuple[KernelStatus, List[str]]]
    ) -> GeodesicSearchResult:
        """
        Searches for a proof certificate from initial_goal to Q.E.D.
        kernel_eval_fn: (goal_str, tactic_str) -> (KernelStatus, new_subgoals)
        """
        root = SearchNode(state_id="0", goal_str=initial_goal, depth=0)
        open_nodes: List[SearchNode] = [root]

        total_steps = 0
        kernel_calls = 0
        repl_calls_saved = 0
        proof_path: List[str] = []

        while open_nodes and total_steps < self.max_steps:
            current = open_nodes.pop(0)
            total_steps += 1

            if current.depth >= self.max_depth:
                continue

            # 1. Predict continuous geodesic vector for active goal
            tangent_v = policy_fn(current.goal_str)

            # 2. Rank candidates and filter via margin threshold
            ranked = self.router.rank_candidates(tangent_v, action_dict)
            queue, margin, is_ambiguous = self.router.select_speculative_queue(ranked)

            # 3. Speculative execution against Lean kernel
            step_succeeded = False
            for idx, candidate in enumerate(queue):
                kernel_calls += 1
                status, new_goals = kernel_eval_fn(current.goal_str, candidate.name)

                if status == KernelStatus.GoalsClosed:
                    proof_path.append(candidate.name)
                    # If candidate 1 closed the goal, candidate 2 was never sent to Lean REPL!
                    if idx == 0 and len(queue) > 1:
                        repl_calls_saved += 1
                    return GeodesicSearchResult(
                        status=SearchStatus.SuccessProven,
                        proof_steps=proof_path,
                        total_steps=total_steps,
                        kernel_calls=kernel_calls,
                        repl_calls_saved=repl_calls_saved,
                        depth_reached=current.depth + 1
                    )
                elif status == KernelStatus.SubgoalsOpen:
                    proof_path.append(candidate.name)
                    step_succeeded = True
                    for sub in new_goals:
                        child = SearchNode(
                            state_id=f"{current.state_id}.{len(open_nodes)}",
                            goal_str=sub,
                            depth=current.depth + 1,
                            parent=current,
                            applied_tactic=candidate.name
                        )
                        open_nodes.append(child)
                    if idx == 0 and len(queue) > 1:
                        repl_calls_saved += 1
                    break
                else:
                    # KernelError: Tactic rejected by Lean. Fall back to next candidate in queue.
                    continue

            if not step_succeeded and not open_nodes:
                return GeodesicSearchResult(
                    status=SearchStatus.SearchFailed,
                    proof_steps=proof_path,
                    total_steps=total_steps,
                    kernel_calls=kernel_calls,
                    repl_calls_saved=repl_calls_saved,
                    depth_reached=current.depth
                )

        status = SearchStatus.BudgetExhausted if total_steps >= self.max_steps else SearchStatus.SearchFailed
        return GeodesicSearchResult(
            status=status,
            proof_steps=proof_path,
            total_steps=total_steps,
            kernel_calls=kernel_calls,
            repl_calls_saved=repl_calls_saved,
            depth_reached=current.depth if 'current' in locals() else 0
        )
