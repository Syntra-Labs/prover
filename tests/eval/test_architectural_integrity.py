"""
Commit-Time Architectural Integrity & Anti-Regression Gate Suite.
Enforces permanent executable regressions for:
1. `stage2-response-mask`: Stage 2 launcher passes `train_mask=` to `geometric-stack train` -> `model.weighted_loss`.
2. Zero ground-truth tactic head leakage (`+ [t_head]`) in evaluation candidate pools.
3. `r4-weight-parity`: 100% (78/78) learned tensor execution in R4 forward pass; zero dummy 0.95 moving averages.
4. `anti-tautology-falsification`: Zero test-side `re.sub` scrubbing or synthetic `RandomState` trajectories in falsification tests.
5. `ir-alpha-bijection`: Reversible declaration-order alpha-canonicalization (`invert_sigma(canonicalize(g, t)) == t`).
6. `param-tactic-synthesis`: Remains an active blocker in the Task DAG until Decision 2 is finalized.
"""

import ast
import functools
from pathlib import Path
from scripts.lint_tasks import scan_tasks, validate_dag
from src.ir.canonicalizer import canonicalize, invert_sigma

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=1)
def _active_tasks_by_id():
    tasks, errors = scan_tasks()
    dag_errors = validate_dag(tasks)
    assert not errors, f"Task linter parse errors: {errors}"
    assert not dag_errors, f"Task DAG validation errors: {dag_errors}"
    return {t["id"]: t for t in tasks}


def test_stage2_response_mask_never_disregarded():
    """
    Permanent Gate 1 (Resolves `stage2-response-mask`):
    Stage 2 training launcher must pass `train_mask=` to `geometric-stack train`,
    and `geometric-stack.rs` must route `train_mask` through `model.weighted_loss`.
    """
    script_path = REPO_ROOT / "scripts/colab_stage2_train.sh"
    content = script_path.read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = "\n".join(active_lines)

    assert "dialogue-train" not in joined, (
        "Regression: scripts/colab_stage2_train.sh must not use natural-language chatbot `dialogue-train`."
    )
    for required_arg in ("train=", "train_mask=", "valid=", "init="):
        assert required_arg in joined, (
            f"Regression in stage2-response-mask: scripts/colab_stage2_train.sh missing `{required_arg}`."
        )

    gs_rs = (REPO_ROOT / "vendor/uor-r4/crates/uor-r4-training/examples/geometric-stack.rs").read_text(encoding="utf-8")
    assert "model.weighted_loss(" in gs_rs and '"train_mask"' in gs_rs, (
        "Regression in stage2-response-mask: geometric-stack.rs must wire `train_mask` to `model.weighted_loss`."
    )


def test_zero_ground_truth_candidate_leakage():
    """
    Permanent Gate 2: Evaluation scripts in src/eval/ must never inject ground-truth labels
    (`t_head`, `ground_truth_tactic`, `expected_tactics`) into candidate pools at test time.
    """
    forbidden_vars = {"t_head", "tactic", "ground_truth_tactic", "expected_tactics", "target_proof"}
    for eval_file in (REPO_ROOT / "src/eval/authentic_benchmark.py", REPO_ROOT / "src/eval/authentic_lean_prover.py"):
        source = eval_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(eval_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.BitOr)):
                segment = ast.get_source_segment(source, node) or ""
                if "CANONICAL_TACTIC_POOL" in segment:
                    names_in_expr = {
                        n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                    }
                    leaked = names_in_expr.intersection(forbidden_vars)
                    assert not leaked, (
                        f"Ground-truth leakage detected in {eval_file.name}: "
                        f"CANONICAL_TACTIC_POOL combined with ground-truth variable(s) {leaked} in `{segment}`"
                    )


def test_r4_weight_parity_gate():
    """
    Permanent Gate 3 (Resolves `r4-weight-parity`):
    Verifies zero hardcoded 0.95 moving average in `src/eval/tactic_eval.py` and `src/eval/r4_runner.py`,
    and verifies live forward pass consumes 100% (all 78) of the learned tensors in `model.safetensors`.
    """
    from src.eval.r4_runner import find_latest_checkpoint
    from src.eval.tactic_eval import TacticEvaluator

    for rel in ("src/eval/tactic_eval.py", "src/eval/r4_runner.py"):
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "0.95 * state" not in src and "0.1 * x_norm" not in src, (
            f"Regression in {rel}: hardcoded dummy moving average detected (violates r4-weight-parity)."
        )

    evaluator = TacticEvaluator(find_latest_checkpoint(), device="cpu")
    res = evaluator.evaluate_tactic_likelihood("theorem t (n : Nat) : n = n := by", "rfl")
    assert len(evaluator.r4.consumed_tensors) == len(evaluator.r4.tensors) == 78, (
        f"r4-weight-parity violation: consumed {len(evaluator.r4.consumed_tensors)}/{len(evaluator.r4.tensors)} tensors."
    )
    assert res["cross_entropy_loss"] > 0.0


def test_anti_tautology_falsification_gate():
    """
    Permanent Gate 4 (Resolves `anti-tautology-falsification`):
    Verifies zero test-side `re.sub` prompt scrubbing in `test_tier_b_stability.py`
    and zero synthetic `RandomState` / `.randn(` trajectories in `test_tier_c_search.py`.
    """
    tier_b_src = (REPO_ROOT / "tests/geometric_falsification/test_tier_b_stability.py").read_text(encoding="utf-8")
    tier_c_src = (REPO_ROOT / "tests/geometric_falsification/test_tier_c_search.py").read_text(encoding="utf-8")

    assert "re.sub(" not in tier_b_src, (
        "anti-tautology-falsification violation: test_tier_b_stability.py contains test-side `re.sub` scrubbing."
    )
    assert "0.95 * state" not in tier_b_src, (
        "anti-tautology-falsification violation: test_tier_b_stability.py contains hardcoded 0.95 loop."
    )
    assert "RandomState(" not in tier_c_src and ".randn(" not in tier_c_src, (
        "anti-tautology-falsification violation: test_tier_c_search.py contains synthetic random trajectories."
    )


def test_ir_bijection_and_param_tactic_dag_gates():
    """
    Permanent Gate 5 (Resolves `ir-alpha-bijection` via Option 1A; tracks `param-tactic-synthesis`):
    Verifies declaration-order alpha-canonicalization is strictly bijective and invariant under renaming.
    """
    g1 = "m n : Nat\nh1 : m + n = n + m\nh2 : n = 0\n⊢ m + 0 = m"
    g2 = "a b : Nat\nhx : a + b = b + a\nhy : b = 0\n⊢ a + 0 = a"
    tac1 = "rw [h2] at h1; exact h1"
    tac2 = "rw [hy] at hx; exact hx"

    cg1, ct1, sig1 = canonicalize(g1, tac1)
    cg2, ct2, sig2 = canonicalize(g2, tac2)

    assert cg1 == cg2, f"ir-alpha-bijection violation: canonical goals differ:\n{cg1}\nvs\n{cg2}"
    assert ct1 == ct2 == "rw [h1] at h0; exact h0"
    assert invert_sigma(ct1, sig1) == tac1
    assert invert_sigma(ct2, sig2) == tac2

    tasks = _active_tasks_by_id()
    assert "param-tactic-synthesis" in tasks, "Missing `param-tactic-synthesis` task in DAG"
    assert "macro-decider" in tasks["param-tactic-synthesis"]["blocks"]
