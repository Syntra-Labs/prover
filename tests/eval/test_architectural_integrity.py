"""
Commit-Time Architectural Integrity & Anti-Regression Gate Suite.
Enforces:
1. Stage 2 never disregards train.mask (must use `dialogue-train` with `train_mask` and `dev_mask`).
2. Benchmark scripts never leak ground-truth tactic heads (`+ [t_head]`) into candidate pools.
3. All 5 architectural audit tasks remain active blockers in the Task DAG until their
   executable verification criteria pass 100% (preventing fake or comment-only resolutions).
"""

import ast
import functools
import re
from pathlib import Path
from scripts.lint_tasks import scan_tasks, validate_dag

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
    Permanent Gate 1: Stage 2 training launcher must never fall back to unmasked `train`
    and must explicitly pass `train_mask` and `dev_mask` to `dialogue-train`.
    """
    script_path = REPO_ROOT / "scripts/colab_stage2_train.sh"
    content = script_path.read_text(encoding="utf-8")

    # Strip comment lines before checking active commands
    active_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = "\n".join(active_lines)

    assert 'dialogue-train' in joined, (
        "Regression detected: scripts/colab_stage2_train.sh must invoke `dialogue-train`, "
        "not unmasked `train`."
    )
    assert not re.search(r'"\$GS_BIN"\s+train\b', joined), (
        "Regression detected: scripts/colab_stage2_train.sh invoked unmasked `train`, "
        "which disregards train.mask."
    )
    for required_arg in ("train_tokens=", "train_mask=", "train_manifest=", "dev_tokens=", "dev_mask=", "dev_manifest="):
        assert required_arg in joined, (
            f"Regression detected: scripts/colab_stage2_train.sh is missing required masked SFT argument `{required_arg}`."
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
    Gate 3: Until `r4-weight-parity` is genuinely resolved (zero hardcoded 0.95 moving average
    and full 78-tensor utilization or direct Rust binary execution), it must remain in the
    Task DAG blocking downstream stability/policy/search tasks.
    """
    tasks = _active_tasks_by_id()
    tactic_eval_src = (REPO_ROOT / "src/eval/tactic_eval.py").read_text(encoding="utf-8")

    has_dummy_ema = ("0.95 * state" in tactic_eval_src) or ("0.1 * x_norm" in tactic_eval_src)
    if has_dummy_ema:
        assert "r4-weight-parity" in tasks, (
            "Premature Resolution Violation: `r4-weight-parity` was removed from the Task DAG, "
            "but `src/eval/tactic_eval.py` still contains hardcoded weightless dummy operations (`0.95 * state` / `0.1 * x_norm`)."
        )
        expected_blocks = {"stab-alpha", "stab-jitter", "policy-r4", "search-action", "search-dead-tactic"}
        actual_blocks = set(tasks["r4-weight-parity"]["blocks"])
        assert expected_blocks.issubset(actual_blocks), (
            f"`r4-weight-parity` must block {expected_blocks}, found {actual_blocks}"
        )


def test_anti_tautology_falsification_gate():
    """
    Gate 4: Until `anti-tautology-falsification` is resolved (zero test-side `re.sub` prompt
    scrubbing in `test_tier_b_stability.py` and zero synthetic `RandomState` trajectories in
    `test_tier_c_search.py`), it must remain an active Tier A blocker in the Task DAG.
    """
    tasks = _active_tasks_by_id()
    tier_b_src = (REPO_ROOT / "tests/geometric_falsification/test_tier_b_stability.py").read_text(encoding="utf-8")
    tier_c_src = (REPO_ROOT / "tests/geometric_falsification/test_tier_c_search.py").read_text(encoding="utf-8")

    has_test_side_resub = "re.sub(" in tier_b_src
    has_synthetic_randn = "RandomState(" in tier_c_src or ".randn(" in tier_c_src

    if has_test_side_resub or has_synthetic_randn:
        assert "anti-tautology-falsification" in tasks, (
            "Premature Resolution Violation: `anti-tautology-falsification` was removed from the Task DAG, "
            "but falsification tests still contain test-side `re.sub` scrubbing or synthetic `randn` vectors."
        )
        expected_blocks = {"stab-alpha", "stab-jitter", "search-action"}
        actual_blocks = set(tasks["anti-tautology-falsification"]["blocks"])
        assert expected_blocks.issubset(actual_blocks), (
            f"`anti-tautology-falsification` must block {expected_blocks}, found {actual_blocks}"
        )


def test_ir_bijection_and_param_tactic_dag_gates():
    """
    Gate 5: Verify `stage2-response-mask`, `ir-alpha-bijection`, and `param-tactic-synthesis`
    are tracked with strict dependency edges in the Task DAG until their implementations exist.
    """
    tasks = _active_tasks_by_id()

    # Check stage2-response-mask contract
    uort_src = (REPO_ROOT / "scripts/prepare_dialogue_uort.py").read_text(encoding="utf-8")
    if "Assistant:" not in uort_src:
        assert "stage2-response-mask" in tasks, (
            "Premature Resolution Violation: `stage2-response-mask` removed before literal-role "
            "`User: ... Assistant: ...` episode encoding was implemented."
        )
        assert "policy-r4" in tasks["stage2-response-mask"]["blocks"]
        assert "param-tactic-synthesis" in tasks["stage2-response-mask"]["blocks"]

    # Check ir-alpha-bijection contract
    prep_src = (REPO_ROOT / "scripts/prepare_mathlib_for_r4.py").read_text(encoding="utf-8")
    if "invert_sigma" not in prep_src:
        assert "ir-alpha-bijection" in tasks, (
            "Premature Resolution Violation: `ir-alpha-bijection` removed before reversible "
            "symbol-table (`sigma` / `invert_sigma`) canonicalization was implemented."
        )
        assert {"stab-alpha", "stab-jitter", "policy-r4"}.issubset(set(tasks["ir-alpha-bijection"]["blocks"]))

    # Check param-tactic-synthesis contract
    assert "param-tactic-synthesis" in tasks, "Missing `param-tactic-synthesis` task in DAG"
    assert "macro-decider" in tasks["param-tactic-synthesis"]["blocks"]
