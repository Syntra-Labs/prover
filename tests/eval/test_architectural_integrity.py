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

    # Verify both Colab launchers auto-persist checkpoints & benchmark_history.csv to GitHub/HF/GDrive
    for launcher in ("scripts/colab_train.sh", "scripts/colab_stage2_train.sh"):
        l_src = (REPO_ROOT / launcher).read_text(encoding="utf-8")
        assert "persist_checkpoint_cloud.sh" in l_src, (
            f"Regression: {launcher} must invoke `persist_checkpoint_cloud.sh` for unattended cloud persistence."
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
    Permanent Gate 5 (Resolves `ir-alpha-bijection` via Option 1A and `param-tactic-synthesis` via Native Lean 4 Syntax AST):
    1. Verifies declaration-order alpha-canonicalization is strictly bijective and invariant under renaming.
    2. Verifies `LeanASTBridge` queries Lean 4's real `Lean.Parser.runParserCategory` and `Syntax.reprint`.
    3. Verifies `SpeculativeGeodesicRouter.synthesize_ast_tactics` routes over native `Lean.Parser.Tactic.*`
       AST node kinds and strictly blocks out-of-scope hypotheses.
    """
    import numpy as np
    from src.ir.tactic_ast import LeanASTBridge
    from src.search.geodesic_search import SpeculativeGeodesicRouter

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

    # Verify real Lean 4 `Lean.Syntax` parser and reprinter via LeanASTBridge
    parsed = LeanASTBridge.parse_tactic("rw [← hy] at hx", sig2)
    assert parsed["ok"] is True
    assert parsed["root_kind"] == "Lean.Parser.Tactic.rwSeq"
    assert "NODE:Lean.Parser.Tactic.rwRule" in parsed["preorder"]
    assert parsed["canonical_tactic"] == "rw [← h1] at h0"

    # Test Native Lean 4 Syntax AST Geodesic Synthesis (`param-tactic-synthesis`)
    router = SpeculativeGeodesicRouter(margin_threshold=0.05, max_beam_width=2)
    target_ast_path = ["NODE:Lean.Parser.Tactic.rwSeq", "ATOM:←", "IDENT:h1", "IDENT:h0"]

    def slot_geometry_fn(canon_goal, prefix_tokens, valid_nodes):
        valid_tokens = {n.token for n in valid_nodes}
        assert "IDENT:h9" not in valid_tokens
        step_idx = len(prefix_tokens)
        desired = target_ast_path[step_idx]
        tangent = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        action_dict = {
            n.token: (np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) if n.token == desired else np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            for n in valid_nodes
        }
        return tangent, action_dict

    synthesized = router.synthesize_ast_tactics(g2, slot_geometry_fn)
    assert synthesized[0][0] == "rw [← hy] at hx", (
        f"param-tactic-synthesis violation: expected 'rw [← hy] at hx', got {synthesized[0][0]!r}"
    )

    # 4. Verify unified turnstile format + blind-inference LemmaTable (L0, L1, ...) bijection
    from src.ir.lemma_table import LemmaTable
    lt = LemmaTable()
    thm_decl = "theorem prop_id (p : Prop) (h : p) : p := by"
    c_decl, c_dtac, decl_sigma = canonicalize(thm_decl, "exact h", lemma_table=lt)
    assert c_decl == "v0 : Prop\nh0 : v0\n⊢ v0"
    assert c_dtac == "exact h0"
    assert invert_sigma(c_dtac, decl_sigma, lemma_table=lt) == "exact h"

    eisen_state = "K : Type u_1\nhp : Prime p\nhn : ¬x ∈ adjoin K {y}\n⊢ False"
    eisen_tac = "exact hn (mem_adjoin_of_smul_prime_smul_of_minpoly_isEisensteinAt hp)"
    c_es, c_etac, _ = canonicalize(eisen_state, eisen_tac, lemma_table=lt)
    assert c_etac == "exact h0 (L0 v1)"
    _, _, blind_sigma = canonicalize(eisen_state, lemma_table=lt)
    assert "L0" not in blind_sigma.values()
    assert invert_sigma(c_etac, blind_sigma, lemma_table=lt) == eisen_tac


def test_kernel_expr_and_admissibility_gate():
    """
    Permanent Gate for `kernel-expr-ir` and `ast-admissibility-gate`:
    1. Verifies Lean 4 kernel Expr DAG extractor elaborates closed goals into 11-constructor DAG
       with zero surface BPE tokens.
    2. Verifies structural admissibility gate achieves >= 98% search-space reduction on the whole
       environment lemma universe while retaining ground truth.
    """
    from src.ir.kernel_expr import ExprBridge
    from src.ir.admissibility import admissible_moves, Move

    decl = "theorem prop_id (p : Prop) (h : p) : p := by"
    st = ExprBridge.state(decl, ["And.intro", "Nat.add_comm"])
    assert len(st.nodes) > 0
    assert len(st.hyps) == 2
    assert st.hyps[0].user == "p"
    assert st.hyps[1].user == "h"
    assert st.hyps[1].type == st.target

    # Gate on focused universe
    moves = admissible_moves(st, decl=decl)
    assert Move("exact", "fv1") in moves
    assert Move("apply", "And.intro") not in moves

    # Gate on full environment universe
    st_full = ExprBridge.state(decl, "*")
    moves_full = admissible_moves(st_full, decl=decl)
    universe = len(st_full.consts) + len(st_full.hyps) + 1
    reduction = 1 - (len(moves_full) / universe)
    assert reduction >= 0.99
    assert Move("exact", "fv1") in moves_full


def test_tree_rotor_operator_invariants():
    """
    Permanent Gate for `tree-rotor-operator`:
    1. Zero fixed sliding-window truncation: AST DAG is folded completely without truncation.
    2. Non-commutative asymmetry: d_S3(App(A, B), App(B, A)) > 0.
    """
    import subprocess
    import ctypes

    build_dir = REPO_ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    my_bin_dir = build_dir / "my_bin"
    my_bin_dir.mkdir(exist_ok=True)

    bin_path = my_bin_dir / "test_tree_rotor"
    if not bin_path.exists():
        subprocess.run(
            ["g++", "-std=c++17", "-O2", str(REPO_ROOT / "src/tree_rotor.cpp"),
             str(REPO_ROOT / "tests/test_tree_rotor.cpp"), "-o", str(bin_path)],
            check=True
        )
    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "ALL S^3 TREE ROTOR OPERATOR TESTS PASSED" in res.stdout

    # Test C-ABI invariants directly via ctypes
    so_path = build_dir / "libtree_rotor.so"
    if not so_path.exists():
        subprocess.run(
            ["g++", "-std=c++17", "-fPIC", "-shared", str(REPO_ROOT / "src/tree_rotor.cpp"),
             "-o", str(so_path)],
            check=True
        )
    lib = ctypes.CDLL(str(so_path))
    lib.rotor_check_invariants.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.rotor_check_invariants.restype = ctypes.c_int

    asym = ctypes.c_double(0.0)
    trunc = ctypes.c_double(1.0)
    rc = lib.rotor_check_invariants(ctypes.byref(asym), ctypes.byref(trunc))

    window_truncation_rate = trunc.value
    non_commutative_asymmetry = asym.value

    assert rc == 0
    assert window_truncation_rate == 0.0
    assert non_commutative_asymmetry > 0.0


def test_least_action_prover_invariants():
    """
    Permanent Gate for `least-action-prover`:
    1. Certified QED Rate >= 0.80 across benchmark theorems.
    2. Zero no-op tactics selected (noop_tactic_selected == 0).
    """
    import subprocess

    build_dir = REPO_ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    my_bin_dir = build_dir / "my_bin"
    my_bin_dir.mkdir(exist_ok=True)

    bin_path = my_bin_dir / "eval_least_action_prover"
    if not bin_path.exists():
        subprocess.run(
            ["g++", "-std=c++11", "-O2",
             str(REPO_ROOT / "src/eval_least_action_prover.cpp"),
             str(REPO_ROOT / "src/tree_rotor.cpp"),
             str(REPO_ROOT / "src/search/geodesic_search.cpp"),
             "-o", str(bin_path)],
            check=True
        )
    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "Certified QED Rate: 100.000%" in res.stdout
    assert "No-Op Tactics Selected: 0" in res.stdout
    assert "[PASS] Task 'least-action-prover' invariants satisfied in pure C++." in res.stdout


def test_legacy_test_script_gates():
    """
    Permanent Gate for `legacy-test-hardcoded` and `legacy-test-multilang`:
    1. `scripts/test.sh` accepts target source file as $1 and defaults to `src/process.cpp`.
    2. `scripts/test.sh` branches on file extension to call appropriate compiler (.cpp, .lean, .rs).
    """
    import subprocess
    script = REPO_ROOT / "scripts/test.sh"
    content = script.read_text(encoding="utf-8")

    # Verify no hardcoded /home/yassine paths
    assert "/home/yassine" not in content, "Hardcoded absolute user paths found in scripts/test.sh"

    # Verify $1 parameterization and extension branching
    assert "$1" in content or "${1" in content, "scripts/test.sh must accept target source file as $1"
    assert "src/process.cpp" in content, "scripts/test.sh must default to src/process.cpp"
    assert "cpp" in content and "lean" in content and "rs" in content, "scripts/test.sh must branch on .cpp, .lean, and .rs"

    # Test execution with --help
    res_help = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True, check=True)
    assert "Usage:" in res_help.stdout

    # Test execution with default C++ file
    res_default = subprocess.run(["bash", str(script)], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
    assert res_default.returncode == 0

    # Test execution with unsupported extension
    res_unsupported = subprocess.run(["bash", str(script), "README.md"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res_unsupported.returncode != 0
    assert "unsupported file extension" in res_unsupported.stderr


def test_macro_decider_and_policy_r4_gates():
    """
    Permanent Gate for `macro-decider` and `policy-r4`:
    1. Executes compiled C++ MacroActionDecider binary and verifies all routing/feedback invariants pass.
    2. Runs authentic Mathlib 4 benchmark with macro-action gating and verifies:
       top1_accuracy >= 0.30 and top5_accuracy >= 0.65.
    """
    import subprocess
    from src.eval.authentic_benchmark import run_authentic_benchmark

    build_dir = REPO_ROOT / "build"
    my_bin_dir = build_dir / "my_bin"
    bin_path = my_bin_dir / "test_macro_decider"

    if not bin_path.exists():
        subprocess.run(
            ["g++", "-std=c++11", "-O2",
             str(REPO_ROOT / "src/tree_rotor.cpp"),
             str(REPO_ROOT / "src/search/macro_decider.cpp"),
             str(REPO_ROOT / "tests/test_macro_decider.cpp"),
             "-o", str(bin_path)],
            check=True
        )
    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "ALL MACRO-ACTION DECIDER TESTS PASSED" in res.stdout
    assert "Candidate admissibility filtering verified." in res.stdout

    # Policy R4 + Macro Decider validation holdout verification
    top1, top5 = run_authentic_benchmark(num_steps=20)
    assert top1 >= 0.30, f"Policy-R4 Top-1 accuracy {top1:.4f} < 0.30"
    assert top5 >= 0.65, f"Policy-R4 Top-5 accuracy {top5:.4f} < 0.65"


def test_live_lean4_kernel_prover_invariants():
    """
    Permanent Gate for Live Lean 4 Compiler Verification (100% QED):
    1. Executes compiled C++ live Lean 4 prover binary (`eval_live_lean_prover`).
    2. Verifies 100.0% (5/5) theorems certified on the real Lean 4 compiler binary (`lean`).
    """
    import subprocess

    build_dir = REPO_ROOT / "build"
    my_bin_dir = build_dir / "my_bin"
    bin_path = my_bin_dir / "eval_live_lean_prover"

    if not bin_path.exists():
        subprocess.run(
            ["g++", "-std=c++11", "-O2",
             str(REPO_ROOT / "src/eval_live_lean_prover.cpp"),
             str(REPO_ROOT / "src/tree_rotor.cpp"),
             str(REPO_ROOT / "src/search/macro_decider.cpp"),
             str(REPO_ROOT / "src/search/geodesic_search.cpp"),
             "-o", str(bin_path)],
            check=True
        )
    res = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    assert "LIVE LEAN 4 VERIFICATION RESULTS: 5 / 5 (100.0%) certified." in res.stdout
    assert "[PASS] 100% of benchmark theorems certified by Lean 4 kernel in pure C++." in res.stdout

