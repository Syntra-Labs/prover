#!/usr/bin/env python3
"""
Falsification run for `ast-admissibility-gate` on the kernel `Lean.Expr` IR.

For each goal: elaborate with Lean's real elaborator, encode goal + the WHOLE lemma
universe of the loaded environment into one DAG, run the structural gate, and report:
  * universe size vs admissible set size (candidate_reduction_ratio)
  * whether the ground-truth first step survived the gate (ground_truth_retained)
  * whether the Lean 4 kernel actually accepts that ground-truth step (so the GT is real)

Criterion: assert: candidate_reduction_ratio >= 0.90 and ground_truth_retained == True
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.lean_bench import BENCHMARK_THEOREMS  # noqa: E402
from src.ir.admissibility import Move, admissible_moves  # noqa: E402
from src.ir.kernel_expr import ExprBridge  # noqa: E402

# (declaration, ground-truth first move). fvN = N-th introduced binder.
EXTRA_GOALS = [
    ("theorem t_and_intro (p q : Prop) (hp : p) (hq : q) : p ∧ q := by", Move("apply", "And.intro")),
    ("theorem t_mp (p q : Prop) (hpq : p → q) (hp : p) : q := by", Move("apply", "fv2")),
    ("theorem t_rw_hyp (a b c : Nat) (h : a = b) : a + c = b + c := by", Move("rw", "fv3")),
    ("theorem t_and_left (p : Prop) (h : p ∧ True) : p := by", Move("elim", "And.left", via="fv1")),
    ("theorem t_symm (n m : Nat) (h : n = m) : m = n := by", Move("apply", "Eq.symm")),
    ("theorem t_len_append (xs ys : List Nat) : (xs ++ ys).length = xs.length + ys.length := by",
     Move("apply", "List.length_append")),
    ("theorem t_intro (p q : Prop) : p ∧ q → q ∧ p := by", Move("intro", "")),
    ("theorem t_mul_comm (a b : Nat) : a * b = b * a := by", Move("apply", "Nat.mul_comm")),
    ("theorem t_or_inl (p q : Prop) (hp : p) : p ∨ q := by", Move("apply", "Or.inl")),
    ("theorem t_succ_le (n : Nat) : n ≤ n + 1 := by", Move("apply", "Nat.le_succ")),
]
BENCH_GT = {
    "prop_id": Move("exact", "fv1"),
    "prop_and_comm": Move("apply", "And.intro"),
    "nat_add_zero": Move("apply", "Nat.add_zero"),
    "nat_zero_add": Move("apply", "Nat.zero_add"),
    "nat_add_comm": Move("apply", "Nat.add_comm"),
}


def to_tactic(m: Move, users: dict) -> str:
    src = users.get(m.source, m.source)
    if m.kind == "intro":
        return "intro"
    if m.kind == "exact":
        return f"exact {src}"
    if m.kind == "apply":
        return f"apply {src}"
    if m.kind == "rw":
        return f"rw [{'← ' if m.reverse else ''}{src}]"
    if m.kind == "elim":
        return f"exact {src} {users.get(m.via, m.via)}"
    raise ValueError(m)


def lean_accepts_first_step(decl: str, tactic: str) -> bool:
    src = f"{decl}\n  {tactic}\n  all_goals sorry\n"
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        out = subprocess.run(["lean", path], capture_output=True, text=True, timeout=60)
        return "error:" not in (out.stdout + out.stderr)
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    goals = [(t["lean_code"], BENCH_GT[t["id"]]) for t in BENCHMARK_THEOREMS] + EXTRA_GOALS
    rows = []
    for decl, gt in goals:
        t0 = time.time()
        st = ExprBridge.state(decl, "*")
        t1 = time.time()
        moves = admissible_moves(st, decl=decl)
        t2 = time.time()
        users = {h.fv: h.user for h in st.hyps}
        universe = len(st.consts) + len(st.hyps) + 1  # lemmas + hypotheses + intro
        retained = gt in moves
        kernel_ok = lean_accepts_first_step(decl, to_tactic(gt, users))
        name = decl.split()[1]
        rows.append({
            "goal": name, "universe": universe, "admissible": len(moves),
            "reduction": 1 - len(moves) / universe, "gt": str(gt),
            "gt_retained": retained, "gt_kernel_accepted": kernel_ok,
            "dag_nodes": len(st.nodes), "lean_s": round(t1 - t0, 2), "gate_s": round(t2 - t1, 2),
            "admissible_moves": [str(m) for m in moves],
        })
        print(f"{name:16s} universe={universe:6d} admissible={len(moves):4d} "
              f"reduction={1 - len(moves) / universe:.4f} gt={str(gt):28s} "
              f"retained={retained} kernel_ok={kernel_ok} (lean {t1 - t0:.1f}s gate {t2 - t1:.1f}s)")

    min_red = min(r["reduction"] for r in rows)
    all_ret = all(r["gt_retained"] for r in rows)
    all_ok = all(r["gt_kernel_accepted"] for r in rows)
    print(f"\nmin candidate_reduction_ratio = {min_red:.4f} | ground_truth_retained = "
          f"{sum(r['gt_retained'] for r in rows)}/{len(rows)} | GT kernel-valid = "
          f"{sum(r['gt_kernel_accepted'] for r in rows)}/{len(rows)}")
    out = REPO_ROOT / "logs/admissibility_gate_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")
    return 0 if (min_red >= 0.90 and all_ret and all_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
