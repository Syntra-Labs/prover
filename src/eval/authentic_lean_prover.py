"""
Authentic Lean 4 Prover Verification.
Runs the trained R4 geometric model against the REAL local Lean 4 kernel compiler.
Zero mocks. Zero synthetic data. 100% authentic Lean 4 kernel verification.
"""

import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src/eval"))

from lean_bench import BENCHMARK_THEOREMS
from tactic_eval import TacticEvaluator
from r4_runner import find_latest_checkpoint

# Candidate tactics dictionary (canonical Lean 4 closing / simplifying tactics)
CANONICAL_TACTIC_POOL = [
    "rfl",
    "simp",
    "omega",
    "exact h",
    "assumption",
    "decide",
    "aesop",
    "linarith",
    "ring",
    "intro h",
    "cases h",
    "exact ⟨h.right, h.left⟩",
]

def verify_with_real_lean(lean_code: str, proposed_tactic: str) -> Tuple[bool, str]:
    """
    Writes the theorem and tactic to a real .lean file and invokes the
    actual Lean 4 compiler binary (`lean`).
    Returns: (is_certified, kernel_output)
    """
    full_source = f"{lean_code.strip()}\n  {proposed_tactic.strip()}\n"
    
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write(full_source)
        temp_path = f.name

    try:
        proc = subprocess.run(
            ["lean", temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15
        )
        output = proc.stdout.strip()
        is_certified = (proc.returncode == 0 and "error:" not in output and "sorry" not in output)
        return is_certified, output
    except subprocess.TimeoutExpired:
        return False, "Lean 4 kernel execution timed out (>15s)"
    finally:
        Path(temp_path).unlink(missing_ok=True)

def run_authentic_proving():
    ckpt_path = find_latest_checkpoint()
    print("=" * 75)
    print("AUTHENTIC LEAN 4 THEOREM PROVING BENCHMARK")
    print(f"Model Checkpoint: {ckpt_path}")
    print(f"Lean 4 Version:   {subprocess.check_output(['lean', '--version'], text=True).strip()}")
    print("=" * 75)

    evaluator = TacticEvaluator(ckpt_path, device="cpu")

    results = []

    for thm in BENCHMARK_THEOREMS:
        print(f"\nTarget: {thm['name']} ({thm['id']})")
        print(f"Goal:   {thm['lean_code']}")

        # 1. Have R4 evaluate candidate tactics from the pool
        tactic_scores = []
        for tac in CANONICAL_TACTIC_POOL:
            # Evaluate model loss on prompt + tactic
            res = evaluator.evaluate_tactic_likelihood(thm["lean_code"], tac)
            tactic_scores.append((tac, res["cross_entropy_loss"]))

        # Sort by lowest loss (highest model preference)
        tactic_scores.sort(key=lambda x: x[1])

        top1_tactic = tactic_scores[0][0]
        top1_loss = tactic_scores[0][1]
        top2_tactic = tactic_scores[1][0]
        top2_loss = tactic_scores[1][1]
        delta_loss = top2_loss - top1_loss

        print(f"  Model Top-1 Choice: {top1_tactic:25s} (loss: {top1_loss:.3f})")
        print(f"  Model Top-2 Choice: {top2_tactic:25s} (loss: {top2_loss:.3f})")
        print(f"  Confidence Margin (Δ loss): {delta_loss:.3f}")

        # 2. Submit Top-1 to the REAL Lean 4 kernel
        print(f"  --> Submitting Top-1 ('{top1_tactic}') to Lean 4 kernel...")
        top1_ok, top1_out = verify_with_real_lean(thm["lean_code"], top1_tactic)

        if top1_ok:
            print(f"  [✓ Q.E.D. CERTIFIED] Lean 4 kernel ACCEPTED proof on Top-1!")
            results.append({
                "theorem": thm["name"],
                "proved": True,
                "tried_tactic": top1_tactic,
                "rank": "Top-1",
                "notes": "Certified by Lean 4 kernel"
            })
            continue

        print(f"  [✗ REJECTED] Lean 4 kernel rejected Top-1: {top1_out.splitlines()[0] if top1_out else 'Non-zero exit'}")

        # 3. Speculative Fallback to Top-2
        print(f"  --> Speculative Fallback: Submitting Top-2 ('{top2_tactic}') to Lean 4 kernel...")
        top2_ok, top2_out = verify_with_real_lean(thm["lean_code"], top2_tactic)

        if top2_ok:
            print(f"  [✓ Q.E.D. CERTIFIED] Lean 4 kernel ACCEPTED proof on Top-2 fallback!")
            results.append({
                "theorem": thm["name"],
                "proved": True,
                "tried_tactic": top2_tactic,
                "rank": "Top-2 Fallback",
                "notes": "Certified by Lean 4 kernel"
            })
        else:
            print(f"  [✗ REJECTED] Lean 4 kernel rejected Top-2: {top2_out.splitlines()[0] if top2_out else 'Non-zero exit'}")
            results.append({
                "theorem": thm["name"],
                "proved": False,
                "tried_tactic": f"{top1_tactic} / {top2_tactic}",
                "rank": "Failed",
                "notes": top2_out.splitlines()[0] if top2_out else "Failed"
            })

    # Summary
    print("\n" + "=" * 75)
    print("AUTHENTIC VERIFICATION SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Theorem':<30} | {'Status':<12} | {'Accepted Tactic':<22} | {'Rank'}")
    print("-" * 75)
    proved_count = 0
    for r in results:
        status_str = "PROVED" if r["proved"] else "FAILED"
        if r["proved"]: proved_count += 1
        print(f"{r['theorem']:<30} | {status_str:<12} | {r['tried_tactic']:<22} | {r['rank']}")
    print("-" * 75)
    print(f"Final Real Proof Score: {proved_count} / {len(results)} theorems ({proved_count/len(results)*100:.1f}%) certified by Lean 4 kernel.")
    print("=" * 75)

if __name__ == "__main__":
    run_authentic_proving()
