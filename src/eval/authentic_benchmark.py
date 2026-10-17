# TODO(policy-r4 | tier:B | depends_on:[stab-alpha, stab-jitter, r4-weight-parity, stage2-response-mask, ir-alpha-bijection] | blocks:[macro-decider]):
# Description: Train linear/MLP classification head over R4 1024-dim hidden states mapping to 50 discrete tactics.
# Criterion: assert: top1_accuracy >= 0.30 and top5_accuracy >= 0.65 on Mathlib 4 validation holdout

# TODO(unify-bundle | tier:C | depends_on:[gate-e8, gate-r4-token] | blocks:[search-action]):
# Description: Connect R4 S3/Hopf and E8 lattice discrete quantization to Lean tactic emission.
# Criterion: review: "End-to-end forward pass maps Lean goal to discrete tactic via E8 Voronoi snapping on Cloud A100"

# TODO(macro-decider | tier:C | depends_on:[policy-r4, search-action, search-dead-tactic, param-tactic-synthesis]):
# Description: Implement Tier 1 Macro-Action Decider gating terminating procedures vs forward lemma invention.
# Criterion: review: "Decider routes proof search between Tier 2A direct tactics and Tier 2B forward invention with kernel feedback"

import sys
import json
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT_DIR / "src/eval"
sys.path.insert(0, str(EVAL_DIR))

from tactic_vocab import parse_tactic_step, TACTIC_MAP
from tokenizers import Tokenizer

def run_authentic_benchmark(num_steps: int = 20):
    print("=" * 70)
    print("AUTHENTIC R4 GEOMETRIC MODEL BENCHMARK ON MATHLIB 4")
    print("=" * 70)

    # 1. Load Mathlib 4 Validation Sample
    sample_file = ROOT_DIR / "vendor/benchsets/mathlib4/val_sample.json"
    if not sample_file.exists():
        sample_file = Path("/content/prover/vendor/benchsets/mathlib4/val_sample.json")
        
    with open(sample_file, "r", encoding="utf-8") as f:
        theorems = json.load(f)

    # Extract proof steps
    steps = []
    for thm in theorems:
        t_name = thm.get("full_name", "unknown")
        for st in thm.get("traced_tactics", []):
            sb = st.get("state_before", "").strip()
            tac = st.get("tactic", "").strip()
            if sb and tac:
                tac_id, tac_head, tac_args = parse_tactic_step(tac)
                steps.append({
                    "theorem": t_name,
                    "state_before": sb,
                    "ground_truth_tactic": tac,
                    "tactic_id": tac_id,
                    "tactic_head": tac_head,
                    "tactic_args": tac_args
                })
                if len(steps) >= num_steps:
                    break
        if len(steps) >= num_steps:
            break

    print(f"[DATA] Prepared {len(steps)} authentic proof steps from Mathlib 4.\n")

    # 2. Initialize Authentic R4 Model
    from r4_runner import find_latest_checkpoint
    from tactic_eval import TacticEvaluator
    from authentic_lean_prover import CANONICAL_TACTIC_POOL

    ckpt_path = find_latest_checkpoint()
    print(f"[MODEL] Loading R4 checkpoint: {ckpt_path}")
    evaluator = TacticEvaluator(ckpt_path, device="cpu")

    results = []
    top1_correct = 0
    top5_correct = 0

    print("-" * 70)
    for idx, step in enumerate(steps):
        thm = step["theorem"]
        state = step["state_before"]
        tactic = step["ground_truth_tactic"]
        t_id = step["tactic_id"]
        t_head = step["tactic_head"]

        print(f"\n[Step {idx+1}/{len(steps)}] {thm}")
        print(f"  Target: '{tactic}' -> Discrete Tactic ID: {t_id} ({t_head})")

        # Evaluate candidate pool (strictly fixed pool; zero ground-truth leakage)
        prompt = f"Goal:\n{state}\n\nTactic:\n"
        eval_candidates = list(CANONICAL_TACTIC_POOL)
        scores = []
        for cand in eval_candidates:
            res = evaluator.evaluate_tactic_likelihood(prompt, cand)
            scores.append((cand, res["cross_entropy_loss"]))
        scores.sort(key=lambda x: x[1])

        ranked_heads = [s[0] for s in scores]
        is_top1 = (ranked_heads[0] == t_head)
        is_top5 = (t_head in ranked_heads[:5])
        if is_top1: top1_correct += 1
        if is_top5: top5_correct += 1

        print(f"  Ranked Candidates: Top-1='{ranked_heads[0]}' | Top-5={ranked_heads[:5]}")
        print(f"  Match: Top-1={is_top1} | Top-5={is_top5}")

        results.append({
            "step": idx,
            "theorem": thm,
            "tactic": tactic,
            "tactic_id": t_id,
            "tactic_head": t_head,
            "top_1_prediction": ranked_heads[0],
            "top_5_predictions": ranked_heads[:5],
            "top_1_correct": is_top1,
            "top_5_correct": is_top5
        })

    # Save authentic run log
    log_file = ROOT_DIR / "logs/authentic_mathlib_baseline.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[COMPLETE] Authentic evaluation finished. Results saved to: {log_file}")
    print(f"Top-1 Accuracy: {top1_correct}/{len(steps)} ({top1_correct/len(steps)*100:.1f}%)")
    print(f"Top-5 Accuracy: {top5_correct}/{len(steps)} ({top5_correct/len(steps)*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_authentic_benchmark(num_steps=20)
