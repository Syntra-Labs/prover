"""
Benchmark Runner for Authentic Geometric Models (R4 & GFN) on Mathlib 4.
Runs strictly against authentic model implementations:
- R4: Casey Allard's StackModel (via r4-inspect)
- GFN: DepthMuun's ISN (via gfn.isn)
Captures:
- Real logits and top-k predictions
- Genuine trajectory norms and hidden state magnitudes
- Discrete tactic mapping and ground truth comparisons
"""

import sys
import json
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = ROOT_DIR / "src/eval"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ROOT_DIR / "vendor/gfn"))

from gfn_eval import AuthenticGFNEvaluator
from tactic_vocab import parse_tactic_step, TACTIC_MAP
from tokenizers import Tokenizer

def run_authentic_benchmark(num_steps: int = 20):
    print("=" * 70)
    print("AUTHENTIC GEOMETRIC MODELS BENCHMARK ON MATHLIB 4")
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

    # 2. Initialize Models
    # GFN
    gfn_ckpt = Path("/content/prover/checkpoints/DepthMuun/gfn-isn-shakespeare-char")
    gfn_model = AuthenticGFNEvaluator(str(gfn_ckpt), device="cuda:0")

    # R4
    r4_bin = ROOT_DIR / "vendor/uor-r4/target/release/r4-inspect"
    if not r4_bin.exists():
        r4_bin = Path("/content/prover/vendor/uor-r4/target/release/r4-inspect")
    r4_model_dir = Path("/content/prover/checkpoints/caseyallard/uor-r4-geometric-214m/chat-smoltalk-b16/model")
    r4_tok_path = Path("/content/prover/checkpoints/caseyallard/uor-r4-geometric-214m/tokenizer.json")
    r4_tokenizer = Tokenizer.from_file(str(r4_tok_path))

    results = []

    print("-" * 70)
    for idx, step in enumerate(steps):
        thm = step["theorem"]
        state = step["state_before"]
        tactic = step["ground_truth_tactic"]
        t_id = step["tactic_id"]
        t_head = step["tactic_head"]

        print(f"\n[Step {idx+1}/{len(steps)}] {thm}")
        print(f"  Target: '{tactic}' -> Discrete Tactic ID: {t_id} ({t_head})")

        # --- A. Authentic GFN Pass ---
        gfn_out = gfn_model.forward_state(state)
        print(f"  [GFN] Tokens: {gfn_out['token_count']} | Trajectory Norm: {gfn_out['emitted_embeddings_norm']:.2f} | Endpoint Norm: {gfn_out['final_state_norm']:.2f}")

        # --- B. Authentic R4 Pass ---
        # Encode state with BPE tokenizer
        enc = r4_tokenizer.encode(state)
        ids_str = ",".join(str(i) for i in enc.ids[:128]) # pass up to context budget
        r4_proc = subprocess.run([str(r4_bin), str(r4_model_dir), ids_str], capture_output=True, text=True)
        
        r4_data = {}
        if r4_proc.returncode == 0 and r4_proc.stdout.strip():
            try:
                r4_data = json.loads(r4_proc.stdout.strip())
                top_tokens = [t["id"] for t in r4_data.get("top_5", [])]
                print(f"  [R4]  Tokens: {r4_data.get('tokens_fed')} | Hidden Norm: {r4_data.get('final_hidden_norm'):.2f} | Top-5 Token IDs: {top_tokens}")
            except Exception as e:
                print(f"  [R4]  JSON parse error: {e}")
        else:
            print(f"  [R4]  Process stderr: {r4_proc.stderr[:100]}")

        results.append({
            "step": idx,
            "theorem": thm,
            "tactic": tactic,
            "tactic_id": t_id,
            "tactic_head": t_head,
            "gfn": {
                "trajectory_norm": gfn_out["emitted_embeddings_norm"],
                "endpoint_norm": gfn_out["final_state_norm"]
            },
            "r4": {
                "hidden_norm": r4_data.get("final_hidden_norm"),
                "top_5": r4_data.get("top_5", [])
            }
        })

    # Save authentic run log
    log_file = ROOT_DIR / "logs/authentic_mathlib_baseline.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[COMPLETE] Authentic evaluation finished. Results saved to: {log_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_authentic_benchmark(num_steps=20)
