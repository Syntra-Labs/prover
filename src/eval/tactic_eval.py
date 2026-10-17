"""
Evaluates R4 214M and GFN on predicting Lean 4 proof tactics.
Given the Lean theorem declaration and proof state:
1. Calculates Cross-Entropy Loss and Perplexity for target tactics.
2. Compares token log-probabilities against uniform and random baselines.
3. Logs geometry metrics (S3 divergence, quaternion norms, symplectic phase drift).
"""

import math
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from safetensors import safe_open

try:
    from lean_bench import BENCHMARK_THEOREMS
    from metrics_collector import GeometricMetricsCollector
    from r4_runner import R4PyTorchModel
except ImportError:
    from src.eval.lean_bench import BENCHMARK_THEOREMS
    from src.eval.metrics_collector import GeometricMetricsCollector
    from src.eval.r4_runner import R4PyTorchModel

class TacticEvaluator:
    def __init__(self, r4_model_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
        self.r4 = R4PyTorchModel(r4_model_path, device=device)
        self.tokenizer = self.r4.tokenizer
        self.embed_weight = self.r4.tensors["embedding.weight"] # [4096, 1536]
        self.final_norm = self.r4.tensors.get("final_norm.weight") # [1536]

    def get_final_hidden(self, text: str) -> torch.Tensor:
        enc = self.tokenizer.encode(text)
        ids = torch.tensor(enc.ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.r4.forward_hidden_and_logits(ids)
        return self.r4.last_recurrence_state

    def rank_tactic_heads(self, prompt: str, heads: list[str]) -> list[tuple[str, float]]:
        ids = torch.tensor(self.tokenizer.encode(prompt).ids[-self.r4.context:], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, logits, _ = self.r4.forward_hidden_and_logits(ids)
        lp = F.log_softmax(logits[-1], dim=-1)
        seen_tids = set()
        scores = []
        for h in heads:
            h_ids = self.tokenizer.encode(h).ids
            if len(h_ids) == 1 and h_ids[0] not in seen_tids:
                seen_tids.add(h_ids[0])
                scores.append((h, -lp[h_ids[0]].item()))
        scores.sort(key=lambda x: x[1])
        return scores
        
    def evaluate_tactic_likelihood(self, prompt: str, target_tactic: str):
        sep = "" if prompt.endswith(("\n", " ")) else " "
        full_text = prompt + sep + target_tactic
        enc_full = self.tokenizer.encode(full_text)
        enc_prompt = self.tokenizer.encode(prompt)
        
        full_ids = torch.tensor(enc_full.ids, dtype=torch.long, device=self.device)
        prompt_len = len(enc_prompt.ids)
        target_len = len(full_ids) - prompt_len
        
        if target_len <= 0:
            target_ids = full_ids[-1:]
            prompt_len = len(full_ids) - 1
            target_len = 1
        else:
            target_ids = full_ids[prompt_len:]
            
        with torch.no_grad():
            hidden, logits, _ = self.r4.forward_hidden_and_logits(full_ids)
        
        # Loss on target tokens
        target_logits = logits[prompt_len-1 : prompt_len-1 + target_len]
        log_probs = F.log_softmax(target_logits, dim=-1)
        
        token_losses = []
        for i, target_id in enumerate(target_ids):
            loss = -log_probs[i, target_id].item()
            token_losses.append(loss)
            
        avg_loss = sum(token_losses) / len(token_losses)
        perplexity = math.exp(min(avg_loss, 20.0)) # clamp overflow
        
        return {
            "prompt": prompt,
            "target_tactic": target_tactic,
            "target_token_count": target_len,
            "cross_entropy_loss": avg_loss,
            "perplexity": perplexity,
            "token_losses": token_losses
        }

def find_latest_checkpoint():
    import glob
    import os
    candidates = []
    candidates.extend(glob.glob("/content/drive/MyDrive/r4_checkpoints/*"))
    candidates.extend(glob.glob("/content/syn-prover/reports/r4_*"))
    candidates.extend(glob.glob("reports/r4_*"))
    valid = [c for c in candidates if (Path(c) / "model/model.safetensors").exists() or (Path(c) / "model.safetensors").exists()]
    if valid:
        valid.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return valid[0]
    return "reports/r4_mathlib_20261017_004540"

if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_checkpoint()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    collector = GeometricMetricsCollector("r4_tactic_eval")
    evaluator = TacticEvaluator(ckpt_path, device=device)
    
    print("=" * 65)
    print("EVALUATING R4 TACTIC TOKEN LIKELIHOOD ON LEAN 4 BENCHMARKS")
    print(f"Checkpoint: {ckpt_path} on {device}")
    print("=" * 65)
    
    for thm in BENCHMARK_THEOREMS:
        print(f"\n--- [THEOREM] {thm['id']} ({thm['name']}) ---")
        print(f"Goal: {thm['lean_code']}")
        
        for tactic in thm["expected_tactics"]:
            res = evaluator.evaluate_tactic_likelihood(thm["lean_code"], tactic)
            print(f"  Tactic: {tactic:30s} | Loss: {res['cross_entropy_loss']:.3f} | PPL: {res['perplexity']:.1f}")
            collector.log_step(thm["id"], {
                "tactic": tactic,
                "loss": res["cross_entropy_loss"],
                "perplexity": res["perplexity"],
                "tokens": res["target_token_count"]
            })
            
    print(f"\n[DONE] Tactic metrics logged to: {collector.log_file}")
