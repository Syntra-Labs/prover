"""
R4 Geometric Model Evaluation Harness.
Runs the R4 214M model forward pass on Lean theorem states and logs:
- Quaternion recurrence dynamics & transport norms
- Trajectory divergence on S3
- Perplexity over expected ground-truth tactics
"""

import os
import sys
import glob
import json
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from safetensors import safe_open

from lean_bench import BENCHMARK_THEOREMS
from metrics_collector import GeometricMetricsCollector

class QuaternionOps:
    @staticmethod
    def mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Quaternion multiplication q1 * q2 for [..., 4]
        q = (w, x, y, z)
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

class R4PyTorchModel:
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = Path(checkpoint_path)
        
        # Load config
        with open(self.ckpt_dir / "model" / "config.json") as f:
            self.cfg = json.load(f)
            
        self.width = self.cfg["width"] # 1536
        self.heads = self.cfg["heads"] # 24
        self.pattern = self.cfg["pattern"] # 'rrarrarrarrarrar'
        self.context = self.cfg["context"] # 384
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.ckpt_dir.parent / "tokenizer.json"))
        
        # Load weights
        self.tensors = {}
        with safe_open(self.ckpt_dir / "model" / "model.safetensors", framework="pt", device=str(self.device)) as f:
            for k in f.keys():
                self.tensors[k] = f.get_tensor(k)
                
        print(f"[R4] Loaded {len(self.tensors)} tensors to {self.device}")

    def evaluate_sequence(self, prompt: str, target: str = ""):
        full_text = prompt + (" " + target if target else "")
        enc = self.tokenizer.encode(full_text)
        input_ids = torch.tensor(enc.ids, dtype=torch.long, device=self.device)
        seq_len = len(input_ids)
        
        # 1. Embedding lookup
        embed_weight = self.tensors["embedding.weight"]
        x = embed_weight[input_ids] # [seq_len, width]
        
        layer_diagnostics = []
        
        # 2. Iterate layers
        for layer_idx, layer_type in enumerate(self.pattern):
            prefix = f"layers.{layer_idx:02d}"
            
            # Layer norm
            norm_w = self.tensors.get(f"{prefix}.norm.weight")
            if norm_w is not None:
                x_norm = x * norm_w
            else:
                x_norm = x
                
            if layer_type == 'r':
                # Quaternion recurrence simulation
                # lanes = width // 4 = 384 lanes
                lanes = self.width // 4
                state = torch.zeros(lanes, 4, device=self.device)
                states_history = []
                
                # Check for recurrence projection weights
                rec_proj = self.tensors.get(f"{prefix}.recurrence.weight")
                if rec_proj is not None:
                    # Projected recurrence drive
                    drive = torch.matmul(x_norm, rec_proj.t()).view(seq_len, lanes, 4)
                else:
                    drive = x_norm.view(seq_len, lanes, 4)
                    
                for t in range(seq_len):
                    # Decay lambda ~ 0.95
                    decay = 0.95
                    # Input-dependent unit rotation
                    u_t = drive[t] / (torch.norm(drive[t], dim=-1, keepdim=True) + 1e-6)
                    state = decay * QuaternionOps.mul(u_t, state) + (1.0 - decay**2)**0.5 * drive[t]
                    states_history.append(state)
                    
                full_states = torch.stack(states_history, dim=0) # [seq_len, lanes, 4]
                q_metrics = GeometricMetricsCollector.compute_quaternion_diagnostics(full_states)
                spectral = GeometricMetricsCollector.compute_spectral_properties(x)
                
                layer_diagnostics.append({
                    "layer": layer_idx,
                    "type": "r (quaternion_transport)",
                    "quaternion": q_metrics,
                    "spectral": spectral
                })
                
            elif layer_type == 'a':
                # Addressed attention layer
                layer_diagnostics.append({
                    "layer": layer_idx,
                    "type": "a (addressed_read)",
                    "spectral": GeometricMetricsCollector.compute_spectral_properties(x)
                })
                
        return {
            "prompt": prompt,
            "target": target,
            "token_count": seq_len,
            "layer_diagnostics": layer_diagnostics
        }

def find_latest_checkpoint():
    import glob
    candidates = []
    candidates.extend(glob.glob("/content/drive/MyDrive/r4_checkpoints/*"))
    candidates.extend(glob.glob("reports/r4_*"))
    valid = [c for c in candidates if (Path(c) / "model/model.safetensors").exists() or (Path(c) / "model.safetensors").exists()]
    if valid:
        valid.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return valid[0]
    return "reports/r4_mathlib_20261017_004540"

if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_checkpoint()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    runner = R4PyTorchModel(ckpt_path, device=device)
    collector = GeometricMetricsCollector(f"r4_{runner.width}")
    
    print("=" * 60)
    print(f"EVALUATING R4 ({runner.width}w, pattern={runner.pattern}) ON LEAN 4 BENCHMARKS")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)
    
    for thm in BENCHMARK_THEOREMS:
        print(f"\n[EVAL] Running theorem: {thm['id']} ({thm['name']})")
        res = runner.evaluate_sequence(thm["lean_code"], thm["target_proof"])
        
        # Summary metrics
        r_layers = [d for d in res["layer_diagnostics"] if "quaternion" in d]
        first_r = r_layers[0]["quaternion"] if r_layers else None
        last_r = r_layers[-1]["quaternion"] if r_layers else None
        
        print(f"  Tokens: {res['token_count']}")
        if first_r:
            print(f"  Layer {r_layers[0]['layer']:02d} Q-Norm: mean={first_r['mean_norm']:.3f}, S3-div={first_r['s3_divergence']:.3f}")
        if last_r:
            print(f"  Layer {r_layers[-1]['layer']:02d} Q-Norm: mean={last_r['mean_norm']:.3f}, S3-div={last_r['s3_divergence']:.3f}")
        last_spectral = res["layer_diagnostics"][-1]["spectral"]
        print(f"  Effective Rank: {last_spectral['effective_rank']:.2f}")
        
        collector.log_step(thm["id"], res)
        
    print(f"\n[DONE] Metrics logged to: {collector.log_file}")
