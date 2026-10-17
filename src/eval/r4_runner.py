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
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from safetensors import safe_open

torch.set_num_threads(2)

try:
    from lean_bench import BENCHMARK_THEOREMS
    from metrics_collector import GeometricMetricsCollector
except ImportError:
    from src.eval.lean_bench import BENCHMARK_THEOREMS
    from src.eval.metrics_collector import GeometricMetricsCollector

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

def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mean_sq = torch.mean(x.to(torch.float64) ** 2, dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_sq + 1e-5).to(x.dtype) * weight

def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(0.7978846 * (x + 0.044715 * x * x * x)))

class R4PyTorchModel:
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
        self.ckpt_dir = Path(checkpoint_path)
        model_dir = self.ckpt_dir / "model" if (self.ckpt_dir / "model").exists() else self.ckpt_dir
        
        # Load config
        with open(model_dir / "config.json") as f:
            self.cfg = json.load(f)
            
        self.width = self.cfg["width"] # 1536
        self.heads = self.cfg["heads"] # 24
        self.pattern = self.cfg["pattern"] # 'rrarrarrarrarrar'
        self.context = self.cfg["context"] # 384
        self.rotation = self.cfg.get("rotation", True)
        
        # Load tokenizer
        tok_candidates = [
            self.ckpt_dir / "tokenizer.json",
            self.ckpt_dir.parent / "tokenizer.json",
            self.ckpt_dir.parent.parent / "data/mathlib4/tokenizer.json",
            Path("/content/syn-prover/data/mathlib4/tokenizer.json"),
            Path("data/mathlib4/tokenizer.json"),
        ]
        tok_path = next((p for p in tok_candidates if p.exists()), None)
        if tok_path is None:
            raise FileNotFoundError(f"Could not find tokenizer.json in candidates: {tok_candidates}")
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        
        # Load weights
        self.tensors = {}
        with safe_open(model_dir / "model.safetensors", framework="pt", device=str(self.device)) as f:
            for k in f.keys():
                self.tensors[k] = f.get_tensor(k).to(torch.float32)
        self.consumed_tensors = set()

    def _get(self, key: str) -> torch.Tensor:
        self.consumed_tensors.add(key)
        return self.tensors[key]

    def _recurrence_layer(self, layer_idx: int, x: torch.Tensor):
        p = f"layers.{layer_idx:02d}"
        seq_len, width = x.shape
        lanes = width // 4
        u = _rms_norm(x, self._get(f"{p}.rec_norm.weight"))
        branches = F.linear(u, self._get(f"{p}.rec.in.weight"))
        gates = F.linear(u, self._get(f"{p}.rec.gate.weight"), self._get(f"{p}.rec.gate.bias"))
        conv_w, conv_b = self._get(f"{p}.rec.conv.weight"), self._get(f"{p}.rec.conv.bias")
        log_a = -F.softplus(-self._get(f"{p}.rec.decay").to(torch.float64)).to(torch.float32)
        a_branch, g_branch = branches[:, :width], branches[:, width:]
        drive = conv_b.unsqueeze(0).repeat(seq_len, 1)
        for s in range(4):
            if s < seq_len:
                drive[s:] = drive[s:] + a_branch[:seq_len - s] * conv_w[s].unsqueeze(0)
        drive_lanes = drive.view(seq_len, lanes, 4)
        lambdas = torch.exp(8.0 * torch.sigmoid(gates[:, :lanes]) * log_a.unsqueeze(0))
        comp = 1.0 - lambdas * lambdas
        keeps = torch.where(comp < 1e-6, torch.full_like(comp, 1e-3), torch.sqrt(torch.clamp(comp, min=1e-6)))
        raw_q = gates[:, lanes:].view(seq_len, lanes, 4)
        unit_q = raw_q / torch.sqrt(torch.sum(raw_q * raw_q, dim=-1, keepdim=True) + 1e-6)
        q_trans = unit_q * lambdas.unsqueeze(-1)
        state = torch.zeros(lanes, 4, device=self.device)
        states = []
        for t in range(seq_len):
            state = QuaternionOps.mul(q_trans[t], state) + keeps[t].unsqueeze(-1) * drive_lanes[t]
            states.append(state)
        full_states = torch.stack(states, dim=0)
        out = F.linear(full_states.view(seq_len, width) * _gelu_tanh(g_branch), self._get(f"{p}.rec.out.weight"))
        return out, full_states

    def _geometric_read_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{layer_idx:02d}"
        seq_len, width = x.shape
        hd = width // self.heads
        u = _rms_norm(x, self._get(f"{p}.read_norm.weight"))
        q = F.linear(u, self._get(f"{p}.read.query.weight")).view(seq_len, self.heads, hd).transpose(0, 1)
        k = F.linear(u, self._get(f"{p}.read.key.weight")).view(seq_len, self.heads, hd).transpose(0, 1)
        v = F.linear(u, self._get(f"{p}.read.value.weight")).view(seq_len, self.heads, hd).transpose(0, 1)
        null_logits = F.linear(u, self._get(f"{p}.read.null.weight"), self._get(f"{p}.read.null.bias")).transpose(0, 1)
        age_table = self._get(f"{p}.read.age")[:, :seq_len]
        beta = torch.exp(self._get(f"{p}.read.log_beta")).to(torch.float64).view(self.heads, 1, 1)
        offset = self._get(f"{p}.read.offset").to(torch.float64).view(self.heads, 1, 1)
        q64, k64 = q.to(torch.float64), k.to(torch.float64)
        dist = torch.sqrt(torch.clamp(
            torch.sum(q64 * q64, dim=-1, keepdim=True) + torch.sum(k64 * k64, dim=-1).unsqueeze(1) - 2.0 * torch.matmul(q64, k64.transpose(-1, -2)),
            min=1e-7
        ))
        t_idx = torch.arange(seq_len, device=self.device).unsqueeze(1)
        j_idx = torch.arange(seq_len, device=self.device).unsqueeze(0)
        scores = (-beta * (dist - offset)).to(torch.float32) + age_table[:, torch.clamp(t_idx - j_idx, min=0)]
        scores = scores.masked_fill(~(j_idx <= t_idx).unsqueeze(0), float("-inf"))
        probs = F.softmax(torch.cat([scores, null_logits.unsqueeze(-1)], dim=-1), dim=-1)[..., :seq_len]
        merged = torch.matmul(probs, v).transpose(0, 1).contiguous().view(seq_len, width)
        return F.linear(merged, self._get(f"{p}.read.out.weight"))

    def _mlp_layer(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{layer_idx:02d}"
        u = _rms_norm(x, self._get(f"{p}.mlp_norm.weight"))
        mixed = F.silu(F.linear(u, self._get(f"{p}.mlp.gate.weight"))) * F.linear(u, self._get(f"{p}.mlp.up.weight"))
        return F.linear(mixed, self._get(f"{p}.mlp.down.weight"))

    def forward_hidden_and_logits(self, input_ids: torch.Tensor):
        embed_weight = self._get("embedding.weight")
        x = embed_weight[input_ids]
        layer_diagnostics = []
        for layer_idx, layer_type in enumerate(self.pattern):
            if layer_type == 'r':
                mixed, full_states = self._recurrence_layer(layer_idx, x)
                self.last_recurrence_state = full_states[-1].reshape(-1)
                x = x + mixed
                layer_diagnostics.append({
                    "layer": layer_idx,
                    "type": "r (quaternion_transport)",
                    "quaternion": GeometricMetricsCollector.compute_quaternion_diagnostics(full_states),
                    "spectral": GeometricMetricsCollector.compute_spectral_properties(x)
                })
            elif layer_type == 'a':
                x = x + self._geometric_read_layer(layer_idx, x)
                layer_diagnostics.append({
                    "layer": layer_idx,
                    "type": "a (addressed_read)",
                    "spectral": GeometricMetricsCollector.compute_spectral_properties(x)
                })
            x = x + self._mlp_layer(layer_idx, x)
        hidden = _rms_norm(x, self._get("final_norm.weight"))
        logits = F.linear(hidden, embed_weight)
        return hidden, logits, layer_diagnostics

    def evaluate_sequence(self, prompt: str, target: str = ""):
        full_text = prompt + (" " + target if target else "")
        enc = self.tokenizer.encode(full_text)
        input_ids = torch.tensor(enc.ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, _, layer_diagnostics = self.forward_hidden_and_logits(input_ids)
        return {
            "prompt": prompt,
            "target": target,
            "token_count": len(input_ids),
            "layer_diagnostics": layer_diagnostics
        }

def find_latest_checkpoint():
    import glob
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
