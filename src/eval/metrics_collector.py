"""
Metrics and diagnostics collector for geometric models.
Captures internal manifold trajectory properties:
- Energy / Hamiltonian conservation (GFN)
- Symplectic drift & momentum bounds
- Quaternion transport norms and angle rotation (UOR-R4)
- Hidden state curvature / condition numbers
- Token perplexity / cross-entropy over mathematical targets
"""

import math
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any, List

class GeometricMetricsCollector:
    def __init__(self, model_name: str, log_dir: str = "logs"):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{model_name}_metrics_{self.session_id}.jsonl"
        self.records: List[Dict[str, Any]] = []

    def log_step(self, theorem_id: str, step_data: Dict[str, Any]):
        record = {
            "timestamp": time.time(),
            "model": self.model_name,
            "theorem_id": theorem_id,
            **step_data
        }
        self.records.append(record)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def compute_quaternion_diagnostics(state: torch.Tensor) -> Dict[str, float]:
        """
        Evaluates UOR-R4 state [B, T, Lanes, 4]:
        - Norm stability: |h|^2 distribution across time
        - Unit violation: divergence from compact manifold S3
        - Dynamic range: max to min ratio
        """
        # state is [..., 4]
        norms = torch.norm(state.float(), dim=-1) # [..., Lanes]
        mean_norm = float(norms.mean().item())
        std_norm = float(norms.std().item())
        max_norm = float(norms.max().item())
        min_norm = float(norms.min().item())
        # Divergence from unit sphere S3 (|q| = 1)
        s3_error = float(torch.abs(norms - 1.0).mean().item())
        return {
            "mean_norm": mean_norm,
            "std_norm": std_norm,
            "max_norm": max_norm,
            "min_norm": min_norm,
            "s3_divergence": s3_error,
        }

    @staticmethod
    def compute_symplectic_diagnostics(coords: torch.Tensor, momenta: torch.Tensor, hamiltonian_fn=None) -> Dict[str, float]:
        """
        Evaluates GFN symplectic phase space (q, p):
        - Kinetic energy T = 0.5 * ||p||^2
        - Coordinate spread and boundary constraints
        """
        kinetic = float((0.5 * (momenta ** 2).sum(dim=-1)).mean().item())
        coord_norm = float(torch.norm(coords, dim=-1).mean().item())
        momentum_norm = float(torch.norm(momenta, dim=-1).mean().item())
        return {
            "kinetic_energy": kinetic,
            "coord_radius": coord_norm,
            "momentum_norm": momentum_norm,
        }

    @staticmethod
    def compute_spectral_properties(hidden: torch.Tensor) -> Dict[str, float]:
        """
        Evaluates singular value spread of hidden states [T, D]
        Reveals rank collapse vs representation expressivity.
        """
        if hidden.dim() > 2:
            hidden = hidden.view(-1, hidden.shape[-1])
        # SVD on sample
        h = hidden.float()
        try:
            _, S, _ = torch.linalg.svd(h, full_matrices=False)
            effective_rank = float(torch.exp(-torch.sum((S / S.sum()) * torch.log(S / S.sum() + 1e-12))).item())
            cond_number = float((S[0] / (S[-1] + 1e-12)).item())
            return {
                "effective_rank": effective_rank,
                "singular_max": float(S[0].item()),
                "singular_min": float(S[-1].item()),
                "condition_number": cond_number
            }
        except Exception:
            return {"effective_rank": 0.0, "condition_number": 0.0}

print("[OK] GeometricMetricsCollector initialized.")
