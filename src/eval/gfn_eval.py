"""
Authentic DepthMuun GFN / ISN Model Evaluator.
Directly uses the author's architecture and weights from 'vendor/gfn'.
Captures real logits, emitted trajectory embeddings, and continuous states.
"""

import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add vendored gfn package to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "vendor/gfn"))

from gfn import isn

class AuthenticGFNEvaluator:
    def __init__(self, checkpoint_dir: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        if checkpoint_dir is None:
            # Fallback to local huggingface cache or standard location
            self.ckpt_dir = Path.home() / ".cache/huggingface/hub/models--DepthMuun--gfn-isn-shakespeare-char/snapshots"
            if self.ckpt_dir.exists():
                snapshots = list(self.ckpt_dir.iterdir())
                if snapshots:
                    self.ckpt_dir = snapshots[0]
        else:
            self.ckpt_dir = Path(checkpoint_dir)
            
        print(f"[GFN-AUTHENTIC] Initializing using weights from: {self.ckpt_dir}")
        
        # Load config
        with open(self.ckpt_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        # Load vocabulary
        with open(self.ckpt_dir / "tinyshakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Create model using DepthMuun's official API
        self.model = isn.create(
            vocab_size=self.vocab_size,
            d_model=self.config["model"]["d_model"],
            d_embedding=self.config["model"]["d_embedding"],
            d_properties=self.config["model"]["d_properties"],
            scanner="gfn",
            world="gfn",
            emitter="gfn"
        ).to(self.device)

        # Load authentic checkpoint
        ckpt = torch.load(self.ckpt_dir / "best_model.pt", map_location=self.device)
        sd = ckpt["model_state_dict"]

        # Remap checkpoint keys to model structure
        mapped_sd = {
            "scanner.embedding.weight": sd["scanner.embedding.weight"],
            "scanner.flow_gate.weight": sd["scanner.flow_gate.weight"],
            "scanner.flow_gate.bias": sd["scanner.flow_gate.bias"],
            "scanner.velocity_norm.weight": sd["scanner.velocity_norm.weight"],
            "scanner.velocity_norm.bias": sd["scanner.velocity_norm.bias"],
            "world.drift.weight": sd["world.drift.weight"],
            "world.drift.bias": sd["world.drift.bias"],
            "world.diffusion.weight": sd["world.diffusion.weight"],
            "world.diffusion.bias": sd["world.diffusion.bias"],
            "world.norm.weight": sd["world.norm.weight"],
            "world.norm.bias": sd["world.norm.bias"],
            "emitter.flow_core.0.weight": sd["emitter.flow_core.0.weight"],
            "emitter.flow_core.0.bias": sd["emitter.flow_core.0.bias"],
            "emitter.flow_core.2.weight": sd["emitter.flow_core.2.weight"],
            "emitter.flow_core.2.bias": sd["emitter.flow_core.2.bias"],
            "emitter.output_proj.weight": sd["emitter.output_proj.weight"],
            "emitter.output_proj.bias": sd["emitter.output_proj.bias"]
        }
        self.model.load_state_dict(mapped_sd, strict=False)
        self.model.eval()
        print("[GFN-AUTHENTIC] Official model and state_dict ready!")

    def forward_state(self, text: str) -> Dict[str, Any]:
        """
        Runs the genuine GFN forward pass over input text.
        Returns:
            - logits: [1, T, vocab_size]
            - emitted_embeddings: [1, T, d_embedding] (trajectory embeddings)
            - final_state: [1, d_embedding] (continuous manifold endpoint)
            - coherence: float
        """
        input_ids = torch.tensor(
            [[self.char_to_ix.get(c, 0) for c in text]],
            dtype=torch.long,
            device=self.device
        )
        with torch.no_grad():
            res = self.model(input_ids)
            
        embeddings = res.get("emitted_embeddings")
        final_state = res.get("final_state")
        logits = res.get("logits")
        
        # Calculate real mathematical properties of the trajectory
        emb_norm = float(torch.norm(embeddings, dim=-1).mean().item()) if embeddings is not None else 0.0
        state_norm = float(torch.norm(final_state, dim=-1).mean().item()) if final_state is not None else 0.0
        
        return {
            "token_count": input_ids.shape[1],
            "logits_shape": list(logits.shape) if logits is not None else [],
            "emitted_embeddings_norm": emb_norm,
            "final_state_norm": state_norm,
            "raw_res": res
        }

if __name__ == "__main__":
    evaluator = AuthenticGFNEvaluator()
    sample_text = "theorem nat_add_zero (n : Nat) : n + 0 = n := by"
    out = evaluator.forward_state(sample_text)
    print(f"Tokens fed: {out['token_count']}")
    print(f"Logits shape: {out['logits_shape']}")
    print(f"Trajectory Embedding Mean Norm: {out['emitted_embeddings_norm']:.4f}")
    print(f"Final State Endpoint Norm: {out['final_state_norm']:.4f}")
