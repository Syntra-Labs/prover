"""
tests/geometric_falsification/conftest.py

Shared fixtures and data loaders for geometric falsification tests.
Runs on Cloud A100 GPU (prover-a100) or local environment.
"""

import os
import json
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

@pytest.fixture(scope="session")
def mathlib_sample():
    """Loads authentic Mathlib 4 validation proof steps."""
    sample_file = ROOT_DIR / "vendor/benchsets/mathlib4/val_sample.json"
    if not sample_file.exists():
        sample_file = Path("/content/prover/vendor/benchsets/mathlib4/val_sample.json")
    if not sample_file.exists():
        pytest.skip(f"Mathlib validation sample not found at {sample_file}")
    with open(sample_file, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def gfn_evaluator():
    """Initializes authentic DepthMuun GFN evaluator on GPU if available."""
    import sys
    sys.path.insert(0, str(ROOT_DIR / "src/eval"))
    sys.path.insert(0, str(ROOT_DIR / "vendor/gfn"))
    from gfn_eval import AuthenticGFNEvaluator
    
    ckpt_path = Path("/content/prover/checkpoints/DepthMuun/gfn-isn-shakespeare-char")
    if not ckpt_path.exists():
        ckpt_path = ROOT_DIR / "checkpoints/DepthMuun/gfn-isn-shakespeare-char"
        
    if not ckpt_path.exists():
        pytest.skip(f"GFN checkpoint not found at {ckpt_path}")
        
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return AuthenticGFNEvaluator(str(ckpt_path), device=device)
