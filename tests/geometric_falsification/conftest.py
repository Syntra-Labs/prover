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

@pytest.fixture(scope="session")
def authentic_e8_snap():
    """
    Calls authentic Conway-Sloane E8 lattice snapping directly from
    uor_r4_integer::codec::e8_round via the compiled r4-inspect binary.
    Zero Python re-implementations permitted.
    """
    import subprocess
    import json
    
    r4_inspect_paths = [
        ROOT_DIR / "vendor/uor-r4/target/debug/r4-inspect",
        ROOT_DIR / "vendor/uor-r4/target/release/r4-inspect",
        Path("/content/prover/vendor/uor-r4/target/debug/r4-inspect"),
        Path("/content/prover/vendor/uor-r4/target/release/r4-inspect"),
    ]
    r4_bin = next((p for p in r4_inspect_paths if p.exists() and os.access(p, os.X_OK)), None)
    if r4_bin is None:
        pytest.skip("r4-inspect binary not found or not executable. Compile vendor/uor-r4 first.")

    def snap_fn(blocks_8d, scale=0.5):
        """
        blocks_8d: list of 8D lists/arrays or single 8D list/array
        returns: list of snapped 8D tuples (or single tuple)
        """
        is_single = False
        if isinstance(blocks_8d, (list, tuple)) and len(blocks_8d) == 8 and not isinstance(blocks_8d[0], (list, tuple)):
            blocks_8d = [blocks_8d]
            is_single = True

        blocks_json = json.dumps([list(float(x) for x in b) for b in blocks_8d])
        proc = subprocess.run(
            [str(r4_bin), "snap", "--scale", str(scale)],
            input=blocks_json,
            text=True,
            capture_output=True,
            check=True
        )
        res = json.loads(proc.stdout)
        snapped = [tuple(b) for b in res["snapped"]]
        return snapped[0] if is_single else snapped

    return snap_fn

