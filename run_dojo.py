import os
import sys
import gc
import torch
import nest_asyncio

# Apply nested asyncio for compatibility
nest_asyncio.apply()

print("="*60)
print("=== LeanDojo DeepSeek-Prover (LoRA + bfloat16 on A100) ===")
print("="*60)

# 1. Environment & Path Setup
cargo_bin = os.path.expanduser("~/.cargo/bin")
elan_bin = os.path.expanduser("~/.elan/bin")
for p in [cargo_bin, elan_bin]:
    if p not in os.environ["PATH"]:
        os.environ["PATH"] = f"{p}:{os.environ['PATH']}"

if "/tmp/PyPantograph" not in sys.path:
    sys.path.insert(0, "/tmp/PyPantograph")

# Persistent Tokens
os.environ["GITHUB_ACCESS_TOKEN"] = "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"
os.environ["HF_TOKEN"] = "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"

# PyTorch Memory Allocation settings to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. Clear GPU Memory completely
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Check GPU status
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU] Device: {gpu_name} ({total_mem:.1f} GB VRAM)")
else:
    print("[WARNING] No CUDA GPU detected!")

# 3. Verify Pantograph
import pantograph
print("[OK] Pantograph library loaded successfully.")

# 4. Patch TrainingArguments / SFTConfig for Gradient Checkpointing & Native bfloat16
from transformers import TrainingArguments
from trl import SFTConfig
original_init = TrainingArguments.__init__
def patched_training_args_init(self, *args, **kwargs):
    kwargs.pop('packing', None)
    kwargs.pop('completion_only_loss', None)
    kwargs.pop('packing', None)
    kwargs.pop('completion_only_loss', None)
    kwargs.pop('packing', None)
    kwargs["gradient_checkpointing"] = True
    kwargs["fp16"] = False
    kwargs["bf16"] = True  # Native bfloat16 for NVIDIA A100
    original_init(self, *args, **kwargs)
TrainingArguments.__init__ = patched_training_args_init
SFTConfig.__init__ = patched_training_args_init
print("[OK] Configured for bfloat16 & gradient checkpointing.")

# 5. Initialize Trainer with native LoRA (Memory: ~16 GB / 40 GB)
from peft import LoraConfig, TaskType
from lean_dojo_v2.agent.hf_agent import HFAgent
from lean_dojo_v2.lean_agent.config import ProverConfig
from lean_dojo_v2.trainer.sft_trainer import SFTTrainer

url = "https://github.com/durant42040/lean4-example"
commit = "3e23ab0bfdcfdbd5b11ab53c2cd8b5d16492e9c2"
output_dir = "/content/outputs-deepseek"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

print(f"[INFO] Setting up model deepseek-ai/DeepSeek-Prover-V2-7B with LoRA...")
trainer = SFTTrainer(
    model_name="deepseek-ai/DeepSeek-Prover-V2-7B",
    output_dir=output_dir,
    epochs_per_repo=1,
    batch_size=1,
    lr=2e-5,
    lora_config=lora_config
)

# Enable gradient checkpointing in SFTConfig & model
trainer.config.gradient_checkpointing = True
if hasattr(trainer.model, "gradient_checkpointing_enable"):
    trainer.model.gradient_checkpointing_enable()

prover_conf = ProverConfig(max_expansions=100)
agent = HFAgent(trainer=trainer, prover_config=prover_conf)
print(f"[INFO] Setting up repository: {url} @ {commit}")
agent.setup_github_repository(url=url, commit=commit)

# 6. Train with LoRA (in bfloat16)
print("\n" + "="*50)
print("Starting LoRA training (Peak VRAM: ~16 GB)...")
print("="*50)
agent.train()

# 7. Prepare Model for Inference (Re-enable KV cache for fast tactic generation)
print("\n" + "="*50)
print("Preparing model for proof generation...")
print("="*50)
if hasattr(agent, "model") and hasattr(agent.model, "config"):
    agent.model.config.use_cache = True
    print("[OK] Re-enabled KV-cache (use_cache=True) for fast tactic search.")

# 8. Prove Theorems
print("\n" + "="*50)
print("Running theorem prover...")
print("="*50)
results = agent.prove()
print("\n[DONE] Proving phase complete. Results:", results)
