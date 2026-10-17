"""
Benchmark Ledger: Persistent CSV and JSON Logging for Model Progress.
Tracks authentic metrics across training stages and checkpoints.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV_PATH = REPO_ROOT / "docs/benchmark_history.csv"

CSV_HEADERS = [
    "timestamp",
    "stage",
    "run_id",
    "step",
    "benchmark_type",
    "sample_size",
    "top1_accuracy",
    "top5_recall",
    "certified_qed_rate",
    "loss_or_nll",
    "notes"
]

def ensure_ledger_exists(csv_path: Path = DEFAULT_CSV_PATH) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)

def log_benchmark_entry(
    stage: str,
    run_id: str,
    step: int,
    benchmark_type: str,
    sample_size: int,
    top1_accuracy: Optional[float] = None,
    top5_recall: Optional[float] = None,
    certified_qed_rate: Optional[float] = None,
    loss_or_nll: Optional[float] = None,
    notes: str = "",
    timestamp: Optional[str] = None,
    csv_path: Path = DEFAULT_CSV_PATH
) -> Dict[str, Any]:
    """
    Appends a structured benchmark observation to the persistent CSV ledger.
    """
    ensure_ledger_exists(csv_path)

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp": timestamp,
        "stage": stage,
        "run_id": run_id,
        "step": step,
        "benchmark_type": benchmark_type,
        "sample_size": sample_size,
        "top1_accuracy": f"{top1_accuracy:.4f}" if top1_accuracy is not None else "",
        "top5_recall": f"{top5_recall:.4f}" if top5_recall is not None else "",
        "certified_qed_rate": f"{certified_qed_rate:.4f}" if certified_qed_rate is not None else "",
        "loss_or_nll": f"{loss_or_nll:.4f}" if loss_or_nll is not None else "",
        "notes": notes.strip()
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)

    return row
