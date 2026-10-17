import csv
import tempfile
from pathlib import Path
from src.eval.benchmark_ledger import ensure_ledger_exists, log_benchmark_entry, CSV_HEADERS

def test_ledger_creation_and_headers():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_ledger.csv"
        ensure_ledger_exists(csv_path)
        assert csv_path.exists()
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_HEADERS

def test_log_benchmark_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_ledger.csv"
        
        entry = log_benchmark_entry(
            stage="Stage 1 Pretrain",
            run_id="run_test_001",
            step=1000,
            benchmark_type="authentic_lean4_compiler",
            sample_size=5,
            top1_accuracy=0.8,
            top5_recall=0.8,
            certified_qed_rate=0.8,
            notes="Test entry",
            timestamp="2026-10-17T06:00:00+00:00",
            csv_path=csv_path
        )
        
        assert entry["run_id"] == "run_test_001"
        assert entry["certified_qed_rate"] == "0.8000"
        
        # Verify readback
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            assert reader[0]["top1_accuracy"] == "0.8000"
            assert reader[0]["notes"] == "Test entry"
