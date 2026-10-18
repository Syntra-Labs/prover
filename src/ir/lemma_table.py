"""
Bijective Mathlib 4 Lemma Lookup Table (`LemmaTable`).
Maps external theorem/lemma names to compact numerical IDs (`L0, L1, ...`) and
supports per-goal positional premise canonicalization (`p0, p1, ...`) with
lossless inverse decoding back to Lean 4 identifier syntax.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TABLE_PATH = REPO_ROOT / "data/mathlib4/lemma_table.json"


class LemmaTable:
    """
    Bidirectional lookup table mapping Lean 4 / Mathlib lemma names <-> integer IDs.
    """

    def __init__(self) -> None:
        self.name_to_id: Dict[str, int] = {}
        self.id_to_name: Dict[int, str] = {}

    def __len__(self) -> int:
        return len(self.name_to_id)

    def register(self, name: str) -> int:
        clean = name.strip()
        if clean in self.name_to_id:
            return self.name_to_id[clean]
        idx = len(self.name_to_id)
        self.name_to_id[clean] = idx
        self.id_to_name[idx] = clean
        return idx

    def register_many(self, names: Iterable[str]) -> None:
        for name in names:
            if name and name.strip():
                self.register(name)

    def get_id(self, name: str) -> Optional[int]:
        return self.name_to_id.get(name.strip())

    def get_name(self, idx: int) -> Optional[str]:
        return self.id_to_name.get(idx)

    def encode_global_token(self, name: str) -> str:
        idx = self.register(name)
        return f"L{idx}"

    def decode_global_token(self, token: str) -> str:
        if token.startswith("L") and token[1:].isdigit():
            found = self.get_name(int(token[1:]))
            if found is not None:
                return found
        return token

    def save(self, path: Path | str = DEFAULT_TABLE_PATH) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.name_to_id, f, ensure_ascii=False, indent=2)
        return out_path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_TABLE_PATH) -> "LemmaTable":
        in_path = Path(path)
        table = cls()
        if in_path.exists():
            with open(in_path, "r", encoding="utf-8") as f:
                raw: Dict[str, int] = json.load(f)
            for name, idx in raw.items():
                table.name_to_id[name] = int(idx)
                table.id_to_name[int(idx)] = name
        return table


GLOBAL_LEMMA_TABLE = LemmaTable.load() if DEFAULT_TABLE_PATH.exists() else LemmaTable()
