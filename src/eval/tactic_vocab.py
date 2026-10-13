"""
Canonical 50 Discrete Tactic Vocabulary and Mathlib 4 Step Parser.
Maps Lean 4 tactics into discrete action IDs and extracts slot parameters.
"""

from typing import Dict, Any, Tuple

# Canonical 50 Discrete Tactic Head IDs
TACTIC_MAP = {
    # Decision Procedures & Solvers (Parameterless / Self-contained)
    "rfl": 0,
    "simp": 1,
    "omega": 2,
    "ring": 3,
    "linarith": 4,
    "norm_num": 5,
    "assumption": 6,
    "trivial": 7,
    "aesop": 8,
    "decide": 9,
    "contradiction": 10,
    "positivity": 11,
    "congr": 12,
    "ext": 13,
    "dsimp": 14,
    
    # Structural & Instantiation Tactics (Require Identifier / Hyp Pointer)
    "exact": 15,
    "apply": 16,
    "intro": 17,
    "intros": 18,
    "cases": 19,
    "rcases": 20,
    "induction": 21,
    "obtain": 22,
    "specialize": 23,
    "revert": 24,
    "clear": 25,
    
    # Rewriting & Simplification
    "rw": 26,
    "simp_rw": 27,
    "nth_rw": 28,
    "subst": 29,
    
    # Forward Reasoning & Intermediate Claims
    "have": 30,
    "let": 31,
    "refine": 32,
    "simpa": 33,
    "suffices": 34,
    
    # Calculus & Special Domains
    "continuity": 35,
    "measurability": 36,
    "fun_prop": 37,
    
    # Fallback / Other
    "unknown": 49
}

ID_TO_TACTIC = {v: k for k, v in TACTIC_MAP.items()}

def parse_tactic_step(tactic_str: str) -> Tuple[int, str, str]:
    """
    Parses a raw Lean tactic string into:
    (tactic_id, base_tactic_name, remaining_arguments)
    """
    cleaned = tactic_str.strip()
    if not cleaned:
        return TACTIC_MAP["unknown"], "unknown", ""
        
    parts = cleaned.split(maxsplit=1)
    head = parts[0].strip("',\"")
    args = parts[1].strip() if len(parts) > 1 else ""
    
    # Strip prime modifiers like induction' -> induction
    normalized_head = head.rstrip("'")
    
    tactic_id = TACTIC_MAP.get(normalized_head, TACTIC_MAP["unknown"])
    return tactic_id, normalized_head, args

if __name__ == "__main__":
    test_cases = [
        "rfl",
        "simp only [degrees]",
        "exact hn (mem_adjoin_of_smul ...)",
        "rw [_root_.pow_succ', mul_smul] at hz",
        "induction' n with n hn",
        "omega"
    ]
    for tc in test_cases:
        t_id, head, args = parse_tactic_step(tc)
        print(f"Input: {tc:40s} -> ID {t_id:2d} ({head:10s}) | Args: '{args}'")
