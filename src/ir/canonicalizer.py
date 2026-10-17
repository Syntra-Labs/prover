"""
Lean 4 Intermediate Representation (IR) Reversible Alpha-Canonicalizer (Option 1A).
Maps bound/local variables and hypotheses in declaration order to canonical positional
identifiers (`v0, v1, ...` and `h0, h1, ...`) with a lossless inverse mapping `invert_sigma`.
"""

import re
from typing import Dict, List, Tuple

LEAN_KEYWORDS = {
    "theorem", "lemma", "def", "example", "by", "where", "let", "have", "fun",
    "forall", "exists", "match", "with", "if", "then", "else", "do", "return",
    "Nat", "Int", "Real", "Bool", "Prop", "Type", "Sort", "List", "Set", "Fin",
    "True", "False", "And", "Or", "Not", "Iff", "Eq", "Ne", "LT", "LE", "GT", "GE",
    "rfl", "simp", "omega", "ring", "linarith", "decide", "aesop", "exact", "apply",
    "intro", "intros", "cases", "rcases", "induction", "rw", "rewrite", "assumption",
    "constructor", "left", "right", "ext", "congr", " trivial", "sorry",
}

# Matches explicit binder groups like `(n : Nat)` or `(p q : Prop)` or `(h : p ∧ q)`
BINDER_GROUP_RE = re.compile(r"\(([a-zA-Z_][a-zA-Z0-9_'\s]*?)\s*:\s*([^()]+)\)")
# Matches LeanDojo context lines like `x y : ℕ` or `h1 : x = y` before `⊢`
CONTEXT_LINE_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_'\s]*?)\s*:\s*(.+)$")


def _is_hypothesis_name_or_type(ident: str, type_str: str, known_props: set) -> bool:
    t = type_str.strip()
    if ident.startswith("h") and (len(ident) == 1 or ident[1:].isdigit() or ident[1] == "_"):
        return True
    if any(op in t for op in ("=", "≠", "≤", "<", "≥", ">", "∧", "∨", "↔", "¬", "∈", "⊆")):
        return True
    if t in known_props:
        return True
    return False


def build_alpha_sigma(goal_str: str) -> Dict[str, str]:
    """
    Extracts local variables and hypotheses in declaration order and returns
    a forward renaming map `sigma: {surface_name -> canonical_name}`.
    """
    sigma: Dict[str, str] = {}
    var_idx = 0
    hyp_idx = 0
    known_props = set()

    def register_decl(names_str: str, type_str: str):
        nonlocal var_idx, hyp_idx
        idents = [tok.strip() for tok in names_str.split() if tok.strip()]
        for ident in idents:
            if ident in LEAN_KEYWORDS or ident in sigma or not ident[0].isalpha():
                continue
            if type_str.strip() == "Prop":
                known_props.add(ident)
            if _is_hypothesis_name_or_type(ident, type_str, known_props):
                sigma[ident] = f"h{hyp_idx}"
                hyp_idx += 1
            else:
                sigma[ident] = f"v{var_idx}"
                var_idx += 1

    # 1. Check LeanDojo multiline context state (before `⊢`)
    if "⊢" in goal_str:
        context_part = goal_str.split("⊢", 1)[0]
        for line in context_part.splitlines():
            line_clean = line.strip()
            if not line_clean or line_clean.startswith("case "):
                continue
            m = CONTEXT_LINE_RE.match(line_clean)
            if m:
                register_decl(m.group(1), m.group(2))

    # 2. Check parenthesized binder groups `(x : T)` in theorem declarations
    for m in BINDER_GROUP_RE.finditer(goal_str):
        register_decl(m.group(1), m.group(2))

    return sigma


def _apply_token_map(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    # Sort keys longest-first to avoid partial prefix collisions, matching whole identifiers only
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def normalize_ir_whitespace(text: str) -> str:
    """Canonicalizes horizontal whitespace and strips trailing spaces per line."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def canonicalize(goal_str: str, tactic_str: str = "") -> Tuple[str, str, Dict[str, str]]:
    """
    Canonicalizes `(goal_str, tactic_str)` into `(canon_goal, canon_tactic, sigma)`.
    Guarantees alpha-equivalent goals map to identical `canon_goal` strings.
    """
    norm_goal = normalize_ir_whitespace(goal_str)
    norm_tactic = normalize_ir_whitespace(tactic_str) if tactic_str else ""
    sigma = build_alpha_sigma(norm_goal)
    canon_goal = _apply_token_map(norm_goal, sigma)
    canon_tactic = _apply_token_map(norm_tactic, sigma) if norm_tactic else ""
    return canon_goal, canon_tactic, sigma


def invert_sigma(canon_tactic: str, sigma: Dict[str, str]) -> str:
    """
    Inverts the canonical symbol table `sigma` on a predicted canonical tactic
    using Lean 4's native `Lean.Syntax` parser and `Syntax.reprint` (falling back
    to token mapping only on unparseable fragments).
    """
    inv_map = {canon: surf for surf, canon in sigma.items()}
    if not inv_map or not canon_tactic:
        return canon_tactic
    try:
        from src.ir.tactic_ast import LeanASTBridge
        res = LeanASTBridge.parse_tactic(canon_tactic, inv_map)
        if res.get("ok") and res.get("root_kind"):
            return res["canonical_tactic"]
    except Exception:
        pass
    return _apply_token_map(canon_tactic, inv_map)
