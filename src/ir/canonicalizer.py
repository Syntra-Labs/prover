"""
Lean 4 Intermediate Representation (IR) Reversible Alpha & Premise Canonicalizer.
1. Normalizes all inputs (both `theorem ... := by` declarations and LeanDojo states)
   into the unified turnstile goal format (`v0 : T\\nh0 : P\\n⊢ Target`).
2. Maps bound/local variables and hypotheses in declaration order to `v0, v1, ...` and `h0, h1, ...`.
3. Maps external Mathlib lemma names to canonical premise IDs (`p0, p1, ...`) backed by
   the bidirectional `GLOBAL_LEMMA_TABLE` (`L0, L1, ...`), with lossless inverse `invert_sigma`.
"""

import re
from typing import Dict, Tuple
from src.ir.lemma_table import GLOBAL_LEMMA_TABLE, LemmaTable

LEAN_KEYWORDS = {
    "theorem", "lemma", "def", "example", "by", "where", "let", "have", "have'", "fun",
    "forall", "exists", "match", "with", "if", "then", "else", "do", "return",
    "show", "from", "suffices", "this", "using", "only", "at", "in",
    "Nat", "Int", "Real", "Bool", "Prop", "Type", "Sort", "List", "Set", "Fin",
    "True", "False", "And", "Or", "Not", "Iff", "Eq", "Ne", "LT", "LE", "GT", "GE",
    "rfl", "simp", "simpa", "simp_rw", "dsimp", "omega", "ring", "linarith", "nlinarith", "decide",
    "aesop", "exact", "exact'", "apply", "apply'", "refine", "refine'", "intro", "intros", "rintro", "revert", "clear",
    "cases", "rcases", "obtain", "induction", "induction'", "rw", "erw", "nth_rw",
    "rewrite", "subst", "assumption", "contradiction", "constructor", "left", "right",
    "ext", "ext1", "funext", "congr", "congrm", "trivial", "sorry", "calc", "conv", "gcongr",
    "positivity", "abel", "group", "field_simp", "norm_num", "norm_cast", "push_cast",
    "mod_cast", "push_neg", "contrapose", "by_contra", "by_cases", "exfalso",
    "specialize", "change", "use", "Continuity", "continuity", "measurability", "fun_prop",
    "symm", "mp", "mpr", "fst", "snd", "refl", "trans",
    "filter_upwards", "choose", "split", "convert", "convert_to", "tauto", "itauto", "ac_rfl", "mono",
}

# Matches explicit/implicit binder groups like `(n : Nat)` or `{p q : Prop}` or `(h : p ∧ q)`
BINDER_GROUP_RE = re.compile(r"[\(\{\[]([^()\{\}\[\]:]+?)\s*:\s*([^()\{\}\[\]]+)[\)\}\]]")
# Matches LeanDojo context lines like `x y : ℕ` or `h1 : x = y` before `⊢`
CONTEXT_LINE_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_'₀-₉₁₂₃₄₅₆₇₈₉\s]*?)\s*:\s*(.+)$")
# Matches Lean identifiers (including dotted lemma names and primed identifiers)
TACTIC_IDENT_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_'₀-₉₁₂₃₄₅₆₇₈₉.]*")
CANON_TOKEN_RE = re.compile(r"^(?:[vhpL]\d+)$")
IDENT_CHAR_CLASS = r"a-zA-Z0-9_'₀-₉₁₂₃₄₅₆₇₈₉"


def normalize_ir_whitespace(text: str) -> str:
    """Canonicalizes horizontal whitespace and strips trailing spaces per line."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def to_turnstile_state(goal_str: str) -> str:
    """
    Converts any Lean 4 declaration (`theorem foo (x : T) (h : P) : G := by`) into the
    canonical turnstile state format (`x : T\\nh : P\\n⊢ G`) used in training.
    Leaves existing turnstile states (`⊢`) unchanged after whitespace normalization.
    """
    norm = normalize_ir_whitespace(goal_str)
    if "⊢" in norm:
        return norm

    # Strip trailing `:= by` or `:=`
    cleaned = re.sub(r":=\s*(?:by\b)?\s*$", "", norm).strip()
    binders = list(BINDER_GROUP_RE.finditer(cleaned))
    context_lines = [f"{m.group(1).strip()} : {m.group(2).strip()}" for m in binders]

    if binders:
        remainder = cleaned[binders[-1].end():].strip()
        if remainder.startswith(":"):
            target = remainder[1:].strip()
            return "\n".join([*context_lines, f"⊢ {target}"])

    # Fallback for binderless declarations like `theorem t : 1 + 1 = 2`
    m_colon = re.match(r"^(?:theorem|lemma|example|def)\s+\S+\s*:\s*(.+)$", cleaned)
    if m_colon:
        return f"⊢ {m_colon.group(1).strip()}"

    return norm


def _is_hypothesis_name_or_type(ident: str, type_str: str, known_props: set) -> bool:
    t = type_str.strip()
    if ident.startswith("h") and (len(ident) == 1 or ident[1:].isdigit() or ident[1] in ("_", "'", "₁", "₂", "₃")):
        return True
    if any(op in t for op in ("=", "≠", "≤", "<", "≥", ">", "∧", "∨", "↔", "¬", "∈", "⊆", "∣")):
        return True
    if t in known_props:
        return True
    return False


def build_alpha_sigma(goal_str: str) -> Dict[str, str]:
    """
    Extracts local variables and hypotheses in declaration order and returns
    a forward renaming map `sigma: {surface_name -> canonical_name}`.
    """
    turnstile_goal = to_turnstile_state(goal_str)
    sigma: Dict[str, str] = {}
    var_idx = 0
    hyp_idx = 0
    known_props = set()

    def register_decl(names_str: str, type_str: str):
        nonlocal var_idx, hyp_idx
        idents = [tok.strip() for tok in names_str.split() if tok.strip()]
        for ident in idents:
            if ident in LEAN_KEYWORDS or ident in sigma or not (ident[0].isalpha() or ident[0] == "_"):
                continue
            if type_str.strip() == "Prop":
                known_props.add(ident)
            if _is_hypothesis_name_or_type(ident, type_str, known_props):
                sigma[ident] = f"h{hyp_idx}"
                hyp_idx += 1
            else:
                sigma[ident] = f"v{var_idx}"
                var_idx += 1

    if "⊢" in turnstile_goal:
        context_part = turnstile_goal.split("⊢", 1)[0]
        for line in context_part.splitlines():
            line_clean = line.strip()
            if not line_clean or line_clean.startswith("case "):
                continue
            m = CONTEXT_LINE_RE.match(line_clean)
            if m:
                register_decl(m.group(1), m.group(2))

    for m in BINDER_GROUP_RE.finditer(goal_str):
        register_decl(m.group(1), m.group(2))

    return sigma


def extend_sigma_with_lemmas(
    tactic_str: str,
    sigma: Dict[str, str],
    lemma_table: LemmaTable = GLOBAL_LEMMA_TABLE,
) -> Dict[str, str]:
    """
    Scans `tactic_str` for external Mathlib lemma names (identifiers not in `LEAN_KEYWORDS`
    and not in local `sigma`), registers them in `lemma_table`, and assigns their
    canonical numerical lemma IDs (`L0, L1, ...`) in `sigma`.
    """
    if not tactic_str:
        return sigma
    out_sigma = dict(sigma)

    for m in TACTIC_IDENT_RE.finditer(tactic_str):
        tok = m.group(0).rstrip(".")
        if not tok or tok in LEAN_KEYWORDS or tok in out_sigma or CANON_TOKEN_RE.match(tok):
            continue
        if "." in tok:
            parts = tok.split(".")
            if parts[0] in out_sigma or CANON_TOKEN_RE.match(parts[0]):
                continue
        out_sigma[tok] = lemma_table.encode_global_token(tok)

    return out_sigma


def _apply_token_map(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(
        f"(?<![{IDENT_CHAR_CLASS}])("
        + "|".join(re.escape(k) for k in sorted_keys)
        + f")(?![{IDENT_CHAR_CLASS}])"
    )
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def canonicalize(
    goal_str: str,
    tactic_str: str = "",
    lemma_table: LemmaTable = GLOBAL_LEMMA_TABLE,
) -> Tuple[str, str, Dict[str, str]]:
    """
    Canonicalizes `(goal_str, tactic_str)` into `(canon_goal, canon_tactic, sigma)`.
    - Normalizes `goal_str` into turnstile format (`v0 : T\\n⊢ G`).
    - Maps local variables/hypotheses to `v0, h0` and external lemmas to `L0, L1`
      via `lemma_table`.
    """
    turnstile_goal = to_turnstile_state(goal_str)
    norm_tactic = normalize_ir_whitespace(tactic_str) if tactic_str else ""
    sigma = build_alpha_sigma(turnstile_goal)
    if norm_tactic:
        sigma = extend_sigma_with_lemmas(norm_tactic, sigma, lemma_table=lemma_table)
    canon_goal = _apply_token_map(turnstile_goal, sigma)
    canon_tactic = _apply_token_map(norm_tactic, sigma) if norm_tactic else ""
    return canon_goal, canon_tactic, sigma


def invert_sigma(
    canon_tactic: str,
    sigma: Dict[str, str],
    lemma_table: LemmaTable = GLOBAL_LEMMA_TABLE,
) -> str:
    """
    Inverts the canonical symbol table `sigma` (local `v_i`, `h_i`) and global
    numerical lemma IDs (`L_i` from `lemma_table`) on a predicted canonical tactic
    using Lean 4's native `Lean.Syntax` parser and `Syntax.reprint`.
    Works at blind inference time when `sigma` was built solely from `goal_str`.
    """
    if not canon_tactic:
        return canon_tactic
    inv_map = {canon: surf for surf, canon in sigma.items()}
    for m in re.finditer(r"\bL\d+\b", canon_tactic):
        ltok = m.group(0)
        if ltok not in inv_map:
            decoded = lemma_table.decode_global_token(ltok)
            if decoded != ltok:
                inv_map[ltok] = decoded
    if not inv_map:
        return canon_tactic
    try:
        from src.ir.tactic_ast import LeanASTBridge
        res = LeanASTBridge.parse_tactic(canon_tactic, inv_map)
        if res.get("ok") and res.get("root_kind"):
            return res["canonical_tactic"]
    except Exception:
        pass
    return _apply_token_map(canon_tactic, inv_map)
