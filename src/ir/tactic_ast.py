"""
Prefix IR AST Grammar & Type-Directed Candidate Generator for Lean 4 Tactics.
Represents tactics as pre-order (Polish prefix) AST node sequences with zero syntactic
delimiters, strictly constraining argument leaves to in-scope LocalContext declarations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
from src.ir.canonicalizer import invert_sigma


class NodeSort(Enum):
    OP = "OP"      # Root tactic operator
    DIR = "DIR"    # Rewrite direction (fwd / symm)
    LOC = "LOC"    # Target location (goal or in-scope hypothesis)
    HYP = "HYP"    # In-scope hypothesis (h0, h1, ...)
    VAR = "VAR"    # In-scope variable (v0, v1, ...)
    TERM = "TERM"  # Proof term / combinator / premise


@dataclass(frozen=True)
class ASTNode:
    token: str
    sort: NodeSort
    child_sorts: Tuple[NodeSort, ...] = ()


# Canonical root tactic operators and structural term combinators
AST_OPERATORS: Tuple[ASTNode, ...] = (
    ASTNode("OP:rfl", NodeSort.OP, ()),
    ASTNode("OP:simp", NodeSort.OP, ()),
    ASTNode("OP:omega", NodeSort.OP, ()),
    ASTNode("OP:ring", NodeSort.OP, ()),
    ASTNode("OP:linarith", NodeSort.OP, ()),
    ASTNode("OP:decide", NodeSort.OP, ()),
    ASTNode("OP:aesop", NodeSort.OP, ()),
    ASTNode("OP:assumption", NodeSort.OP, ()),
    ASTNode("OP:exact", NodeSort.OP, (NodeSort.TERM,)),
    ASTNode("OP:apply", NodeSort.OP, (NodeSort.TERM,)),
    ASTNode("OP:intro", NodeSort.OP, (NodeSort.VAR,)),
    ASTNode("OP:cases", NodeSort.OP, (NodeSort.HYP,)),
    ASTNode("OP:induction", NodeSort.OP, (NodeSort.VAR,)),
    ASTNode("OP:rw", NodeSort.OP, (NodeSort.DIR, NodeSort.TERM, NodeSort.LOC)),
)

TERM_COMBINATORS: Tuple[ASTNode, ...] = (
    ASTNode("CTOR:pair", NodeSort.TERM, (NodeSort.TERM, NodeSort.TERM)),
    ASTNode("PROJ:left", NodeSort.TERM, (NodeSort.HYP,)),
    ASTNode("PROJ:right", NodeSort.TERM, (NodeSort.HYP,)),
)

NODE_BY_TOKEN: Dict[str, ASTNode] = {
    n.token: n
    for n in (
        *AST_OPERATORS,
        *TERM_COMBINATORS,
        ASTNode("DIR:fwd", NodeSort.DIR, ()),
        ASTNode("DIR:symm", NodeSort.DIR, ()),
        ASTNode("LOC:goal", NodeSort.LOC, ()),
    )
}


def valid_ast_candidates(
    required_sort: NodeSort,
    sigma: Dict[str, str],
    premises: Optional[Sequence[str]] = None,
    allow_combinators: bool = True,
) -> List[ASTNode]:
    """
    Returns strictly grammar-valid and LocalContext-in-scope ASTNode candidates
    for the given AST slot `required_sort`. Out-of-scope h_i / v_i cannot be emitted.
    """
    canon_vals = sorted(set(sigma.values()))
    in_scope_hyps = [ASTNode(f"HYP:{c}", NodeSort.HYP, ()) for c in canon_vals if c.startswith("h")]
    in_scope_vars = [ASTNode(f"VAR:{c}", NodeSort.VAR, ()) for c in canon_vals if c.startswith("v")]

    if required_sort == NodeSort.OP:
        # Filter operators that require a HYP/VAR if none exist in LocalContext
        valid_ops = []
        for op in AST_OPERATORS:
            if NodeSort.HYP in op.child_sorts and not in_scope_hyps:
                continue
            valid_ops.append(op)
        return valid_ops

    if required_sort == NodeSort.DIR:
        return [NODE_BY_TOKEN["DIR:fwd"], NODE_BY_TOKEN["DIR:symm"]]

    if required_sort == NodeSort.HYP:
        return in_scope_hyps if in_scope_hyps else [ASTNode("HYP:h0", NodeSort.HYP, ())]

    if required_sort == NodeSort.VAR:
        return in_scope_vars if in_scope_vars else [ASTNode("VAR:v0", NodeSort.VAR, ())]

    if required_sort == NodeSort.LOC:
        return [NODE_BY_TOKEN["LOC:goal"], *in_scope_hyps]

    if required_sort == NodeSort.TERM:
        cands: List[ASTNode] = [*in_scope_hyps, *in_scope_vars]
        if allow_combinators and in_scope_hyps:
            cands.extend(TERM_COMBINATORS)
        if premises:
            cands.extend(ASTNode(f"CONST:{p}", NodeSort.TERM, ()) for p in premises)
        return cands if cands else [ASTNode("HYP:h0", NodeSort.HYP, ())]

    raise ValueError(f"Unsupported NodeSort: {required_sort}")


def _render_term(tokens: List[str], idx: int) -> Tuple[str, int]:
    tok = tokens[idx]
    if tok == "CTOR:pair":
        left_str, next_idx = _render_term(tokens, idx + 1)
        right_str, final_idx = _render_term(tokens, next_idx)
        return f"⟨{left_str}, {right_str}⟩", final_idx
    if tok == "PROJ:left":
        arg_str, next_idx = _render_term(tokens, idx + 1)
        return f"{arg_str}.left", next_idx
    if tok == "PROJ:right":
        arg_str, next_idx = _render_term(tokens, idx + 1)
        return f"{arg_str}.right", next_idx
    if ":" in tok:
        return tok.split(":", 1)[1], idx + 1
    return tok, idx + 1


def lower_ast_to_lean(nodes: Sequence[ASTNode], sigma: Optional[Dict[str, str]] = None) -> str:
    """
    Lowers a pre-order sequence of ASTNodes into a Lean 4 tactic string,
    and applies sigma^-1 (`invert_sigma`) to restore surface variable/hypothesis names.
    """
    tokens = [n.token if isinstance(n, ASTNode) else str(n) for n in nodes]
    if not tokens:
        return ""

    root = tokens[0]
    op_name = root.split(":", 1)[1] if ":" in root else root

    if op_name in {"rfl", "simp", "omega", "ring", "linarith", "decide", "aesop", "assumption"}:
        canon_tac = op_name
    elif op_name in {"exact", "apply", "intro", "cases", "induction"}:
        term_str, _ = _render_term(tokens, 1)
        canon_tac = f"{op_name} {term_str}"
    elif op_name == "rw":
        direction = tokens[1] if len(tokens) > 1 else "DIR:fwd"
        rule_str, next_idx = _render_term(tokens, 2)
        loc_tok = tokens[next_idx] if next_idx < len(tokens) else "LOC:goal"
        prefix = "← " if direction == "DIR:symm" else ""
        if loc_tok == "LOC:goal":
            canon_tac = f"rw [{prefix}{rule_str}]"
        else:
            loc_name = loc_tok.split(":", 1)[1]
            canon_tac = f"rw [{prefix}{rule_str}] at {loc_name}"
    else:
        canon_tac = op_name

    return invert_sigma(canon_tac, sigma) if sigma else canon_tac
