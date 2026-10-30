# TODO(ast-admissibility-gate | tier:B):
# Description: Filter LemmaTable and Lean.Expr constructors via structural target/hypothesis head unification to assign E=+inf to inadmissible moves before operator scoring.
# Criterion: assert: candidate_reduction_ratio >= 0.90 and ground_truth_retained == True

# TODO(tree-rotor-operator | tier:B):
# Description: Fold Lean.Expr DAG bottom-up via non-commutative quaternion/rotor operators on S3 without fixed sliding-window truncation.
# Criterion: assert: window_truncation_rate == 0.0 and non_commutative_asymmetry > 0.0
"""
Native Lean 4 `Lean.Syntax` Prefix AST Bridge & Type-Directed Candidate Generator.
Communicates over a persistent JSONL stdio pipe with `src/lean_bridge/ast_bridge.lean`
(`Lean.Parser.runParserCategory` + `Lean.Syntax` + `Syntax.reprint`) so that tactic AST
node kinds (`Lean.Parser.Tactic.*`, `Lean.Parser.Term.*`, `IDENT:*`, `ATOM:*`) and
identifier renamings come directly from Lean 4's compiler backend.
"""

import atexit
import json
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRIDGE_LEAN_FILE = REPO_ROOT / "src/lean_bridge/ast_bridge.lean"


class LeanASTBridge:
    """Persistent JSONL stdio client to `lean --run src/lean_bridge/ast_bridge.lean`."""

    _proc: Optional[subprocess.Popen] = None

    @classmethod
    def _ensure_proc(cls) -> subprocess.Popen:
        if cls._proc is None or cls._proc.poll() is not None:
            cls._proc = subprocess.Popen(
                ["lean", "--run", str(BRIDGE_LEAN_FILE)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            atexit.register(cls.close)
        return cls._proc

    @classmethod
    def request(cls, payload: Dict) -> Dict:
        proc = cls._ensure_proc()
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Lean AST bridge process terminated unexpectedly")
        return json.loads(line)

    @classmethod
    def parse_tactic(cls, tactic: str, sigma: Optional[Dict[str, str]] = None) -> Dict:
        return cls.request({"op": "parse_tactic", "tactic": tactic, "sigma": sigma or {}})

    @classmethod
    def parse_command_binders(cls, command: str) -> Dict:
        return cls.request({"op": "parse_command_binders", "command": command})

    @classmethod
    def close(cls) -> None:
        if cls._proc is not None and cls._proc.poll() is None:
            try:
                if cls._proc.stdin:
                    cls._proc.stdin.close()
                cls._proc.terminate()
            except Exception:
                pass
            cls._proc = None


class NodeSort(Enum):
    OP = "OP"      # Root `Lean.Parser.Tactic.*` node
    DIR = "DIR"    # Rewrite direction (`ATOM:fwd` / `ATOM:←`)
    LOC = "LOC"    # Target location (`NODE:Lean.Parser.Tactic.locationWildcard` or `IDENT:h_i`)
    HYP = "HYP"    # In-scope hypothesis `Syntax.ident` (`IDENT:h0`, ...)
    VAR = "VAR"    # In-scope variable `Syntax.ident` (`IDENT:v0`, ...)
    TERM = "TERM"  # Term `Lean.Parser.Term.*` or `Syntax.ident`


@dataclass(frozen=True)
class ASTNode:
    token: str
    sort: NodeSort
    child_sorts: Tuple[NodeSort, ...] = ()


# Native Lean 4 `Lean.Parser.Tactic.*` and `Lean.Parser.Term.*` SyntaxNodeKinds
AST_OPERATORS: Tuple[ASTNode, ...] = (
    ASTNode("NODE:Lean.Parser.Tactic.tacticRfl", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.simp", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.omega", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.ring", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.linarith", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.decide", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.aesop", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.assumption", NodeSort.OP, ()),
    ASTNode("NODE:Lean.Parser.Tactic.exact", NodeSort.OP, (NodeSort.TERM,)),
    ASTNode("NODE:Lean.Parser.Tactic.apply", NodeSort.OP, (NodeSort.TERM,)),
    ASTNode("NODE:Lean.Parser.Tactic.intro", NodeSort.OP, (NodeSort.VAR,)),
    ASTNode("NODE:Lean.Parser.Tactic.cases", NodeSort.OP, (NodeSort.HYP,)),
    ASTNode("NODE:Lean.Parser.Tactic.induction", NodeSort.OP, (NodeSort.VAR,)),
    ASTNode("NODE:Lean.Parser.Tactic.rwSeq", NodeSort.OP, (NodeSort.DIR, NodeSort.TERM, NodeSort.LOC)),
)

TERM_COMBINATORS: Tuple[ASTNode, ...] = (
    ASTNode("NODE:Lean.Parser.Term.anonymousCtor", NodeSort.TERM, (NodeSort.TERM, NodeSort.TERM)),
    ASTNode("PROJ:left", NodeSort.TERM, (NodeSort.HYP,)),
    ASTNode("PROJ:right", NodeSort.TERM, (NodeSort.HYP,)),
)


def valid_ast_candidates(
    required_sort: NodeSort,
    sigma: Dict[str, str],
    premises: Optional[Sequence[str]] = None,
    allow_combinators: bool = True,
) -> List[ASTNode]:
    """
    Returns strictly grammar-valid and LocalContext-in-scope `Lean.Syntax` ASTNode
    candidates for `required_sort`. Out-of-scope h_i / v_i cannot be emitted.
    """
    canon_vals = sorted(set(sigma.values()))
    in_scope_hyps = [ASTNode(f"IDENT:{c}", NodeSort.HYP, ()) for c in canon_vals if c.startswith("h")]
    in_scope_vars = [ASTNode(f"IDENT:{c}", NodeSort.VAR, ()) for c in canon_vals if c.startswith("v")]

    if required_sort == NodeSort.OP:
        return [
            op for op in AST_OPERATORS
            if not (NodeSort.HYP in op.child_sorts and not in_scope_hyps)
        ]
    if required_sort == NodeSort.DIR:
        return [ASTNode("ATOM:fwd", NodeSort.DIR, ()), ASTNode("ATOM:←", NodeSort.DIR, ())]
    if required_sort == NodeSort.HYP:
        return in_scope_hyps if in_scope_hyps else [ASTNode("IDENT:h0", NodeSort.HYP, ())]
    if required_sort == NodeSort.VAR:
        return in_scope_vars if in_scope_vars else [ASTNode("IDENT:v0", NodeSort.VAR, ())]
    if required_sort == NodeSort.LOC:
        return [ASTNode("NODE:Lean.Parser.Tactic.locationWildcard", NodeSort.LOC, ()), *in_scope_hyps]
    if required_sort == NodeSort.TERM:
        cands: List[ASTNode] = [*in_scope_hyps, *in_scope_vars]
        if allow_combinators and in_scope_hyps:
            cands.extend(TERM_COMBINATORS)
        if premises:
            cands.extend(ASTNode(f"IDENT:{p}", NodeSort.TERM, ()) for p in premises)
        return cands if cands else [ASTNode("IDENT:h0", NodeSort.HYP, ())]
    raise ValueError(f"Unsupported NodeSort: {required_sort}")


def _render_term(tokens: List[str], idx: int) -> Tuple[str, int]:
    tok = tokens[idx]
    if tok in {"NODE:Lean.Parser.Term.anonymousCtor", "CTOR:pair"}:
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


_OP_SURFACE_MAP = {
    "Lean.Parser.Tactic.tacticRfl": "rfl",
    "Lean.Parser.Tactic.simp": "simp",
    "Lean.Parser.Tactic.omega": "omega",
    "Lean.Parser.Tactic.ring": "ring",
    "Lean.Parser.Tactic.linarith": "linarith",
    "Lean.Parser.Tactic.decide": "decide",
    "Lean.Parser.Tactic.aesop": "aesop",
    "Lean.Parser.Tactic.assumption": "assumption",
    "Lean.Parser.Tactic.exact": "exact",
    "Lean.Parser.Tactic.apply": "apply",
    "Lean.Parser.Tactic.intro": "intro",
    "Lean.Parser.Tactic.cases": "cases",
    "Lean.Parser.Tactic.induction": "induction",
    "Lean.Parser.Tactic.rwSeq": "rw",
}


def lower_ast_to_lean(nodes: Sequence[ASTNode], sigma: Optional[Dict[str, str]] = None) -> str:
    """
    Lowers a pre-order sequence of `Lean.Syntax` ASTNodes into a canonical tactic,
    then passes it through `LeanASTBridge.parse_tactic` (`Lean.Parser.runParserCategory`
    + `mapSyntaxIdents` + `Syntax.reprint`) with `sigma^-1` to emit a verified Lean 4 tactic.
    """
    tokens = [n.token if isinstance(n, ASTNode) else str(n) for n in nodes]
    if not tokens:
        return ""

    raw_root = tokens[0].split(":", 1)[1] if ":" in tokens[0] else tokens[0]
    op_name = _OP_SURFACE_MAP.get(raw_root, raw_root)

    if op_name in {"rfl", "simp", "omega", "ring", "linarith", "decide", "aesop", "assumption"}:
        canon_tac = op_name
    elif op_name in {"exact", "apply", "intro", "cases", "induction"}:
        term_str, _ = _render_term(tokens, 1)
        canon_tac = f"{op_name} {term_str}"
    elif op_name == "rw":
        direction = tokens[1] if len(tokens) > 1 else "ATOM:fwd"
        rule_str, next_idx = _render_term(tokens, 2)
        loc_tok = tokens[next_idx] if next_idx < len(tokens) else "NODE:Lean.Parser.Tactic.locationWildcard"
        prefix = "← " if direction in {"ATOM:←", "DIR:symm"} else ""
        if loc_tok in {"NODE:Lean.Parser.Tactic.locationWildcard", "LOC:goal"}:
            canon_tac = f"rw [{prefix}{rule_str}]"
        else:
            loc_name = loc_tok.split(":", 1)[1]
            canon_tac = f"rw [{prefix}{rule_str}] at {loc_name}"
    else:
        canon_tac = op_name

    inv_map = {canon: surf for surf, canon in (sigma or {}).items()}
    res = LeanASTBridge.parse_tactic(canon_tac, inv_map)
    return res.get("canonical_tactic", canon_tac)
