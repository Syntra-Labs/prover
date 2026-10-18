"""
Kernel `Lean.Expr` IR client (`kernel-expr-ir`).

Talks to `src/lean_bridge/expr_bridge.lean` (Lean 4's real elaborator) and returns the
proof state as a hash-consed DAG over the 11 kernel constructors. The model-facing
representation is integer node ids only; no surface text and no BPE tokenization:

    goal "theorem prop_id (p : Prop) (h : p) : p := by"
      -> hyps   fv0 : sort(0)      fv1 : fvar(fv0)
      -> target fvar(fv0)            (same node id as fv1's type => `exact fv1` is visible)
"""

import atexit
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPR_BRIDGE_FILE = REPO_ROOT / "src/lean_bridge/expr_bridge.lean"

CTOR_NAMES = ("bvar", "fvar", "mvar", "sort", "const", "app", "lam",
              "forallE", "letE", "lit", "proj")
BVAR, FVAR, MVAR, SORT, CONST, APP, LAM, FORALL, LET, LIT, PROJ = range(11)


@dataclass(frozen=True)
class Node:
    k: int                 # constructor id in [0, 11)
    label: str             # de Bruijn index / fvN / constant name / literal ("" for binders, app)
    children: Tuple[int, ...]


@dataclass
class Hyp:
    fv: str
    user: str
    type: int
    is_prop: bool


@dataclass
class KernelState:
    """A turnstile goal plus a lemma universe, all in ONE shared hash-consed DAG."""
    nodes: List[Node]
    hyps: List[Hyp]
    target: int
    consts: Dict[str, int] = field(default_factory=dict)

    def subterms(self, root: int) -> Iterator[int]:
        seen, stack = set(), [root]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            yield n
            stack.extend(self.nodes[n].children)

    def head(self, n: int) -> int:
        """Strip `app` spine: head symbol of `f a1 ... an`."""
        while self.nodes[n].k == APP:
            n = self.nodes[n].children[0]
        return n

    def render(self, n: int) -> str:
        """Debug-only pretty print (never fed to a model)."""
        nd = self.nodes[n]
        if nd.k == APP:
            return f"({self.render(nd.children[0])} {self.render(nd.children[1])})"
        if nd.k in (LAM, FORALL):
            sym = "λ" if nd.k == LAM else "∀"
            return f"{sym}:{self.render(nd.children[0])}.{self.render(nd.children[1])}"
        if nd.k == BVAR:
            return f"#{nd.label}"
        if nd.k == SORT:
            return f"Sort({nd.label})"
        if nd.k == PROJ:
            return f"{self.render(nd.children[0])}.{nd.label}"
        return nd.label


def decl_to_pi_type(decl: str) -> Tuple[str, int]:
    """`theorem foo (p : Prop) (h : p) : p := by` -> (`∀ (p : Prop) (h : p), p`, 2).
    Returns (closed_pi_type, num_declared_binders)."""
    s = decl.strip()
    for suffix in (":= by", ":="):
        if s.endswith(suffix):
            s = s[: -len(suffix)].rstrip()
            break
    parts = s.split(None, 2)
    if parts and parts[0] in ("theorem", "lemma"):
        s = parts[2] if len(parts) > 2 else ""
    elif parts and parts[0] == "example":
        s = s[len("example"):].strip()
    depth, split_at = 0, -1
    for i, ch in enumerate(s):
        if ch in "([{⟨":
            depth += 1
        elif ch in ")]}⟩":
            depth -= 1
        elif ch == ":" and depth == 0 and s[i + 1: i + 2] != "=":
            split_at = i
            break
    if split_at < 0:
        return s, -1
    binders, ty = s[:split_at].strip(), s[split_at + 1:].strip()
    if not binders:
        return ty, 0

    # Count individual identifiers declared in binder groups like `(a b c : Nat) {h : P}`
    import re
    n_binders = 0
    for m in re.finditer(r"[\(\{\[]([^:()\{\}\[\]]+):", binders):
        n_binders += len(m.group(1).split())

    return f"∀ {binders}, {ty}", n_binders


class ExprBridge:
    """Persistent JSONL client to `lean --run src/lean_bridge/expr_bridge.lean`."""

    _proc: Optional[subprocess.Popen] = None

    @classmethod
    def _ensure(cls) -> subprocess.Popen:
        if cls._proc is None or cls._proc.poll() is not None:
            cls._proc = subprocess.Popen(
                ["lean", "--run", str(EXPR_BRIDGE_FILE)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            atexit.register(cls.close)
        return cls._proc

    @classmethod
    def close(cls) -> None:
        if cls._proc is not None and cls._proc.poll() is None:
            cls._proc.stdin.close()
            cls._proc.wait(timeout=5)
        cls._proc = None

    @classmethod
    def state(cls, goal: str, names: Union[str, Sequence[str]] = (), intros: Optional[int] = None) -> KernelState:
        """`goal` is a declaration (`theorem ... := by`) or a closed type.
        `names="*"` encodes every public theorem/constructor in the environment."""
        if intros is None:
            if ":=" in goal or goal.lstrip().startswith(("theorem", "lemma", "example")):
                ty, intros = decl_to_pi_type(goal)
            else:
                ty, intros = goal, -1
        else:
            ty = decl_to_pi_type(goal)[0] if (":=" in goal or goal.lstrip().startswith(("theorem", "lemma", "example"))) else goal

        proc = cls._ensure()
        proc.stdin.write(json.dumps({
            "op": "state_dag", "type": ty, "intros": intros,
            "names": names if names == "*" else list(names)
        }) + "\n")
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline())
        if not resp.get("ok"):
            raise ValueError(f"expr_bridge failed on {ty!r}: {resp.get('error')}")
        nodes = [Node(n["k"], n["l"], tuple(n["c"])) for n in resp["nodes"]]
        hyps = [Hyp(h["fv"], h["user"], h["type"], h["isProp"]) for h in resp["hyps"]]
        consts = {c["name"]: c["type"] for c in resp["consts"]}
        return KernelState(nodes, hyps, resp["target"], consts)

    @classmethod
    def apply_check(cls, goal: str, names: Sequence[str], intros: Optional[int] = None) -> List[str]:
        """Tier-2 Gate: queries Lean's real `MVarId.apply` on candidate lemma names."""
        if intros is None:
            if ":=" in goal or goal.lstrip().startswith(("theorem", "lemma", "example")):
                ty, intros = decl_to_pi_type(goal)
            else:
                ty, intros = goal, -1
        else:
            ty = decl_to_pi_type(goal)[0] if (":=" in goal or goal.lstrip().startswith(("theorem", "lemma", "example"))) else goal

        proc = cls._ensure()
        proc.stdin.write(json.dumps({
            "op": "apply_check", "type": ty, "intros": intros,
            "names": list(names)
        }) + "\n")
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline())
        if not resp.get("ok"):
            raise ValueError(f"apply_check failed: {resp.get('error')}")
        return resp.get("accepted", [])
