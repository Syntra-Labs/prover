"""
Structural AST Admissibility Gate over the kernel `Lean.Expr` DAG (`ast-admissibility-gate`).

Given a turnstile goal and a lemma universe in ONE hash-consed DAG, assigns E = +inf to
every one-step move that cannot structurally unify with the current state, BEFORE any
learned operator scores anything. Lemma binders become pattern wildcards (first-order,
consistent bindings); everything else must match constructor + label exactly.

Move vocabulary (each is a single kernel-level step):
  exact  fvI          hypothesis type is the *same node* as the target
  intro               target is a `forallE` (or `Not _`)
  apply  L            L's conclusion (rigid head) unifies with the target
  rw     L / ← L      a side of L's `Eq`/`Iff` conclusion (rigid head) occurs in the target
  rw     fvI / ← fvI  same, for equational hypotheses
  elim   L fvI        L's conclusion is a bare/flex wildcard (And.left, absurd, ...):
                      admissible only if one of its premises unifies with a Prop hypothesis

Limitation (by design, v0): unification is syntactic, not modulo definitional unfolding.
Moves admissible only up to defeq (e.g. `rfl` on `n + 0 = n`) are not recovered here;
those should be re-checked with Lean's `isDefEq` on the (small) survivor set.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.ir.kernel_expr import (APP, BVAR, CONST, FORALL, LAM, KernelState)

Binding = Dict[int, int]


@dataclass(frozen=True)
class Move:
    kind: str                 # exact | intro | apply | rw | elim
    source: str               # lemma name or fvN ("" for intro)
    reverse: bool = False     # rw ← direction
    via: str = ""             # hypothesis consumed by an elim move

    def __str__(self) -> str:
        if self.kind == "intro":
            return "intro"
        if self.kind == "rw":
            return f"rw [{'← ' if self.reverse else ''}{self.source}]"
        if self.kind == "elim":
            return f"apply {self.source} (using {self.via})"
        return f"{self.kind} {self.source}"


def _strip_foralls(st: KernelState, n: int) -> Tuple[List[int], int]:
    binders = []
    while st.nodes[n].k == FORALL:
        dom, body = st.nodes[n].children
        binders.append(dom)
        n = body
    return binders, n


def _match(st: KernelState, p: int, t: int, depth: int, local: int, b: Binding) -> bool:
    """First-order match of pattern node `p` (inside `depth` stripped lemma binders and
    `local` pattern-internal binders) against target node `t`. Mutates `b` on success."""
    pn = st.nodes[p]
    if pn.k == BVAR:
        i = int(pn.label)
        if i < local:
            tn = st.nodes[t]
            return tn.k == BVAR and int(tn.label) == i
        absolute = depth - 1 - (i - local)
        if absolute < 0:
            return False
        if absolute in b:
            return b[absolute] == t
        b[absolute] = t
        return True
    if p == t:
        return True
    tn = st.nodes[t]
    if pn.k != tn.k or pn.label != tn.label or len(pn.children) != len(tn.children):
        return False
    inner = local + 1 if pn.k in (LAM, FORALL) else local
    for idx, (pc, tc) in enumerate(zip(pn.children, tn.children)):
        # binder domain is outside the new scope, body is inside it
        loc = local if (pn.k in (LAM, FORALL) and idx == 0) else inner
        if not _match(st, pc, tc, depth, loc, b):
            return False
    return True


def _rigid(st: KernelState, n: int) -> bool:
    """A pattern has a rigid head if its app-spine head is not a wildcard bvar."""
    return st.nodes[st.head(n)].k != BVAR


def _eq_sides(st: KernelState, concl: int) -> Optional[Tuple[int, int]]:
    """`@Eq α lhs rhs` or `Iff a b` -> (lhs, rhs)."""
    args, h = [], concl
    while st.nodes[h].k == APP:
        f, a = st.nodes[h].children
        args.append(a)
        h = f
    head = st.nodes[h]
    if head.k == CONST and head.label == "Eq" and len(args) == 3:
        return args[1], args[0]
    if head.k == CONST and head.label == "Iff" and len(args) == 2:
        return args[1], args[0]
    return None


def admissible_moves(st: KernelState, decl: Optional[str] = None) -> List[Move]:
    target_subterms = list(st.subterms(st.target))
    tnode = st.nodes[st.target]
    prop_hyps = [h for h in st.hyps if h.is_prop]
    moves: List[Move] = []

    # intro: Π-type target, or `Not P` (which unfolds to P → False)
    head = st.nodes[st.head(st.target)]
    if tnode.k == FORALL or (head.k == CONST and head.label == "Not"):
        moves.append(Move("intro", ""))

    # hypotheses: exact by node identity, rewrite by subterm occurrence
    for h in prop_hyps:
        if h.type == st.target:
            moves.append(Move("exact", h.fv))
        h_binders, h_concl = _strip_foralls(st, h.type)
        if h_binders and _rigid(st, h_concl) and _match(st, h_concl, st.target, len(h_binders), 0, {}):
            moves.append(Move("apply", h.fv))
        sides = _eq_sides(st, h.type)
        if sides:
            for rev, side in ((False, sides[0]), (True, sides[1])):
                if side in target_subterms and side != st.target:
                    moves.append(Move("rw", h.fv, reverse=rev))

    # lemma universe
    for name, root in st.consts.items():
        binders, concl = _strip_foralls(st, root)
        depth = len(binders)
        if _rigid(st, concl):
            if _match(st, concl, st.target, depth, 0, {}):
                moves.append(Move("apply", name))
            sides = _eq_sides(st, concl)
            if sides:
                for rev, side in ((False, sides[0]), (True, sides[1])):
                    if _rigid(st, side) and any(
                        _match(st, side, t, depth, 0, {}) for t in target_subterms
                    ):
                        moves.append(Move("rw", name, reverse=rev))
        else:
            b: Binding = {}
            if not _match(st, concl, st.target, depth, 0, b):
                continue
            for j, dom in enumerate(binders):
                if not _rigid(st, dom):
                    continue
                hit = next((h for h in prop_hyps
                            if _match(st, dom, h.type, j, 0, dict(b))), None)
                if hit is not None:
                    moves.append(Move("elim", name, via=hit.fv))
                    break

    # Tier-2 DefEq Gate: for non-equality relations (e.g. `LE.le`), syntactic matching misses
    # lemmas that unify modulo definitional reduction (e.g. `n ≤ succ n` vs `n ≤ n + 1`).
    # Run apply_check on lemmas sharing the relation head and type domain.
    if decl:
        from src.ir.kernel_expr import ExprBridge
        target_head = st.head(st.target)
        th_nd = st.nodes[target_head]
        # Skip Eq (which has 3800 candidates and already has syntactic rewrite/apply hits)
        if th_nd.k == CONST and th_nd.label != "Eq":
            def get_spine_args(n: int) -> List[int]:
                args = []
                while st.nodes[n].k == APP:
                    args.append(st.nodes[n].children[1])
                    n = st.nodes[n].children[0]
                return args[::-1]

            t_args = get_spine_args(st.target)
            candidates = []
            for name, root in st.consts.items():
                _, concl = _strip_foralls(st, root)
                c_head = st.head(concl)
                ch_nd = st.nodes[c_head]
                if ch_nd.k == CONST and ch_nd.label == th_nd.label:
                    c_args = get_spine_args(concl)
                    # If relation has arguments (e.g. LE.le α inst a b), match type argument α
                    if not t_args or (c_args and (c_args[0] == t_args[0] or st.nodes[c_args[0]].k == BVAR)):
                        if Move("apply", name) not in moves:
                            candidates.append(name)
            if candidates and len(candidates) <= 1000:
                accepted = ExprBridge.apply_check(decl, candidates)
                for acc in accepted:
                    moves.append(Move("apply", acc))

    return moves
