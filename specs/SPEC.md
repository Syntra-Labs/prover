# Prover Geometric Engine & LeanDojo Application Specification

This document is the normative specification for the **Prover Theorem Proving System** (`prover`), integrating Casey Allard's **UOR-R4 214M Geometric Language Model** with **LeanDojo** for automated theorem proving in Lean 4.

## 1. Scope and End-to-End Application Architecture

The Prover application automates formal mathematical reasoning by coupling geometric deep learning with interactive Lean 4 kernel feedback:

```text
               ┌────────────────────────────────────────────────────────┐
               │              Lean 4 Theorem Target                     │
               │         (Statement, Context, Open Subgoals)            │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │                 LeanDojo Interface                     │
               │    (Environment Lifecycle, Goal Extraction, REPL)     │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                        Active Goal State │ Normalized Syntax
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │           UOR-R4 214M Geometric Policy                 │
               │  - Tokenizer: Byte-Pair Encoding without UNK dropping   │
               │  - Associative Scan: S^3 Unit Quaternions / Cl(3,0)    │
               │  - Discrete Quantization: 8D Conway-Sloane E8 Snapping │
               │  - Emission: Top-k Candidate Tactics & Arguments       │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                        Candidate Tactics │ Ranked Actions
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │              Proof Search State Machine                │
               │  - Search Frontier: Best-First Search with Confidence   │
               │  - Depth & Step Budget Bounds                          │
               │  - Kernel Execution via LeanDojo                       │
               │  - Goal Resolution: Proven / Exhausted / Backtrack     │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │           Certified Lean 4 Proof Script                │
               │       (Kernel-replayed with 0 sorry / 0 err)           │
               └────────────────────────────────────────────────────────┘
```

In accordance with the PrismPM standard:
- Application behavior, interaction states, and geometric transformations are derived strictly from formal `.lex.tex` specifications.
- State transitions and mathematical invariants are formally proven in Lean 4 (`leanprover/lean4:v4.32.1`) prior to artifact emission.
- Anti-drift git hooks mechanically enforce task DAG acyclicity (`scripts/lint_tasks.py`) and LexLean formal compiler verification (`lexlean check`) on every commit.

---

## 2. Core Subsystems

### 2.1 LeanDojo Interactive Kernel Interface
- **Environment Management**: Initializes sandbox Lean 4 project environments (`leanprover/lean4`).
- **Goal State Tracking**: Extracts active hypotheses, target propositions, and open subgoal indices (`GoalState`).
- **Tactic Execution**: Sends candidate tactic strings to the Lean 4 kernel via LeanDojo/Pantograph and parses kernel responses:
  - `GoalsClosed`: Current goal discharged with 0 remaining subgoals.
  - `SubgoalsOpen`: Tactic accepted, generating $k \ge 1$ new subgoals.
  - `KernelError`: Tactic rejected by Lean 4 kernel (syntax error, type mismatch, failed unifier).
  - `KernelTimeout`: Execution timed out beyond safety budget.

### 2.2 UOR-R4 Geometric Policy Engine
- **Representation**: Native Rust `uor-r4` runtime using associative prefix scans over unit quaternions $\mathbf{q} \in S^3 \subset \mathbb{H}$.
- **Discrete Quantization**: 64-dimensional hidden states are segmented into eight 8-dimensional blocks:
  $$\mathbf{x} = [\mathbf{b}_0, \mathbf{b}_1, \dots, \mathbf{b}_7], \quad \mathbf{b}_i \in \mathbb{R}^8$$
- **Conway-Sloane $E_8$ Snapping**:
  Each block is mapped to the nearest point in the Gosset root lattice $E_8$:
  $$E_8 = D_8 \cup \left(D_8 + \frac{1}{2}\mathbf{1}\right)$$
  where $D_8 = \left\{ \mathbf{x} \in \mathbb{Z}^8 : \sum_{i=1}^8 x_i \equiv 0 \pmod 2 \right\}$.
- **Tactic Selection**: Snapped lattice Voronoi cells map directly to the 50 canonical Mathlib 4 tactics (`intro`, `exact`, `apply`, `simp`, `rw`, `ring`, `omega`, `rfl`, `cases`, `induction`, etc.).

### 2.3 Search Engine & State Machine
- **State Space**: Search nodes track `(goalCount, depth, steps)` bounded by `SearchLimits(maxSteps, maxDepth)`.
- **Status Invariant**: Engine advances through formal lifecycle states:
  - `Searching`: Active exploration of proof branches.
  - `SuccessProven`: All initial and spawned subgoals closed by verified kernel tactics.
  - `BudgetExhausted`: Step or depth budget exceeded without closure.
  - `SearchFailed`: All search tree branches exhausted without finding a proof.

---

## 3. Formal Types in LexLean (`ProverEngine.lex.tex`)

The formal semantic module defines the exact machine-checked data structures and functions:

| Type / Constructor | Category | Semantics |
| :--- | :--- | :--- |
| `Quaternion4` | Structure | Four-coordinate $(w, x, y, z)$ unit quaternion in $S^3$. |
| `LatticeVector8` | Structure | Eight-coordinate $(x_0, \dots, x_7)$ vector in $\mathbb{R}^8$. |
| `TacticKind` | Inductive | Canonical Lean 4 tactic vocabulary (`Intro`, `Exact`, `Apply`, `Simp`, etc.). |
| `KernelStatus` | Inductive | Lean 4 kernel evaluation outcome (`GoalsClosed`, `SubgoalsOpen`, `KernelError`, `KernelTimeout`). |
| `SearchStatus` | Inductive | Application search state (`Searching`, `SuccessProven`, `BudgetExhausted`, `SearchFailed`). |
| `SearchState` | Structure | Runtime tracking: `(goalCount: int64, depth: int64, steps: int64)`. |
| `SearchLimits` | Structure | Exploration bounds: `(maxSteps: int64, maxDepth: int64)`. |
| `nextSearchStatus` | Definition | Deterministic state transition mapping `(current, feedback) -> next_status`. |
| `isTerminalStatus` | Definition | Boolean predicate identifying terminal search states (`SuccessProven`, `BudgetExhausted`, `SearchFailed`). |

---

## 4. Machine-Checked Theorems (Lean 4 Kernel Verified)

The formal specification proves six fundamental theorems via reflexivity (`rfl`), machine-checked by `lexlean check`, `lexlean build`, and the official Lean 4 kernel:

1. **`calculate_coord_correct`**: Verified reflexivity of checked coordinate arithmetic operations.
2. **`quaternion_identity_w`**: Verified that the $w$ scalar component of the canonical identity rotor $\mathbf{1}_{S^3}$ equals $1$.
3. **`lattice_origin_x0`**: Verified that the $x_0$ coordinate of the $E_8$ lattice origin $\mathbf{0}_{\mathbb{R}^8}$ equals $0$.
4. **`initial_search_state_goals`**: Verified that initializing the search state starts with exactly 1 active goal.
5. **`closed_goals_yields_success`**: Verified that receiving `KernelStatus.GoalsClosed` from LeanDojo when `Searching` unconditionally transitions to `SearchStatus.SuccessProven`.
6. **`terminal_success_is_terminal`**: Verified that `SearchStatus.SuccessProven` evaluates to `true` under `isTerminalStatus`.

---

## 5. Continuous Conformance and Anti-Drift Verification

Every commit to the repository must satisfy:
1. `python3 scripts/lint_tasks.py`: Complete acyclicity of the 12-task DAG and anti-erasure log tracking.
2. `./vendor/prismpm/vendor/lexlean/target/debug/lexlean --project specs/lexlean.toml check`: Complete semantic validation and canonical JSON link check of `specs/src/ProverEngine.lex.tex`.
3. Pre-push gate: Rejection of direct pushes to `leanDojo` or `prover`, requiring audited pull requests.
