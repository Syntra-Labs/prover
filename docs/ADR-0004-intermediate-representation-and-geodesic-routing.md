# ADR-0004: Intermediate Representation (IR) State-Action Conditioning and Geodesic Routing

- **Status:** Accepted
- **Date:** 2026-10-17
- **Deciders:** Syn-Prover Architecture Team
- **Context:** Transition from Stage 1 (Mathlib Language Modeling) to Stage 2 (Goal-Conditioned Tactic Proving)

---

## 1. Context and Problem Statement

Stage 1 training successfully validated the geometric associative scan of UOR-R4 (8M parameters, 288 width, pattern `rrarra`), dropping held-out Mathlib 4 validation NLL from 8.32 (uniform baseline) down to 2.08 over 808,192 tokens.

However, continuing to train with raw text next-token prediction across arbitrary `.lean` source files suffers from fundamental inefficiencies:
1. **Surface Syntax Noise:** >60% of tokens in raw Mathlib files comprise imports, module docstrings, formatting whitespace, and arbitrary identifier names that do not contribute to mathematical deduction.
2. **Alpha-Equivalence Blindness:** Distinct dummy bound variable names (e.g. `(x : Nat)` vs `(y : Nat)`) are treated as separate tokens, forcing the model to expend parameter capacity learning equivalence relations that are trivial in Lean's core representations.
3. **Mismatched Optimization Target:** Standard next-token loss penalizes the network for failing to predict goal statements or hypotheses, whereas an automated theorem prover only needs to predict the *deductive action* that advances the goal toward QED.

---

## 2. Decision

We pivot the training and inference pipeline from unstructured next-token text prediction to an **Intermediate Representation (IR) State-Action Framework**:

### 2.1 Level 1: Supervised State-Action Conditioning with Response Masking
- The training corpus is structured as discrete state-action pairs:
  $$\text{Prompt: } [\text{Goal State } \mathcal{S}_t] \implies \text{Response: } [\text{Tactic } \mathcal{A}_t]$$
- Training utilizes **Response Masking** (`train.mask`), setting cross-entropy loss to $0$ on prompt tokens ($\mathcal{S}_t$) and $1$ on tactic tokens ($\mathcal{A}_t$). 100% of gradient updates are focused on valid deduction steps.

### 2.2 Level 2: Kernel AST / de Bruijn Normalization
- Proof goal states are normalized to the Lean 4 Kernel expression representation (`Expr`), mapping bound variables to de Bruijn indices (`BVar`).
- This guarantees absolute alpha-invariance: theorems with identical proof structures map to identical semantic embeddings regardless of surface variable names.

### 2.3 Level 3: Speculative Adaptive-Margin Geodesic Snapping
- In place of unguided tree search (BFS/MCTS) or static Top-2 branching:
  1. The continuous geometric policy predicts a tangent direction vector $\mathbf{v} \in \mathbb{R}^d$ towards the QED state.
  2. The vector is projected onto the discrete action vocabulary via cosine alignment:
     $$\Delta \cos = \cos(\mathbf{v}, \mathbf{e}_{(1)}) - \cos(\mathbf{v}, \mathbf{e}_{(2)})$$
  3. **Adaptive Gating:** If $\Delta \cos \ge \tau$ (high confidence), collapse to Top-1. If $\Delta \cos < \tau$ (decision boundary ambiguity), admit Top-2.
  4. **Speculative Short-Circuiting:** Evaluate candidate #1 in the Lean 4 REPL first; if accepted with subgoals reduced or closed, immediately drop candidate #2 to conserve kernel elaboration time.

---

## 3. Consequences and Invariants

- **Sample Efficiency:** Gradients directly supervise tactic policy without wasting capacity on Lean surface syntax.
- **Search Complexity:** Effective branching factor drops to $b_{\text{eff}} \approx 1.05\text{--}1.15$ while preserving the 100% path discovery rate proven in `tests/geometric_falsification/test_geodesic_discretization.py`.
- **Specification Compliance:** Fully aligns with the formal discrete state machine specified in `specs/src/ProverEngine.lex.tex` (`TacticKind`, `SearchStatus`, and `KernelStatus`).
