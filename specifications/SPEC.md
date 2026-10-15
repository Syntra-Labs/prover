# Prover Geometric Engine Specification

This document is the normative specification for the **Prover Geometric Engine** (`prover-engine`), bridging continuous geometric flow dynamics with discrete lattice quantization for automated theorem proving in Lean 4.

## 1. Scope and Architecture Overview

The Prover Geometric Engine provides mathematically verified coordination between two core representations:
1. **Continuous Phase Space ($\mathbb{T}^d \times \mathbb{R}^d$)**: Symplectic manifold trajectory representations of Lean 4 proof states (via Geometric Flow Networks / ISN).
2. **Discrete Algebraic Lattice ($S^3 \times E_8$)**: Conway-Sloane $E_8$ root lattice Voronoi cells and unit quaternion associative scans (via UOR-R4 214M).

In accordance with the PrismPM standard:
- Application behavior and geometric transformations are derived strictly from formal `.lex.tex` specifications.
- Mathematical invariants are formally proven in Lean 4 (`leanprover/lean4:v4.32.1`) prior to artifact emission.
- All release artifacts (Rust crates, Wasm guests) are emitted deterministically with content-addressed provenance.

```text
Lean 4 Goal State (Syntax / AST)
       │
       ▼
Symplectic Trajectory Manifold (GFN: Position q, Velocity p on T^d)
       │
       ▼
Hamiltonian Action Minimization (Discrete Lagrangian Path)
       │
       ▼
Conway-Sloane E8 Lattice Snap (Union of D8 and D8 + 1/2)
       │
       ▼
Discrete Tactic Candidate Selection (Top-k Mathlib 4 Tactics)
```

---

## 2. Formal Types and Data Contracts

### 2.1 E8 Vector and Codeword
An 8-dimensional Euclidean vector $\mathbf{x} \in \mathbb{R}^8$ snapped to the nearest point in the Gosset root lattice $E_8$:
- **Definition**: $E_8 = D_8 \cup \left(D_8 + \frac{1}{2}\mathbf{1}\right)$
  where $D_8 = \left\{ \mathbf{x} \in \mathbb{Z}^8 : \sum_{i=1}^8 x_i \equiv 0 \pmod 2 \right\}$.
- **RMS Normalization**:
  $$\text{rms}(\mathbf{x}) = \sqrt{\frac{1}{8}\sum_{i=1}^8 x_i^2}$$
  $$\mathbf{x}_{\text{scaled}} = \text{scale} \cdot \frac{\mathbf{x}}{\max(\text{rms}(\mathbf{x}), 10^{-7})}$$
- **Root Codebook ($2E_8$)**:
  - Type 1 Roots: All permutations of $(\pm 2, \pm 2, 0, 0, 0, 0, 0, 0)$ (norm squared = 8, 112 roots).
  - Type 2 Roots: $(\pm 1, \pm 1, \dots, \pm 1)$ with an even number of minus signs (norm squared = 8, 128 roots).
  - Total Minimal Roots: 240.

### 2.2 Proof State Manifold Embedding
- Continuous embedding dimension: $D = 64$ (composed of 8 consecutive 8D blocks $\mathbf{b}_0, \dots, \mathbf{b}_7$).
- Discrete cell key: $\mathbf{k} = \left(\text{snap}_{E8}(\mathbf{b}_0), \dots, \text{snap}_{E8}(\mathbf{b}_7)\right) \in (E_8)^8$.

### 2.3 Tactic Discrete Action Space
- Set of 50 canonical Mathlib 4 tactic primitives (e.g., `intro`, `exact`, `apply`, `simp`, `ring`, `linarith`, `omega`, `aesop`, `cases`, `induction`).
- Action assignment via Voronoi cell distance metric minimizing empirical divergence on validation proof trees.

---

## 3. Mathematical Invariants (Lean 4 Verified Claims)

The formal specification mandates proof of three core invariants in Lean 4:

1. **Nearest Lattice Point Optimality (Conway-Sloane Theorem)**:
   For any $\mathbf{y} \in \mathbb{R}^8$, the decoded point $\hat{\mathbf{x}} = \text{decode}_{E8}(\mathbf{y})$ satisfies:
   $$\forall \mathbf{z} \in E_8, \quad \|\mathbf{y} - \hat{\mathbf{x}}\|_2 \le \|\mathbf{y} - \mathbf{z}\|_2$$

2. **Symplectic Phase Volume Preservation (Liouville's Invariant)**:
   The flow operator $\Phi_t: (q, p) \mapsto (q', p')$ preserves the differential 2-form $\omega = \sum dq_i \wedge dp_i$:
   $$\det \left( \frac{\partial (q', p')}{\partial (q, p)} \right) = 1$$

3. **Quaternion Normalization (Hopf $S^3$ Invariant)**:
   For all rotor compositions $R = q_1 \otimes q_2$, if $\|q_1\| = 1$ and $\|q_2\| = 1$, then:
   $$\|R\| = 1$$

---

## 4. Acceptance Test Vectors

Every verified build must validate against deterministic acceptance test vectors:

| Vector ID | Input Tensor ($\mathbf{x}_0 \dots \mathbf{x}_7$) | Scale | Expected Snapped Point | Norm $L_2^2$ |
| :--- | :--- | :--- | :--- | :--- |
| `vec-e8-zero` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | 0.5 | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | 0.0 |
| `vec-e8-half` | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]` | 0.5 | `[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` | 2.0 |
| `vec-e8-root` | `[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | 0.5 | `[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` | 4.0 |

---

## 5. Build and Conformance Pipeline

Execution of `prismpm verify` on this project requires:
1. Replay and proof checking of all theorems via `leanchecker`.
2. Zero handwritten unsafe code or raw pointer manipulations in generated crates.
3. Passing 100% of acceptance vectors with zero rounding epsilon tolerance.
