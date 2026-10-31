# Prover

An authentic neurosymbolic theorem-proving framework and live Lean 4 compiler verification loop designed as a reference substrate for self-learning and self-improving reasoning environments.

## Overview

The core objective of this repository is to close the loop between language models (e.g., DeepSeek-Prover, Qwen-Coder) and the formal Lean 4 kernel:

```
                  +---------------------------+
                  |  Theorem Goal & Context   |
                  +-------------+-------------+
                                |
                                v
                  +---------------------------+
                  |   Model / Search Policy   |
                  |  - Local LLM Process      |
                  |  - S^3 Tree Rotor & AST   |
                  |  - Speculative Router     |
                  +-------------+-------------+
                                |
                   [Top-k Candidate Tactics]
                                |
                                v
                  +---------------------------+
                  |    Live Lean 4 Compiler   |
                  |      Verification Loop    |
                  +-------------+-------------+
                                |
                +---------------+---------------+
                |                               |
          [✓ Certified]                   [✗ Rejected]
                |                               |
                v                               v
    +-----------------------+       +-----------------------+
    |   Q.E.D. Accepted     |       |  Speculative Fallback |
    | Positive Reward / Data|       | Error Feedback Trace  |
    +-----------------------+       +-----------------------+
```

1. **Model & Generation Layer**: Generates tactic candidates from theorem goals using local LLMs or geometric policy heads.
2. **Speculative Routing**: Ranks candidates with geodesic margins and speculative short-circuiting to minimize compiler invocations.
3. **Formal Verification Loop**: Dispatches proof candidates directly to the live Lean 4 compiler binary (`lean`). Zero mocks, 100% authentic compiler validation.
4. **Self-Improvement Signal**: The deterministic binary ground truth (certified vs rejected) together with Lean 4 error diagnostics supplies the exact environment reward needed for self-improving search, MCTS, and reinforcement learning.

---

## Core Components

### 1. Live Lean 4 Compiler Verification Loop
- **C++ Verification Harness** (`src/eval_live_lean_prover.cpp`): Invokes the real `lean` compiler binary via process execution and pipes, validates output for errors/sorry markers, and executes automated fallback across candidate pools.
- **Python Prover Loop** (`src/eval/authentic_lean_prover.py`): Python verification harness implementing speculative candidate submission, error extraction, and full benchmark tracking against Mathlib theorems.

### 2. Search & Speculative Routing
- **Speculative Geodesic Router** (`src/search/geodesic_search.cpp`, `src/search/geodesic_search.hpp`): Implements dynamic margin gating ($\tau$). When model confidence is high, it submits only the Top-1 candidate; when ambiguous, it prepares a speculative queue that recovers automatically if the primary candidate fails.
- **Macro-Action Decider** (`src/search/macro_decider.cpp`, `src/search/macro_decider.hpp`): Decides between direct tactic closing steps and structural macro rewrites with state feedback.

### 3. S^3 Quaternion Tree Rotor
- **Tree Rotor Operator** (`src/tree_rotor.cpp`, `src/tree_rotor.hpp`): Implements non-commutative unit quaternion ($S^3$) folding over the 11 Lean 4 `Lean.Expr` / AST constructors without tokenization window truncation.

### 4. Local Model Process Control
- **Process Controller** (`src/process.cpp`, `src/process.hpp`): Spawns local inference runtimes (such as `llama-cli` hosting DeepSeek-Prover or Qwen-Coder) via `fork()`, `execv()`, and bidirectional pipes, reading streaming generation and handling process termination upon turn completion.

---

## Quick Start

### Prerequisites
- **Lean 4**: Ensure `lean` is available in your `$PATH` (installed via `elan`).
  ```bash
  lean --version
  ```
- **CMake & C++17 Compiler**:
  ```bash
  sudo apt install cmake build-essential
  ```

### Build (C++)
```bash
mkdir -p build && cd build
cmake ..
make
```

### Run C++ Lean 4 Verification Benchmark
```bash
./my_bin/eval_live_lean_prover
```
Output:
```
===========================================================================
LIVE LEAN 4 KERNEL VERIFICATION BENCHMARK (C++)
Zero mocks. Zero synthetic data. 100% authentic Lean 4 kernel compilation.
===========================================================================

Target: Identity Proposition (prop_id)
Goal:   theorem prop_id (p : Prop) (h : p) : p := by
  Model Top-1 Choice: exact h
  --> Submitting Top-1 ('exact h') to Lean 4 kernel...
  [✓ Q.E.D. CERTIFIED] Lean 4 kernel ACCEPTED proof!
...
LIVE LEAN 4 VERIFICATION RESULTS: 5 / 5 (100.0%) certified.
```

### Run CTest Suite
```bash
ctest --output-on-failure
```

### Run Python Verification Loop
```bash
PYTHONPATH=. python3 src/eval/authentic_lean_prover.py
```

---

## Directory Structure

```
├── CMakeLists.txt              # C++ build configuration and CTest registration
├── README.md                   # Architecture & documentation
├── scripts/                    # Environment and runner scripts
│   ├── configure.sh            # Setup environment
│   ├── prover.sh               # DeepSeek-Prover local CLI launcher
│   ├── qwen.sh                 # Qwen-Coder local CLI launcher
│   └── test.sh                 # Test runner script
├── src/
│   ├── eval_live_lean_prover.cpp   # Live C++ Lean 4 compiler verification
│   ├── eval_least_action_prover.cpp# Least action geodesic prover
│   ├── process.cpp / .hpp      # C++ process control & LLM pipe execution
│   ├── parseJsonl.cpp / .hpp   # JSONL theorem dataset parsing
│   ├── tree_rotor.cpp / .hpp   # S^3 unit quaternion AST tree rotor
│   ├── eval/
│   │   ├── authentic_lean_prover.py# Python live Lean 4 compiler prover
│   │   ├── lean_bench.py       # Benchmark theorems & ground truth
│   │   ├── tactic_eval.py      # Tactic loss and likelihood evaluation
│   │   └── tactic_vocab.py     # Canonical Lean 4 tactic vocabulary
│   ├── ir/                     # Intermediate representation & AST canonicalization
│   ├── lean_bridge/            # Lean 4 AST and Expr bridges
│   └── search/
│       ├── geodesic_search.cpp / .hpp # Speculative geodesic search
│       └── macro_decider.cpp / .hpp   # Macro-action routing decider
├── tests/                      # C++ unit and invariant test suite
│   ├── test_tree_rotor.cpp
│   ├── test_speculative_geodesic_search.cpp
│   └── test_macro_decider.cpp
└── vendor/
    └── benchsets/mathlib4/     # Sample Mathlib 4 validation dataset
```

---

## Reference for Self-Learning Environments

This repository can serve as a minimal, clean blueprint for an agent self-improvement loop:
1. **Action Generation**: An actor model proposes a sequence or tree of Lean 4 tactics.
2. **Environment Execution**: The environment executes `verify_with_real_lean()` using the real Lean 4 compiler.
3. **Reward Signal**: If Lean 4 certifies the proof (`exit_code == 0` without `sorry` or `error`), return positive reward $+1.0$. If rejected, return negative reward and feed Lean compiler error messages back to the actor for reflection/retry.
4. **Policy Update**: Successful trajectories are collected to fine-tune the actor, continuously expanding theorem coverage.
