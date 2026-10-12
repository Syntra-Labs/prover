"""
Benchmark suite of Lean 4 theorems ranging from basic propositional logic
to inductive equality and algebraic properties.
"""

BENCHMARK_THEOREMS = [
    {
        "id": "prop_id",
        "name": "Identity Proposition",
        "category": "propositional",
        "lean_code": "theorem prop_id (p : Prop) (h : p) : p := by",
        "target_proof": "exact h",
        "expected_tactics": ["exact h", "assumption"]
    },
    {
        "id": "prop_and_comm",
        "name": "Conjunction Commutativity",
        "category": "propositional",
        "lean_code": "theorem and_comm (p q : Prop) (h : p ∧ q) : q ∧ p := by",
        "target_proof": "cases h with | intro hp hq => exact ⟨hq, hp⟩",
        "expected_tactics": ["rcases h with ⟨hp, hq⟩; exact ⟨hq, hp⟩", "exact ⟨h.right, h.left⟩"]
    },
    {
        "id": "nat_add_zero",
        "name": "Natural Add Zero",
        "category": "algebraic",
        "lean_code": "theorem nat_add_zero (n : Nat) : n + 0 = n := by",
        "target_proof": "rfl",
        "expected_tactics": ["rfl", "simp", "omega"]
    },
    {
        "id": "nat_zero_add",
        "name": "Natural Zero Add",
        "category": "inductive",
        "lean_code": "theorem nat_zero_add (n : Nat) : 0 + n = n := by",
        "target_proof": "simp",
        "expected_tactics": ["simp", "omega", "induction n <;> simp [*]"]
    },
    {
        "id": "nat_add_comm",
        "name": "Natural Add Commutativity",
        "category": "algebraic",
        "lean_code": "theorem nat_add_comm (n m : Nat) : n + m = m + n := by",
        "target_proof": "omega",
        "expected_tactics": ["omega", "simp [Nat.add_comm]"]
    }
]
