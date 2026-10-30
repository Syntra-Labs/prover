#include "ir/kernel_expr.hpp"
#include "ir/tactic_ast.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <unordered_map>

void test_decl_to_pi_type() {
    std::cout << "[TEST] decl_to_pi_type ... ";
    std::string decl = "theorem prop_id (p : Prop) (h : p) : p := by";
    auto res = syn::ir::decl_to_pi_type(decl);
    assert(res.first == "∀ (p : Prop) (h : p), p");
    assert(res.second == 2);

    decl = "example (n : Nat) : n + 0 = n := by";
    res = syn::ir::decl_to_pi_type(decl);
    assert(res.first == "∀ (n : Nat), n + 0 = n");
    assert(res.second == 1);

    decl = "theorem t : True := by";
    res = syn::ir::decl_to_pi_type(decl);
    assert(res.first == "True");
    assert(res.second == 0);
    std::cout << "PASSED\n";
}

void test_kernel_state_dag() {
    std::cout << "[TEST] kernel_state_dag ... ";
    using namespace syn::ir;
    // Build a small DAG: (app (app Eq a) b)
    std::vector<Node> nodes = {
        Node(static_cast<uint8_t>(ExprKind::CONST), "Eq", {}),       // 0
        Node(static_cast<uint8_t>(ExprKind::BVAR), "0", {}),         // 1
        Node(static_cast<uint8_t>(ExprKind::APP), "", {0, 1}),       // 2
        Node(static_cast<uint8_t>(ExprKind::BVAR), "1", {}),         // 3
        Node(static_cast<uint8_t>(ExprKind::APP), "", {2, 3})        // 4
    };
    std::vector<Hyp> hyps = {
        Hyp("fv0", "h", 1, true)
    };
    KernelState st(nodes, hyps, 4);

    assert(st.head(4) == 0);
    assert(st.head(2) == 0);
    assert(st.head(1) == 1);

    auto subs = st.subterms(4);
    assert(subs.size() == 5);
    assert(subs[0] == 4);

    std::string rendered = st.render(4);
    assert(rendered == "((Eq #0) #1)");
    std::cout << "PASSED\n";
}

void test_tactic_ast() {
    std::cout << "[TEST] tactic_ast ... ";
    using namespace syn::ir;
    std::unordered_map<std::string, std::string> sigma = {
        {"my_hyp", "h0"},
        {"x", "v0"}
    };

    auto ops = valid_ast_candidates(NodeSort::OP, sigma);
    assert(!ops.empty());

    auto hyps = valid_ast_candidates(NodeSort::HYP, sigma);
    assert(hyps.size() == 1);
    assert(hyps[0].token == "IDENT:h0");

    auto vars = valid_ast_candidates(NodeSort::VAR, sigma);
    assert(vars.size() == 1);
    assert(vars[0].token == "IDENT:v0");

    // Test lower_ast_to_lean: exact h0 -> exact my_hyp
    std::vector<ASTNode> exact_nodes = {
        ASTNode("NODE:Lean.Parser.Tactic.exact", NodeSort::OP, {NodeSort::TERM}),
        ASTNode("IDENT:h0", NodeSort::TERM, {})
    };
    std::string lowered = lower_ast_to_lean(exact_nodes, sigma);
    assert(lowered == "exact my_hyp");

    // Test lower_ast_to_lean: rw [← h0] -> rw [← my_hyp]
    std::vector<ASTNode> rw_nodes = {
        ASTNode("NODE:Lean.Parser.Tactic.rwSeq", NodeSort::OP, {}),
        ASTNode("ATOM:←", NodeSort::DIR, {}),
        ASTNode("IDENT:h0", NodeSort::TERM, {}),
        ASTNode("NODE:Lean.Parser.Tactic.locationWildcard", NodeSort::LOC, {})
    };
    lowered = lower_ast_to_lean(rw_nodes, sigma);
    assert(lowered == "rw [← my_hyp]");

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "RUNNING C++ KERNEL EXPR & TACTIC AST TESTS\n";
    std::cout << "===========================================\n";
    test_decl_to_pi_type();
    test_kernel_state_dag();
    test_tactic_ast();
    std::cout << "ALL C++ IR TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
