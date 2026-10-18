#include "../src/search/macro_decider.hpp"
#include <cassert>
#include <iostream>
#include <unordered_set>

using namespace syn;

void test_metric_and_head_routing() {
    MacroActionDecider decider(0.5);
    Quat z_near = Quat(0.99, 0.05, 0.05, 0.05).normalize();
    Quat z_far = Quat(0.0, 1.0, 0.0, 0.0).normalize();

    assert(decider.decide_kind(z_near, "P ∧ Q") == MacroActionKind::Tier2A_DirectTerminating);
    assert(decider.decide_kind(z_far, "P ∧ Q") == MacroActionKind::Tier2B_ForwardInvention);

    // Reflexive and terminal heads route to Tier 2A even at large geodesic distance
    assert(decider.decide_kind(z_far, "n = n") == MacroActionKind::Tier2A_DirectTerminating);
    assert(decider.decide_kind(z_far, "⊢ a + b = a + b") == MacroActionKind::Tier2A_DirectTerminating);
    assert(decider.decide_kind(z_far, "rfl") == MacroActionKind::Tier2A_DirectTerminating);
    assert(decider.decide_kind(z_far, "True") == MacroActionKind::Tier2A_DirectTerminating);
    assert(decider.decide_kind(z_far, "n = 0") == MacroActionKind::Tier2B_ForwardInvention);
    std::cout << "[PASS] Metric convergence and head routing verified." << std::endl;
}

void test_feedback_loop() {
    MacroActionDecider decider(0.5);
    GoalState st(Quat(0.99, 0.05, 0.05, 0.05).normalize(), "P ∧ Q");

    auto reject_all = [](const std::string&) { return false; };
    auto d_reject = decider.route_with_feedback(st, reject_all);
    assert(d_reject.fallback_triggered && d_reject.kind == MacroActionKind::Tier2B_ForwardInvention);
    assert(d_reject.candidate_tactics[0] == "apply");

    auto accept_rfl = [](const std::string& t) { return t == "rfl"; };
    auto d_accept = decider.route_with_feedback(GoalState(st.z, "x = x"), accept_rfl);
    assert(!d_accept.fallback_triggered && d_accept.kind == MacroActionKind::Tier2A_DirectTerminating);

    assert(decider.route_step(st, true, false) == MacroActionKind::Tier2B_ForwardInvention);
    assert(decider.route_step(st, false, true) == MacroActionKind::Tier2B_ForwardInvention);
    assert(decider.route_step(st, false, false) == MacroActionKind::Tier2A_DirectTerminating);
    std::cout << "[PASS] Feedback loop and fallback verified." << std::endl;
}

void test_falsifications() {
    const auto& t2a = MacroActionDecider::get_tier2a_tactics();
    const auto& t2b = MacroActionDecider::get_tier2b_tactics();
    assert(t2a.size() == 7 && t2b.size() == 5);
    std::unordered_set<std::string> s2a(t2a.begin(), t2a.end());
    for (const auto& tac : t2b) assert(s2a.find(tac) == s2a.end());

    MacroActionDecider decider(0.5);
    GoalState div_st(Quat(0.2, 0.8, 0.4, 0.4).normalize(), "⊢ n ≤ n + 1");
    auto d = decider.route(div_st);
    assert(d.kind == MacroActionKind::Tier2B_ForwardInvention && !d.fallback_triggered);
    std::cout << "[PASS] Falsification: disjoint vocab and non-terminal divergence verified." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "TIER 1 MACRO-ACTION DECIDER TESTS (C++)" << std::endl;
    std::cout << "========================================" << std::endl;
    test_metric_and_head_routing();
    test_feedback_loop();
    test_falsifications();
    std::cout << "========================================" << std::endl;
    std::cout << "ALL MACRO-ACTION DECIDER TESTS PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
