#include "search/geodesic_search.hpp"
#include <cassert>
#include <iostream>
#include <vector>

void test_margin_gating_collapse_to_top1() {
    using namespace syn;
    SpeculativeGeodesicRouter router(0.05);
    std::vector<double> tangent = {1.0, 0.0, 0.0};
    std::unordered_map<std::string, std::vector<double>> actions = {
        {"exact", {0.99, 0.01, 0.0}},
        {"simp",  {0.50, 0.50, 0.0}},
        {"omega", {0.10, 0.90, 0.0}}
    };

    auto ranked = router.rank_candidates(tangent, actions);
    auto res = router.select_speculative_queue(ranked);
    const auto& queue = std::get<0>(res);
    double margin = std::get<1>(res);
    bool is_ambiguous = std::get<2>(res);

    assert(!is_ambiguous);
    assert(queue.size() == 1);
    assert(queue[0].name == "exact");
    assert(margin > 0.05);
    std::cout << "[PASS] Dynamic margin gating collapses to Top-1 when margin >= tau." << std::endl;
}

void test_margin_gating_triggers_top2() {
    using namespace syn;
    SpeculativeGeodesicRouter router(0.05);
    std::vector<double> tangent = {1.0, 0.0, 0.0};
    std::unordered_map<std::string, std::vector<double>> actions = {
        {"distractor_simp",    {0.985, 0.05, 0.0}},
        {"ground_truth_exact", {0.980, 0.06, 0.0}},
        {"omega",              {0.20, 0.80, 0.0}}
    };

    auto ranked = router.rank_candidates(tangent, actions);
    auto res = router.select_speculative_queue(ranked);
    const auto& queue = std::get<0>(res);
    double margin = std::get<1>(res);
    bool is_ambiguous = std::get<2>(res);

    assert(is_ambiguous);
    assert(queue.size() == 2);
    assert(queue[0].name == "distractor_simp");
    assert(queue[1].name == "ground_truth_exact");
    assert(margin < 0.05);
    std::cout << "[PASS] Angular ambiguity triggers Top-2 speculative queue when margin < tau." << std::endl;
}

void test_speculative_short_circuiting_saves_repl_call() {
    using namespace syn;
    SpeculativeGeodesicRouter router(0.10); // wide threshold to force top-2 queue
    SpeculativeGeodesicProver prover(router, 10);

    std::unordered_map<std::string, std::vector<double>> actions = {
        {"candidate_1", {0.98, 0.02}},
        {"candidate_2", {0.95, 0.05}}
    };

    std::vector<std::string> kernel_calls;
    auto mock_kernel = [&](const std::string&, const std::string& tactic)
        -> std::pair<KernelStatus, std::vector<std::string>> {
        kernel_calls.push_back(tactic);
        if (tactic == "candidate_1") {
            return {KernelStatus::GoalsClosed, {}};
        }
        return {KernelStatus::KernelError, {}};
    };

    auto policy_fn = [](const std::string&) -> std::vector<double> {
        return {1.0, 0.0};
    };

    auto result = prover.search("⊢ P", policy_fn, actions, mock_kernel);

    assert(result.status == SearchStatus::SuccessProven);
    assert(result.proof_steps.size() == 1 && result.proof_steps[0] == "candidate_1");
    assert(result.kernel_calls == 1);
    assert(result.repl_calls_saved == 1);
    assert(kernel_calls.size() == 1 && kernel_calls[0] == "candidate_1");
    std::cout << "[PASS] Speculative short-circuiting skips candidate #2 on kernel success." << std::endl;
}

void test_speculative_fallback_on_shadowed_distractor() {
    using namespace syn;
    SpeculativeGeodesicRouter router(0.05);
    SpeculativeGeodesicProver prover(router, 10);

    std::unordered_map<std::string, std::vector<double>> actions = {
        {"shadow_distractor", {0.995, 0.05}},
        {"valid_lemma",       {0.985, 0.06}}
    };

    std::vector<std::string> kernel_calls;
    auto mock_kernel = [&](const std::string&, const std::string& tactic)
        -> std::pair<KernelStatus, std::vector<std::string>> {
        kernel_calls.push_back(tactic);
        if (tactic == "shadow_distractor") {
            return {KernelStatus::KernelError, {}};
        }
        if (tactic == "valid_lemma") {
            return {KernelStatus::GoalsClosed, {}};
        }
        return {KernelStatus::KernelError, {}};
    };

    auto policy_fn = [](const std::string&) -> std::vector<double> {
        return {1.0, 0.0};
    };

    auto result = prover.search("⊢ P", policy_fn, actions, mock_kernel);

    assert(result.status == SearchStatus::SuccessProven);
    assert(result.proof_steps.size() == 1 && result.proof_steps[0] == "valid_lemma");
    assert(result.kernel_calls == 2);
    assert(kernel_calls.size() == 2);
    std::cout << "[PASS] Speculative fallback successfully recovers from candidate #1 rejection." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "SPECULATIVE GEODESIC SEARCH (C++) TESTS" << std::endl;
    std::cout << "========================================" << std::endl;
    test_margin_gating_collapse_to_top1();
    test_margin_gating_triggers_top2();
    test_speculative_short_circuiting_saves_repl_call();
    test_speculative_fallback_on_shadowed_distractor();
    std::cout << "========================================" << std::endl;
    std::cout << "ALL SPECULATIVE GEODESIC SEARCH TESTS PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
