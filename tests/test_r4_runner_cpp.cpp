#include "geometry/r4_runner.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "===========================================\n";
    std::cout << "RUNNING C++ R4 RUNNER TEST\n";
    std::cout << "===========================================\n";

    // Small test config to verify inference math quickly
    syn::R4Config cfg;
    cfg.width = 64;   // 16 lanes
    cfg.heads = 4;    // 16 dim per head
    cfg.context = 32;
    cfg.vocab_size = 128;
    cfg.pattern = "ra"; // 1 recurrence layer, 1 addressed read layer

    syn::R4RunnerCpp runner(cfg);
    runner.load_dummy_weights();

    std::vector<uint16_t> tokens = {1, 5, 12, 42};
    std::vector<float> hidden, logits;
    runner.forward(tokens, hidden, logits);

    assert(hidden.size() == static_cast<size_t>(cfg.width));
    assert(logits.size() == static_cast<size_t>(cfg.vocab_size));

    std::cout << "[PASS] Forward pass produced hidden state of size: " << hidden.size() << "\n";
    std::cout << "[PASS] Forward pass produced logits of size: " << logits.size() << "\n";

    auto embed_64 = runner.embed_goal_64d(tokens);
    assert(embed_64.size() == 64);
    std::cout << "[PASS] Goal 64D embedding extracted successfully!\n";

    std::cout << "ALL C++ R4 RUNNER TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
