// Resolution record: unify-bundle, least-action-prover, kernel-expr-ir, policy-r4
// retired during pivot from Lean 4 theorem proving to codebase manifold & deterministic code oracle.
#include "oracle/code_oracle.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "========================================\n";
    std::cout << "CODE ORACLE TEST SUITE\n";
    std::cout << "========================================\n";

    // 1. Basic command execution test
    auto res = syn::CodeOracle::execute("echo 'hello oracle'", ".");
    assert(res.success == true);
    assert(res.exit_code == 0);
    assert(res.stdout_output.find("hello oracle") != std::string::npos);
    std::cout << "[PASS] Basic subprocess execution & output capture verified ("
              << res.elapsed_ms << " ms).\n";

    // 2. Failure code handling test
    auto fail_res = syn::CodeOracle::execute("false", ".");
    assert(fail_res.success == false);
    assert(fail_res.exit_code != 0);
    std::cout << "[PASS] Deterministic error propagation verified (exit code "
              << fail_res.exit_code << ").\n";

    // 3. Hoare contract verification test
    syn::HoareContract contract;
    contract.id = "test-echo-gate";
    contract.test_command = "echo '1 passed' && true";
    contract.action_desc = "Verify echo contract";
    contract.ensures_spec = "Outputs 1 passed and returns 0";

    syn::CodeExecutionResult contract_res;
    bool contract_ok = syn::CodeOracle::verify_contract(contract, contract_res);
    assert(contract_ok == true);
    assert(contract_res.tests_passed == 1);
    std::cout << "[PASS] Hoare code contract gate verification verified.\n";

    std::cout << "========================================\n";
    std::cout << "ALL CODE ORACLE TESTS PASSED\n";
    std::cout << "========================================\n";
    return 0;
}
