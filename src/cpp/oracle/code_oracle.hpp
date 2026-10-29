#ifndef SYN_CODE_ORACLE_HPP
#define SYN_CODE_ORACLE_HPP

#include <chrono>
#include <string>
#include <vector>

namespace syn {

struct CodeExecutionResult {
    int exit_code{-1};
    bool success{false};
    double elapsed_ms{0.0};
    std::string stdout_output;
    std::string stderr_output;
    size_t tests_passed{0};
    size_t tests_failed{0};
};

struct HoareContract {
    std::string id;
    std::vector<std::string> requires_deps;
    std::string test_command;
    std::string action_desc;
    std::string ensures_spec;
};

class CodeOracle {
public:
    // Execute an arbitrary shell/build command in a child process
    static CodeExecutionResult execute(
        const std::string& command,
        const std::string& working_dir = ".",
        int timeout_sec = 60
    );

    // Execute cargo test on a Rust crate/workspace deterministically
    static CodeExecutionResult run_cargo_test(
        const std::string& manifest_path = "Cargo.toml",
        const std::string& package = "",
        const std::string& test_filter = "",
        int timeout_sec = 120
    );

    // Execute cmake / ctest on a C++ build directory
    static CodeExecutionResult run_ctest(
        const std::string& build_dir = "build",
        const std::string& test_name = "",
        int timeout_sec = 60
    );

    // Validate a Hoare contract (verifies pre-checks, runs test, ensures postcondition holds)
    static bool verify_contract(
        const HoareContract& contract,
        CodeExecutionResult& out_result,
        const std::string& working_dir = "."
    );
};

} // namespace syn

#endif // SYN_CODE_ORACLE_HPP
