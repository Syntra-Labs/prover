#include "oracle/code_oracle.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>

extern "C" {
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
}

namespace syn {

CodeExecutionResult CodeOracle::execute(
    const std::string& command,
    const std::string& working_dir,
    int timeout_sec
) {
    CodeExecutionResult res;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Construct command wrapped in working directory cd and output capture
    std::string full_cmd = "cd \"" + working_dir + "\" && " + command + " 2>&1";

    std::array<char, 512> buffer;
    FILE* pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) {
        res.exit_code = -1;
        res.success = false;
        res.stderr_output = "Failed to spawn popen pipe";
        return res;
    }

    std::ostringstream out_stream;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        out_stream << buffer.data();
    }

    int status = pclose(pipe);
    if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        res.exit_code = 128 + WTERMSIG(status);
    } else {
        res.exit_code = status;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    res.stdout_output = out_stream.str();
    res.success = (res.exit_code == 0);

    // Parse standard test results (Rust cargo test, pytest, ctest)
    // 1. Rust: "test result: ok. 14 passed; 0 failed"
    std::regex rust_regex(R"(test result: (ok|FAILED)\. (\d+) passed; (\d+) failed)");
    std::smatch match;
    if (std::regex_search(res.stdout_output, match, rust_regex)) {
        res.tests_passed = std::stoull(match[2].str());
        res.tests_failed = std::stoull(match[3].str());
    }

    // 2. Pytest: "14 passed, 2 warnings in 12.14s"
    std::regex pytest_regex(R"((\d+) passed(?:, (\d+) failed)?)");
    if (std::regex_search(res.stdout_output, match, pytest_regex)) {
        res.tests_passed = std::stoull(match[1].str());
        if (match[2].matched) {
            res.tests_failed = std::stoull(match[2].str());
        }
    }

    return res;
}

CodeExecutionResult CodeOracle::run_cargo_test(
    const std::string& manifest_path,
    const std::string& package,
    const std::string& test_filter,
    int timeout_sec
) {
    std::string cmd = "cargo test --manifest-path \"" + manifest_path + "\"";
    if (!package.empty()) {
        cmd += " -p " + package;
    }
    if (!test_filter.empty()) {
        cmd += " -- " + test_filter;
    }
    return execute(cmd, ".", timeout_sec);
}

CodeExecutionResult CodeOracle::run_ctest(
    const std::string& build_dir,
    const std::string& test_name,
    int timeout_sec
) {
    std::string cmd = "ctest --output-on-failure";
    if (!test_name.empty()) {
        cmd += " -R " + test_name;
    }
    return execute(cmd, build_dir, timeout_sec);
}

bool CodeOracle::verify_contract(
    const HoareContract& contract,
    CodeExecutionResult& out_result,
    const std::string& working_dir
) {
    out_result = execute(contract.test_command, working_dir);
    return out_result.success && out_result.tests_failed == 0;
}

} // namespace syn
