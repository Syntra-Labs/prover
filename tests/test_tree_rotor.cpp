#include "geometry/tree_rotor.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

void test_quaternion_algebra() {
    using namespace syn;
    Quat q1(1.0, 0.0, 0.0, 0.0);
    assert(std::abs(q1.norm() - 1.0) < 1e-9);

    Quat i(0.0, 1.0, 0.0, 0.0);
    Quat j(0.0, 0.0, 1.0, 0.0);
    Quat k = i.mul(j);
    // i * j = k
    assert(std::abs(k.w) < 1e-9);
    assert(std::abs(k.x) < 1e-9);
    assert(std::abs(k.y) < 1e-9);
    assert(std::abs(k.z - 1.0) < 1e-9);

    // j * i = -k (non-commutative)
    Quat minus_k = j.mul(i);
    assert(std::abs(minus_k.z - (-1.0)) < 1e-9);

    // Geodesic distance on S^3
    double d_self = i.geodesic_dist(i);
    assert(std::abs(d_self) < 1e-9);

    double d_ij = i.geodesic_dist(j);
    assert(std::abs(d_ij - 3.141592653589793) < 1e-4);
    std::cout << "[PASS] Quaternion Hamilton algebra & S^3 metric verified." << std::endl;
}

void test_zero_window_truncation() {
    using namespace syn;
    TreeRotorOperator op;

    // Build a deep DAG with 1023 nodes (depth 10 complete binary tree of Applications)
    std::vector<ExprNode> nodes;
    nodes.reserve(1023);

    // 512 leaves (constants and bvars)
    for (uint32_t i = 0; i < 512; ++i) {
        if (i % 2 == 0) {
            nodes.push_back({i, ExprKind::CONST, 0, "Const_" + std::to_string(i), {}});
        } else {
            nodes.push_back({i, ExprKind::BVAR, i, "", {}});
        }
    }

    // Binary application layers
    uint32_t prev_start = 0;
    uint32_t prev_count = 512;
    uint32_t curr_id = 512;

    while (prev_count > 1) {
        uint32_t next_count = prev_count / 2;
        for (uint32_t j = 0; j < next_count; ++j) {
            uint32_t left_child = prev_start + 2 * j;
            uint32_t right_child = prev_start + 2 * j + 1;
            nodes.push_back({curr_id + j, ExprKind::APP, 0, "", {left_child, right_child}});
        }
        prev_start = curr_id;
        prev_count = next_count;
        curr_id += next_count;
    }

    uint32_t root_id = nodes.back().id;
    assert(nodes.size() == 1023);

    // Fold entire tree
    size_t visited = 0;
    Quat root_q = op.fold_dag(nodes, root_id, &visited);

    // Verify it reached S^3 with norm 1.0
    assert(std::abs(root_q.norm() - 1.0) < 1e-9);

    // window_truncation_rate is strictly 0.0 (all 1023 nodes visited and folded)
    double truncation_rate = (nodes.empty()) ? 0.0 : (1.0 - static_cast<double>(visited) / static_cast<double>(nodes.size()));
    assert(truncation_rate == 0.0);
    std::cout << "[PASS] Zero window truncation verified on 1,023-node AST DAG (window_truncation_rate = 0.0)." << std::endl;
}

void test_non_commutative_asymmetry() {
    using namespace syn;
    TreeRotorOperator op;

    ExprNode a{1, ExprKind::CONST, 0, "Nat.add", {}};
    ExprNode b{2, ExprKind::BVAR, 0, "", {}};

    double asym = op.non_commutative_asymmetry(a, b);
    std::cout << "[ASYMMETRY] d_S3(App(A, B), App(B, A)) = " << asym << " rad" << std::endl;

    assert(asym > 0.01); // Strictly asymmetric
    std::cout << "[PASS] Non-commutative asymmetry verified (asymmetry > 0.0)." << std::endl;
}

void test_c_abi() {
    double asym = 0.0;
    double trunc = 1.0;
    int rc = rotor_check_invariants(&asym, &trunc);
    assert(rc == 0);
    assert(asym > 0.0);
    assert(trunc == 0.0);
    std::cout << "[PASS] C-ABI invariants check passed." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "TREE ROTOR OPERATOR TEST HARNESS" << std::endl;
    std::cout << "========================================" << std::endl;
    test_quaternion_algebra();
    test_zero_window_truncation();
    test_non_commutative_asymmetry();
    test_c_abi();
    std::cout << "========================================" << std::endl;
    std::cout << "ALL S^3 TREE ROTOR OPERATOR TESTS PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
