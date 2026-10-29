#include "tree_rotor.hpp"
#include <functional>

namespace syn {

static Quat make_rotor(double angle_rad, double ux, double uy, double uz) {
    double norm_u = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (norm_u < 1e-12) {
        return Quat::identity();
    }
    double half = angle_rad * 0.5;
    double s = std::sin(half) / norm_u;
    return Quat(std::cos(half), s * ux, s * uy, s * uz).normalize();
}

RotorCodebook::RotorCodebook() {
    // 11 distinct constructor anchors on S^3
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < 11; ++i) {
        double theta = (i + 1) * (2.0 * pi / 13.0);
        double ux = std::cos(theta * 1.5);
        double uy = std::sin(theta * 2.1);
        double uz = std::cos(theta * 0.7);
        constr_rotors[i] = make_rotor(theta, ux, uy, uz);
    }

    // Branch rotors: strictly asymmetric Left vs Right
    r_left = make_rotor(pi / 3.0, 1.0, 0.0, 0.0);
    r_right = make_rotor(pi / 4.0, 0.0, 1.0, 0.0);
    r_dom = make_rotor(pi / 5.0, 0.0, 0.0, 1.0);
    r_body = make_rotor(2.0 * pi / 5.0, 1.0, 1.0, 0.0);
    r_let_type = make_rotor(pi / 6.0, 1.0, 0.0, 1.0);
    r_let_val = make_rotor(pi / 7.0, 0.0, 1.0, 1.0);
    r_let_body = make_rotor(pi / 8.0, 1.0, 1.0, 1.0);
}

TreeRotorOperator::TreeRotorOperator(RotorCodebook codebook)
    : codebook_(codebook) {}

Quat TreeRotorOperator::fold_dag(const std::vector<ExprNode>& nodes, uint32_t root_id, size_t* out_visited_count) const {
    std::unordered_map<uint32_t, const ExprNode*> node_map;
    node_map.reserve(nodes.size());
    for (const auto& n : nodes) {
        node_map[n.id] = &n;
    }
    std::unordered_map<uint32_t, Quat> memo;
    Quat res = fold_recursive(node_map, root_id, memo);
    if (out_visited_count) {
        *out_visited_count = memo.size();
    }
    return res;
}

Quat TreeRotorOperator::fold_recursive(
    const std::unordered_map<uint32_t, const ExprNode*>& node_map,
    uint32_t curr_id,
    std::unordered_map<uint32_t, Quat>& memo
) const {
    auto m_it = memo.find(curr_id);
    if (m_it != memo.end()) {
        return m_it->second;
    }

    auto it = node_map.find(curr_id);
    if (it == node_map.end()) {
        return Quat::identity();
    }
    const ExprNode& node = *(it->second);
    Quat res = Quat::identity();
    uint8_t k_idx = static_cast<uint8_t>(node.kind);
    Quat base_anchor = (k_idx < 11) ? codebook_.constr_rotors[k_idx] : Quat::identity();

    switch (node.kind) {
        case ExprKind::BVAR:
        case ExprKind::FVAR:
        case ExprKind::MVAR:
        case ExprKind::SORT: {
            double angle = (node.payload + 1) * 0.314159;
            Quat p_rotor = make_rotor(angle, 0.577, 0.577, 0.577);
            res = base_anchor.rotate(p_rotor);
            break;
        }
        case ExprKind::CONST: {
            uint64_t h = std::hash<std::string>{}(node.symbol);
            double angle = (h % 1000) * (6.2831853 / 1000.0);
            Quat c_rotor = make_rotor(angle, 0.8, 0.6, 0.0);
            res = base_anchor.rotate(c_rotor);
            break;
        }
        case ExprKind::APP: {
            if (node.children.size() >= 2) {
                Quat q_fn = fold_recursive(node_map, node.children[0], memo);
                Quat q_arg = fold_recursive(node_map, node.children[1], memo);
                Quat rot_fn = q_fn.rotate(codebook_.r_left);
                Quat rot_arg = q_arg.rotate(codebook_.r_right);
                Quat sum(rot_fn.w + rot_arg.w, rot_fn.x + rot_arg.x,
                         rot_fn.y + rot_arg.y, rot_fn.z + rot_arg.z);
                res = sum.normalize();
            } else if (!node.children.empty()) {
                res = fold_recursive(node_map, node.children[0], memo);
            }
            break;
        }
        case ExprKind::LAM:
        case ExprKind::FORALLE: {
            if (node.children.size() >= 2) {
                Quat q_dom = fold_recursive(node_map, node.children[0], memo);
                Quat q_body = fold_recursive(node_map, node.children[1], memo);
                Quat rot_dom = q_dom.rotate(codebook_.r_dom);
                Quat rot_body = q_body.rotate(codebook_.r_body);
                Quat sum(rot_dom.w + rot_body.w, rot_dom.x + rot_body.x,
                         rot_dom.y + rot_body.y, rot_dom.z + rot_body.z);
                res = sum.normalize();
            }
            break;
        }
        case ExprKind::LETE: {
            if (node.children.size() >= 3) {
                Quat q_t = fold_recursive(node_map, node.children[0], memo).rotate(codebook_.r_let_type);
                Quat q_v = fold_recursive(node_map, node.children[1], memo).rotate(codebook_.r_let_val);
                Quat q_b = fold_recursive(node_map, node.children[2], memo).rotate(codebook_.r_let_body);
                Quat sum(q_t.w + q_v.w + q_b.w, q_t.x + q_v.x + q_b.x,
                         q_t.y + q_v.y + q_b.y, q_t.z + q_v.z + q_b.z);
                res = sum.normalize();
            }
            break;
        }
        case ExprKind::LIT: {
            double angle = (node.payload % 500) * (3.14159 / 250.0);
            Quat l_rotor = make_rotor(angle, 0.0, 0.707, 0.707);
            res = base_anchor.rotate(l_rotor);
            break;
        }
        case ExprKind::MDATA: {
            if (!node.children.empty()) {
                Quat q_sub = fold_recursive(node_map, node.children[0], memo);
                res = q_sub.rotate(base_anchor);
            }
            break;
        }
    }

    memo[curr_id] = res;
    return res;
}

double TreeRotorOperator::non_commutative_asymmetry(const ExprNode& node_a, const ExprNode& node_b) const {
    ExprNode app_ab{100, ExprKind::APP, 0, "", {node_a.id, node_b.id}};
    ExprNode app_ba{101, ExprKind::APP, 0, "", {node_b.id, node_a.id}};

    std::vector<ExprNode> nodes_ab = {node_a, node_b, app_ab};
    std::vector<ExprNode> nodes_ba = {node_a, node_b, app_ba};

    Quat q_ab = fold_dag(nodes_ab, 100);
    Quat q_ba = fold_dag(nodes_ba, 101);

    return q_ab.geodesic_dist(q_ba);
}

} // namespace syn

extern "C" {

void* rotor_operator_create() {
    return new syn::TreeRotorOperator();
}

void rotor_operator_destroy(void* op) {
    delete static_cast<syn::TreeRotorOperator*>(op);
}

double rotor_geodesic_distance(double w1, double x1, double y1, double z1,
                               double w2, double x2, double y2, double z2) {
    syn::Quat p(w1, x1, y1, z1);
    syn::Quat q(w2, x2, y2, z2);
    return p.geodesic_dist(q);
}

int rotor_check_invariants(double* out_asymmetry, double* out_truncation_rate) {
    syn::TreeRotorOperator op;
    syn::ExprNode a{1, syn::ExprKind::CONST, 0, "Nat.add", {}};
    syn::ExprNode b{2, syn::ExprKind::BVAR, 0, "", {}};
    double asym = op.non_commutative_asymmetry(a, b);
    if (out_asymmetry) *out_asymmetry = asym;

    // Verify zero window truncation on a 63-node tree (depth 6 DAG)
    std::vector<syn::ExprNode> nodes;
    for (uint32_t i = 0; i < 32; ++i) {
        nodes.push_back({i, syn::ExprKind::CONST, i, "c" + std::to_string(i), {}});
    }
    uint32_t prev_start = 0;
    uint32_t prev_count = 32;
    uint32_t curr_id = 32;
    while (prev_count > 1) {
        uint32_t next_count = prev_count / 2;
        for (uint32_t j = 0; j < next_count; ++j) {
            nodes.push_back({curr_id + j, syn::ExprKind::APP, 0, "", {prev_start + 2 * j, prev_start + 2 * j + 1}});
        }
        prev_start = curr_id;
        prev_count = next_count;
        curr_id += next_count;
    }
    size_t visited = 0;
    op.fold_dag(nodes, nodes.back().id, &visited);
    double trunc_rate = (nodes.empty()) ? 0.0 : (1.0 - static_cast<double>(visited) / static_cast<double>(nodes.size()));
    if (out_truncation_rate) *out_truncation_rate = trunc_rate;

    return (asym > 0.0 && trunc_rate == 0.0) ? 0 : 1;
}

}
