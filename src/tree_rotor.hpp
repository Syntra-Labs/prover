#ifndef TREE_ROTOR_HPP
#define TREE_ROTOR_HPP

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace syn {

// 4D Unit Quaternion representing a point/rotor in S^3
struct Quat {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Quat() = default;
    Quat(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}

    static Quat identity() { return Quat(1.0, 0.0, 0.0, 0.0); }

    double norm_sq() const {
        return w * w + x * x + y * y + z * z;
    }

    double norm() const {
        return std::sqrt(norm_sq());
    }

    // Projection onto unit 3-sphere S^3
    Quat normalize() const {
        double n = norm();
        if (n < 1e-12) {
            return identity();
        }
        return Quat(w / n, x / n, y / n, z / n);
    }

    Quat conj() const {
        return Quat(w, -x, -y, -z);
    }

    // Hamilton product: this * q
    Quat mul(const Quat& q) const {
        return Quat(
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        );
    }

    // Rotor sandwich transformation: R * v * R*
    Quat rotate(const Quat& r) const {
        return r.mul(*this).mul(r.conj());
    }

    double dot(const Quat& q) const {
        return w * q.w + x * q.x + y * q.y + z * q.z;
    }

    // Geodesic distance on S^3 taking antipodal identification into account
    // d_S3(p, q) = 2 * arccos(|<p, q>|) in [0, pi]
    double geodesic_dist(const Quat& q) const {
        double d = std::abs(dot(q));
        if (d > 1.0) d = 1.0;
        return 2.0 * std::acos(d);
    }
};

// The 11 constructors of Lean 4 kernel Lean.Expr
enum class ExprKind : uint8_t {
    BVAR = 0,    // de Bruijn bound variable
    FVAR = 1,    // free variable / local hyp
    MVAR = 2,    // meta variable
    SORT = 3,    // universe Sort
    CONST = 4,   // global constant
    APP = 5,     // function application (fn, arg)
    LAM = 6,     // lambda abstraction (name, type, body)
    FORALLE = 7, // Pi / forall (name, type, body)
    LETE = 8,    // let binding (name, type, val, body)
    LIT = 9,     // literal (nat or string)
    MDATA = 10   // metadata wrapper
};

// An AST/DAG Node in the Lean.Expr intermediate representation
struct ExprNode {
    uint32_t id{0};
    ExprKind kind{ExprKind::BVAR};
    uint64_t payload{0};              // hash or index (e.g. bvar index, level, lit value)
    std::string symbol{""};           // identifier name if applicable (e.g. const name)
    std::vector<uint32_t> children{}; // child node IDs

    ExprNode() = default;
    ExprNode(uint32_t id, ExprKind kind, uint64_t payload, std::string symbol, std::vector<uint32_t> children)
        : id(id), kind(kind), payload(payload), symbol(std::move(symbol)), children(std::move(children)) {}
};

// Rotor codebook providing asymmetric unit quaternions on S^3
struct RotorCodebook {
    Quat constr_rotors[11];
    Quat r_left;       // Left branch of application (fn)
    Quat r_right;      // Right branch of application (arg) - asymmetric from r_left
    Quat r_dom;        // Domain of lambda / Pi
    Quat r_body;       // Body of lambda / Pi
    Quat r_let_type;   // Type of let
    Quat r_let_val;    // Val of let
    Quat r_let_body;   // Body of let

    RotorCodebook();
};

// Evaluation / folding engine
class TreeRotorOperator {
public:
    explicit TreeRotorOperator(RotorCodebook codebook = RotorCodebook());

    // Evaluates an Expr DAG bottom-up without sliding-window truncation.
    // Guaranteed window_truncation_rate == 0.0.
    Quat fold_dag(const std::vector<ExprNode>& nodes, uint32_t root_id, size_t* out_visited_count = nullptr) const;

    // Evaluates non-commutative asymmetry:
    // d_S3(fold(app(A, B)), fold(app(B, A)))
    double non_commutative_asymmetry(const ExprNode& node_a, const ExprNode& node_b) const;

    const RotorCodebook& codebook() const { return codebook_; }

private:
    RotorCodebook codebook_;
    Quat fold_recursive(
        const std::unordered_map<uint32_t, const ExprNode*>& node_map,
        uint32_t curr_id,
        std::unordered_map<uint32_t, Quat>& memo
    ) const;
};

} // namespace syn

// C ABI exports for interop and verification
extern "C" {
    void* rotor_operator_create();
    void rotor_operator_destroy(void* op);
    double rotor_geodesic_distance(double w1, double x1, double y1, double z1,
                                   double w2, double x2, double y2, double z2);
    int rotor_check_invariants(double* out_asymmetry, double* out_truncation_rate);
}

#endif // TREE_ROTOR_HPP
