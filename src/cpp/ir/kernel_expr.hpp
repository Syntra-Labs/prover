#ifndef SYN_IR_KERNEL_EXPR_HPP
#define SYN_IR_KERNEL_EXPR_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <memory>

namespace syn {
namespace ir {

enum class ExprKind : uint8_t {
    BVAR = 0,
    FVAR = 1,
    MVAR = 2,
    SORT = 3,
    CONST = 4,
    APP = 5,
    LAM = 6,
    FORALLE = 7,
    LETE = 8,
    LIT = 9,
    PROJ = 10
};

const char* expr_kind_name(ExprKind k);
ExprKind expr_kind_from_name(const std::string& name);

struct Node {
    uint8_t k{0}; // constructor id in [0, 11)
    std::string label{""};
    std::vector<int> children{};

    Node() = default;
    Node(uint8_t k, std::string label, std::vector<int> children)
        : k(k), label(std::move(label)), children(std::move(children)) {}
};

struct Hyp {
    std::string fv{""};
    std::string user{""};
    int type{0};
    bool is_prop{false};

    Hyp() = default;
    Hyp(std::string fv, std::string user, int type, bool is_prop)
        : fv(std::move(fv)), user(std::move(user)), type(type), is_prop(is_prop) {}
};

class KernelState {
public:
    std::vector<Node> nodes{};
    std::vector<Hyp> hyps{};
    int target{0};
    std::unordered_map<std::string, int> consts{};

    KernelState() = default;
    KernelState(std::vector<Node> nodes, std::vector<Hyp> hyps, int target,
                std::unordered_map<std::string, int> consts = {})
        : nodes(std::move(nodes)), hyps(std::move(hyps)), target(target), consts(std::move(consts)) {}

    std::vector<int> subterms(int root) const;
    int head(int n) const;
    std::string render(int n) const;
};

// Decl to Pi type parser
std::pair<std::string, int> decl_to_pi_type(const std::string& decl);

// Persistent stdio JSON client to `lean --run src/lean_bridge/expr_bridge.lean`
class KernelExprBridge {
public:
    static KernelExprBridge& instance();
    ~KernelExprBridge();

    KernelState elaborate_command(const std::string& decl);
    KernelState elaborate_state(const std::string& turnstile_str);
    void close();

private:
    KernelExprBridge() = default;
    void ensure_proc();
    std::string send_request(const std::string& json_payload);

    pid_t pid_{-1};
    int in_fd_{-1};
    int out_fd_{-1};
};

} // namespace ir
} // namespace syn

#endif // SYN_IR_KERNEL_EXPR_HPP
