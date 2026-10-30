#ifndef SYN_IR_TACTIC_AST_HPP
#define SYN_IR_TACTIC_AST_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace syn {
namespace ir {

enum class NodeSort {
    OP,
    DIR,
    LOC,
    HYP,
    VAR,
    TERM
};

const char* node_sort_name(NodeSort s);
NodeSort node_sort_from_name(const std::string& s);

struct ASTNode {
    std::string token{""};
    NodeSort sort{NodeSort::OP};
    std::vector<NodeSort> child_sorts{};

    ASTNode() = default;
    ASTNode(std::string token, NodeSort sort, std::vector<NodeSort> child_sorts = {})
        : token(std::move(token)), sort(sort), child_sorts(std::move(child_sorts)) {}

    bool operator==(const ASTNode& other) const {
        return token == other.token && sort == other.sort && child_sorts == other.child_sorts;
    }
};

extern const std::vector<ASTNode> AST_OPERATORS;
extern const std::vector<ASTNode> TERM_COMBINATORS;

std::vector<ASTNode> valid_ast_candidates(
    NodeSort required_sort,
    const std::unordered_map<std::string, std::string>& sigma,
    const std::vector<std::string>& premises = {},
    bool allow_combinators = true
);

std::string lower_ast_to_lean(
    const std::vector<ASTNode>& nodes,
    const std::unordered_map<std::string, std::string>& sigma = {}
);

// Client to ast_bridge.lean over JSONL stdio
class LeanASTBridge {
public:
    static LeanASTBridge& instance();
    ~LeanASTBridge();

    std::string parse_tactic(const std::string& tactic, const std::unordered_map<std::string, std::string>& sigma = {});
    void close();

private:
    LeanASTBridge() = default;
    void ensure_proc();
    std::string send_request(const std::string& json_payload);

    pid_t pid_{-1};
    int in_fd_{-1};
    int out_fd_{-1};
};

} // namespace ir
} // namespace syn

#endif // SYN_IR_TACTIC_AST_HPP
