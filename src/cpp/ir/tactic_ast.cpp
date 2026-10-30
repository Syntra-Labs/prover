#include "tactic_ast.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <set>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <algorithm>

namespace syn {
namespace ir {

const char* node_sort_name(NodeSort s) {
    switch (s) {
        case NodeSort::OP: return "OP";
        case NodeSort::DIR: return "DIR";
        case NodeSort::LOC: return "LOC";
        case NodeSort::HYP: return "HYP";
        case NodeSort::VAR: return "VAR";
        case NodeSort::TERM: return "TERM";
    }
    return "UNKNOWN";
}

NodeSort node_sort_from_name(const std::string& s) {
    if (s == "OP") return NodeSort::OP;
    if (s == "DIR") return NodeSort::DIR;
    if (s == "LOC") return NodeSort::LOC;
    if (s == "HYP") return NodeSort::HYP;
    if (s == "VAR") return NodeSort::VAR;
    if (s == "TERM") return NodeSort::TERM;
    return NodeSort::OP;
}

const std::vector<ASTNode> AST_OPERATORS = {
    ASTNode("NODE:Lean.Parser.Tactic.tacticRfl", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.simp", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.omega", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.ring", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.linarith", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.decide", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.aesop", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.assumption", NodeSort::OP, {}),
    ASTNode("NODE:Lean.Parser.Tactic.exact", NodeSort::OP, {NodeSort::TERM}),
    ASTNode("NODE:Lean.Parser.Tactic.apply", NodeSort::OP, {NodeSort::TERM}),
    ASTNode("NODE:Lean.Parser.Tactic.intro", NodeSort::OP, {NodeSort::VAR}),
    ASTNode("NODE:Lean.Parser.Tactic.cases", NodeSort::OP, {NodeSort::HYP}),
    ASTNode("NODE:Lean.Parser.Tactic.induction", NodeSort::OP, {NodeSort::VAR}),
    ASTNode("NODE:Lean.Parser.Tactic.rwSeq", NodeSort::OP, {NodeSort::DIR, NodeSort::TERM, NodeSort::LOC})
};

const std::vector<ASTNode> TERM_COMBINATORS = {
    ASTNode("NODE:Lean.Parser.Term.anonymousCtor", NodeSort::TERM, {NodeSort::TERM, NodeSort::TERM}),
    ASTNode("PROJ:left", NodeSort::TERM, {NodeSort::HYP}),
    ASTNode("PROJ:right", NodeSort::TERM, {NodeSort::HYP})
};

std::vector<ASTNode> valid_ast_candidates(
    NodeSort required_sort,
    const std::unordered_map<std::string, std::string>& sigma,
    const std::vector<std::string>& premises,
    bool allow_combinators
) {
    std::set<std::string> canon_vals;
    for (const auto& kv : sigma) {
        canon_vals.insert(kv.second);
    }

    std::vector<ASTNode> in_scope_hyps;
    std::vector<ASTNode> in_scope_vars;
    for (const auto& c : canon_vals) {
        if (!c.empty() && c[0] == 'h') {
            in_scope_hyps.emplace_back("IDENT:" + c, NodeSort::HYP, std::vector<NodeSort>{});
        } else if (!c.empty() && c[0] == 'v') {
            in_scope_vars.emplace_back("IDENT:" + c, NodeSort::VAR, std::vector<NodeSort>{});
        }
    }

    if (required_sort == NodeSort::OP) {
        std::vector<ASTNode> res;
        for (const auto& op : AST_OPERATORS) {
            bool needs_hyp = false;
            for (auto s : op.child_sorts) {
                if (s == NodeSort::HYP) {
                    needs_hyp = true;
                    break;
                }
            }
            if (needs_hyp && in_scope_hyps.empty()) {
                continue;
            }
            res.push_back(op);
        }
        return res;
    }

    if (required_sort == NodeSort::DIR) {
        return {
            ASTNode("ATOM:fwd", NodeSort::DIR, {}),
            ASTNode("ATOM:←", NodeSort::DIR, {})
        };
    }

    if (required_sort == NodeSort::HYP) {
        return !in_scope_hyps.empty() ? in_scope_hyps : std::vector<ASTNode>{ASTNode("IDENT:h0", NodeSort::HYP, {})};
    }

    if (required_sort == NodeSort::VAR) {
        return !in_scope_vars.empty() ? in_scope_vars : std::vector<ASTNode>{ASTNode("IDENT:v0", NodeSort::VAR, {})};
    }

    if (required_sort == NodeSort::LOC) {
        std::vector<ASTNode> res = {ASTNode("NODE:Lean.Parser.Tactic.locationWildcard", NodeSort::LOC, {})};
        res.insert(res.end(), in_scope_hyps.begin(), in_scope_hyps.end());
        return res;
    }

    if (required_sort == NodeSort::TERM) {
        std::vector<ASTNode> cands;
        cands.insert(cands.end(), in_scope_hyps.begin(), in_scope_hyps.end());
        cands.insert(cands.end(), in_scope_vars.begin(), in_scope_vars.end());
        if (allow_combinators && !in_scope_hyps.empty()) {
            cands.insert(cands.end(), TERM_COMBINATORS.begin(), TERM_COMBINATORS.end());
        }
        for (const auto& p : premises) {
            cands.emplace_back("IDENT:" + p, NodeSort::TERM, std::vector<NodeSort>{});
        }
        return !cands.empty() ? cands : std::vector<ASTNode>{ASTNode("IDENT:h0", NodeSort::HYP, {})};
    }

    return {};
}

static std::pair<std::string, size_t> render_term_seq(const std::vector<std::string>& tokens, size_t idx) {
    if (idx >= tokens.size()) return {"", idx};
    const std::string& tok = tokens[idx];
    if (tok == "NODE:Lean.Parser.Term.anonymousCtor" || tok == "CTOR:pair") {
        auto left = render_term_seq(tokens, idx + 1);
        auto right = render_term_seq(tokens, left.second);
        return {"⟨" + left.first + ", " + right.first + "⟩", right.second};
    }
    if (tok == "PROJ:left") {
        auto arg = render_term_seq(tokens, idx + 1);
        return {arg.first + ".left", arg.second};
    }
    if (tok == "PROJ:right") {
        auto arg = render_term_seq(tokens, idx + 1);
        return {arg.first + ".right", arg.second};
    }
    size_t colon = tok.find(':');
    if (colon != std::string::npos) {
        return {tok.substr(colon + 1), idx + 1};
    }
    return {tok, idx + 1};
}

std::string lower_ast_to_lean(
    const std::vector<ASTNode>& nodes,
    const std::unordered_map<std::string, std::string>& sigma
) {
    if (nodes.empty()) return "";
    std::vector<std::string> tokens;
    tokens.reserve(nodes.size());
    for (const auto& n : nodes) {
        tokens.push_back(n.token);
    }

    std::string root = tokens[0];
    size_t colon = root.find(':');
    std::string raw_root = (colon != std::string::npos) ? root.substr(colon + 1) : root;

    std::string op = raw_root;
    if (raw_root == "Lean.Parser.Tactic.tacticRfl") op = "rfl";
    else if (raw_root == "Lean.Parser.Tactic.simp") op = "simp";
    else if (raw_root == "Lean.Parser.Tactic.omega") op = "omega";
    else if (raw_root == "Lean.Parser.Tactic.ring") op = "ring";
    else if (raw_root == "Lean.Parser.Tactic.linarith") op = "linarith";
    else if (raw_root == "Lean.Parser.Tactic.decide") op = "decide";
    else if (raw_root == "Lean.Parser.Tactic.aesop") op = "aesop";
    else if (raw_root == "Lean.Parser.Tactic.assumption") op = "assumption";
    else if (raw_root == "Lean.Parser.Tactic.exact") op = "exact";
    else if (raw_root == "Lean.Parser.Tactic.apply") op = "apply";
    else if (raw_root == "Lean.Parser.Tactic.intro") op = "intro";
    else if (raw_root == "Lean.Parser.Tactic.cases") op = "cases";
    else if (raw_root == "Lean.Parser.Tactic.induction") op = "induction";
    else if (raw_root == "Lean.Parser.Tactic.rwSeq") op = "rw";

    std::string canon_tac = op;
    if (op == "rfl" || op == "simp" || op == "omega" || op == "ring" ||
        op == "linarith" || op == "decide" || op == "aesop" || op == "assumption") {
        canon_tac = op;
    } else if (op == "exact" || op == "apply" || op == "intro" || op == "cases" || op == "induction") {
        auto term = render_term_seq(tokens, 1);
        canon_tac = op + " " + term.first;
    } else if (op == "rw") {
        std::string dir = (tokens.size() > 1) ? tokens[1] : "ATOM:fwd";
        auto rule = render_term_seq(tokens, 2);
        std::string loc = (rule.second < tokens.size()) ? tokens[rule.second] : "NODE:Lean.Parser.Tactic.locationWildcard";
        std::string prefix = (dir == "ATOM:←" || dir == "DIR:symm") ? "← " : "";
        if (loc == "NODE:Lean.Parser.Tactic.locationWildcard" || loc == "LOC:goal") {
            canon_tac = "rw [" + prefix + rule.first + "]";
        } else {
            size_t c_loc = loc.find(':');
            std::string loc_name = (c_loc != std::string::npos) ? loc.substr(c_loc + 1) : loc;
            canon_tac = "rw [" + prefix + rule.first + "] at " + loc_name;
        }
    }

    // Invert canonical identifiers (h0 -> original_h) if sigma supplied
    if (!sigma.empty()) {
        std::unordered_map<std::string, std::string> inv_map;
        for (const auto& kv : sigma) {
            inv_map[kv.second] = kv.first;
        }
        for (const auto& kv : inv_map) {
            size_t pos = 0;
            while ((pos = canon_tac.find(kv.first, pos)) != std::string::npos) {
                // Ensure word boundary
                bool left_ok = (pos == 0 || !isalnum(canon_tac[pos - 1]));
                bool right_ok = (pos + kv.first.size() >= canon_tac.size() || !isalnum(canon_tac[pos + kv.first.size()]));
                if (left_ok && right_ok) {
                    canon_tac.replace(pos, kv.first.size(), kv.second);
                    pos += kv.second.size();
                } else {
                    pos += kv.first.size();
                }
            }
        }
    }

    return canon_tac;
}

LeanASTBridge& LeanASTBridge::instance() {
    static LeanASTBridge inst;
    return inst;
}

LeanASTBridge::~LeanASTBridge() {
    close();
}

void LeanASTBridge::ensure_proc() {
    if (pid_ > 0) return;

    int p_in[2];
    int p_out[2];
    if (pipe(p_in) < 0 || pipe(p_out) < 0) {
        throw std::runtime_error("LeanASTBridge: failed to create pipes");
    }

    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("LeanASTBridge: failed to fork");
    }

    if (pid == 0) {
        dup2(p_in[0], STDIN_FILENO);
        dup2(p_out[1], STDOUT_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDERR_FILENO);
            ::close(devnull);
        }
        ::close(p_in[0]);
        ::close(p_in[1]);
        ::close(p_out[0]);
        ::close(p_out[1]);

        execlp("lean", "lean", "--run", "src/lean_bridge/ast_bridge.lean", (char*)nullptr);
        _exit(127);
    } else {
        ::close(p_in[0]);
        ::close(p_out[1]);
        pid_ = pid;
        in_fd_ = p_in[1];
        out_fd_ = p_out[0];
    }
}

void LeanASTBridge::close() {
    if (pid_ > 0) {
        if (in_fd_ >= 0) ::close(in_fd_);
        if (out_fd_ >= 0) ::close(out_fd_);
        kill(pid_, SIGTERM);
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
        in_fd_ = -1;
        out_fd_ = -1;
    }
}

std::string LeanASTBridge::send_request(const std::string& json_payload) {
    ensure_proc();
    std::string msg = json_payload + "\n";
    ssize_t written = write(in_fd_, msg.c_str(), msg.size());
    if (written < static_cast<ssize_t>(msg.size())) {
        throw std::runtime_error("LeanASTBridge: pipe write failed");
    }

    std::string line;
    char ch;
    while (read(out_fd_, &ch, 1) > 0) {
        if (ch == '\n') break;
        line += ch;
    }
    return line;
}

std::string LeanASTBridge::parse_tactic(
    const std::string& tactic,
    const std::unordered_map<std::string, std::string>& sigma
) {
    std::string req = "{\"op\":\"parse_tactic\",\"tactic\":\"" + tactic + "\",\"sigma\":{}}";
    std::string resp = send_request(req);
    size_t pos = resp.find("\"canonical_tactic\":\"");
    if (pos != std::string::npos) {
        pos += 20;
        size_t end = resp.find("\"", pos);
        if (end != std::string::npos) {
            return resp.substr(pos, end - pos);
        }
    }
    return tactic;
}

} // namespace ir
} // namespace syn
