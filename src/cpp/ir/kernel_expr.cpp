#include "kernel_expr.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <algorithm>

namespace syn {
namespace ir {

static const char* CTOR_NAMES[] = {
    "bvar", "fvar", "mvar", "sort", "const", "app", "lam", "forallE", "letE", "lit", "proj"
};

const char* expr_kind_name(ExprKind k) {
    uint8_t idx = static_cast<uint8_t>(k);
    if (idx < 11) {
        return CTOR_NAMES[idx];
    }
    return "unknown";
}

ExprKind expr_kind_from_name(const std::string& name) {
    for (uint8_t i = 0; i < 11; ++i) {
        if (name == CTOR_NAMES[i]) {
            return static_cast<ExprKind>(i);
        }
    }
    return ExprKind::BVAR;
}

std::vector<int> KernelState::subterms(int root) const {
    std::vector<int> res;
    if (root < 0 || root >= static_cast<int>(nodes.size())) {
        return res;
    }
    std::unordered_set<int> seen;
    std::vector<int> stack = {root};
    while (!stack.empty()) {
        int n = stack.back();
        stack.pop_back();
        if (seen.find(n) != seen.end()) {
            continue;
        }
        seen.insert(n);
        res.push_back(n);
        if (n >= 0 && n < static_cast<int>(nodes.size())) {
            const auto& ch = nodes[n].children;
            for (auto it = ch.rbegin(); it != ch.rend(); ++it) {
                stack.push_back(*it);
            }
        }
    }
    return res;
}

int KernelState::head(int n) const {
    while (n >= 0 && n < static_cast<int>(nodes.size()) && nodes[n].k == static_cast<uint8_t>(ExprKind::APP)) {
        if (nodes[n].children.empty()) break;
        n = nodes[n].children[0];
    }
    return n;
}

std::string KernelState::render(int n) const {
    if (n < 0 || n >= static_cast<int>(nodes.size())) {
        return "<invalid>";
    }
    const auto& nd = nodes[n];
    ExprKind kind = static_cast<ExprKind>(nd.k);
    if (kind == ExprKind::APP && nd.children.size() >= 2) {
        return "(" + render(nd.children[0]) + " " + render(nd.children[1]) + ")";
    }
    if ((kind == ExprKind::LAM || kind == ExprKind::FORALLE) && nd.children.size() >= 2) {
        std::string sym = (kind == ExprKind::LAM) ? "λ" : "∀";
        return sym + ":" + render(nd.children[0]) + "." + render(nd.children[1]);
    }
    if (kind == ExprKind::BVAR) {
        return "#" + nd.label;
    }
    if (kind == ExprKind::SORT) {
        return "Sort(" + nd.label + ")";
    }
    if (kind == ExprKind::PROJ && !nd.children.empty()) {
        return render(nd.children[0]) + "." + nd.label;
    }
    return nd.label;
}

std::pair<std::string, int> decl_to_pi_type(const std::string& decl) {
    std::string s = decl;
    // trim
    size_t first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return {"", 0};
    size_t last = s.find_last_not_of(" \t\n\r");
    s = s.substr(first, last - first + 1);

    const std::string s1 = ":= by";
    const std::string s2 = ":=";
    if (s.size() >= s1.size() && s.rfind(s1) == s.size() - s1.size()) {
        s = s.substr(0, s.size() - s1.size());
    } else if (s.size() >= s2.size() && s.rfind(s2) == s.size() - s2.size()) {
        s = s.substr(0, s.size() - s2.size());
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();

    // split command prefix (theorem/lemma/example)
    std::string kw;
    size_t first_space = s.find_first_of(" \t");
    if (first_space != std::string::npos) {
        kw = s.substr(0, first_space);
        if (kw == "theorem" || kw == "lemma") {
            size_t second_word_start = s.find_first_not_of(" \t", first_space);
            if (second_word_start != std::string::npos) {
                size_t second_space = s.find_first_of(" \t", second_word_start);
                if (second_space != std::string::npos) {
                    s = s.substr(second_space + 1);
                } else {
                    s = "";
                }
            }
        } else if (kw == "example") {
            s = s.substr(first_space + 1);
        }
    }

    int depth = 0;
    int split_at = -1;
    for (size_t i = 0; i < s.size(); ++i) {
        char ch = s[i];
        if (ch == '(' || ch == '[' || ch == '{') depth++;
        else if (ch == ')' || ch == ']' || ch == '}') depth--;
        else if (ch == ':' && depth == 0) {
            split_at = static_cast<int>(i);
            break;
        }
    }

    if (split_at < 0) {
        return {s, 0};
    }

    std::string binders = s.substr(0, split_at);
    std::string ret_ty = s.substr(split_at + 1);

    // trim
    first = binders.find_first_not_of(" \t\n\r");
    binders = (first == std::string::npos) ? "" : binders.substr(first, binders.find_last_not_of(" \t\n\r") - first + 1);

    first = ret_ty.find_first_not_of(" \t\n\r");
    ret_ty = (first == std::string::npos) ? "" : ret_ty.substr(first, ret_ty.find_last_not_of(" \t\n\r") - first + 1);

    if (binders.empty()) {
        return {ret_ty, 0};
    }

    int n_binders = 0;
    for (char c : binders) {
        if (c == '(' || c == '[' || c == '{') n_binders++;
    }
    return {"∀ " + binders + ", " + ret_ty, n_binders};
}

KernelExprBridge& KernelExprBridge::instance() {
    static KernelExprBridge inst;
    return inst;
}

KernelExprBridge::~KernelExprBridge() {
    close();
}

void KernelExprBridge::ensure_proc() {
    if (pid_ > 0) return;

    int p_in[2];
    int p_out[2];
    if (pipe(p_in) < 0 || pipe(p_out) < 0) {
        throw std::runtime_error("KernelExprBridge: failed to create pipes");
    }

    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("KernelExprBridge: failed to fork");
    }

    if (pid == 0) {
        // Child
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

        execlp("lean", "lean", "--run", "src/lean_bridge/expr_bridge.lean", (char*)nullptr);
        _exit(127);
    } else {
        // Parent
        ::close(p_in[0]);
        ::close(p_out[1]);
        pid_ = pid;
        in_fd_ = p_in[1];
        out_fd_ = p_out[0];
    }
}

void KernelExprBridge::close() {
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

static std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

std::string KernelExprBridge::send_request(const std::string& json_payload) {
    ensure_proc();
    std::string msg = json_payload + "\n";
    ssize_t written = write(in_fd_, msg.c_str(), msg.size());
    if (written < static_cast<ssize_t>(msg.size())) {
        throw std::runtime_error("KernelExprBridge: pipe write failed");
    }

    std::string line;
    char ch;
    while (read(out_fd_, &ch, 1) > 0) {
        if (ch == '\n') break;
        line += ch;
    }
    return line;
}

static std::string extract_json_str(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.size();
    size_t end = json.find("\"", pos);
    if (end == std::string::npos) return "";
    return json.substr(pos, end - pos);
}

static int extract_json_int(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return -1;
    pos += pattern.size();
    return std::atoi(json.c_str() + pos);
}

// Lightweight JSON parser for state_dag response
static KernelState parse_state_dag_response(const std::string& resp_json) {
    KernelState st;
    if (resp_json.empty() || resp_json.find("\"ok\":true") == std::string::npos) {
        return st;
    }

    // Extract target
    st.target = extract_json_int(resp_json, "target");

    // Parse nodes: [{"k":0,"l":"0","c":[]}, ...]
    size_t nodes_pos = resp_json.find("\"nodes\":[");
    if (nodes_pos != std::string::npos) {
        nodes_pos += 9;
        size_t end_nodes = resp_json.find("],\"hyps\"", nodes_pos);
        if (end_nodes == std::string::npos) end_nodes = resp_json.find("],\"target\"", nodes_pos);
        if (end_nodes != std::string::npos) {
            std::string nodes_str = resp_json.substr(nodes_pos, end_nodes - nodes_pos);
            size_t p = 0;
            while ((p = nodes_str.find('{', p)) != std::string::npos) {
                size_t p_end = nodes_str.find('}', p);
                if (p_end == std::string::npos) break;
                std::string obj = nodes_str.substr(p, p_end - p + 1);
                int k = extract_json_int(obj, "k");
                std::string label = extract_json_str(obj, "l");
                
                std::vector<int> children;
                size_t c_pos = obj.find("\"c\":[");
                if (c_pos != std::string::npos) {
                    c_pos += 5;
                    size_t c_end = obj.find(']', c_pos);
                    if (c_end != std::string::npos && c_end > c_pos) {
                        std::string c_list = obj.substr(c_pos, c_end - c_pos);
                        std::istringstream c_iss(c_list);
                        std::string tok;
                        while (std::getline(c_iss, tok, ',')) {
                            if (!tok.empty()) children.push_back(std::atoi(tok.c_str()));
                        }
                    }
                }
                st.nodes.emplace_back(static_cast<uint8_t>(k), label, children);
                p = p_end + 1;
            }
        }
    }

    // Parse hyps: [{"fv":"fv0","user":"p","type":3,"isProp":true}, ...]
    size_t hyps_pos = resp_json.find("\"hyps\":[");
    if (hyps_pos != std::string::npos) {
        hyps_pos += 8;
        size_t end_hyps = resp_json.find("],\"target\"", hyps_pos);
        if (end_hyps != std::string::npos) {
            std::string hyps_str = resp_json.substr(hyps_pos, end_hyps - hyps_pos);
            size_t p = 0;
            while ((p = hyps_str.find('{', p)) != std::string::npos) {
                size_t p_end = hyps_str.find('}', p);
                if (p_end == std::string::npos) break;
                std::string obj = hyps_str.substr(p, p_end - p + 1);
                std::string fv = extract_json_str(obj, "fv");
                std::string user = extract_json_str(obj, "user");
                int type = extract_json_int(obj, "type");
                bool is_prop = (obj.find("\"isProp\":true") != std::string::npos);
                st.hyps.emplace_back(fv, user, type, is_prop);
                p = p_end + 1;
            }
        }
    }

    return st;
}

KernelState KernelExprBridge::elaborate_command(const std::string& decl) {
    auto parsed = decl_to_pi_type(decl);
    std::string req = "{\"op\":\"state_dag\",\"type\":\"" + escape_json(parsed.first) +
                      "\",\"intros\":" + std::to_string(parsed.second) + ",\"names\":\"*\"}";
    std::string resp = send_request(req);
    return parse_state_dag_response(resp);
}

KernelState KernelExprBridge::elaborate_state(const std::string& turnstile_str) {
    return elaborate_command(turnstile_str);
}

} // namespace ir
} // namespace syn
