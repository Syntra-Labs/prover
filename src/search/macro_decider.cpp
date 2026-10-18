#include "macro_decider.hpp"

namespace syn {

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start, s.find_last_not_of(" \t\n\r") - start + 1);
}

bool MacroActionDecider::is_reflexive_or_terminal_head(const std::string& conclusion) {
    std::string s = trim(conclusion);
    size_t turnstile = s.find("⊢");
    if (turnstile != std::string::npos) s = trim(s.substr(turnstile + 3));
    if (s.empty()) return false;
    if (s == "True" || s == "true" || s == "⊤" || s == "rfl" || s == "Eq.refl" || s == "refl") return true;

    size_t eq = s.find(" = ");
    size_t len = 3;
    if (eq == std::string::npos) { eq = s.find("=="); len = 2; }
    if (eq == std::string::npos) {
        eq = s.find('='); len = 1;
        if (eq != std::string::npos && eq > 0 && (s[eq - 1] == '!' || s[eq - 1] == '<' || s[eq - 1] == '>'))
            eq = std::string::npos;
    }
    if (eq != std::string::npos) {
        std::string lhs = trim(s.substr(0, eq)), rhs = trim(s.substr(eq + len));
        return !lhs.empty() && lhs == rhs;
    }
    return false;
}

const std::vector<std::string>& MacroActionDecider::get_tier2a_tactics() {
    static const std::vector<std::string> t2a = {"exact", "rfl", "simp", "omega", "ring", "linarith", "decide"};
    return t2a;
}

const std::vector<std::string>& MacroActionDecider::get_tier2b_tactics() {
    static const std::vector<std::string> t2b = {"apply", "rw", "intro", "cases", "induction"};
    return t2b;
}

MacroActionKind MacroActionDecider::decide_kind(const Quat& z_s, const std::string& conclusion) const {
    if (is_reflexive_or_terminal_head(conclusion) || z_s.normalize().geodesic_dist(z_qed_) <= theta_conv_)
        return MacroActionKind::Tier2A_DirectTerminating;
    return MacroActionKind::Tier2B_ForwardInvention;
}

MacroRouteDecision MacroActionDecider::route(const GoalState& state) const {
    MacroRouteDecision dec;
    dec.geodesic_distance = state.z.normalize().geodesic_dist(z_qed_);
    dec.matched_terminal_head = is_reflexive_or_terminal_head(state.conclusion);
    if (dec.matched_terminal_head || dec.geodesic_distance <= theta_conv_) {
        dec.kind = MacroActionKind::Tier2A_DirectTerminating;
        dec.candidate_tactics = get_tier2a_tactics();
        dec.reason = dec.matched_terminal_head ? "Terminal/reflexive head" : "Geodesic convergence";
    } else {
        dec.kind = MacroActionKind::Tier2B_ForwardInvention;
        dec.candidate_tactics = get_tier2b_tactics();
        dec.reason = "Divergent state requiring forward invention";
    }
    return dec;
}

MacroRouteDecision MacroActionDecider::route_with_feedback(
    const GoalState& state,
    const std::function<bool(const std::string&)>& kernel_eval
) const {
    MacroRouteDecision dec = route(state);
    if (dec.kind == MacroActionKind::Tier2A_DirectTerminating) {
        bool ok = false;
        for (const auto& tac : dec.candidate_tactics) {
            if (kernel_eval(tac)) { ok = true; break; }
        }
        if (!ok) {
            dec.kind = MacroActionKind::Tier2B_ForwardInvention;
            dec.candidate_tactics = get_tier2b_tactics();
            dec.fallback_triggered = true;
            dec.reason = "Kernel feedback rejection: fell back to Tier 2B";
        }
    }
    return dec;
}

MacroActionKind MacroActionDecider::route_step(const GoalState& s, bool rejected, bool non_terminating) const {
    MacroActionKind k = decide_kind(s);
    return (k == MacroActionKind::Tier2A_DirectTerminating && (rejected || non_terminating))
        ? MacroActionKind::Tier2B_ForwardInvention : k;
}

} // namespace syn
