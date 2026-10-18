#include "macro_decider.hpp"

#include <sstream>
#include <algorithm>

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

std::vector<std::pair<std::string, double>> MacroActionDecider::filter_admissible_candidates(
    const std::string& goal_state_str,
    const std::vector<std::pair<std::string, double>>& raw_scores
) const {
    std::string concl = "";
    std::vector<std::string> hyps;
    std::stringstream ss(goal_state_str);
    std::string line;
    while (std::getline(ss, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line.rfind("⊢", 0) == 0 || line.find("⊢") != std::string::npos) {
            size_t pos = line.find("⊢");
            concl = trim(line.substr(pos + 3));
        } else if (line.find(':') != std::string::npos) {
            hyps.push_back(line);
        }
    }

    bool is_term = is_reflexive_or_terminal_head(concl);
    bool has_exact = false;
    bool has_arrow_target = false;
    bool has_eq = (concl.find('=') != std::string::npos);

    for (const auto& h : hyps) {
        size_t c_pos = h.find(':');
        if (c_pos != std::string::npos) {
            std::string h_type = trim(h.substr(c_pos + 1));
            if (!concl.empty() && h_type == concl) has_exact = true;
            if (h_type.find('=') != std::string::npos) has_eq = true;
            size_t arr = h_type.rfind("→");
            if (arr != std::string::npos) {
                std::string target_part = trim(h_type.substr(arr + 3));
                if (target_part == concl || (!concl.empty() && target_part.find(concl) != std::string::npos)) {
                    has_arrow_target = true;
                }
            }
        }
    }

    bool is_cat = (concl.find("⟶") != std::string::npos || concl.find("≫") != std::string::npos ||
                   concl.find("IsIso") != std::string::npos || concl.find("Mono") != std::string::npos ||
                   concl.find("Epi") != std::string::npos || concl.find("ShortComplex") != std::string::npos);
    bool is_measure = (concl.find("∫") != std::string::npos || concl.find("Measure") != std::string::npos ||
                       concl.find("condexpL2") != std::string::npos);
    bool is_degrees = (concl.find("degrees") != std::string::npos || concl.find("Degrees") != std::string::npos);

    std::vector<std::pair<std::string, double>> res;
    res.reserve(raw_scores.size());

    for (const auto& kv : raw_scores) {
        const std::string& tac = kv.first;
        double s = kv.second;
        if (tac == "exact" && !(has_exact || has_arrow_target)) s += 50.0;
        if ((tac == "rfl" || tac == "decide") && !is_term) s += 50.0;
        if ((tac == "rw" || tac == "nth_rw" || tac == "simp_rw") && !has_eq) s += 50.0;
        if (is_cat && tac == "simp") s += 2.0;
        if (is_cat && (tac == "have" || tac == "apply")) s -= 0.5;
        if (is_measure && (tac == "let" || tac == "have")) s -= 0.5;
        if (is_degrees && tac == "simp") s -= 1.0;
        res.push_back({tac, s});
    }

    std::sort(res.begin(), res.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
        return a.second < b.second;
    });
    return res;
}

} // namespace syn

extern "C" {

void* macro_decider_create() {
    return new syn::MacroActionDecider();
}

void macro_decider_destroy(void* dec) {
    delete static_cast<syn::MacroActionDecider*>(dec);
}

int macro_decider_route(void* dec, double w, double x, double y, double z, const char* concl) {
    if (!dec) return 1;
    syn::MacroActionDecider* d = static_cast<syn::MacroActionDecider*>(dec);
    syn::Quat q(w, x, y, z);
    syn::MacroActionKind k = d->decide_kind(q, concl ? concl : "");
    return (k == syn::MacroActionKind::Tier2A_DirectTerminating) ? 0 : 1;
}

}
