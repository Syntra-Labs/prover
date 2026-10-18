#ifndef MACRO_DECIDER_HPP
#define MACRO_DECIDER_HPP

#include "../tree_rotor.hpp"
#include <functional>
#include <string>
#include <vector>

namespace syn {

enum class MacroActionKind {
    Tier2A_DirectTerminating,
    Tier2B_ForwardInvention
};

struct GoalState {
    Quat z{Quat::identity()};
    std::string conclusion{""};
    GoalState() = default;
    GoalState(Quat z_val, std::string conc = "") : z(z_val), conclusion(std::move(conc)) {}
};

struct MacroRouteDecision {
    MacroActionKind kind{MacroActionKind::Tier2B_ForwardInvention};
    double geodesic_distance{0.0};
    bool matched_terminal_head{false};
    bool fallback_triggered{false};
    std::vector<std::string> candidate_tactics{};
    std::string reason{""};
};

class MacroActionDecider {
public:
    explicit MacroActionDecider(double theta_conv = 0.5, Quat z_qed = Quat::identity())
        : theta_conv_(theta_conv), z_qed_(z_qed) {}

    MacroActionKind decide_kind(const Quat& z_s, const std::string& conclusion = "") const;
    MacroActionKind decide_kind(const GoalState& s) const { return decide_kind(s.z, s.conclusion); }
    MacroRouteDecision route(const GoalState& state) const;
    MacroRouteDecision route_with_feedback(
        const GoalState& state,
        const std::function<bool(const std::string&)>& kernel_eval
    ) const;
    MacroActionKind route_step(const GoalState& s, bool rejected, bool non_terminating = false) const;

    static bool is_reflexive_or_terminal_head(const std::string& conclusion);
    static const std::vector<std::string>& get_tier2a_tactics();
    static const std::vector<std::string>& get_tier2b_tactics();
    std::vector<std::pair<std::string, double>> filter_admissible_candidates(
        const std::string& goal_state_str,
        const std::vector<std::pair<std::string, double>>& raw_scores
    ) const;
    double theta_conv() const { return theta_conv_; }

private:
    double theta_conv_{0.5};
    Quat z_qed_{Quat::identity()};
};

} // namespace syn
#endif // MACRO_DECIDER_HPP
