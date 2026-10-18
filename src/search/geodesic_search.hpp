#ifndef GEODESIC_SEARCH_HPP
#define GEODESIC_SEARCH_HPP

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace syn {

enum class KernelStatus {
    GoalsClosed,
    SubgoalsOpen,
    KernelError,
    KernelTimeout
};

enum class SearchStatus {
    Searching,
    SuccessProven,
    BudgetExhausted,
    SearchFailed
};

struct TacticCandidate {
    std::string name;
    std::vector<double> vector;
    double score{0.0};

    TacticCandidate() = default;
    TacticCandidate(std::string name, std::vector<double> vector, double score = 0.0)
        : name(std::move(name)), vector(std::move(vector)), score(score) {}
};

struct SearchNode {
    std::string state_id;
    std::string goal_str;
    int depth{0};
    std::shared_ptr<SearchNode> parent{nullptr};
    std::string applied_tactic{""};
    std::vector<std::string> subgoals{};

    SearchNode() = default;
    SearchNode(std::string id, std::string goal, int d = 0, std::shared_ptr<SearchNode> p = nullptr,
               std::string tac = "", std::vector<std::string> subs = {})
        : state_id(std::move(id)), goal_str(std::move(goal)), depth(d), parent(std::move(p)),
          applied_tactic(std::move(tac)), subgoals(std::move(subs)) {}
};

struct GeodesicSearchResult {
    SearchStatus status{SearchStatus::Searching};
    std::vector<std::string> proof_steps{};
    int total_steps{0};
    int kernel_calls{0};
    int repl_calls_saved{0};
    int depth_reached{0};

    GeodesicSearchResult() = default;
    GeodesicSearchResult(SearchStatus st, std::vector<std::string> steps, int total, int k_calls, int saved, int depth)
        : status(st), proof_steps(std::move(steps)), total_steps(total), kernel_calls(k_calls),
          repl_calls_saved(saved), depth_reached(depth) {}
};

class SpeculativeGeodesicRouter {
public:
    explicit SpeculativeGeodesicRouter(double margin_threshold = 0.05, int max_beam_width = 2)
        : margin_threshold_(margin_threshold), max_beam_width_(max_beam_width) {}

    std::vector<TacticCandidate> rank_candidates(
        const std::vector<double>& tangent_vector,
        const std::unordered_map<std::string, std::vector<double>>& action_dict
    ) const;

    std::tuple<std::vector<TacticCandidate>, double, bool> select_speculative_queue(
        const std::vector<TacticCandidate>& ranked
    ) const;

    double margin_threshold() const { return margin_threshold_; }
    int max_beam_width() const { return max_beam_width_; }

private:
    double margin_threshold_{0.05};
    int max_beam_width_{2};
};

class SpeculativeGeodesicProver {
public:
    using PolicyFn = std::function<std::vector<double>(const std::string&)>;
    using KernelEvalFn = std::function<std::pair<KernelStatus, std::vector<std::string>>(const std::string&, const std::string&)>;

    explicit SpeculativeGeodesicProver(
        SpeculativeGeodesicRouter router = SpeculativeGeodesicRouter(),
        int max_steps = 100,
        int max_depth = 20
    ) : router_(std::move(router)), max_steps_(max_steps), max_depth_(max_depth) {}

    GeodesicSearchResult search(
        const std::string& initial_goal,
        const PolicyFn& policy_fn,
        const std::unordered_map<std::string, std::vector<double>>& action_dict,
        const KernelEvalFn& kernel_eval_fn
    );

    int max_steps() const { return max_steps_; }
    int max_depth() const { return max_depth_; }

private:
    SpeculativeGeodesicRouter router_;
    int max_steps_{100};
    int max_depth_{20};
};

} // namespace syn

#endif // GEODESIC_SEARCH_HPP
