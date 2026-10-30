#include "geodesic_search.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace syn {

static double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static double vector_norm(const std::vector<double>& v) {
    return std::sqrt(dot_product(v, v));
}

std::vector<TacticCandidate> SpeculativeGeodesicRouter::rank_candidates(
    const std::vector<double>& tangent_vector,
    const std::unordered_map<std::string, std::vector<double>>& action_dict
) const {
    double norm_v = vector_norm(tangent_vector);
    if (norm_v < 1e-9) {
        throw std::invalid_argument("Zero tangent vector cannot determine geodesic direction.");
    }

    std::vector<TacticCandidate> candidates;
    candidates.reserve(action_dict.size());

    for (const auto& kv : action_dict) {
        const std::string& name = kv.first;
        const std::vector<double>& vec = kv.second;
        double norm_w = vector_norm(vec);
        double sim = 0.0;
        if (norm_w >= 1e-9) {
            sim = dot_product(tangent_vector, vec) / (norm_v * norm_w);
        }
        candidates.emplace_back(name, vec, sim);
    }

    std::sort(candidates.begin(), candidates.end(), [](const TacticCandidate& a, const TacticCandidate& b) {
        return a.score > b.score;
    });

    return candidates;
}

// NOTE: Rank candidates by cosine similarity to tangent vector (geodesic direction on S³).
// Higher score = closer alignment with the direction of steepest descent in proof space.
std::tuple<std::vector<TacticCandidate>, double, bool> SpeculativeGeodesicRouter::select_speculative_queue(
    const std::vector<TacticCandidate>& ranked
) const {
    if (ranked.empty()) {
        return std::make_tuple(std::vector<TacticCandidate>{}, 0.0, false);
    }
    if (ranked.size() == 1) {
        return std::make_tuple(std::vector<TacticCandidate>{ranked[0]}, 1.0, false);
    }

    double margin = ranked[0].score - ranked[1].score;
    if (margin >= margin_threshold_ || max_beam_width_ < 2) {
        return std::make_tuple(std::vector<TacticCandidate>{ranked[0]}, margin, false);
    } else {
        return std::make_tuple(std::vector<TacticCandidate>{ranked[0], ranked[1]}, margin, true);
    }
}

GeodesicSearchResult SpeculativeGeodesicProver::search(
    const std::string& initial_goal,
    const PolicyFn& policy_fn,
    const std::unordered_map<std::string, std::vector<double>>& action_dict,
    const KernelEvalFn& kernel_eval_fn
) {
    auto root = std::make_shared<SearchNode>("0", initial_goal, 0);
    std::vector<std::shared_ptr<SearchNode>> open_nodes = {root};

    int total_steps = 0;
    int kernel_calls = 0;
    int repl_calls_saved = 0;
    std::vector<std::string> proof_path;
    int max_depth_reached = 0;

    while (!open_nodes.empty() && total_steps < max_steps_) {
        auto current = open_nodes.front();
        open_nodes.erase(open_nodes.begin());
        total_steps++;
        if (current->depth > max_depth_reached) {
            max_depth_reached = current->depth;
        }

        if (current->depth >= max_depth_) {
            continue;
        }

        std::vector<double> tangent_v = policy_fn(current->goal_str);
        std::vector<TacticCandidate> ranked = router_.rank_candidates(tangent_v, action_dict);
        auto queue_info = router_.select_speculative_queue(ranked);
        const auto& queue = std::get<0>(queue_info);

        bool step_succeeded = false;
        for (size_t idx = 0; idx < queue.size(); ++idx) {
            kernel_calls++;
            const auto& candidate = queue[idx];
            auto result = kernel_eval_fn(current->goal_str, candidate.name);
            KernelStatus status = result.first;
            const auto& new_goals = result.second;

            if (status == KernelStatus::GoalsClosed) {
                proof_path.push_back(candidate.name);
                if (idx == 0 && queue.size() > 1) {
                    repl_calls_saved++;
                }
                return GeodesicSearchResult{
                    SearchStatus::SuccessProven,
                    proof_path,
                    total_steps,
                    kernel_calls,
                    repl_calls_saved,
                    current->depth + 1
                };
            } else if (status == KernelStatus::SubgoalsOpen) {
                proof_path.push_back(candidate.name);
                step_succeeded = true;
                for (size_t g_idx = 0; g_idx < new_goals.size(); ++g_idx) {
                    std::string child_id = current->state_id + "." + std::to_string(open_nodes.size());
                    open_nodes.push_back(std::make_shared<SearchNode>(
                        child_id, new_goals[g_idx], current->depth + 1, current, candidate.name
                    ));
                }
                if (idx == 0 && queue.size() > 1) {
                    repl_calls_saved++;
                }
                break;
            } else {
                // KernelError: fallback to next speculative candidate
                continue;
            }
        }

        if (!step_succeeded && open_nodes.empty()) {
            return GeodesicSearchResult{
                SearchStatus::SearchFailed,
                proof_path,
                total_steps,
                kernel_calls,
                repl_calls_saved,
                current->depth
            };
        }
    }

    SearchStatus final_status = (total_steps >= max_steps_) ? SearchStatus::BudgetExhausted : SearchStatus::SearchFailed;
    return GeodesicSearchResult{
        final_status,
        proof_path,
        total_steps,
        kernel_calls,
        repl_calls_saved,
        max_depth_reached
    };
}

} // namespace syn
