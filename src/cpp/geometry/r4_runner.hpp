#ifndef R4_RUNNER_HPP
#define R4_RUNNER_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <cstdint>

#include "geometry/tree_rotor.hpp"

namespace syn {

struct R4Config {
    int width{1536};
    int heads{24};
    int context{384};
    int vocab_size{4096};
    std::string pattern{"rrarrarrarrarrar"};
    bool rotation{true};

    int lanes() const { return width / 4; }
    int head_dim() const { return width / heads; }
};

// Raw tensor weight storage for C++ inference
struct TensorWeight {
    std::vector<int64_t> shape;
    std::vector<float> data;

    TensorWeight() = default;
    TensorWeight(std::vector<int64_t> shape, std::vector<float> data)
        : shape(std::move(shape)), data(std::move(data)) {}

    size_t size() const { return data.size(); }
};

class R4RunnerCpp {
public:
    explicit R4RunnerCpp(R4Config config = R4Config());

    const R4Config& config() const { return cfg_; }
    void load_dummy_weights(); // Initializer for standalone testing/verification
    void set_weight(const std::string& name, const std::vector<int64_t>& shape, const std::vector<float>& data);

    // Forward pass: token sequence -> final hidden state (width) and next-token logits (vocab_size)
    void forward(
        const std::vector<uint16_t>& token_ids,
        std::vector<float>& out_hidden,
        std::vector<float>& out_logits
    );

    // Convenience evaluation of state string -> 64D goal embedding
    std::vector<float> embed_goal_64d(const std::vector<uint16_t>& token_ids);

private:
    R4Config cfg_;
    std::unordered_map<std::string, TensorWeight> weights_;

    // Core layer primitives
    void rms_norm(std::vector<float>& out, const std::vector<float>& x, const float* weight, int d);
    void linear(std::vector<float>& out, const std::vector<float>& x, const float* weight, const float* bias, int in_d, int out_d);
    
    // Sub-layer forward passes
    void recurrence_layer(int layer_idx, const std::vector<float>& in_x, int seq_len, std::vector<float>& out_x);
    void addressed_read_layer(int layer_idx, const std::vector<float>& in_x, int seq_len, std::vector<float>& out_x);
    void mlp_layer(int layer_idx, const std::vector<float>& in_x, int seq_len, std::vector<float>& out_x);
};

} // namespace syn

#endif // R4_RUNNER_HPP
