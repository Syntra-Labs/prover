#include "r4_runner.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <iostream>

namespace syn {

static inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978846f * (x + 0.044715f * x * x * x)));
}

static inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float softplus(float x) {
    return (x > 20.0f) ? x : std::log(1.0f + std::exp(x));
}

R4RunnerCpp::R4RunnerCpp(R4Config config) : cfg_(config) {}

void R4RunnerCpp::set_weight(const std::string& name, const std::vector<int64_t>& shape, const std::vector<float>& data) {
    weights_[name] = TensorWeight(shape, data);
}

void R4RunnerCpp::rms_norm(std::vector<float>& out, const std::vector<float>& x, const float* weight, int d) {
    out.resize(x.size());
    size_t count = x.size() / d;
    for (size_t b = 0; b < count; ++b) {
        const float* x_ptr = x.data() + b * d;
        float* o_ptr = out.data() + b * d;
        double sum_sq = 0.0;
        for (int i = 0; i < d; ++i) {
            sum_sq += static_cast<double>(x_ptr[i]) * x_ptr[i];
        }
        float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / d) + 1e-5f);
        for (int i = 0; i < d; ++i) {
            float w = weight ? weight[i] : 1.0f;
            o_ptr[i] = x_ptr[i] * inv_rms * w;
        }
    }
}

void R4RunnerCpp::linear(std::vector<float>& out, const std::vector<float>& x, const float* weight, const float* bias, int in_d, int out_d) {
    size_t batch = x.size() / in_d;
    out.assign(batch * out_d, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        const float* in_ptr = x.data() + b * in_d;
        float* out_ptr = out.data() + b * out_d;
        for (int o = 0; o < out_d; ++o) {
            float val = bias ? bias[o] : 0.0f;
            const float* w_row = weight ? (weight + o * in_d) : nullptr;
            if (w_row) {
                for (int i = 0; i < in_d; ++i) {
                    val += in_ptr[i] * w_row[i];
                }
            }
            out_ptr[o] = val;
        }
    }
}

void R4RunnerCpp::load_dummy_weights() {
    int W = cfg_.width;
    int H = cfg_.heads;
    int hd = cfg_.head_dim();
    int lanes = cfg_.lanes();

    // Embedding & LM Head
    weights_["embedding.weight"] = TensorWeight({cfg_.vocab_size, W}, std::vector<float>(cfg_.vocab_size * W, 0.01f));
    weights_["final_norm.weight"] = TensorWeight({W}, std::vector<float>(W, 1.0f));

    // For each layer in pattern
    for (size_t l = 0; l < cfg_.pattern.size(); ++l) {
        std::string p = "layers." + (l < 10 ? "0" + std::to_string(l) : std::to_string(l));
        char type = cfg_.pattern[l];
        if (type == 'r') {
            weights_[p + ".rec_norm.weight"] = TensorWeight({W}, std::vector<float>(W, 1.0f));
            weights_[p + ".rec.in.weight"] = TensorWeight({2 * W, W}, std::vector<float>(2 * W * W, 0.01f));
            weights_[p + ".rec.gate.weight"] = TensorWeight({5 * lanes, W}, std::vector<float>(5 * lanes * W, 0.01f));
            weights_[p + ".rec.gate.bias"] = TensorWeight({5 * lanes}, std::vector<float>(5 * lanes, 0.0f));
            weights_[p + ".rec.conv.weight"] = TensorWeight({4, W}, std::vector<float>(4 * W, 0.25f));
            weights_[p + ".rec.conv.bias"] = TensorWeight({W}, std::vector<float>(W, 0.0f));
            weights_[p + ".rec.decay"] = TensorWeight({lanes}, std::vector<float>(lanes, 1.0f));
            weights_[p + ".rec.out.weight"] = TensorWeight({W, W}, std::vector<float>(W * W, 0.01f));
        } else if (type == 'a') {
            weights_[p + ".read_norm.weight"] = TensorWeight({W}, std::vector<float>(W, 1.0f));
            weights_[p + ".read.query.weight"] = TensorWeight({W, W}, std::vector<float>(W * W, 0.01f));
            weights_[p + ".read.key.weight"] = TensorWeight({W, W}, std::vector<float>(W * W, 0.01f));
            weights_[p + ".read.value.weight"] = TensorWeight({W, W}, std::vector<float>(W * W, 0.01f));
            weights_[p + ".read.null.weight"] = TensorWeight({H, W}, std::vector<float>(H * W, 0.01f));
            weights_[p + ".read.null.bias"] = TensorWeight({H}, std::vector<float>(H, 0.0f));
            weights_[p + ".read.age"] = TensorWeight({H, cfg_.context}, std::vector<float>(H * cfg_.context, 0.0f));
            weights_[p + ".read.log_beta"] = TensorWeight({H}, std::vector<float>(H, 0.0f));
            weights_[p + ".read.offset"] = TensorWeight({H}, std::vector<float>(H, 1.0f));
            weights_[p + ".read.out.weight"] = TensorWeight({W, W}, std::vector<float>(W * W, 0.01f));
        }
        weights_[p + ".mlp_norm.weight"] = TensorWeight({W}, std::vector<float>(W, 1.0f));
        int mlp_hidden = W * 8 / 3;
        weights_[p + ".mlp.gate.weight"] = TensorWeight({mlp_hidden, W}, std::vector<float>(mlp_hidden * W, 0.01f));
        weights_[p + ".mlp.up.weight"] = TensorWeight({mlp_hidden, W}, std::vector<float>(mlp_hidden * W, 0.01f));
        weights_[p + ".mlp.down.weight"] = TensorWeight({W, mlp_hidden}, std::vector<float>(W * mlp_hidden, 0.01f));
    }
}

void R4RunnerCpp::recurrence_layer(int layer_idx, const std::vector<float>& in_x, int seq_len, std::vector<float>& out_x) {
    int W = cfg_.width;
    int lanes = cfg_.lanes();
    std::string p = "layers." + (layer_idx < 10 ? "0" + std::to_string(layer_idx) : std::to_string(layer_idx));

    std::vector<float> u;
    rms_norm(u, in_x, weights_[p + ".rec_norm.weight"].data.data(), W);

    std::vector<float> branches;
    linear(branches, u, weights_[p + ".rec.in.weight"].data.data(), nullptr, W, 2 * W);

    std::vector<float> gates;
    linear(gates, u, weights_[p + ".rec.gate.weight"].data.data(), weights_[p + ".rec.gate.bias"].data.data(), W, 5 * lanes);

    const float* conv_w = weights_[p + ".rec.conv.weight"].data.data();
    const float* conv_b = weights_[p + ".rec.conv.bias"].data.data();
    const float* decay = weights_[p + ".rec.decay"].data.data();

    std::vector<float> log_a(lanes);
    for (int l = 0; l < lanes; ++l) {
        log_a[l] = -softplus(-decay[l]);
    }

    std::vector<float> drive(seq_len * W);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < W; ++i) {
            float val = conv_b[i];
            for (int s = 0; s < 4; ++s) {
                if (t >= s) {
                    float a_val = branches[(t - s) * (2 * W) + i];
                    val += a_val * conv_w[s * W + i];
                }
            }
            drive[t * W + i] = val;
        }
    }

    // Recurrent quaternion state scan
    std::vector<Quat> lane_states(lanes, Quat(0, 0, 0, 0));
    std::vector<float> full_states(seq_len * W);

    for (int t = 0; t < seq_len; ++t) {
        const float* g_t = gates.data() + t * (5 * lanes);
        for (int l = 0; l < lanes; ++l) {
            float lambda_val = std::exp(8.0f * sigmoid(g_t[l]) * log_a[l]);
            float comp = 1.0f - lambda_val * lambda_val;
            float keep = (comp < 1e-6f) ? 1e-3f : std::sqrt(std::max(comp, 1e-6f));

            const float* rq = g_t + lanes + l * 4;
            Quat raw(rq[0], rq[1], rq[2], rq[3]);
            Quat unit_q = raw.normalize();

            // Drive quaternion for lane l
            const float* dl = drive.data() + t * W + l * 4;
            Quat d_quat(dl[0], dl[1], dl[2], dl[3]);

            Quat q_trans(unit_q.w * lambda_val, unit_q.x * lambda_val, unit_q.y * lambda_val, unit_q.z * lambda_val);

            // Hamilton multiplication: q_trans * state + keep * drive
            Quat updated = q_trans.mul(lane_states[l]);
            lane_states[l] = Quat(
                updated.w + keep * d_quat.w,
                updated.x + keep * d_quat.x,
                updated.y + keep * d_quat.y,
                updated.z + keep * d_quat.z
            );

            full_states[t * W + l * 4 + 0] = static_cast<float>(lane_states[l].w);
            full_states[t * W + l * 4 + 1] = static_cast<float>(lane_states[l].x);
            full_states[t * W + l * 4 + 2] = static_cast<float>(lane_states[l].y);
            full_states[t * W + l * 4 + 3] = static_cast<float>(lane_states[l].z);
        }
    }

    // Multiply by gelu_tanh(g_branch)
    std::vector<float> gated(seq_len * W);
    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < W; ++i) {
            float g_val = branches[t * (2 * W) + W + i];
            gated[t * W + i] = full_states[t * W + i] * gelu_tanh(g_val);
        }
    }

    linear(out_x, gated, weights_[p + ".rec.out.weight"].data.data(), nullptr, W, W);
}

void R4RunnerCpp::addressed_read_layer(int layer_idx, const std::vector<float>& in_x, int seq_len, std::vector<float>& out_x) {
    int W = cfg_.width;
    int H = cfg_.heads;
    int hd = cfg_.head_dim();
    std::string p = "layers." + (layer_idx < 10 ? "0" + std::to_string(layer_idx) : std::to_string(layer_idx));

    std::vector<float> u;
    rms_norm(u, in_x, weights_[p + ".read_norm.weight"].data.data(), W);

    std::vector<float> q, k, v, null_logits;
    linear(q, u, weights_[p + ".read.query.weight"].data.data(), nullptr, W, W);
    linear(k, u, weights_[p + ".read.key.weight"].data.data(), nullptr, W, W);
    linear(v, u, weights_[p + ".read.value.weight"].data.data(), nullptr, W, W);
    linear(null_logits, u, weights_[p + ".read.null.weight"].data.data(), weights_[p + ".read.null.bias"].data.data(), W, H);

    const float* log_beta = weights_[p + ".read.log_beta"].data.data();
    const float* offset = weights_[p + ".read.offset"].data.data();
    const float* age_table = weights_[p + ".read.age"].data.data();

    std::vector<float> merged(seq_len * W, 0.0f);

    for (int h = 0; h < H; ++h) {
        float beta = std::exp(log_beta[h]);
        float off = offset[h];

        for (int t = 0; t < seq_len; ++t) {
            const float* qt = q.data() + t * W + h * hd;
            std::vector<float> scores(seq_len);
            float max_score = -1e9f;

            for (int j = 0; j <= t; ++j) {
                const float* kj = k.data() + j * W + h * hd;
                double d_sq = 0.0;
                for (int d = 0; d < hd; ++d) {
                    double diff = static_cast<double>(qt[d]) - kj[d];
                    d_sq += diff * diff;
                }
                float dist = static_cast<float>(std::sqrt(std::max(d_sq, 1e-7)));
                float age = age_table[h * cfg_.context + (t - j)];
                float s = -beta * (dist - off) + age;
                scores[j] = s;
                if (s > max_score) max_score = s;
            }

            float null_val = null_logits[t * H + h];
            if (null_val > max_score) max_score = null_val;

            double exp_sum = std::exp(null_val - max_score);
            for (int j = 0; j <= t; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                exp_sum += scores[j];
            }

            // Weighted sum of values
            for (int d = 0; d < hd; ++d) {
                double val_acc = 0.0;
                for (int j = 0; j <= t; ++j) {
                    float prob = static_cast<float>(scores[j] / exp_sum);
                    val_acc += prob * v[j * W + h * hd + d];
                }
                merged[t * W + h * hd + d] = static_cast<float>(val_acc);
            }
        }
    }

    linear(out_x, merged, weights_[p + ".read.out.weight"].data.data(), nullptr, W, W);
}

void R4RunnerCpp::mlp_layer(int layer_idx, const std::vector<float>& in_x, int seq_len, std::vector<float>& out_x) {
    int W = cfg_.width;
    int mlp_hidden = W * 8 / 3;
    std::string p = "layers." + (layer_idx < 10 ? "0" + std::to_string(layer_idx) : std::to_string(layer_idx));

    std::vector<float> u;
    rms_norm(u, in_x, weights_[p + ".mlp_norm.weight"].data.data(), W);

    std::vector<float> gate, up;
    linear(gate, u, weights_[p + ".mlp.gate.weight"].data.data(), nullptr, W, mlp_hidden);
    linear(up, u, weights_[p + ".mlp.up.weight"].data.data(), nullptr, W, mlp_hidden);

    std::vector<float> mixed(seq_len * mlp_hidden);
    for (size_t i = 0; i < mixed.size(); ++i) {
        mixed[i] = silu(gate[i]) * up[i];
    }

    linear(out_x, mixed, weights_[p + ".mlp.down.weight"].data.data(), nullptr, mlp_hidden, W);
}

void R4RunnerCpp::forward(
    const std::vector<uint16_t>& token_ids,
    std::vector<float>& out_hidden,
    std::vector<float>& out_logits
) {
    if (token_ids.empty()) return;
    int W = cfg_.width;
    size_t seq_len = std::min(token_ids.size(), static_cast<size_t>(cfg_.context));
    size_t start_idx = token_ids.size() - seq_len;

    // Embedding lookup
    const float* embed_w = weights_["embedding.weight"].data.data();
    std::vector<float> x(seq_len * W);
    for (size_t t = 0; t < seq_len; ++t) {
        uint16_t tid = token_ids[start_idx + t];
        if (tid < cfg_.vocab_size) {
            std::memcpy(x.data() + t * W, embed_w + tid * W, W * sizeof(float));
        } else {
            std::fill(x.data() + t * W, x.data() + (t + 1) * W, 0.0f);
        }
    }

    // Layer stack
    for (size_t l = 0; l < cfg_.pattern.size(); ++l) {
        char type = cfg_.pattern[l];
        std::vector<float> mixed;
        if (type == 'r') {
            recurrence_layer(l, x, seq_len, mixed);
        } else if (type == 'a') {
            addressed_read_layer(l, x, seq_len, mixed);
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += mixed[i];

        std::vector<float> mlp_out;
        mlp_layer(l, x, seq_len, mlp_out);
        for (size_t i = 0; i < x.size(); ++i) x[i] += mlp_out[i];
    }

    // Final RMSNorm
    std::vector<float> norm_out;
    rms_norm(norm_out, x, weights_["final_norm.weight"].data.data(), W);

    // Extract last token hidden state
    out_hidden.assign(norm_out.data() + (seq_len - 1) * W, norm_out.data() + seq_len * W);

    // Compute next-token logits: Linear(last_hidden, embed_weight)
    out_logits.resize(cfg_.vocab_size);
    for (int v = 0; v < cfg_.vocab_size; ++v) {
        float dot = 0.0f;
        const float* v_ptr = embed_w + v * W;
        for (int i = 0; i < W; ++i) {
            dot += out_hidden[i] * v_ptr[i];
        }
        out_logits[v] = dot;
    }
}

std::vector<float> R4RunnerCpp::embed_goal_64d(const std::vector<uint16_t>& token_ids) {
    std::vector<float> hidden, logits;
    forward(token_ids, hidden, logits);
    if (hidden.size() >= 64) {
        return std::vector<float>(hidden.begin(), hidden.begin() + 64);
    }
    std::vector<float> res(64, 0.0f);
    std::copy(hidden.begin(), hidden.end(), res.begin());
    return res;
}

} // namespace syn
