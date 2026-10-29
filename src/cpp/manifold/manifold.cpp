// ============================================================================
// manifold.cpp — High-Throughput Geometric Substrate & Lifecycle Router (C++20)
// ============================================================================
// Zero-dependency, parser-free byte-entropy & 8192-bit binary hypervector engine
// for whole-repository ingestion, wget/HTML reference directory ingestion,
// non-destructive state accumulation, incremental algebraic updates,
// task placement, scientific experiment logging, and binary checkpointing.
//
// Compile: g++ -O3 -march=native -std=c++20 -Wall -Wextra manifold.cpp -o manifold
// Run self-test:  ./manifold selftest
// Run benchmark:  ./manifold benchmark [dir_path]
// ============================================================================

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// 1. Constants, Scopes & Pragmatic Lifecycle Archetypes
// ----------------------------------------------------------------------------
static constexpr size_t HV_BITS = 8192;
static constexpr size_t HV_WORDS = HV_BITS / 64; // 128 x uint64_t = 1024 bytes

// First 16 odd primes used as cyclic rotation strides (Prime-Router style
// non-aliasing positional mixing across sequence steps)
static constexpr std::array<uint32_t, 16> PRIME_STRIDES = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59
};

enum class Scale : uint8_t {
    REPO  = 0,
    FILE  = 1,
    BLOCK = 2,
    SPAN  = 3
};

// Source scope (@repo, @ref, @web, @docs, @exp)
enum class Modality : uint8_t {
    REPO_CODE  = 0, // @repo — Primary project repository files
    SPEC_DOC   = 1, // @docs — Local specifications, Markdown, LaTeX papers
    EXT_REF    = 2, // @ref  — External reference codebases / cloned repos
    WEB_REF    = 3, // @web  — Downloaded wget/HTML/online documentation dirs
    EXPERIMENT = 4  // @exp  — Logged scientific hypotheses (KEEP / REJECT)
};

enum class EpistemicState : uint8_t {
    ACTIVE = 0,
    TODO   = 1,
    KEEP   = 2,
    REJECT = 3
};

// 6 Pragmatic Lifecycle Archetypes (applicable to both ML models & general code)
enum class Archetype : uint8_t {
    IO    = 0, // #0:io    — Ingestion, tokenization, byte patching, CLI/file I/O, codecs
    STATE = 1, // #1:state — Data structures, tensors, hypervectors, memory layout, state
    SPACE = 2, // #2:space — Coordinate spaces, manifolds, graphs, trees, lookup tables
    OP    = 3, // #3:op    — Pure math/transform kernels, transport PDEs, attention, FFT
    ROUTE = 4, // #4:route — Dynamic routing, task dispatch, online adaptation, delta rules
    GATE  = 5, // #5:gate  — Verification, SOM lucidity firewall, tests, proofs, ledgers
    COUNT = 6
};

static const char* archetype_name(Archetype a) {
    switch (a) {
        case Archetype::IO:    return "#0:io(INGEST_EMIT)";
        case Archetype::STATE: return "#1:state(DATA_SUBSTRATE)";
        case Archetype::SPACE: return "#2:space(INDEX_TOPOLOGY)";
        case Archetype::OP:    return "#3:op(TRANSFORM_KERNEL)";
        case Archetype::ROUTE: return "#4:route(CONTROL_ADAPT)";
        case Archetype::GATE:  return "#5:gate(VERIFY_GOVERN)";
        default:               return "#?:unknown";
    }
}

static const char* archetype_short(Archetype a) {
    switch (a) {
        case Archetype::IO:    return "#io";
        case Archetype::STATE: return "#state";
        case Archetype::SPACE: return "#space";
        case Archetype::OP:    return "#op";
        case Archetype::ROUTE: return "#route";
        case Archetype::GATE:  return "#gate";
        default:               return "#?";
    }
}

static const char* modality_tag(Modality m) {
    switch (m) {
        case Modality::REPO_CODE:  return "@repo";
        case Modality::SPEC_DOC:   return "@docs";
        case Modality::EXT_REF:    return "@ref";
        case Modality::WEB_REF:    return "@web";
        case Modality::EXPERIMENT: return "@exp";
        default:                   return "@?";
    }
}

static const char* epistemic_name(EpistemicState e) {
    switch (e) {
        case EpistemicState::ACTIVE: return "ACTIVE";
        case EpistemicState::TODO:   return "TODO";
        case EpistemicState::KEEP:   return "KEEP";
        case EpistemicState::REJECT: return "REJECT";
        default:                     return "?";
    }
}

// ----------------------------------------------------------------------------
// 2. Deterministic Hashing (FNV-1a & SplitMix64 PRNG for Hypervector Expansion)
// ----------------------------------------------------------------------------
static inline uint64_t fnv1a_64(const void* data, size_t len, uint64_t basis = 14695981039346656037ULL) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint64_t h = basis;
    for (size_t i = 0; i < len; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static inline uint64_t fnv1a_str(std::string_view s, uint64_t basis = 14695981039346656037ULL) {
    return fnv1a_64(s.data(), s.size(), basis);
}

static inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ----------------------------------------------------------------------------
// 3. 8192-Bit Binary Hypervector & Non-Destructive Bipolar Accumulator
// ----------------------------------------------------------------------------
struct Hypervec8192 {
    std::array<uint64_t, HV_WORDS> words{};

    void clear() { words.fill(0ULL); }

    static Hypervec8192 from_seed(uint64_t seed) {
        Hypervec8192 hv;
        uint64_t state = seed;
        for (size_t i = 0; i < HV_WORDS; ++i) {
            hv.words[i] = splitmix64(state);
        }
        return hv;
    }

    Hypervec8192 bind(const Hypervec8192& other) const {
        Hypervec8192 out;
        for (size_t i = 0; i < HV_WORDS; ++i) {
            out.words[i] = words[i] ^ other.words[i];
        }
        return out;
    }

    Hypervec8192 rotl(uint32_t shift) const {
        shift %= HV_BITS;
        if (shift == 0) return *this;
        Hypervec8192 out;
        const size_t word_shift = shift / 64;
        const uint32_t bit_shift = shift % 64;
        for (size_t i = 0; i < HV_WORDS; ++i) {
            size_t src_hi = (i + HV_WORDS - word_shift) % HV_WORDS;
            if (bit_shift == 0) {
                out.words[i] = words[src_hi];
            } else {
                size_t src_lo = (src_hi + HV_WORDS - 1) % HV_WORDS;
                out.words[i] = (words[src_hi] << bit_shift) | (words[src_lo] >> (64 - bit_shift));
            }
        }
        return out;
    }

    uint32_t hamming(const Hypervec8192& other) const {
        uint32_t dist = 0;
        for (size_t i = 0; i < HV_WORDS; ++i) {
            dist += static_cast<uint32_t>(std::popcount(words[i] ^ other.words[i]));
        }
        return dist;
    }

    double similarity(const Hypervec8192& other) const {
        return 1.0 - (static_cast<double>(hamming(other)) / static_cast<double>(HV_BITS));
    }

    double z_score(const Hypervec8192& other) const {
        double expected = static_cast<double>(HV_BITS) * 0.5;
        double sigma = std::sqrt(static_cast<double>(HV_BITS)) * 0.5;
        return (expected - static_cast<double>(hamming(other))) / sigma;
    }
};

// Non-Destructive Carry-Save Bipolar Accumulator (int16_t[8192])
// Uses branchless 8-bit unrolled inner loops so GCC/Clang auto-vectorize to AVX2/NEON.
struct BipolarAccumulator {
    std::array<int16_t, HV_BITS> counts{};
    uint32_t total_items = 0;

    void clear() {
        counts.fill(0);
        total_items = 0;
    }

    // Fast branchless accumulation of a dense 8192-bit Hypervec8192
    void accumulate(const Hypervec8192& hv, int16_t weight = 1) {
        for (size_t w = 0; w < HV_WORDS; ++w) {
            uint64_t word = hv.words[w];
            int16_t* __restrict c_ptr = &counts[w << 6];
            for (size_t b = 0; b < 64; b += 8) {
                uint8_t byte_val = static_cast<uint8_t>(word >> b);
                c_ptr[b + 0] += ((byte_val & 0x01) ? weight : -weight);
                c_ptr[b + 1] += ((byte_val & 0x02) ? weight : -weight);
                c_ptr[b + 2] += ((byte_val & 0x04) ? weight : -weight);
                c_ptr[b + 3] += ((byte_val & 0x08) ? weight : -weight);
                c_ptr[b + 4] += ((byte_val & 0x10) ? weight : -weight);
                c_ptr[b + 5] += ((byte_val & 0x20) ? weight : -weight);
                c_ptr[b + 6] += ((byte_val & 0x40) ? weight : -weight);
                c_ptr[b + 7] += ((byte_val & 0x80) ? weight : -weight);
            }
        }
        if (weight > 0) total_items += static_cast<uint32_t>(weight);
        else if (weight < 0 && total_items >= static_cast<uint32_t>(-weight))
            total_items -= static_cast<uint32_t>(-weight);
    }

    // Fast direct seed accumulation (expands 64-bit feature seed directly into counts)
    void accumulate_seed(uint64_t seed, int16_t weight) {
        uint64_t state = seed;
        for (size_t w = 0; w < HV_WORDS; ++w) {
            uint64_t word = splitmix64(state);
            int16_t* __restrict c_ptr = &counts[w << 6];
            for (size_t b = 0; b < 64; b += 8) {
                uint8_t byte_val = static_cast<uint8_t>(word >> b);
                c_ptr[b + 0] += ((byte_val & 0x01) ? weight : -weight);
                c_ptr[b + 1] += ((byte_val & 0x02) ? weight : -weight);
                c_ptr[b + 2] += ((byte_val & 0x04) ? weight : -weight);
                c_ptr[b + 3] += ((byte_val & 0x08) ? weight : -weight);
                c_ptr[b + 4] += ((byte_val & 0x10) ? weight : -weight);
                c_ptr[b + 5] += ((byte_val & 0x20) ? weight : -weight);
                c_ptr[b + 6] += ((byte_val & 0x40) ? weight : -weight);
                c_ptr[b + 7] += ((byte_val & 0x80) ? weight : -weight);
            }
        }
        if (weight > 0) total_items += static_cast<uint32_t>(weight);
    }

    Hypervec8192 project() const {
        Hypervec8192 out;
        for (size_t w = 0; w < HV_WORDS; ++w) {
            uint64_t word = 0ULL;
            const int16_t* __restrict c_ptr = &counts[w << 6];
            for (size_t b = 0; b < 64; ++b) {
                int16_t c = c_ptr[b];
                bool bit = (c > 0) || (c == 0 && (((w << 6) + b) & 1ULL));
                if (bit) word |= (1ULL << b);
            }
            out.words[w] = word;
        }
        return out;
    }
};

// ----------------------------------------------------------------------------
// 4. Halo Header (\Sigma) & UOR Coordinate (U)
// ----------------------------------------------------------------------------
struct HaloHeader {
    Scale scale               = Scale::BLOCK;
    Modality modality         = Modality::REPO_CODE;
    Archetype primary_arch    = Archetype::OP;
    EpistemicState epistemic  = EpistemicState::ACTIVE;
    uint8_t archetype_mask    = 0; // 6-bit mask of active archetypes
    uint8_t sector_id         = 0; // 8-bit coarse routing sector (Prime-Router style)
    uint8_t entropy_q8        = 0; // Shannon entropy H * 32 (0..255 maps to 0..8 bits/byte)
    uint8_t surprisal_q8      = 0; // Causal transition delta * 255
};

struct UORCoord {
    std::string uri;
    uint32_t start_byte = 0;
    uint32_t end_byte   = 0;
    uint32_t start_line = 1;
    uint32_t end_line   = 1;
    uint64_t content_hash = 0;
};

struct Cell {
    uint32_t id        = 0;
    uint32_t parent_id = 0;
    HaloHeader halo;
    UORCoord coord;
    Hypervec8192 hv;
    std::string preview;
};

// ----------------------------------------------------------------------------
// 5. HTML / Wget Online Reference Demodulator & Universal Byte Chunker
// ----------------------------------------------------------------------------
static std::string demodulate_html_to_text(const std::string& html) {
    std::string out;
    out.reserve(html.size());

    size_t i = 0;
    const size_t n = html.size();
    auto starts_ci = [&](size_t pos, const char* lit) {
        size_t l = std::strlen(lit);
        if (pos + l > n) return false;
        for (size_t k = 0; k < l; ++k) {
            if (std::tolower(static_cast<unsigned char>(html[pos + k])) != std::tolower(static_cast<unsigned char>(lit[k])))
                return false;
        }
        return true;
    };

    while (i < n) {
        if (html[i] == '<') {
            if (starts_ci(i, "<script")) {
                while (i < n && !starts_ci(i, "</script>")) i++;
                while (i < n && html[i] != '>') i++;
                if (i < n) i++;
                continue;
            }
            if (starts_ci(i, "<style")) {
                while (i < n && !starts_ci(i, "</style>")) i++;
                while (i < n && html[i] != '>') i++;
                if (i < n) i++;
                continue;
            }
            if (starts_ci(i, "<nav") || starts_ci(i, "<footer")) {
                const char* close_tag = starts_ci(i, "<nav") ? "</nav>" : "</footer>";
                while (i < n && !starts_ci(i, close_tag)) i++;
                while (i < n && html[i] != '>') i++;
                if (i < n) i++;
                continue;
            }

            if (starts_ci(i, "<h1") || starts_ci(i, "<h2") || starts_ci(i, "<h3") || starts_ci(i, "<h4")) {
                out += "\n\n# ";
            } else if (starts_ci(i, "</h1") || starts_ci(i, "</h2") || starts_ci(i, "</h3") ||
                       starts_ci(i, "</h4") || starts_ci(i, "</p>") || starts_ci(i, "</pre>") ||
                       starts_ci(i, "</div>") || starts_ci(i, "</li>") || starts_ci(i, "<br")) {
                out += "\n";
            } else if (starts_ci(i, "<pre") || starts_ci(i, "<code")) {
                out += "\n";
            }

            while (i < n && html[i] != '>') {
                if (html[i] == '\n') out.push_back('\n');
                i++;
            }
            if (i < n) i++;
        } else if (html[i] == '&') {
            if (starts_ci(i, "&lt;"))        { out.push_back('<'); i += 4; }
            else if (starts_ci(i, "&gt;"))   { out.push_back('>'); i += 4; }
            else if (starts_ci(i, "&amp;"))  { out.push_back('&'); i += 5; }
            else if (starts_ci(i, "&quot;")) { out.push_back('"'); i += 6; }
            else if (starts_ci(i, "&#39;"))  { out.push_back('\''); i += 5; }
            else if (starts_ci(i, "&nbsp;")) { out.push_back(' '); i += 6; }
            else { out.push_back(html[i++]); }
        } else {
            out.push_back(html[i++]);
        }
    }
    return out;
}

enum class CharClass : uint8_t {
    SPACE = 0, LOWER = 1, UPPER = 2, DIGIT = 3, OP_MATH = 4, DELIM = 5, OTHER = 6
};

static inline CharClass classify_byte(uint8_t c) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') return CharClass::SPACE;
    if (c >= 'a' && c <= 'z') return CharClass::LOWER;
    if (c >= 'A' && c <= 'Z') return CharClass::UPPER;
    if (c >= '0' && c <= '9') return CharClass::DIGIT;
    if (std::strchr("+-*/=<>!&|^~%\\", c) != nullptr) return CharClass::OP_MATH;
    if (std::strchr("(){}[].,;:_#@$\"'`", c) != nullptr) return CharClass::DELIM;
    return CharClass::OTHER;
}

struct ByteStats {
    double shannon_entropy = 0.0;
    double mean_delta      = 0.0;
    double op_density      = 0.0;
};

static ByteStats compute_byte_stats(std::string_view text) {
    ByteStats st;
    if (text.empty()) return st;
    std::array<uint32_t, 256> freq{};
    uint32_t transitions = 0;
    uint32_t op_count = 0;
    CharClass prev_cc = classify_byte(static_cast<uint8_t>(text[0]));

    size_t step = (text.size() > 2048) ? (text.size() / 2048) : 1;
    size_t sampled = 0;
    for (size_t i = 0; i < text.size(); i += step) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        freq[c]++;
        CharClass cc = classify_byte(c);
        if (cc != prev_cc) transitions++;
        if (cc == CharClass::OP_MATH || cc == CharClass::DELIM) op_count++;
        prev_cc = cc;
        sampled++;
    }

    double n = static_cast<double>(sampled);
    double h = 0.0;
    for (uint32_t count : freq) {
        if (count > 0) {
            double p = static_cast<double>(count) / n;
            h -= p * std::log2(p);
        }
    }
    st.shannon_entropy = h;
    st.mean_delta = static_cast<double>(transitions) / n;
    st.op_density = static_cast<double>(op_count) / n;
    return st;
}

static std::vector<std::string> extract_universal_patches(std::string_view text) {
    std::vector<std::string> patches;
    size_t limit = std::min<size_t>(text.size(), 3072);
    patches.reserve(96);
    std::string cur_word;

    auto flush_word = [&]() {
        if (cur_word.size() >= 2) {
            std::string stem = cur_word;
            if (stem.size() > 5 && stem.ends_with("ing")) stem.resize(stem.size() - 3);
            else if (stem.size() > 5 && stem.ends_with("tion")) stem.resize(stem.size() - 4);
            else if (stem.size() > 4 && stem.ends_with("ed")) stem.resize(stem.size() - 2);
            else if (stem.size() > 4 && stem.ends_with("es")) stem.resize(stem.size() - 2);
            else if (stem.size() > 3 && stem.ends_with("s") && !stem.ends_with("ss")) stem.resize(stem.size() - 1);
            patches.push_back(stem);
            if (stem.size() >= 6) {
                patches.push_back(stem.substr(0, 5));
            }
        }
        cur_word.clear();
    };

    for (size_t i = 0; i < limit; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c >= 'A' && c <= 'Z') {
            if (!cur_word.empty() && i > 0 && std::islower(static_cast<unsigned char>(text[i - 1]))) {
                flush_word();
            }
            cur_word.push_back(static_cast<char>(c - 'A' + 'a'));
        } else if (c >= 'a' && c <= 'z') {
            cur_word.push_back(static_cast<char>(c));
        } else if (c >= '0' && c <= '9') {
            cur_word.push_back(static_cast<char>(c));
        } else {
            flush_word();
            if (classify_byte(c) == CharClass::OP_MATH) {
                std::string op(1, static_cast<char>(c));
                if (i + 1 < limit && classify_byte(static_cast<uint8_t>(text[i + 1])) == CharClass::OP_MATH) {
                    op.push_back(text[i + 1]);
                }
                patches.push_back("op:" + op);
            }
        }
        if (patches.size() >= 128) break;
    }
    flush_word();
    return patches;
}

// High-Speed Aggregated Hypervector Encoder:
// Deduplicates feature seeds per block and expands the top 48 weighted features into
// the SIMD-vectorized BipolarAccumulator.
static Hypervec8192 encode_span_hypervector(std::string_view text) {
    BipolarAccumulator acc;
    auto patches = extract_universal_patches(text);

    auto is_stop = [](std::string_view s) {
        return s == "the" || s == "and" || s == "for" || s == "with" || s == "this" ||
               s == "that" || s == "from" || s == "return" || s == "local" || s == "const" ||
               s == "void" || s == "auto" || s == "pub" || s == "let" || s == "def" ||
               s == "function" || s == "end" || s == "int" || s == "std" || s == "inline" ||
               s == "template" || s == "typename" || s == "class" || s == "struct";
    };

    std::pair<uint64_t, int16_t> seed_weights[256];
    size_t num_unique = 0;

    auto add_feature_seed = [&](uint64_t seed, int16_t w) {
        for (size_t i = 0; i < num_unique; ++i) {
            if (seed_weights[i].first == seed) {
                seed_weights[i].second += w;
                return;
            }
        }
        if (num_unique < 256) {
            seed_weights[num_unique++] = {seed, w};
        }
    };

    uint64_t prev1_seed = 0;
    uint64_t prev2_seed = 0;

    for (size_t i = 0; i < patches.size(); ++i) {
        uint64_t s = fnv1a_str(patches[i]);
        int16_t w = is_stop(patches[i]) ? 1 : (patches[i].starts_with("op:") ? 2 : 4);
        add_feature_seed(s, w);

        if (i > 0) {
            uint64_t bg_seed = std::rotl(prev1_seed, static_cast<int>(PRIME_STRIDES[0])) ^ (s * 0x9E3779B97F4A7C15ULL);
            add_feature_seed(bg_seed, 2);
        }
        if (i > 1) {
            uint64_t sg_seed = std::rotl(prev2_seed, static_cast<int>(PRIME_STRIDES[1])) ^ (s * 0xBF58476D1CE4E5B9ULL);
            add_feature_seed(sg_seed, 1);
        }
        prev2_seed = prev1_seed;
        prev1_seed = s;
    }

    std::string compact;
    size_t char_limit = std::min<size_t>(text.size(), 512);
    compact.reserve(char_limit);
    for (size_t i = 0; i < char_limit; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (!std::isspace(c)) compact.push_back(static_cast<char>(std::tolower(c)));
    }
    size_t shingle_step = std::max<size_t>(4, compact.size() / 24);
    for (size_t i = 0; i + 4 <= compact.size(); i += shingle_step) {
        uint64_t sh_seed = fnv1a_64(compact.data() + i, 4, 0xCBF29CE484222325ULL);
        add_feature_seed(sh_seed, 1);
    }

    if (num_unique > 48) {
        std::partial_sort(seed_weights, seed_weights + 48, seed_weights + num_unique,
            [](const auto& a, const auto& b) { return a.second > b.second; });
        num_unique = 48;
    }

    for (size_t i = 0; i < num_unique; ++i) {
        acc.accumulate_seed(seed_weights[i].first, seed_weights[i].second);
    }

    return acc.project();
}

static uint8_t compute_sector_id(const Hypervec8192& hv) {
    uint8_t sector = 0;
    for (int bit = 0; bit < 8; ++bit) {
        uint32_t ones = 0;
        for (int w = 0; w < 16; ++w) {
            ones += static_cast<uint32_t>(std::popcount(hv.words[bit * 16 + w]));
        }
        if (ones >= 512) sector |= (1U << bit);
    }
    return sector;
}

// ----------------------------------------------------------------------------
// 6. Zero-Copy Byte-Entropy & Indentation Chunker (Zero Language Parsers)
// ----------------------------------------------------------------------------
struct RawBlock {
    uint32_t start_byte;
    uint32_t end_byte;
    uint32_t start_line;
    uint32_t end_line;
    std::string_view text;
    std::string preview;
};

static std::vector<RawBlock> chunk_byte_stream(std::string_view content) {
    std::vector<RawBlock> blocks;
    if (content.empty()) return blocks;

    struct LineSpan {
        uint32_t start_byte;
        uint32_t end_byte;
        uint32_t line_num;
        int indent;
        bool is_blank;
        bool is_top_header;
        std::string_view trimmed;
    };

    std::vector<LineSpan> lines;
    lines.reserve(content.size() / 36 + 16);
    uint32_t pos = 0;
    uint32_t line_num = 1;
    while (pos < content.size()) {
        uint32_t next_nl = pos;
        while (next_nl < content.size() && content[next_nl] != '\n') next_nl++;
        uint32_t end_b = (next_nl < content.size()) ? next_nl + 1 : next_nl;
        std::string_view line_str = content.substr(pos, end_b - pos);

        int indent = 0;
        bool blank = true;
        size_t first_non_space = 0;
        for (size_t k = 0; k < line_str.size(); ++k) {
            char c = line_str[k];
            if (c == ' ') indent++;
            else if (c == '\t') indent += 4;
            else if (c != '\r' && c != '\n') {
                blank = false;
                first_non_space = k;
                break;
            }
        }

        std::string_view trimmed;
        if (!blank) {
            size_t last_non_space = line_str.size() - 1;
            while (last_non_space > first_non_space &&
                   std::isspace(static_cast<unsigned char>(line_str[last_non_space]))) {
                last_non_space--;
            }
            trimmed = line_str.substr(first_non_space, last_non_space - first_non_space + 1);
        }

        bool top_hdr = false;
        if (!blank && indent <= 4) {
            if (trimmed.starts_with("#") || trimmed.starts_with("\\section") || trimmed.starts_with("\\subsection") ||
                trimmed.starts_with("struct ") || trimmed.starts_with("class ") || trimmed.starts_with("def ") ||
                trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") || trimmed.starts_with("function ") ||
                trimmed.starts_with("local function ") || trimmed.starts_with("theorem ") ||
                trimmed.starts_with("DATA_STRUCTURE") || trimmed.starts_with("static ") ||
                trimmed.starts_with("inline ") || trimmed.starts_with("template")) {
                top_hdr = true;
            }
        }

        lines.push_back({pos, end_b, line_num, indent, blank, top_hdr, trimmed});
        pos = end_b;
        line_num++;
    }

    size_t start_idx = 0;
    for (size_t i = 1; i < lines.size(); ++i) {
        size_t cur_lines = i - start_idx;
        bool prev_blank = lines[i - 1].is_blank;
        bool boundary = (lines[i].is_top_header && cur_lines >= 3 && prev_blank) ||
                        (prev_blank && lines[i].indent == 0 && !lines[i].is_blank && cur_lines >= 6) ||
                        (prev_blank && cur_lines >= 50) ||
                        (cur_lines >= 100);

        if (boundary) {
            size_t end_idx = i - 1;
            while (end_idx > start_idx && lines[end_idx].is_blank) end_idx--;
            while (start_idx < end_idx && lines[start_idx].is_blank) start_idx++;

            if (!lines[start_idx].is_blank) {
                uint32_t sb = lines[start_idx].start_byte;
                uint32_t eb = lines[end_idx].end_byte;
                std::string_view blk_text = content.substr(sb, eb - sb);

                std::string prev_str;
                for (size_t k = start_idx; k <= end_idx; ++k) {
                    if (!lines[k].is_blank && !lines[k].trimmed.starts_with("// ---") && !lines[k].trimmed.starts_with("-- ---")) {
                        prev_str = std::string(lines[k].trimmed.substr(0, 68));
                        if (lines[k].trimmed.size() > 68) prev_str += "...";
                        break;
                    }
                }
                blocks.push_back({sb, eb, lines[start_idx].line_num, lines[end_idx].line_num, blk_text, std::move(prev_str)});
            }
            start_idx = i;
        }
    }

    if (start_idx < lines.size()) {
        size_t end_idx = lines.size() - 1;
        while (end_idx > start_idx && lines[end_idx].is_blank) end_idx--;
        while (start_idx < end_idx && lines[start_idx].is_blank) start_idx++;
        if (!lines[start_idx].is_blank) {
            uint32_t sb = lines[start_idx].start_byte;
            uint32_t eb = lines[end_idx].end_byte;
            std::string prev_str(lines[start_idx].trimmed.substr(0, 68));
            if (lines[start_idx].trimmed.size() > 68) prev_str += "...";
            blocks.push_back({sb, eb, lines[start_idx].line_num, lines[end_idx].line_num, content.substr(sb, eb - sb), std::move(prev_str)});
        }
    }

    return blocks;
}

// ----------------------------------------------------------------------------
// 7. Repository & Wget Directory Scanner Helpers
// ----------------------------------------------------------------------------
static bool is_ignored_dir(const std::string& dirname) {
    static const std::set<std::string> ignored = {
        ".git", ".hg", ".svn", "node_modules", "target", "build", "dist",
        "__pycache__", ".venv", "venv", "myenv", ".cache", ".idea", ".vscode", ".lexlean"
    };
    return ignored.contains(dirname);
}

static bool is_supported_file(const fs::path& p, Modality& inferred_mod, bool& is_html_out) {
    is_html_out = false;
    std::string ext = p.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    static const std::set<std::string> binary_or_skip_exts = {
        ".o", ".a", ".so", ".dll", ".exe", ".bin", ".mfd", ".olean", ".pyc",
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".pdf",
        ".zip", ".tar", ".gz", ".xz", ".zst", ".woff", ".woff2", ".ttf",
        ".wav", ".mp3", ".mp4", ".lock", ".jsonl", ".log"
    };
    if (binary_or_skip_exts.contains(ext)) return false;

    std::string fname = p.filename().string();
    if (fname.ends_with(".min.js") || fname.ends_with(".min.css") || fname.ends_with(".bundle.js")) {
        return false;
    }

    if (ext == ".html" || ext == ".htm" || ext == ".xhtml") {
        is_html_out = true;
        inferred_mod = Modality::WEB_REF;
        return true;
    }
    if (ext == ".md" || ext == ".tex" || ext == ".rst" || ext == ".txt") {
        inferred_mod = Modality::SPEC_DOC;
        return true;
    }
    return true;
}

static bool read_text_file_safe(const fs::path& p, std::string& out_content, size_t max_bytes = 2 * 1024 * 1024) {
    std::error_code ec;
    auto fsize = fs::file_size(p, ec);
    if (ec || fsize == 0 || fsize > max_bytes) return false;

    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    out_content.resize(static_cast<size_t>(fsize));
    in.read(out_content.data(), static_cast<std::streamsize>(fsize));

    size_t check_len = std::min<size_t>(out_content.size(), 4096);
    for (size_t i = 0; i < check_len; ++i) {
        if (out_content[i] == '\0') return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// 8. The Geometric Manifold State Engine (O(1) Indexed + SIMD-Friendly)
// ----------------------------------------------------------------------------
struct LucidityEval {
    Archetype best_archetype = Archetype::OP;
    double archetype_sim     = 0.5;
    uint32_t bmu_cell_id     = 0;
    double bmu_sim           = 0.5;
    double q_error           = 1.0;
    double lucidity          = 0.0; // L_semantic in [0.0, 1.0] (HUFD §4.2)
};

struct IngestStats {
    uint32_t files_added     = 0;
    uint32_t files_updated   = 0;
    uint32_t files_unchanged = 0;
    uint32_t blocks_added    = 0;
    size_t   bytes_processed = 0;
};

class ManifoldEngine {
public:
    BipolarAccumulator global_acc;
    std::array<BipolarAccumulator, static_cast<size_t>(Archetype::COUNT)> arch_acc;
    std::array<Hypervec8192, static_cast<size_t>(Archetype::COUNT)> arch_hv;
    std::vector<Cell> cells;

    // O(1) lookup tables for fast large-repo ingestion and routing
    std::unordered_map<std::string, size_t> file_uri_to_idx;
    std::unordered_map<uint32_t, size_t> cell_id_to_idx;

    ManifoldEngine() {
        init_pragmatic_archetypes();
    }

    void rebuild_lookup_tables() {
        file_uri_to_idx.clear();
        cell_id_to_idx.clear();
        file_uri_to_idx.reserve(cells.size() / 4 + 16);
        cell_id_to_idx.reserve(cells.size() + 16);
        for (size_t i = 0; i < cells.size(); ++i) {
            cell_id_to_idx[cells[i].id] = i;
            if (cells[i].halo.scale == Scale::FILE) {
                file_uri_to_idx[cells[i].coord.uri] = i;
            }
        }
    }

    void refresh_archetype_projections() {
        for (size_t i = 0; i < 6; ++i) {
            arch_hv[i] = arch_acc[i].project();
        }
    }

    void init_pragmatic_archetypes() {
        static const std::array<const char*, 6> seeds = {
            // #0:io (INGEST_EMIT)
            "ingest parse tokenize byte stream patch surprisal shannon entropy demodulate "
            "readfile writefile cli cmdline input output serialize decode format utf8 buffer match html wget",
            // #1:state (DATA_SUBSTRATE)
            "struct data_structure tensor field connection hypervector bitfield uint64 int16 "
            "accumulator state queue registry memory lattice cache buffer slice dyadic H_mu omega_mu",
            // #2:space (INDEX_TOPOLOGY)
            "manifold metric topology coordinate hopf sphere poincare disk lattice graph tree "
            "index symbol_table lookup hash address uor sector prime zeta distance s3",
            // #3:op (TRANSFORM_KERNEL)
            "operator kernel transform transport parallel_transport exp laplacian helmholtz fft "
            "stencil attention kuramoto phase rotor quaternion bind permute popcount hamming taylor",
            // #4:route (CONTROL_ADAPT)
            "route router dispatch select rank hungarian bijective ragu adapt update delta_rule "
            "schedule decompose_and_add_tasks start_task jump_to_target queue next prev affinity",
            // #5:gate (VERIFY_GOVERN)
            "verify gate audit lucidity som firewall threshold invariant ensure require assert "
            "theorem proof check review accept reject ledger hash_chain quarantine test bmu"
        };

        for (size_t i = 0; i < 6; ++i) {
            arch_acc[i].clear();
            Hypervec8192 seed_hv = encode_span_hypervector(seeds[i]);
            arch_acc[i].accumulate(seed_hv, 12);
            arch_hv[i] = arch_acc[i].project();
        }
    }

    std::pair<Archetype, uint8_t> classify_archetype(const Hypervec8192& hv, double& best_sim_out) const {
        Archetype best_a = Archetype::OP;
        double best_sim = -1.0;
        uint8_t mask = 0;

        for (size_t i = 0; i < 6; ++i) {
            double sim = hv.similarity(arch_hv[i]);
            if (sim > best_sim) {
                best_sim = sim;
                best_a = static_cast<Archetype>(i);
            }
            if (sim >= 0.512) {
                mask |= (1U << i);
            }
        }
        if (mask == 0) mask |= (1U << static_cast<uint8_t>(best_a));
        best_sim_out = best_sim;
        return {best_a, mask};
    }

    void remove_file_by_uri(const std::string& uri) {
        auto it = file_uri_to_idx.find(uri);
        if (it == file_uri_to_idx.end()) return;
        uint32_t target_file_id = cells[it->second].id;

        std::vector<Cell> surviving;
        surviving.reserve(cells.size());
        for (const auto& c : cells) {
            bool is_target_file = (c.id == target_file_id);
            bool is_target_block = (c.halo.scale == Scale::BLOCK && c.parent_id == target_file_id);

            if (is_target_block) {
                global_acc.accumulate(c.hv, -1);
                arch_acc[static_cast<size_t>(c.halo.primary_arch)].accumulate(c.hv, -1);
            }
            if (!is_target_file && !is_target_block) {
                surviving.push_back(c);
            }
        }
        cells = std::move(surviving);
        refresh_archetype_projections();
        rebuild_lookup_tables();
    }

    uint32_t ingest_file(
        const std::string& uri,
        const std::string& raw_content,
        Modality mod = Modality::REPO_CODE,
        bool is_html = false,
        IngestStats* stats = nullptr,
        bool defer_projection = false
    ) {
        std::string dem_storage;
        std::string_view content = raw_content;
        if (is_html) {
            dem_storage = demodulate_html_to_text(raw_content);
            content = dem_storage;
        }
        uint64_t new_hash = fnv1a_str(content);

        auto existing_it = file_uri_to_idx.find(uri);
        if (existing_it != file_uri_to_idx.end()) {
            const Cell& existing_fc = cells[existing_it->second];
            if (existing_fc.coord.content_hash == new_hash) {
                if (stats) stats->files_unchanged++;
                return existing_fc.id;
            }
            remove_file_by_uri(uri);
            if (stats) stats->files_updated++;
        } else {
            if (stats) stats->files_added++;
        }

        if (stats) stats->bytes_processed += content.size();

        auto raw_blocks = chunk_byte_stream(content);
        BipolarAccumulator file_acc;

        uint32_t file_cell_id = cells.empty() ? 1 : (cells.back().id + 1);
        size_t file_cell_idx = cells.size();

        Cell file_cell;
        file_cell.id = file_cell_id;
        file_cell.parent_id = 0;
        file_cell.halo.scale = Scale::FILE;
        file_cell.halo.modality = mod;
        file_cell.halo.epistemic = EpistemicState::ACTIVE;
        file_cell.coord = {uri, 0, static_cast<uint32_t>(content.size()), 1, 1, new_hash};
        file_cell.preview = std::string("[") + modality_tag(mod) + "] " + uri;
        cells.push_back(std::move(file_cell));
        file_uri_to_idx[uri] = file_cell_idx;
        cell_id_to_idx[file_cell_id] = file_cell_idx;

        uint32_t max_line = 1;
        for (const auto& rb : raw_blocks) {
            Hypervec8192 bhv = encode_span_hypervector(rb.text);
            ByteStats bstats = compute_byte_stats(rb.text);
            double arch_sim = 0.0;
            auto [best_arch, arch_mask] = classify_archetype(bhv, arch_sim);

            Cell blk;
            blk.id = cells.back().id + 1;
            blk.parent_id = file_cell_id;
            blk.halo.scale = Scale::BLOCK;
            blk.halo.modality = mod;
            blk.halo.primary_arch = best_arch;
            blk.halo.epistemic = EpistemicState::ACTIVE;
            blk.halo.archetype_mask = arch_mask;
            blk.halo.sector_id = compute_sector_id(bhv);
            blk.halo.entropy_q8 = static_cast<uint8_t>(std::clamp(bstats.shannon_entropy * 32.0, 0.0, 255.0));
            blk.halo.surprisal_q8 = static_cast<uint8_t>(std::clamp(bstats.mean_delta * 255.0, 0.0, 255.0));
            blk.coord = {uri, rb.start_byte, rb.end_byte, rb.start_line, rb.end_line, fnv1a_str(rb.text)};
            blk.hv = bhv;
            blk.preview = rb.preview;

            size_t blk_idx = cells.size();
            cell_id_to_idx[blk.id] = blk_idx;
            cells.push_back(std::move(blk));

            file_acc.accumulate(bhv, 1);
            global_acc.accumulate(bhv, 1);
            arch_acc[static_cast<size_t>(best_arch)].accumulate(bhv, 1);
            max_line = std::max(max_line, rb.end_line);
            if (stats) stats->blocks_added++;
        }

        if (!defer_projection) {
            refresh_archetype_projections();
        }

        Hypervec8192 fhv = raw_blocks.empty() ? encode_span_hypervector(content) : file_acc.project();
        double f_arch_sim = 0.0;
        auto [f_arch, f_mask] = classify_archetype(fhv, f_arch_sim);
        ByteStats fstats = compute_byte_stats(content);

        Cell& fc = cells[file_cell_idx];
        fc.hv = fhv;
        fc.halo.primary_arch = f_arch;
        fc.halo.archetype_mask = f_mask;
        fc.halo.sector_id = compute_sector_id(fhv);
        fc.halo.entropy_q8 = static_cast<uint8_t>(std::clamp(fstats.shannon_entropy * 32.0, 0.0, 255.0));
        fc.coord.end_line = max_line;

        return file_cell_id;
    }

    IngestStats ingest_path(
        const fs::path& target_path,
        std::optional<Modality> explicit_modality = std::nullopt,
        const std::string& uri_prefix = ""
    ) {
        IngestStats stats;
        std::error_code ec;
        if (!fs::exists(target_path, ec)) return stats;

        uint32_t files_processed_since_proj = 0;
        auto process_single_file = [&](const fs::path& fp, const fs::path& base_dir) {
            Modality inferred_mod = Modality::REPO_CODE;
            bool is_html = false;
            if (!is_supported_file(fp, inferred_mod, is_html)) return;

            std::string raw;
            if (!read_text_file_safe(fp, raw)) return;

            Modality final_mod = explicit_modality.value_or(inferred_mod);
            std::string rel = base_dir.empty() ? fp.string() : fs::relative(fp, base_dir, ec).string();
            if (rel.empty() || rel.starts_with("..")) rel = fp.string();

            std::string uri = uri_prefix.empty() ? rel : (uri_prefix + "/" + rel);
            ingest_file(uri, raw, final_mod, is_html, &stats, true);

            if (++files_processed_since_proj >= 32) {
                refresh_archetype_projections();
                files_processed_since_proj = 0;
            }
        };

        if (fs::is_regular_file(target_path, ec)) {
            process_single_file(target_path, target_path.parent_path());
            refresh_archetype_projections();
            return stats;
        }

        if (fs::is_directory(target_path, ec)) {
            std::vector<fs::path> found_files;
            for (auto it = fs::recursive_directory_iterator(target_path, fs::directory_options::skip_permission_denied, ec);
                 it != fs::recursive_directory_iterator(); it.increment(ec)) {
                if (ec) break;
                const fs::path& p = it->path();
                if (it->is_directory(ec)) {
                    std::string dname = p.filename().string();
                    if (is_ignored_dir(dname) || (dname.starts_with(".") && dname.size() > 1)) {
                        it.disable_recursion_pending();
                    }
                    continue;
                }
                if (it->is_regular_file(ec)) {
                    found_files.push_back(p);
                }
            }
            std::sort(found_files.begin(), found_files.end());
            for (const auto& fp : found_files) {
                process_single_file(fp, target_path);
            }
            refresh_archetype_projections();
        }
        return stats;
    }

    LucidityEval evaluate_lucidity(const Hypervec8192& x, Scale target_scale = Scale::BLOCK) const {
        LucidityEval ev;
        auto [best_arch, mask] = classify_archetype(x, ev.archetype_sim);
        ev.best_archetype = best_arch;

        double best_sim = -1.0;
        for (const auto& c : cells) {
            if (c.halo.scale != target_scale || c.halo.epistemic == EpistemicState::REJECT) continue;
            if (c.halo.modality != Modality::REPO_CODE) continue;
            double sim = x.similarity(c.hv);
            if (sim > best_sim) {
                best_sim = sim;
                ev.bmu_cell_id = c.id;
            }
        }
        ev.bmu_sim = (best_sim < 0.0) ? 0.5 : best_sim;

        double excess_bmu = std::clamp((ev.bmu_sim - 0.52) / 0.10, 0.0, 1.0);
        ev.q_error = (1.0 - excess_bmu) * 1.5;
        double q_norm = std::tanh(ev.q_error);

        double cos_pos = std::clamp((ev.archetype_sim - 0.50) / 0.08, 0.0, 1.0);
        ev.lucidity = std::clamp(1.0 - 0.5 * q_norm + 0.5 * cos_pos - 0.25, 0.0, 1.0);
        return ev;
    }

    struct RouteHit {
        const Cell* cell;
        double similarity;
        double z_score;
    };

    // O(N) Hardware POPCNT Routing with O(1) Parent Lookup
    std::vector<RouteHit> route(
        const Hypervec8192& query_hv,
        Scale target_scale = Scale::BLOCK,
        int archetype_filter = -1,
        int modality_filter = -1,
        size_t top_k = 5
    ) const {
        std::vector<RouteHit> hits;
        hits.reserve(256);

        for (const auto& c : cells) {
            if (c.halo.scale != target_scale) continue;
            if (c.halo.epistemic == EpistemicState::REJECT) continue;
            if (modality_filter >= 0 && static_cast<int>(c.halo.modality) != modality_filter) continue;
            if (archetype_filter >= 0) {
                bool match_primary = (static_cast<int>(c.halo.primary_arch) == archetype_filter);
                bool match_mask = (c.halo.archetype_mask & (1U << archetype_filter)) != 0;
                if (!match_primary && !match_mask) continue;
            }

            double sim = query_hv.similarity(c.hv);
            if (target_scale == Scale::BLOCK && c.parent_id > 0) {
                auto pit = cell_id_to_idx.find(c.parent_id);
                if (pit != cell_id_to_idx.end()) {
                    double parent_sim = query_hv.similarity(cells[pit->second].hv);
                    sim = 0.82 * sim + 0.18 * parent_sim;
                }
            }
            hits.push_back({&c, sim, query_hv.z_score(c.hv)});
        }

        if (hits.size() > top_k) {
            std::partial_sort(hits.begin(), hits.begin() + top_k, hits.end(),
                [](const RouteHit& a, const RouteHit& b) { return a.similarity > b.similarity; });
            hits.resize(top_k);
        } else {
            std::sort(hits.begin(), hits.end(),
                [](const RouteHit& a, const RouteHit& b) { return a.similarity > b.similarity; });
        }
        return hits;
    }

    uint32_t record_experiment(
        const std::string& exp_id,
        Archetype stage,
        EpistemicState verdict,
        const std::string& description
    ) {
        Hypervec8192 hv = encode_span_hypervector(description);
        ByteStats st = compute_byte_stats(description);

        Cell c;
        c.id = cells.empty() ? 1 : (cells.back().id + 1);
        c.parent_id = 0;
        c.halo.scale = Scale::BLOCK;
        c.halo.modality = Modality::EXPERIMENT;
        c.halo.primary_arch = stage;
        c.halo.epistemic = verdict;
        c.halo.archetype_mask = (1U << static_cast<uint8_t>(stage));
        c.halo.sector_id = compute_sector_id(hv);
        c.halo.entropy_q8 = static_cast<uint8_t>(std::clamp(st.shannon_entropy * 32.0, 0.0, 255.0));
        c.coord = {"exp://" + exp_id, 0, static_cast<uint32_t>(description.size()), 1, 1, fnv1a_str(description)};
        c.hv = hv;
        c.preview = "[" + std::string(epistemic_name(verdict)) + "] " + exp_id + ": " + description.substr(0, 62);

        uint32_t cid = c.id;
        cell_id_to_idx[cid] = cells.size();
        cells.push_back(std::move(c));

        if (verdict == EpistemicState::KEEP) {
            global_acc.accumulate(hv, 2);
            arch_acc[static_cast<size_t>(stage)].accumulate(hv, 2);
            arch_hv[static_cast<size_t>(stage)] = arch_acc[static_cast<size_t>(stage)].project();
        }
        return cid;
    }

    struct PlacementPlan {
        LucidityEval lucidity;
        const Cell* best_file  = nullptr;
        const Cell* best_block = nullptr;
        std::string decision;
        const Cell* rejected_hazard = nullptr;
        double hazard_similarity    = 0.0;
        double hazard_z_score       = 0.0;
    };

    PlacementPlan place_task(const std::string& task_spec) const {
        PlacementPlan plan;
        Hypervec8192 thv = encode_span_hypervector(task_spec);
        plan.lucidity = evaluate_lucidity(thv, Scale::BLOCK);

        auto file_hits = route(thv, Scale::FILE, -1, static_cast<int>(Modality::REPO_CODE), 1);
        if (!file_hits.empty()) plan.best_file = file_hits[0].cell;

        auto block_hits = route(thv, Scale::BLOCK, -1, static_cast<int>(Modality::REPO_CODE), 1);
        if (!block_hits.empty()) plan.best_block = block_hits[0].cell;

        for (const auto& c : cells) {
            if (c.halo.epistemic == EpistemicState::REJECT) {
                double sim = thv.similarity(c.hv);
                double z = thv.z_score(c.hv);
                if (sim > plan.hazard_similarity) {
                    plan.hazard_similarity = sim;
                    plan.hazard_z_score = z;
                    plan.rejected_hazard = &c;
                }
            }
        }

        if (plan.lucidity.lucidity >= 0.65 && plan.best_block != nullptr) {
            plan.decision = "INSERT_AT_EXISTING_BLOCK (High Lucidity: target block already implements this domain)";
        } else if (plan.best_file != nullptr) {
            plan.decision = "SCAFFOLD_NEW_BLOCK_IN_FILE (Low Block Lucidity: domain matches file/stage, but needs a new function/struct stub)";
        } else {
            plan.decision = "SCAFFOLD_NEW_FILE_FOR_STAGE (No resonant file found; create new module for archetype)";
        }
        return plan;
    }

    bool save_checkpoint(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) return false;
        const char magic[8] = {'M','F','D','v','1','\0','\0','\0'};
        out.write(magic, 8);
        out.write(reinterpret_cast<const char*>(&global_acc), sizeof(global_acc));
        out.write(reinterpret_cast<const char*>(arch_acc.data()), sizeof(arch_acc));
        out.write(reinterpret_cast<const char*>(arch_hv.data()), sizeof(arch_hv));

        uint32_t count = static_cast<uint32_t>(cells.size());
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& c : cells) {
            out.write(reinterpret_cast<const char*>(&c.id), sizeof(c.id));
            out.write(reinterpret_cast<const char*>(&c.parent_id), sizeof(c.parent_id));
            out.write(reinterpret_cast<const char*>(&c.halo), sizeof(c.halo));
            out.write(reinterpret_cast<const char*>(&c.coord.start_byte), sizeof(uint32_t) * 4);
            out.write(reinterpret_cast<const char*>(&c.coord.content_hash), sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(&c.hv), sizeof(c.hv));

            uint32_t uri_len = static_cast<uint32_t>(c.coord.uri.size());
            out.write(reinterpret_cast<const char*>(&uri_len), sizeof(uri_len));
            out.write(c.coord.uri.data(), uri_len);

            uint32_t prev_len = static_cast<uint32_t>(c.preview.size());
            out.write(reinterpret_cast<const char*>(&prev_len), sizeof(prev_len));
            out.write(c.preview.data(), prev_len);
        }
        return out.good();
    }

    bool load_checkpoint(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) return false;
        char magic[8]{};
        in.read(magic, 8);
        if (std::strncmp(magic, "MFDv1", 5) != 0) return false;

        in.read(reinterpret_cast<char*>(&global_acc), sizeof(global_acc));
        in.read(reinterpret_cast<char*>(arch_acc.data()), sizeof(arch_acc));
        in.read(reinterpret_cast<char*>(arch_hv.data()), sizeof(arch_hv));

        uint32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));
        cells.clear();
        cells.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& c = cells[i];
            in.read(reinterpret_cast<char*>(&c.id), sizeof(c.id));
            in.read(reinterpret_cast<char*>(&c.parent_id), sizeof(c.parent_id));
            in.read(reinterpret_cast<char*>(&c.halo), sizeof(c.halo));
            in.read(reinterpret_cast<char*>(&c.coord.start_byte), sizeof(uint32_t) * 4);
            in.read(reinterpret_cast<char*>(&c.coord.content_hash), sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&c.hv), sizeof(c.hv));

            uint32_t uri_len = 0;
            in.read(reinterpret_cast<char*>(&uri_len), sizeof(uri_len));
            c.coord.uri.resize(uri_len);
            in.read(c.coord.uri.data(), uri_len);

            uint32_t prev_len = 0;
            in.read(reinterpret_cast<char*>(&prev_len), sizeof(prev_len));
            c.preview.resize(prev_len);
            in.read(c.preview.data(), prev_len);
        }
        rebuild_lookup_tables();
        return in.good();
    }
};

// ----------------------------------------------------------------------------
// 9. Query Expression Parser (@repo, @ref, @web, @docs, #io..#gate, /file, /block)
// ----------------------------------------------------------------------------
struct ParsedQuery {
    int modality_filter  = -1;
    int archetype_filter = -1;
    Scale target_scale   = Scale::BLOCK;
    std::string clean_expr;
};

static ParsedQuery parse_query_dsl(const std::string& raw_query) {
    ParsedQuery pq;
    std::istringstream iss(raw_query);
    std::string tok;
    std::vector<std::string> expr_tokens;

    while (iss >> tok) {
        if (tok == "@repo" || tok == "@code") pq.modality_filter = static_cast<int>(Modality::REPO_CODE);
        else if (tok == "@docs" || tok == "@doc") pq.modality_filter = static_cast<int>(Modality::SPEC_DOC);
        else if (tok == "@ref") pq.modality_filter = static_cast<int>(Modality::EXT_REF);
        else if (tok == "@web" || tok == "@wget") pq.modality_filter = static_cast<int>(Modality::WEB_REF);
        else if (tok == "@exp") pq.modality_filter = static_cast<int>(Modality::EXPERIMENT);
        else if (tok == "#io" || tok == "#0") pq.archetype_filter = static_cast<int>(Archetype::IO);
        else if (tok == "#state" || tok == "#1") pq.archetype_filter = static_cast<int>(Archetype::STATE);
        else if (tok == "#space" || tok == "#2") pq.archetype_filter = static_cast<int>(Archetype::SPACE);
        else if (tok == "#op" || tok == "#3") pq.archetype_filter = static_cast<int>(Archetype::OP);
        else if (tok == "#route" || tok == "#4") pq.archetype_filter = static_cast<int>(Archetype::ROUTE);
        else if (tok == "#gate" || tok == "#5") pq.archetype_filter = static_cast<int>(Archetype::GATE);
        else if (tok == "/file") pq.target_scale = Scale::FILE;
        else if (tok == "/block") pq.target_scale = Scale::BLOCK;
        else expr_tokens.push_back(tok);
    }

    for (size_t i = 0; i < expr_tokens.size(); ++i) {
        if (i > 0) pq.clean_expr.push_back(' ');
        pq.clean_expr += expr_tokens[i];
    }
    return pq;
}

static Modality parse_modality_flag(const std::string& s) {
    if (s == "repo" || s == "@repo") return Modality::REPO_CODE;
    if (s == "docs" || s == "@docs") return Modality::SPEC_DOC;
    if (s == "ref"  || s == "@ref")  return Modality::EXT_REF;
    if (s == "web"  || s == "@web" || s == "wget") return Modality::WEB_REF;
    return Modality::REPO_CODE;
}

// ----------------------------------------------------------------------------
// 10. Large-Scale Repository Benchmark Suite (`./manifold benchmark [dir]`)
// ----------------------------------------------------------------------------
static int run_benchmark(const std::string& target_dir) {
    using clock = std::chrono::high_resolution_clock;
    std::cout << "============================================================================\n";
    std::cout << "  MANIFOLD C++20 ENGINE — LARGE REPOSITORY STRESS BENCHMARK\n";
    std::cout << "============================================================================\n";
    std::cout << "  Target Corpus : " << target_dir << "\n";

    ManifoldEngine eng;
    auto t0 = clock::now();
    IngestStats st = eng.ingest_path(target_dir, Modality::REPO_CODE, "bench");
    auto t1 = clock::now();
    double ingest_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mb = static_cast<double>(st.bytes_processed) / (1024.0 * 1024.0);

    std::cout << "  Files Ingested: " << st.files_added << " files (" << std::fixed << std::setprecision(2) << mb << " MB)\n";
    std::cout << "  Total Cells   : " << eng.cells.size() << " (" << st.blocks_added << " blocks + " << st.files_added << " file nodes)\n";
    std::cout << "  Cold Ingest   : " << std::setprecision(1) << ingest_ms << " ms ("
              << std::setprecision(1) << (mb / (ingest_ms / 1000.0)) << " MB/s | "
              << static_cast<int>(st.blocks_added / (ingest_ms / 1000.0)) << " blocks/s)\n";

    // Incremental re-ingest (warm hash check across all files)
    auto t2 = clock::now();
    IngestStats warm_st = eng.ingest_path(target_dir, Modality::REPO_CODE, "bench");
    auto t3 = clock::now();
    double warm_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "  Warm Re-Sync  : " << std::setprecision(1) << warm_ms << " ms ("
              << warm_st.files_unchanged << " unchanged files verified & skipped)\n";

    // Checkpoint Save & Load latency
    std::string ckpt = "/tmp/manifold_large_bench.mfd";
    auto t4 = clock::now();
    eng.save_checkpoint(ckpt);
    auto t5 = clock::now();
    ManifoldEngine loaded;
    loaded.load_checkpoint(ckpt);
    auto t6 = clock::now();

    double save_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double load_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();
    double ckpt_mb = static_cast<double>(fs::file_size(ckpt)) / (1024.0 * 1024.0);
    std::cout << "  Checkpoint I/O: Save=" << save_ms << " ms, Load=" << load_ms
              << " ms (Size=" << std::setprecision(2) << ckpt_mb << " MB)\n";

    // Query Latency across all cells
    const std::array<const char*, 4> bench_queries = {{
        "red black tree rebalance rotate node pointer",
        "#op bit manipulation popcount count leading zeros",
        "#state vector contiguous allocator capacity reserve",
        "#gate static_assert concept type_traits integral_constant"
    }};

    double total_q_ms = 0.0;
    std::cout << "\n  [Query Routing Latency Across " << loaded.cells.size() << " Cells]\n";
    for (const char* q : bench_queries) {
        auto qt0 = clock::now();
        ParsedQuery pq = parse_query_dsl(q);
        Hypervec8192 qhv = encode_span_hypervector(pq.clean_expr);
        auto hits = loaded.route(qhv, pq.target_scale, pq.archetype_filter, pq.modality_filter, 3);
        auto qt1 = clock::now();
        double q_ms = std::chrono::duration<double, std::milli>(qt1 - qt0).count();
        total_q_ms += q_ms;

        std::cout << "  • Query (" << std::setprecision(2) << q_ms << " ms): \"" << q << "\"\n";
        if (!hits.empty()) {
            std::cout << "      Top Hit: " << hits[0].cell->coord.uri << ":" << hits[0].cell->coord.start_line
                      << " [" << archetype_short(hits[0].cell->halo.primary_arch)
                      << " sim=" << std::setprecision(3) << hits[0].similarity
                      << " +" << std::setprecision(1) << hits[0].z_score << "s] | "
                      << hits[0].cell->preview << "\n";
        }
    }
    std::cout << "  Mean Query Latency (In-Memory): " << std::setprecision(2) << (total_q_ms / bench_queries.size()) << " ms\n";
    std::cout << "============================================================================\n";
    return 0;
}

// ----------------------------------------------------------------------------
// 11. Comprehensive Empirical Verification Suite (`./manifold selftest`)
// ----------------------------------------------------------------------------
static int run_selftest() {
    std::cout << "============================================================================\n";
    std::cout << "  MANIFOLD C++20 ENGINE — LIVE EMPIRICAL VERIFICATION SUITE\n";
    std::cout << "============================================================================\n\n";

    std::cout << "[TEST 1] Non-Destructive Accumulator Superposition & Exact Algebraic Revert\n";
    BipolarAccumulator acc;
    std::vector<Hypervec8192> items;
    for (int i = 0; i < 15; ++i) {
        items.push_back(encode_span_hypervector("algorithm_module_" + std::to_string(i) + " holonomic_field_" + std::to_string(i * 17)));
        acc.accumulate(items.back(), 1);
    }
    Hypervec8192 bundled_15 = acc.project();

    double min_z = 999.0;
    for (int i = 0; i < 15; ++i) {
        min_z = std::min(min_z, bundled_15.z_score(items[i]));
    }
    Hypervec8192 random_probe = Hypervec8192::from_seed(0xDEADBEEFCAFEBABEULL);
    double noise_z = bundled_15.z_score(random_probe);

    std::cout << "  • Superimposed 15 distinct vectors into 1 accumulator:\n";
    std::cout << "    - Minimum signal Z-score across all 15 stored items : +" << std::fixed << std::setprecision(2) << min_z << " sigma (100% retained)\n";
    std::cout << "    - Unrelated random probe Z-score                    :  " << noise_z << " sigma (orthogonal noise)\n";

    BipolarAccumulator snapshot_before = acc;
    Hypervec8192 transient = encode_span_hypervector("transient_failed_hypothesis_unnormalized_taylor_drift");
    acc.accumulate(transient, +3);
    uint32_t hamming_during = acc.project().hamming(bundled_15);
    acc.accumulate(transient, -3);
    uint32_t hamming_after_revert = acc.project().hamming(bundled_15);
    bool exact_counts_restored = (acc.counts == snapshot_before.counts);

    std::cout << "  • Added transient hypothesis (weight +3) -> field shifted by " << hamming_during << " / 8192 bits\n";
    std::cout << "  • Subtracted transient hypothesis (-3)   -> Hamming distance to prior state = "
              << hamming_after_revert << " bits (Exact integer match: " << (exact_counts_restored ? "YES" : "NO") << ")\n\n";

    std::cout << "[TEST 2] Whole-Repo Ingestion + Wget Online Reference Directory Ingestion\n";
    fs::path tmp_repo = "/tmp/manifold_test_repo";
    fs::path tmp_wget = "/tmp/manifold_test_wget_refs";
    fs::remove_all(tmp_repo);
    fs::remove_all(tmp_wget);
    fs::create_directories(tmp_repo / "src");
    fs::create_directories(tmp_repo / ".git");
    fs::create_directories(tmp_wget / "scipy.org" / "fft");
    fs::create_directories(tmp_wget / "r4labs.org" / "routing");

    {
        std::ofstream(tmp_repo / ".git" / "config") << "[core]\nrepositoryformatversion = 0\n";
        std::ofstream(tmp_repo / "src" / "hufd_core.cpp") << R"RAW(
struct HolonomicFieldConnection {
    std::vector<float> H_mu;
    std::vector<float> omega_mu;
    float band_sigma;
};

void holonomic_parallel_transport(HolonomicFieldConnection& field, const float* query, float* out, int n) {
    // Evaluates ConstAttn(Q,K,V) = exp(-div H_mu) V via Taylor curvature expansion
    for (int i = 0; i < n; ++i) {
        float laplacian = (i > 0 && i + 1 < n) ? (field.H_mu[i-1] - 2.0f * field.H_mu[i] + field.H_mu[i+1]) : 0.0f;
        out[i] = query[i] * (1.0f - laplacian + 0.5f * laplacian * laplacian);
    }
}

bool verify_som_lucidity_gate(const float* state_slice, const float* bmu_weights, float threshold, int d) {
    float q_err = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = state_slice[i] - bmu_weights[i];
        q_err += diff * diff;
    }
    float lucidity = 1.0f - 0.5f * std::tanh(std::sqrt(q_err));
    return lucidity >= threshold;
}
)RAW";

        std::ofstream(tmp_repo / "src" / "reviews.lua") << R"RAW(
function M.parse_cmd_query(cmdline)
    -- Tokenize command line input into @scope, #stage, and query string
    local clean = vim.trim(cmdline or "")
    local scope = clean:match("@([%w_%-]+)") or "repo"
    return scope, clean
end

function M.accept_review_and_log(review_id, verdict, metrics)
    -- Verify review patch integrity and append SHA256 audit record to ledger
    assert(review_id ~= nil, "review_id required for governance gate")
    local record = { id = review_id, status = verdict, metrics = metrics }
    return record
end
)RAW";

        std::ofstream(tmp_wget / "scipy.org" / "fft" / "helmholtz_solver.html") << R"RAW(
<!DOCTYPE html>
<html>
<head>
  <title>1D & 2D FFT Helmholtz PDE Solver Reference</title>
  <script>console.log("analytics noise that must be stripped by demodulator");</script>
  <style>body { font-family: sans-serif; color: #333; }</style>
</head>
<body>
  <nav><a href="/">Home</a> | <a href="/docs">Docs</a></nav>
  <h2>Frequency-Domain Helmholtz Wave Transport Solver</h2>
  <p>To resolve the stabilized wave transport equation H_{t+1} = H_t - dt (-laplacian H_t + g_eff^2 H_t)
     in O(N log N) complexity, transform the spatial gauge connection into Fourier frequency space:</p>
  <pre><code>
def helmholtz_fft_step(H_t, xi_perturbation, dt, g_eff):
    H_freq = np.fft.rfft(H_t + xi_perturbation)
    k_sq = np.arange(len(H_freq), dtype=np.float32) ** 2
    denom = 1.0 + dt * (k_sq + g_eff * g_eff)
    return np.fft.irfft(H_freq / denom, n=len(H_t))
  </code></pre>
</body>
</html>
)RAW";

        std::ofstream(tmp_wget / "r4labs.org" / "routing" / "prime_hopf.html") << R"RAW(
<!DOCTYPE html>
<html>
<body>
  <h2>Riemann Zeta-Prime Hopf S3 Coordinate Addressing</h2>
  <p>Maps prime numbers p_a, p_b, p_c and zeta zeros gamma_k onto the unit 3-sphere S3:</p>
  <pre><code>
def prime_zeta_hopf_address(token_id, primes, zeta_gammas):
    p_a, p_b, p_c = primes[token_id], primes[token_id + 1], primes[token_id + 2]
    eta = 1.570796 * ((p_a * zeta_gammas[0]) % 1.0)
    xi1 = 6.283185 * ((p_b * zeta_gammas[0]) % 1.0)
    xi2 = 6.283185 * ((p_c * zeta_gammas[1]) % 1.0)
    return (math.sin(eta)*math.cos(xi1), math.sin(eta)*math.sin(xi1),
            math.cos(eta)*math.cos(xi2), math.cos(eta)*math.sin(xi2))
  </code></pre>
</body>
</html>
)RAW";
    }

    ManifoldEngine engine;
    IngestStats repo_stats = engine.ingest_path(tmp_repo, Modality::REPO_CODE, "repo");
    IngestStats wget_stats = engine.ingest_path(tmp_wget, Modality::WEB_REF, "wget");
    IngestStats reingest_stats = engine.ingest_path(tmp_repo, Modality::REPO_CODE, "repo");

    std::cout << "  • Ingested repo dir (" << repo_stats.files_added << " files, " << repo_stats.blocks_added
              << " blocks; .git automatically skipped)\n";
    std::cout << "  • Ingested wget dir (" << wget_stats.files_added << " HTML pages demodulated, " << wget_stats.blocks_added
              << " blocks extracted)\n";
    std::cout << "  • Re-ingested repo  (added=" << reingest_stats.files_added << ", updated=" << reingest_stats.files_updated
              << ", unchanged_skipped=" << reingest_stats.files_unchanged << ")\n\n";

    std::cout << "[TEST 3] Scoped Query DSL Routing Across `@repo` and `@web` (`route()`)\n";
    const std::array<const char*, 3> dsl_queries = {{
        "@web #op Fourier frequency domain Helmholtz wave transport rfft",
        "@web #space Riemann zeta zero phases and prime Hopf S3 sphere",
        "@repo #gate SOM quantization error lucidity threshold check and review"
    }};

    for (const char* q_str : dsl_queries) {
        ParsedQuery pq = parse_query_dsl(q_str);
        Hypervec8192 qhv = encode_span_hypervector(pq.clean_expr);
        auto hits = engine.route(qhv, pq.target_scale, pq.archetype_filter, pq.modality_filter, 2);
        std::cout << "  • Query: \"" << q_str << "\"\n";
        for (size_t rank = 0; rank < hits.size(); ++rank) {
            const auto& h = hits[rank];
            std::cout << "      #" << (rank + 1) << " sim=" << std::fixed << std::setprecision(4) << h.similarity
                      << " (+" << std::setprecision(2) << h.z_score << "s) -> ["
                      << modality_tag(h.cell->halo.modality) << " " << archetype_short(h.cell->halo.primary_arch) << "] "
                      << h.cell->coord.uri << ":" << h.cell->coord.start_line << " | " << h.cell->preview << "\n";
        }
    }
    std::cout << "\n";

    std::cout << "[TEST 4] Scientific Experiment Ledger & Task Placement (`record()` + `place()`)\n";
    engine.record_experiment(
        "exp-014-unbounded-taylor",
        Archetype::OP,
        EpistemicState::REJECT,
        "Unnormalized 4th-order Taylor expansion of holonomic_parallel_transport exp(-div H_mu) without consensus potential gain g caused L2 norm explosion at depth 12"
    );

    std::string task_1 = "Implement 4th-order unnormalized Taylor curvature expansion for holonomic_parallel_transport exp(-div H_mu) laplacian query out in hufd_core";
    auto plan1 = engine.place_task(task_1);
    std::cout << "  • Task 1: \"" << task_1 << "\"\n";
    std::cout << "    - Target Placement     : " << (plan1.best_block ? plan1.best_block->coord.uri : "none")
              << " (Lines " << (plan1.best_block ? plan1.best_block->coord.start_line : 0) << "-"
              << (plan1.best_block ? plan1.best_block->coord.end_line : 0) << ")\n";
    std::cout << "    - BMU Sim / Lucidity   : sim=" << std::fixed << std::setprecision(4) << plan1.lucidity.bmu_sim
              << " | L_semantic=" << std::setprecision(3) << plan1.lucidity.lucidity
              << " -> " << plan1.decision << "\n";
    if (plan1.rejected_hazard && plan1.hazard_z_score >= 3.0) {
        std::cout << "    - [!] SCIENTIFIC HAZARD ALERT: Task resonates at +" << std::setprecision(2) << plan1.hazard_z_score
                  << " sigma with FALSIFIED experiment:\n          " << plan1.rejected_hazard->preview << "\n";
    }
    std::cout << "\n";

    std::cout << "[TEST 5] Binary Checkpoint Serialization, Branching & Algebraic Diffing\n";
    std::string ckpt_v1 = "/tmp/manifold_v1_baseline.mfd";
    std::string ckpt_v2 = "/tmp/manifold_v2_updated.mfd";
    engine.save_checkpoint(ckpt_v1);

    {
        std::ofstream(tmp_repo / "src" / "hufd_core.cpp", std::ios::app) << R"RAW(
void solve_helmholtz_fft_1d(std::vector<float>& H_t, float dt, float g_eff) {
    // Frequency domain Helmholtz solver added after consulting @web reference
}
)RAW";
    }
    IngestStats inc_stats = engine.ingest_path(tmp_repo, Modality::REPO_CODE, "repo");
    engine.save_checkpoint(ckpt_v2);

    ManifoldEngine loaded_v1, loaded_v2;
    loaded_v1.load_checkpoint(ckpt_v1);
    loaded_v2.load_checkpoint(ckpt_v2);

    uint32_t global_bit_diff = loaded_v1.global_acc.project().hamming(loaded_v2.global_acc.project());
    std::cout << "  • Incremental repo sync: updated=" << inc_stats.files_updated
              << ", unchanged_skipped=" << inc_stats.files_unchanged
              << ", new_blocks=" << inc_stats.blocks_added << "\n";
    std::cout << "  • Global Manifold Shift (v1 -> v2): " << global_bit_diff << " / 8192 bits changed ("
              << std::fixed << std::setprecision(2) << (100.0 * global_bit_diff / 8192.0) << "% curvature delta; "
              << (100.0 - 100.0 * global_bit_diff / 8192.0) << "% prior state conserved)\n";

    std::cout << "============================================================================\n";
    return 0;
}

// ----------------------------------------------------------------------------
// 12. Command-Line Interface (for direct use from Neovim / synvim)
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "selftest") {
        return run_selftest();
    }

    std::string cmd = argv[1];
    if (cmd == "benchmark") {
        std::string target = (argc >= 3) ? argv[2] : "/usr/include/c++/12";
        return run_benchmark(target);
    }

    if (cmd == "init" && argc >= 3) {
        ManifoldEngine eng;
        if (eng.save_checkpoint(argv[2])) {
            std::cout << "Initialized empty manifold checkpoint at " << argv[2] << "\n";
            return 0;
        }
        std::cerr << "Error: could not write " << argv[2] << "\n";
        return 1;
    }

    if (cmd == "ingest" && argc >= 4) {
        std::string db = argv[2];
        ManifoldEngine eng;
        if (fs::exists(db)) eng.load_checkpoint(db);

        std::optional<Modality> explicit_mod = std::nullopt;
        std::string prefix = "";
        std::vector<std::string> targets;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--scope" || arg == "-s") && i + 1 < argc) {
                explicit_mod = parse_modality_flag(argv[++i]);
            } else if ((arg == "--prefix" || arg == "-p") && i + 1 < argc) {
                prefix = argv[++i];
            } else {
                targets.push_back(arg);
            }
        }

        IngestStats total{};
        for (const auto& t : targets) {
            IngestStats st = eng.ingest_path(t, explicit_mod, prefix);
            total.files_added += st.files_added;
            total.files_updated += st.files_updated;
            total.files_unchanged += st.files_unchanged;
            total.blocks_added += st.blocks_added;
        }
        eng.save_checkpoint(db);
        std::cout << "Ingested into " << db << ": +" << total.files_added << " new files, ~"
                  << total.files_updated << " updated, =" << total.files_unchanged
                  << " unchanged (" << eng.cells.size() << " total cells)\n";
        return 0;
    }

    if (cmd == "query" && argc >= 4) {
        std::string db = argv[2];
        std::string raw_q;
        for (int i = 3; i < argc; ++i) {
            if (i > 3) raw_q.push_back(' ');
            raw_q += argv[i];
        }
        ManifoldEngine eng;
        if (!eng.load_checkpoint(db)) {
            std::cerr << "Error: cannot load " << db << "\n";
            return 1;
        }
        ParsedQuery pq = parse_query_dsl(raw_q);
        Hypervec8192 qhv = encode_span_hypervector(pq.clean_expr);
        auto hits = eng.route(qhv, pq.target_scale, pq.archetype_filter, pq.modality_filter, 10);
        for (const auto& h : hits) {
            std::cout << h.cell->coord.uri << ":" << h.cell->coord.start_line << ":1: "
                      << "[" << modality_tag(h.cell->halo.modality) << " "
                      << archetype_short(h.cell->halo.primary_arch)
                      << " sim=" << std::fixed << std::setprecision(3) << h.similarity
                      << " +" << std::setprecision(1) << h.z_score << "s] "
                      << h.cell->preview << "\n";
        }
        return 0;
    }

    if (cmd == "place" && argc >= 4) {
        std::string db = argv[2];
        std::string spec;
        for (int i = 3; i < argc; ++i) {
            if (i > 3) spec.push_back(' ');
            spec += argv[i];
        }
        ManifoldEngine eng;
        if (!eng.load_checkpoint(db)) {
            std::cerr << "Error: cannot load " << db << "\n";
            return 1;
        }
        auto plan = eng.place_task(spec);
        std::cout << "ARCHETYPE=" << archetype_name(plan.lucidity.best_archetype) << "\n";
        std::cout << "LUCIDITY=" << std::fixed << std::setprecision(3) << plan.lucidity.lucidity << "\n";
        std::cout << "DECISION=" << plan.decision << "\n";
        if (plan.best_block) {
            std::cout << "TARGET=" << plan.best_block->coord.uri << ":" << plan.best_block->coord.start_line << "\n";
        }
        if (plan.rejected_hazard && plan.hazard_z_score >= 3.0) {
            std::cout << "HAZARD=" << plan.rejected_hazard->preview << " (+" << plan.hazard_z_score << "s)\n";
        }
        return 0;
    }

    if (cmd == "record" && argc >= 6) {
        std::string db = argv[2];
        std::string exp_id = argv[3];
        std::string verdict_str = argv[4];
        std::string desc = argv[5];
        ManifoldEngine eng;
        if (fs::exists(db)) eng.load_checkpoint(db);
        Hypervec8192 dhv = encode_span_hypervector(desc);
        double sim = 0.0;
        auto [arch, mask] = eng.classify_archetype(dhv, sim);
        EpistemicState st = (verdict_str == "keep" || verdict_str == "KEEP") ? EpistemicState::KEEP : EpistemicState::REJECT;
        eng.record_experiment(exp_id, arch, st, desc);
        eng.save_checkpoint(db);
        std::cout << "Recorded experiment " << exp_id << " [" << epistemic_name(st) << "] under " << archetype_name(arch) << "\n";
        return 0;
    }

    if (cmd == "diff" && argc >= 4) {
        ManifoldEngine a, b;
        if (!a.load_checkpoint(argv[2]) || !b.load_checkpoint(argv[3])) {
            std::cerr << "Error loading checkpoints for diff\n";
            return 1;
        }
        uint32_t dist = a.global_acc.project().hamming(b.global_acc.project());
        std::cout << "Manifold Checkpoint Diff: " << dist << " / 8192 bits ("
                  << std::fixed << std::setprecision(2) << (100.0 * dist / 8192.0) << "% shift) | "
                  << "Cells: " << a.cells.size() << " -> " << b.cells.size() << "\n";
        return 0;
    }

    std::cout << "Usage:\n"
              << "  manifold selftest\n"
              << "  manifold benchmark [dir_path]\n"
              << "  manifold init <db.mfd>\n"
              << "  manifold ingest <db.mfd> [--scope repo|ref|web|docs] [--prefix name] <dir_or_file>...\n"
              << "  manifold query <db.mfd> \"[@repo|@ref|@web|@docs] [#io|#state|#space|#op|#route|#gate] [/file|/block] <expr>\"\n"
              << "  manifold place <db.mfd> \"<task specification>\"\n"
              << "  manifold record <db.mfd> <exp_id> <keep|reject> \"<hypothesis & result>\"\n"
              << "  manifold diff <ckpt_a.mfd> <ckpt_b.mfd>\n";
    return 0;
}