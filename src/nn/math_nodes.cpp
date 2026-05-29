#include "sub0llm/nn/math_nodes.hpp"

#include "sub0llm/autograd/embedding_ops.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>

namespace sub0llm::nn {

using namespace autograd;

// ── apply_math_op ─────────────────────────────────────────────────────────────

MathResult apply_math_op(RouteType op, float a, float b) {
    auto overflow_check = [](float result) -> MathResult {
        const auto iv = static_cast<int32_t>(std::round(result));
        if (iv < NumericTokenizer::kIntMin || iv > NumericTokenizer::kIntMax)
            return {0.0f, false, true};
        return {static_cast<float>(iv), false, false};
    };

    switch (op) {
        case RouteType::Add:
            return overflow_check(a + b);
        case RouteType::Sub:
            return overflow_check(a - b);
        case RouteType::Mul:
            return overflow_check(a * b);
        case RouteType::Div:
            if (std::abs(b) < 1e-9f)
                return {0.0f, true, false};
            return overflow_check(std::round(a / b));
        case RouteType::IsLessThan:
            return {a < b ? 1.0f : 0.0f, false, false};
        case RouteType::IsGreaterThan:
            return {a > b ? 1.0f : 0.0f, false, false};
        case RouteType::IsEqual:
            return {a == b ? 1.0f : 0.0f, false, false};
        // Unary: b = the operand (most-recent numeric); a is unused
        case RouteType::Increment:
            return overflow_check(b + 1.0f);
        case RouteType::Decrement:
            return overflow_check(b - 1.0f);
        case RouteType::FFN:
        default:
            return {0.0f, true, false};
    }
}

// ── MathLayer ─────────────────────────────────────────────────────────────────

MathLayer::MathLayer(int64_t D, int64_t d_ff, std::uint64_t seed)
    : D_(D),
      norm_(D),
      router_(D, seed),
      ffn_(D, d_ff, seed + 100) {
    if (D <= 0)
        throw std::runtime_error(
            std::format("MathLayer: D={} must be positive", D));
}

autograd::Variable MathLayer::forward(
    const autograd::Variable& h,
    const std::vector<float>& reg,
    const autograd::Variable& emb_weight,
    const NumericTokenizer&   ntok) const
{
    // Delegate to forward_with_boost with unit scale so the two paths stay in sync.
    auto [ffn_residual, math_inject] =
        forward_with_boost(h, reg, emb_weight, ntok, 1.0f);
    return add(ffn_residual, math_inject);
}

// ── MathLayer::forward_with_boost ────────────────────────────────────────────
// Like forward(), but separates the FFN residual (for the block residual
// stream) from the math-result injection (to be added AFTER ln_f).
// Injecting after ln_f bypasses layer-norm magnitude suppression:
// the result embedding contribution isn't normalised away before the
// final weight-tied projection.

std::pair<autograd::Variable, autograd::Variable>
MathLayer::forward_with_boost(
    const autograd::Variable& h,
    const std::vector<float>& reg,
    const autograd::Variable& emb_weight,
    const NumericTokenizer&   ntok,
    float                     kMathBoost) const
{
    const auto& hd = h.data();
    if (hd.ndim() != 2)
        throw std::runtime_error("MathLayer::forward_with_boost: h must be 2D (T, D)");
    const int64_t T = hd.shape(0);
    if (T < 1)
        throw std::runtime_error("MathLayer::forward_with_boost: T must be >= 1");
    if (static_cast<int64_t>(reg.size()) != T)
        throw std::runtime_error(std::format(
            "MathLayer::forward_with_boost: reg.size()={} != T={}", reg.size(), T));

    const auto T_sz = static_cast<std::size_t>(T);

    // Pre-norm + routing (identical to forward)
    Variable xn = norm_.forward(h);
    auto [soft_probs, hard_mask] = router_.forward(xn);
    Variable gates   = mul(soft_probs, Variable(hard_mask, false));  // (T, K) STE
    Variable gates_T = transpose2d(gates);                            // (K, T)

    // ── FFN branch (goes into the block residual stream, as before) ───────────
    Variable ffn_out  = ffn_.forward(xn);
    Variable gate_row0    = narrow(gates_T, 0, 1);
    Variable gate_col0    = transpose2d(gate_row0);
    Variable ffn_residual = row_scale(ffn_out, gate_col0);

    // ── Operand positions ─────────────────────────────────────────────────────
    std::vector<int64_t> op1_pos(T_sz, -1);
    std::vector<int64_t> op2_pos(T_sz, -1);
    for (std::size_t t = 0; t < T_sz; ++t) {
        int count = 0;
        for (int64_t s = static_cast<int64_t>(t) - 1; s >= 0 && count < 2; --s) {
            if (!std::isnan(reg[static_cast<std::size_t>(s)])) {
                if (count == 0) op1_pos[t] = s;
                else            op2_pos[t] = s;
                ++count;
            }
        }
    }

    // ── Math-result injection: (T, D) scaled result embeddings ───────────────
    // Accumulate gate_k * emb[result_k] for each math op k (NaN/overflow zeroed
    // out via valid_mask).  After the loop, scale by kMathBoost so that validity
    // (0/1) and the boost magnitude are kept separate — kMathBoost=0 explicitly
    // disables injection without corrupting the validity logic.
    // Caller adds math_inject to x AFTER ln_f, bypassing layer-norm suppression.
    Variable math_inject(zeros({T, D_}, DType::Float32), false);

    const int32_t nan_id = ntok.nan_token();
    const int32_t ovf_id = ntok.overflow_token();

    // result_ids is an Int32 index tensor (not autograd-tracked); reuse across ops.
    Tensor result_ids({T}, DType::Int32);
    auto   ids_sp = result_ids.data_as<int32_t>();

    for (int k = 1; k < kNumRouteTypes; ++k) {
        const RouteType route_k = static_cast<RouteType>(k);
        const bool is_unary = (route_k == RouteType::Increment ||
                               route_k == RouteType::Decrement);

        // valid_mask[t, 0] = 1.0f when a real numeric result exists, 0 otherwise.
        // A fresh tensor is required each iteration: Variable holds a reference to
        // the underlying storage, so the previous iteration's autograd node would
        // be corrupted if we reset and reuse the same buffer.
        Tensor valid_data = zeros({T, 1}, DType::Float32);
        float* vp = valid_data.data_as<float>().data();

        for (std::size_t t = 0; t < T_sz; ++t) {
            if (op1_pos[t] < 0 || (!is_unary && op2_pos[t] < 0)) {
                ids_sp[t] = nan_id;
            } else {
                const float rhs = reg[static_cast<std::size_t>(op1_pos[t])];
                const float lhs = (!is_unary && op2_pos[t] >= 0)
                                  ? reg[static_cast<std::size_t>(op2_pos[t])] : 0.0f;
                const MathResult mr = apply_math_op(route_k, lhs, rhs);
                if (mr.is_nan || mr.is_overflow) {
                    ids_sp[t] = (mr.is_overflow) ? ovf_id : nan_id;
                } else {
                    ids_sp[t] = ntok.encode_int(static_cast<int32_t>(mr.value));
                    vp[t] = 1.0f;  // validity flag only — boost applied after loop
                }
            }
        }

        Variable result_embs = embedding_lookup(emb_weight, result_ids);    // (T, D)
        Variable gate_row    = narrow(gates_T, static_cast<int64_t>(k), 1); // (1, T)
        Variable gate_col    = transpose2d(gate_row);                        // (T, 1)
        Variable valid_scale(valid_data, false);
        Variable masked_gate = mul(gate_col, valid_scale);                   // (T, 1)

        math_inject = add(math_inject, row_scale(result_embs, masked_gate));
    }

    // Apply boost as a single scalar multiply: keeps validity (0/1) and scale separate.
    return {ffn_residual, scale(math_inject, kMathBoost)};
}

RouteInfo MathLayer::route_info(const autograd::Variable& h) const {
    if (h.data().ndim() != 2)
        throw std::runtime_error("MathLayer::route_info: h must be 2D (T, D)");
    const int64_t T = h.data().shape(0);
    const int64_t K = static_cast<int64_t>(kNumRouteTypes);

    Variable xn = norm_.forward(h);
    auto [soft_probs, hard_mask] = router_.forward(xn);

    auto soft_sp = soft_probs.data().data_as<float>();
    auto mask_sp = hard_mask.data_as<float>();

    RouteInfo info;
    info.routes.reserve(static_cast<std::size_t>(T));
    info.entropy.reserve(static_cast<std::size_t>(T));

    for (int64_t t = 0; t < T; ++t) {
        // Find hard route from one-hot mask
        RouteType route = RouteType::FFN;
        for (int64_t k = 0; k < K; ++k) {
            if (mask_sp[static_cast<std::size_t>(t * K + k)] > 0.5f) {
                route = static_cast<RouteType>(k);
                break;
            }
        }
        info.routes.push_back(route);

        // Compute entropy H = -sum(p * log(p)) from softmax probs
        float H = 0.f;
        for (int64_t k = 0; k < K; ++k) {
            float p = soft_sp[static_cast<std::size_t>(t * K + k)];
            if (p > 1e-9f) H -= p * std::log(p);
        }
        info.entropy.push_back(H);
    }
    return info;
}

std::vector<autograd::Variable*> MathLayer::parameters() {
    std::vector<Variable*> p;
    for (auto* v : norm_.parameters())   p.push_back(v);
    for (auto* v : router_.parameters()) p.push_back(v);
    for (auto* v : ffn_.parameters())    p.push_back(v);
    return p;
}

// ── MathTransformerBlock ──────────────────────────────────────────────────────

MathTransformerBlock::MathTransformerBlock(int64_t D, std::size_t n_heads,
                                           std::size_t n_kv_heads,
                                           int64_t d_ff, std::uint64_t seed)
    : norm1_(D),
      attn_(static_cast<std::size_t>(D), n_heads, n_kv_heads, 10000.0f, seed),
      math_ffn_(D, d_ff, seed + 500) {}

autograd::Variable MathTransformerBlock::forward_math(
    const autograd::Variable& x,
    const std::vector<float>& reg,
    const autograd::Variable& emb_weight,
    const NumericTokenizer&   ntok) const
{
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    return add(h, math_ffn_.forward(h, reg, emb_weight, ntok));
}

std::pair<autograd::Variable, autograd::Variable>
MathTransformerBlock::forward_math_with_boost(
    const autograd::Variable& x,
    const std::vector<float>& reg,
    const autograd::Variable& emb_weight,
    const NumericTokenizer&   ntok,
    float                     kMathBoost) const
{
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    auto [ffn_residual, math_inject] =
        math_ffn_.forward_with_boost(h, reg, emb_weight, ntok, kMathBoost);
    return {add(h, ffn_residual), math_inject};
}

RouteInfo MathTransformerBlock::route_info(const autograd::Variable& x) const {
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    return math_ffn_.route_info(h);
}

autograd::Variable MathLayer::router_logits(const autograd::Variable& h) const {
    if (h.data().ndim() != 2)
        throw std::runtime_error("MathLayer::router_logits: h must be 2D (T, D)");
    Variable xn = norm_.forward(h);
    return router_.router_logits(xn);
}

autograd::Variable MathTransformerBlock::router_logits(const autograd::Variable& x) const {
    auto h = add(x, attn_.forward(norm1_.forward(x), /*causal=*/true));
    // Detach so supervision CE loss does not back-prop through attention weights.
    return math_ffn_.router_logits(detach(h));
}

std::vector<autograd::Variable*> MathTransformerBlock::parameters() {
    std::vector<Variable*> p;
    for (auto* v : norm1_.parameters())     p.push_back(v);
    for (auto* v : attn_.parameters())      p.push_back(v);
    for (auto* v : math_ffn_.parameters())  p.push_back(v);
    return p;
}

// ── MathGPT ───────────────────────────────────────────────────────────────────

namespace {

int64_t mathgpt_validate(int64_t total_vocab, int64_t embed_dim,
                          std::size_t n_heads, std::size_t n_kv_heads,
                          int64_t n_layers) {
    if (total_vocab <= 0)
        throw std::runtime_error(
            std::format("MathGPT: total_vocab={} must be positive", total_vocab));
    if (embed_dim <= 0)
        throw std::runtime_error(
            std::format("MathGPT: embed_dim={} must be positive", embed_dim));
    if (n_layers <= 0)
        throw std::runtime_error(
            std::format("MathGPT: n_layers={} must be positive", n_layers));
    if (n_heads == 0)
        throw std::runtime_error(
            std::format("MathGPT: n_heads={} must be positive", n_heads));
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0)
        throw std::runtime_error(std::format(
            "MathGPT: n_heads={} must be divisible by n_kv_heads={}",
            n_heads, n_kv_heads));
    return embed_dim;
}

int64_t compute_l_math(int64_t l_math, int64_t n_layers) {
    if (l_math < 0)
        l_math = static_cast<int64_t>(std::round(0.7 * static_cast<double>(n_layers)));
    if (l_math < 0)         l_math = 0;
    if (l_math >= n_layers) l_math = n_layers - 1;
    return l_math;
}

} // anonymous namespace

MathGPT::MathGPT(int64_t       total_vocab,
                 int64_t       embed_dim,
                 std::size_t   n_heads,
                 std::size_t   n_kv_heads,
                 int64_t       n_layers,
                 int64_t       l_math,
                 int64_t       d_ff,
                 std::uint64_t seed,
                 float         math_boost)
    : tok_emb_(total_vocab,
               mathgpt_validate(total_vocab, embed_dim, n_heads, n_kv_heads, n_layers),
               seed),
      math_block_(embed_dim, n_heads, n_kv_heads, d_ff,
                  seed + 2 + static_cast<std::uint64_t>(
                      compute_l_math(l_math, n_layers)) * 2000),
      ln_f_(embed_dim),
      l_math_(compute_l_math(l_math, n_layers)),
      math_boost_(math_boost)
{
    blocks_.reserve(static_cast<std::size_t>(n_layers));
    for (int64_t li = 0; li < n_layers; ++li)
        blocks_.emplace_back(embed_dim, n_heads, n_kv_heads, d_ff,
                             seed + 2 + static_cast<std::uint64_t>(li) * 2000);
}

autograd::Variable MathGPT::forward(const Tensor& token_ids) const {
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MathGPT::forward: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("MathGPT::forward: token_ids must be non-empty");

    Variable x = tok_emb_.forward(token_ids);
    for (const auto& block : blocks_) x = block.forward(x);
    x = ln_f_.forward(x);
    return matmul(x, transpose2d(tok_emb_.weight()));
}

autograd::Variable MathGPT::forward_math(
    const Tensor& token_ids, const NumericTokenizer& ntok, TokenMode mode) const
{
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MathGPT::forward_math: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("MathGPT::forward_math: token_ids must be non-empty");

    const std::vector<float> reg      = build_register(token_ids, ntok);
    const Tensor             emb_ids  = remap_tokens(token_ids, ntok, mode);

    Variable x = tok_emb_.forward(emb_ids);
    const auto n_layers = static_cast<int64_t>(blocks_.size());

    // math_inject is guaranteed to be assigned: l_math_ is in [0, n_layers).
    Variable math_inject;

    for (int64_t li = 0; li < n_layers; ++li) {
        if (li == l_math_) {
            auto [x_new, inject] = math_block_.forward_math_with_boost(
                x, reg, tok_emb_.weight(), ntok, math_boost_);
            x           = x_new;
            math_inject = inject;
        } else {
            x = blocks_[static_cast<std::size_t>(li)].forward(x);
        }
    }

    x = ln_f_.forward(x);
    // Add the result-embedding injection AFTER ln_f so it is not normalised
    // away before the final weight-tied projection to logits.
    x = add(x, math_inject);

    return matmul(x, transpose2d(tok_emb_.weight()));
}

RouteInfo MathGPT::route_info(
    const Tensor& token_ids, const NumericTokenizer& ntok, TokenMode mode) const
{
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MathGPT::route_info: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("MathGPT::route_info: token_ids must be non-empty");

    const Tensor emb_ids = remap_tokens(token_ids, ntok, mode);
    Variable x = tok_emb_.forward(emb_ids);

    for (int64_t li = 0; li < l_math_; ++li)
        x = blocks_[static_cast<std::size_t>(li)].forward(x);

    return math_block_.route_info(x);
}

autograd::Variable MathGPT::router_logits(
    const Tensor& token_ids, const NumericTokenizer& ntok, TokenMode mode) const
{
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MathGPT::router_logits: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("MathGPT::router_logits: token_ids must be non-empty");

    const Tensor emb_ids = remap_tokens(token_ids, ntok, mode);
    Variable x = tok_emb_.forward(emb_ids);
    for (int64_t li = 0; li < l_math_; ++li)
        x = blocks_[static_cast<std::size_t>(li)].forward(x);
    return math_block_.router_logits(x);
}

autograd::Variable MathGPT::router_logits(const Tensor& token_ids) const
{
    // Real-mode convenience overload — no remapping.
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MathGPT::router_logits: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("MathGPT::router_logits: token_ids must be non-empty");

    Variable x = tok_emb_.forward(token_ids);
    for (int64_t li = 0; li < l_math_; ++li)
        x = blocks_[static_cast<std::size_t>(li)].forward(x);
    return math_block_.router_logits(x);
}

std::vector<autograd::Variable*> MathGPT::parameters() {
    std::vector<Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto& block : blocks_)
        for (auto* v : block.parameters()) p.push_back(v);
    for (auto* v : math_block_.parameters()) p.push_back(v);
    for (auto* v : ln_f_.parameters()) p.push_back(v);
    return p;
}

std::vector<autograd::Variable*> MathGPT::math_parameters() {
    std::vector<Variable*> p;
    p.push_back(&tok_emb_.weight());
    for (auto* v : math_block_.parameters()) p.push_back(v);
    return p;
}

std::vector<autograd::Variable*> MathGPT::math_block_only_parameters() {
    return math_block_.parameters();
}

int64_t     MathGPT::l_math()     const noexcept { return l_math_; }
int64_t     MathGPT::vocab_size() const noexcept { return tok_emb_.vocab_size(); }
int64_t     MathGPT::embed_dim()  const noexcept { return tok_emb_.embed_dim(); }
std::size_t MathGPT::num_layers() const noexcept { return blocks_.size(); }

void MathGPT::import_math_block(MathGPT& source) {
    auto src_p = source.math_block_.parameters();
    auto dst_p = math_block_.parameters();
    if (src_p.size() != dst_p.size())
        throw std::runtime_error(std::format(
            "import_math_block: param count mismatch: {} vs {}",
            src_p.size(), dst_p.size()));
    for (std::size_t i = 0; i < src_p.size(); ++i) {
        const Tensor& src_t = src_p[i]->data();
        Tensor& dst_t = dst_p[i]->data();
        if (src_t.shape() != dst_t.shape())
            throw std::runtime_error(std::format(
                "import_math_block: tensor {} shape mismatch (numel {} vs {})",
                i, src_t.numel(), dst_t.numel()));
        std::memcpy(dst_t.data_as<float>().data(),
                    src_t.data_as<float>().data(),
                    static_cast<std::size_t>(src_t.numel()) * sizeof(float));
    }
}

Tensor MathGPT::remap_tokens(
    const Tensor& token_ids, const NumericTokenizer& ntok, TokenMode mode)
{
    if (mode == TokenMode::Real) return token_ids;

    const int64_t T   = token_ids.shape()[0];
    const auto    src = token_ids.data_as<int32_t>();
    Tensor        out({T}, DType::Int32);
    auto          dst = out.data_as<int32_t>();

    if (mode == TokenMode::Anon) {
        const auto placeholder = static_cast<int32_t>(ntok.num_placeholder_token());
        for (int64_t t = 0; t < T; ++t) {
            const auto id = static_cast<NumericTokenizer::TokenId>(
                src[static_cast<std::size_t>(t)]);
            dst[static_cast<std::size_t>(t)] =
                ntok.is_numeric(id) ? placeholder : src[static_cast<std::size_t>(t)];
        }
    } else {
        // Algebraic / AlgebraicSpecial: assign X0, X1, … to each distinct numeric
        // value in order of first appearance. Same value → same slot.
        // AlgebraicSpecial keeps {-1, 0, 1} as their real token IDs so the
        // transformer can reason structurally about identity elements and boolean
        // outputs without those values being collapsed into opaque slots.
        const bool preserve_special = (mode == TokenMode::AlgebraicSpecial);
        auto is_special = [](float v) {
            return v == -1.0f || v == 0.0f || v == 1.0f;
        };

        std::array<float, NumericTokenizer::kAlgSlots> seen{};
        seen.fill(std::numeric_limits<float>::quiet_NaN());
        int next_slot = 0;
        const auto placeholder = static_cast<int32_t>(ntok.num_placeholder_token());

        for (int64_t t = 0; t < T; ++t) {
            const auto id = static_cast<NumericTokenizer::TokenId>(
                src[static_cast<std::size_t>(t)]);
            if (ntok.is_numeric(id) && !ntok.is_nan_token(id) &&
                !ntok.is_overflow_token(id)) {
                const float val = ntok.numeric_value(id);
                if (preserve_special && is_special(val)) {
                    dst[static_cast<std::size_t>(t)] = src[static_cast<std::size_t>(t)];
                    continue;
                }
                int slot = -1;
                for (int s = 0; s < next_slot; ++s) {
                    if (seen[static_cast<std::size_t>(s)] == val) { slot = s; break; }
                }
                if (slot < 0 && next_slot < static_cast<int>(NumericTokenizer::kAlgSlots)) {
                    seen[static_cast<std::size_t>(next_slot)] = val;
                    slot = next_slot++;
                }
                dst[static_cast<std::size_t>(t)] = (slot >= 0)
                    ? static_cast<int32_t>(ntok.algebraic_token(slot))
                    : placeholder;
            } else {
                dst[static_cast<std::size_t>(t)] = src[static_cast<std::size_t>(t)];
            }
        }
    }
    return out;
}

std::vector<float> MathGPT::build_register(
    const Tensor& token_ids, const NumericTokenizer& ntok)
{
    const int64_t T = token_ids.shape()[0];
    const auto ids_sp = token_ids.data_as<int32_t>();
    std::vector<float> reg(static_cast<std::size_t>(T),
                           std::numeric_limits<float>::quiet_NaN());

    for (int64_t t = 0; t < T; ++t) {
        const auto id = static_cast<NumericTokenizer::TokenId>(
            ids_sp[static_cast<std::size_t>(t)]);
        if (ntok.is_numeric(id) &&
            !ntok.is_nan_token(id) &&
            !ntok.is_overflow_token(id))
        {
            reg[static_cast<std::size_t>(t)] = ntok.numeric_value(id);
        }
    }
    return reg;
}

} // namespace sub0llm::nn
