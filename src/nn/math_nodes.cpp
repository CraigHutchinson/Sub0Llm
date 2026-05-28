#include "sub0llm/nn/math_nodes.hpp"

#include "sub0llm/autograd/embedding_ops.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cmath>
#include <cstring>
#include <format>
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
    const auto& hd = h.data();
    if (hd.ndim() != 2)
        throw std::runtime_error("MathLayer::forward: h must be 2D (T, D)");
    const int64_t T = hd.shape(0);
    if (T < 1)
        throw std::runtime_error("MathLayer::forward: T must be >= 1");
    if (static_cast<int64_t>(reg.size()) != T)
        throw std::runtime_error(std::format(
            "MathLayer::forward: reg.size()={} != T={}", reg.size(), T));

    // Pre-norm
    Variable xn = norm_.forward(h);   // (T, D)

    // STE routing
    auto [soft_probs, hard_mask] = router_.forward(xn);   // soft: (T, K), mask: (T, K)

    // Sparse gates: STE — forward argmax, backward through softmax
    Variable gates = mul(soft_probs, Variable(hard_mask, false));  // (T, K)

    // FFN branch
    Variable ffn_out = ffn_.forward(xn);  // (T, D)

    // Accumulate output
    Variable output(zeros({T, D_}, DType::Float32), false);

    // Extract gate column for FFN (k=0) and scale ffn_out
    {
        Variable gates_T     = transpose2d(gates);               // (K, T)
        Variable gate_row0   = narrow(gates_T, 0, 1);            // (1, T)
        Variable gate_col0   = transpose2d(gate_row0);           // (T, 1)
        output = add(output, row_scale(ffn_out, gate_col0));
    }

    // Find two most recent non-NaN operand positions per token
    const auto T_sz = static_cast<std::size_t>(T);
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

    // Math operation branches (k = 1..5)
    for (int k = 1; k < kNumRouteTypes; ++k) {
        const RouteType route_k = static_cast<RouteType>(k);

        // Build result token IDs for this operation
        Tensor result_ids({T}, DType::Int32);
        auto   ids_sp = result_ids.data_as<int32_t>();

        for (std::size_t t = 0; t < T_sz; ++t) {
            if (op1_pos[t] < 0) {
                ids_sp[t] = ntok.nan_token();
            } else {
                // op1 = most recent token = RIGHT operand in "A op B"
                // op2 = second most recent = LEFT operand
                // apply_math_op contract: (op, a=LEFT, b=RIGHT)
                const float rhs = reg[static_cast<std::size_t>(op1_pos[t])];
                const float lhs = (op2_pos[t] >= 0)
                    ? reg[static_cast<std::size_t>(op2_pos[t])]
                    : rhs;
                const MathResult mr = apply_math_op(route_k, lhs, rhs);
                if (mr.is_nan)
                    ids_sp[t] = ntok.nan_token();
                else if (mr.is_overflow)
                    ids_sp[t] = ntok.overflow_token();
                else
                    ids_sp[t] = ntok.encode_int(static_cast<int32_t>(mr.value));
            }
        }

        // Look up embeddings for the result tokens
        Variable math_emb = embedding_lookup(emb_weight, result_ids);  // (T, D)

        // Extract gate column for this branch
        Variable gates_T   = transpose2d(gates);                     // (K, T)
        Variable gate_row  = narrow(gates_T, static_cast<int64_t>(k), 1);  // (1, T)
        Variable gate_col  = transpose2d(gate_row);                  // (T, 1)

        output = add(output, row_scale(math_emb, gate_col));
    }

    return output;
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
                 std::uint64_t seed)
    : tok_emb_(total_vocab,
               mathgpt_validate(total_vocab, embed_dim, n_heads, n_kv_heads, n_layers),
               seed),
      math_block_(embed_dim, n_heads, n_kv_heads, d_ff,
                  seed + 2 + static_cast<std::uint64_t>(
                      compute_l_math(l_math, n_layers)) * 2000),
      ln_f_(embed_dim),
      l_math_(compute_l_math(l_math, n_layers))
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
    const Tensor& token_ids, const NumericTokenizer& ntok) const
{
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MathGPT::forward_math: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("MathGPT::forward_math: token_ids must be non-empty");

    const std::vector<float> reg = build_register(token_ids, ntok);

    Variable x = tok_emb_.forward(token_ids);
    const auto n_layers = static_cast<int64_t>(blocks_.size());

    for (int64_t li = 0; li < n_layers; ++li) {
        if (li == l_math_)
            x = math_block_.forward_math(x, reg, tok_emb_.weight(), ntok);
        else
            x = blocks_[static_cast<std::size_t>(li)].forward(x);
    }

    x = ln_f_.forward(x);
    return matmul(x, transpose2d(tok_emb_.weight()));
}

RouteInfo MathGPT::route_info(
    const Tensor& token_ids, const NumericTokenizer& ntok) const
{
    if (token_ids.ndim() != 1)
        throw std::runtime_error("MathGPT::route_info: token_ids must be 1D (T,)");
    if (token_ids.shape()[0] < 1)
        throw std::runtime_error("MathGPT::route_info: token_ids must be non-empty");

    Variable x = tok_emb_.forward(token_ids);

    for (int64_t li = 0; li < l_math_; ++li)
        x = blocks_[static_cast<std::size_t>(li)].forward(x);

    return math_block_.route_info(x);
}

autograd::Variable MathGPT::router_logits(const Tensor& token_ids) const
{
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
                    src_t.numel() * sizeof(float));
    }
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
