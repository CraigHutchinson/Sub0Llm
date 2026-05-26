#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Tensor make_token_ids(int64_t T, int64_t vocab, std::mt19937_64& rng) {
    Tensor ids({T}, DType::Int32);
    std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(vocab - 1));
    for (auto& v : ids.data_as<int32_t>()) v = dist(rng);
    return ids;
}

// ── § 10.1  SiLU vs GELU ─────────────────────────────────────────────────────

static void section_activations() {
    std::cout << "\n=== §10.1  SiLU vs GELU ===\n";

    const std::vector<float> xs = {-3.f, -1.f, 0.f, 0.5f, 1.f, 2.f, 3.f};
    std::cout << std::format("{:>8}  {:>10}  {:>10}\n", "x", "SiLU", "GELU");

    for (float xv : xs) {
        Tensor t({1}, DType::Float32);
        t.data_as<float>()[0] = xv;
        const float s = ops::silu(t).data_as<float>()[0];
        const float g = ops::gelu(t).data_as<float>()[0];
        std::cout << std::format("{:>8.2f}  {:>10.6f}  {:>10.6f}\n", xv, s, g);
    }
    std::cout << "  SiLU(x)=x·σ(x) is smoother than GELU near x=0 and slightly cheaper.\n";
}

// ── § 10.2  RMSNorm vs LayerNorm ─────────────────────────────────────────────

static void section_rmsnorm() {
    std::cout << "\n=== §10.2  RMSNorm vs LayerNorm ===\n";

    constexpr int64_t D = 8;
    RMSNorm norm(D);

    Tensor xd({1, D}, DType::Float32);
    {
        auto d = xd.data_as<float>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(D); ++i)
            d[i] = static_cast<float>(i + 1) * 0.5f;
    }

    Variable x(xd, /*requires_grad=*/false);
    auto y = norm.forward(x);

    // Compute row RMS manually to verify.
    float rms2 = 0.0f;
    auto xd_data = xd.data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(D); ++i)
        rms2 += xd_data[i] * xd_data[i];
    rms2 = std::sqrt(rms2 / static_cast<float>(D) + 1e-6f);

    std::cout << std::format("  Input  RMS = {:.6f}\n", rms2);
    float out_rms = 0.0f;
    auto yd = y.data().data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(D); ++i) out_rms += yd[i] * yd[i];
    out_rms = std::sqrt(out_rms / static_cast<float>(D));
    std::cout << std::format("  Output RMS ≈ 1.0  (got {:.6f}) — weight=1 initialisation\n", out_rms);
    std::cout << "  RMSNorm: no mean subtraction, no bias — 2× cheaper than LayerNorm.\n";
}

// ── § 10.3  SwiGLU FFN ────────────────────────────────────────────────────────

static void section_swiglu() {
    std::cout << "\n=== §10.3  SwiGLU Feed-Forward ===\n";

    constexpr int64_t D    = 32;
    constexpr int64_t d_ff = 0;  // auto = 8*D/3 rounded to 64

    SwiGLUFeedForward ffn(D, d_ff, /*seed=*/1);
    auto params = ffn.parameters();
    int64_t n_params = 0;
    for (auto* p : params) n_params += p->data().numel();

    const int64_t d_ff_actual = (((8 * D + 2) / 3 + 63) / 64) * 64;
    std::cout << std::format("  D={}, d_ff={} (auto 8D/3↑64), params={}\n",
                              D, d_ff_actual, n_params);
    std::cout << std::format("  3 projections: gate(D→d_ff) + up(D→d_ff) + down(d_ff→D)\n");

    Tensor xd({4, D}, DType::Float32);
    {
        std::mt19937_64 rng(99);
        std::normal_distribution<float> nd;
        for (auto& v : xd.data_as<float>()) v = nd(rng) * 0.1f;
    }
    Variable x(xd, true);
    auto y = ffn.forward(x);
    std::cout << std::format("  forward OK — output shape ({}, {})\n",
                              y.data().shape()[0], y.data().shape()[1]);
}

// ── § 10.4  RoPE position encoding ───────────────────────────────────────────

static void section_rope() {
    std::cout << "\n=== §10.4  RoPE Position Encoding ===\n";

    // Construct a tiny GQA to observe that RoPE makes Q/K position-dependent.
    constexpr std::size_t Demb = 16, Nh = 2, Nkv = 1;
    GroupedQueryAttention gqa(Demb, Nh, Nkv, 10000.0f, /*seed=*/5);

    Tensor ids({6}, DType::Int32);
    for (auto& v : ids.data_as<int32_t>()) v = 0;

    // Embed via a tiny embedding table so we have real Tensors to feed.
    Tensor emb_w({8, static_cast<int64_t>(Demb)}, DType::Float32);
    {
        std::mt19937_64 rng(3);
        std::normal_distribution<float> nd(0.f, 0.1f);
        for (auto& v : emb_w.data_as<float>()) v = nd(rng);
    }
    Tensor xd({6, static_cast<int64_t>(Demb)}, DType::Float32);
    {
        std::mt19937_64 rng(7);
        std::normal_distribution<float> nd(0.f, 0.1f);
        for (auto& v : xd.data_as<float>()) v = nd(rng);
    }
    Variable x(xd, false);
    auto out = gqa.forward(x, /*causal=*/true);
    std::cout << std::format("  GQA forward OK  (T=6, embed={}): output shape ({},{})\n",
                              Demb, out.data().shape()[0], out.data().shape()[1]);
    std::cout << "  RoPE bakes position into Q,K — no learned position parameters needed.\n";
}

// ── § 10.5  Grouped-Query Attention parameter savings ────────────────────────

static void section_gqa_params() {
    std::cout << "\n=== §10.5  GQA Parameter Efficiency ===\n";

    const std::size_t D = 512, Nh = 8;
    [[maybe_unused]] const std::size_t Dh = D / Nh;

    auto count_gqa = [&](std::size_t nkv) -> int64_t {
        GroupedQueryAttention g(D, Nh, nkv, 10000.0f, 42);
        auto ps = g.parameters();
        int64_t n = 0; for (auto* p : ps) n += p->data().numel();
        return n;
    };

    for (std::size_t nkv : {8u, 4u, 2u, 1u}) {
        const int64_t p = count_gqa(nkv);
        std::cout << std::format("  n_kv_heads={}: {:>7} params  ({}× KV reduction vs MHA)\n",
                                  nkv, p, 8 / nkv);
    }
    std::cout << "  n_kv_heads=1 (MQA) reduces KV params 8× with minimal quality loss.\n";
}

// ── § 10.6  ModernGPT forward + backward ─────────────────────────────────────

static void section_modern_gpt() {
    std::cout << "\n=== §10.6  ModernGPT forward + backward ===\n";

    constexpr int64_t     V      = 64;
    constexpr int64_t     D      = 32;
    constexpr std::size_t Nh     = 2;
    constexpr std::size_t Nkv    = 1;
    constexpr int64_t     Layers = 2;

    ModernGPT model(V, D, Nh, Nkv, Layers, /*d_ff=*/0, /*n_mtp=*/0, /*seed=*/42);

    std::mt19937_64 rng(11);
    Tensor ids = make_token_ids(8, V, rng);

    auto logits = model.forward(ids);
    std::cout << std::format("  forward: logits ({},{})\n",
                              logits.data().shape()[0], logits.data().shape()[1]);

    // Build a simple CE loss and run backward.
    Tensor tgt({8}, DType::Int32);
    {
        std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(V - 1));
        for (auto& v : tgt.data_as<int32_t>()) v = dist(rng);
    }
    auto loss = cross_entropy(logits, tgt);
    loss.backward();

    auto params = model.parameters();
    int64_t n_with_grad = 0, total_params = 0;
    for (auto* p : params) {
        total_params += p->data().numel();
        if (p->grad().numel() > 0) ++n_with_grad;
    }
    std::cout << std::format("  backward OK — {}/{} params have grad, total {} floats\n",
                              n_with_grad, params.size(), total_params);
}

// ── § 10.7  MTP heads ─────────────────────────────────────────────────────────

static void section_mtp() {
    std::cout << "\n=== §10.7  Multi-Token Prediction (MTP) heads ===\n";

    constexpr int64_t     V      = 64;
    constexpr int64_t     D      = 32;
    constexpr std::size_t Nh     = 2;
    constexpr std::size_t Nkv    = 1;
    constexpr int64_t     Layers = 2;
    constexpr int64_t     Nmtp   = 3;

    ModernGPT model(V, D, Nh, Nkv, Layers, 0, Nmtp, /*seed=*/7);

    std::mt19937_64 rng(22);
    Tensor ids = make_token_ids(8, V, rng);

    auto heads = model.forward_mtp(ids);
    std::cout << std::format("  forward_mtp: {} heads returned (head_0 + {} MTP)\n",
                              heads.size(), Nmtp);
    for (std::size_t i = 0; i < heads.size(); ++i)
        std::cout << std::format("    head[{}]: logits ({},{})\n",
                                  i, heads[i].data().shape()[0], heads[i].data().shape()[1]);

    // Accumulate CE losses from all heads — main head for t, MTP head k for t+k+1.
    const int64_t T = static_cast<int64_t>(ids.shape()[0]);
    std::vector<Tensor> targets;
    targets.reserve(1 + Nmtp);
    for (int64_t k = 0; k <= Nmtp; ++k) {
        const int64_t offset = k + 1;
        const int64_t len    = T - offset;
        if (len <= 0) break;
        Tensor tgt_k({len}, DType::Int32);
        auto td = tgt_k.data_as<int32_t>();
        auto id = ids.data_as<int32_t>();
        for (int64_t i = 0; i < len; ++i)
            td[static_cast<std::size_t>(i)] = id[static_cast<std::size_t>(i + offset)];
        targets.push_back(tgt_k);
    }

    if (targets.empty()) {
        std::cout << "  MTP combined loss = N/A (sequence too short for any offset)\n";
        return;
    }

    // Sum losses; use sub-sequence logits matching each target length.
    Variable total_loss;
    bool first = true;
    for (std::size_t k = 0; k < targets.size(); ++k) {
        const int64_t len = targets[k].shape()[0];
        // Take first `len` rows of logits for head k.
        // Workaround: build a sub-tensor by copy (full slice ops in Ch15).
        Tensor sub_logits({len, V}, DType::Float32);
        {
            auto src = heads[k].data().data_as<float>();
            auto dst = sub_logits.data_as<float>();
            for (std::size_t r = 0; r < static_cast<std::size_t>(len); ++r)
                for (std::size_t c = 0; c < static_cast<std::size_t>(V); ++c)
                    dst[r * static_cast<std::size_t>(V) + c] =
                        src[r * static_cast<std::size_t>(V) + c];
        }
        Variable sub_var(sub_logits, false);
        auto l = cross_entropy(sub_var, targets[k]);
        if (first) { total_loss = l; first = false; }
        else         total_loss = add(total_loss, l);
    }
    auto loss_val = total_loss.data().data_as<float>()[0];
    std::cout << std::format("  MTP combined loss = {:.4f}\n", loss_val);
    std::cout << "  MTP heads predict tokens at offsets +1…+k simultaneously during training.\n";
}

// ── § 10.8  Training loop with ModernGPT ─────────────────────────────────────

static void section_training_loop() {
    std::cout << "\n=== §10.8  ModernGPT mini training loop ===\n";

    constexpr int64_t     V      = 128;
    constexpr int64_t     D      = 64;
    constexpr std::size_t Nh     = 4;
    constexpr std::size_t Nkv    = 2;
    constexpr int64_t     Layers = 2;
    constexpr int64_t     T      = 16;
    constexpr float       lr     = 3e-3f;
    constexpr int         steps  = 40;

    ModernGPT model(V, D, Nh, Nkv, Layers, 0, 0, /*seed=*/99);
    auto params = model.parameters();

    std::mt19937_64 rng(55);
    Tensor ids = make_token_ids(T, V, rng);
    Tensor tgt({T}, DType::Int32);
    {
        auto id = ids.data_as<int32_t>();
        auto td = tgt.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
            td[i] = id[(i + 1u) % static_cast<std::size_t>(T)];
    }

    float first_loss = -1.f, last_loss = -1.f;

    for (int step = 0; step < steps; ++step) {
        // Zero gradients.
        for (auto* p : params) p->grad() = zeros(p->data().shape(), DType::Float32);

        auto logits = model.forward(ids);
        auto loss   = cross_entropy(logits, tgt);
        loss.backward();

        const float lv = loss.data().data_as<float>()[0];
        if (step == 0)        first_loss = lv;
        if (step == steps - 1) last_loss  = lv;

        // SGD update.
        for (auto* p : params) {
            if (p->grad().numel() == 0) continue;
            auto pd = p->data().data_as<float>();
            auto gd = p->grad().data_as<float>();
            for (std::size_t i = 0; i < pd.size(); ++i) pd[i] -= lr * gd[i];
        }
    }

    std::cout << std::format("  step  0  loss = {:.4f}\n", first_loss);
    std::cout << std::format("  step {:>2}  loss = {:.4f}  ({})\n",
                              steps, last_loss,
                              last_loss < first_loss ? "converging ✓" : "not converging ✗");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 10 — Modern Architecture (LLaMA/Gemma/Qwen style)\n";
    std::cout << std::string(60, '=') << '\n';

    section_activations();
    section_rmsnorm();
    section_swiglu();
    section_rope();
    section_gqa_params();
    section_modern_gpt();
    section_mtp();
    section_training_loop();

    std::cout << "\nDone.\n";
}
