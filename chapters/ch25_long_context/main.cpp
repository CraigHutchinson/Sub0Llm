// Chapter 25 — Long-Context Inference
//
// Topics:
//   1. The quadratic wall: why naive generation is O(n²) time and memory
//   2. KV cache: reduce generation to O(n) total work
//   3. Sliding-window attention: O(n·W) memory, O(W) per step
//   4. RoPE NTK-aware scaling: extend context beyond training length
//
// Run:  ./build/bin/ch25_long_context

#include "sub0llm/nn/long_context.hpp"
#include "sub0llm/nn/modern_gpt.hpp"
#include "sub0llm/nn/sampler.hpp"

#include <chrono>
#include <cstdint>
#include <format>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;

// ── helpers ──────────────────────────────────────────────────────────────────

static Tensor make_ids(int64_t T, int32_t V) {
    Tensor ids({T}, DType::Int32);
    auto d = ids.data_as<int32_t>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T); ++i)
        d[i] = static_cast<int32_t>(i % static_cast<std::size_t>(V));
    return ids;
}

static double ms_now() {
    using namespace std::chrono;
    return static_cast<double>(
        duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count()) /
        1000.0;
}

// ── §1  Quadratic wall ────────────────────────────────────────────────────────

static void demo_quadratic_wall() {
    std::cout << "\n=== §1  The quadratic wall ===\n"
              << "With naive forward(), each decoding step re-runs the entire\n"
              << "sequence.  Cost grows as O(T²) for T tokens generated.\n\n"
              << std::left
              << std::setw(10) << "Tokens"
              << std::setw(16) << "Naive ops (rel)"
              << std::setw(16) << "KV-cached ops"
              << "\n"
              << std::string(42, '-') << "\n";

    for (int T : {64, 128, 256, 512, 1024, 4096}) {
        const double naive  = static_cast<double>(T) * static_cast<double>(T);
        const double cached = static_cast<double>(T);
        std::cout << std::setw(10) << T
                  << std::setw(16) << std::format("{:.0f}x", naive / cached)
                  << std::setw(16) << "1x (O(n))\n";
    }
    std::cout << "\nKV cache trades memory for time: store K,V once per token "
              << "and reuse.\n";
}

// ── §2  KV-cache speedup ──────────────────────────────────────────────────────

static void demo_kv_cache_speedup() {
    std::cout << "\n=== §2  KV-cache speedup ===\n";

    // Small model for speed
    const int64_t V = 256, D = 64;
    ModernGPT model(V, D, /*n_heads=*/4, /*n_kv_heads=*/2,
                    /*n_layers=*/2, /*d_ff=*/128);

    const int64_t prompt_len  = 16;
    const int64_t new_tokens  = 32;
    const int32_t start_token = 7;

    // ── Naive generation (re-run full sequence each step) ──────────────────
    const double t0_naive = ms_now();
    {
        Tensor ids = make_ids(prompt_len, static_cast<int32_t>(V));
        std::vector<int32_t> out_ids(static_cast<std::size_t>(prompt_len));
        auto d = ids.data_as<int32_t>();
        for (std::size_t i = 0; i < static_cast<std::size_t>(prompt_len); ++i)
            out_ids[i] = static_cast<int32_t>(d[i]);
        out_ids.push_back(start_token);
        for (int64_t step = 0; step < new_tokens; ++step) {
            Tensor cur_ids({static_cast<int64_t>(out_ids.size())}, DType::Int32);
            auto cd = cur_ids.data_as<int32_t>();
            for (std::size_t i = 0; i < out_ids.size(); ++i)
                cd[i] = out_ids[i];
            auto logits = model.forward(cur_ids);
            // greedy: argmax of last row
            const int64_t T = static_cast<int64_t>(out_ids.size());
            auto ld = logits.data().data_as<float>();
            int32_t best = 0;
            for (int64_t v = 1; v < V; ++v)
                if (ld[static_cast<std::size_t>((T-1) * V + v)] >
                    ld[static_cast<std::size_t>((T-1) * V + best)])
                    best = static_cast<int32_t>(v);
            out_ids.push_back(best);
        }
    }
    const double naive_ms = ms_now() - t0_naive;

    // ── KV-cache generation ────────────────────────────────────────────────
    const double t0_cached = ms_now();
    {
        auto cache = model.make_kv_cache(prompt_len + new_tokens + 4);
        std::vector<int32_t> prompt_vec(static_cast<std::size_t>(prompt_len));
        {
            Tensor ids = make_ids(prompt_len, static_cast<int32_t>(V));
            auto d = ids.data_as<int32_t>();
            for (std::size_t i = 0; i < static_cast<std::size_t>(prompt_len); ++i)
                prompt_vec[i] = static_cast<int32_t>(d[i]);
        }
        LongContextConfig cfg;
        cfg.max_new_tokens = new_tokens;
        cfg.temperature    = 0.0f;  // greedy
        [[maybe_unused]] auto gen = generate_cached(model, prompt_vec, cache, cfg);
    }
    const double cached_ms = ms_now() - t0_cached;

    std::cout << std::format("Prompt {:d} + generate {:d} tokens\n",
                             prompt_len, new_tokens)
              << std::format("  Naive (re-run)  : {:.1f} ms\n", naive_ms)
              << std::format("  KV-cached       : {:.1f} ms\n", cached_ms)
              << std::format("  Speedup         : {:.1f}x\n",
                             naive_ms / std::max(cached_ms, 0.001));
}

// ── §3  Sliding-window attention ──────────────────────────────────────────────

static void demo_sliding_window() {
    std::cout << "\n=== §3  Sliding-window attention ===\n"
              << "With window W, each token attends only to the last W tokens.\n"
              << "Memory: O(n·W) instead of O(n²) for the attention matrix.\n\n";

    const int64_t V = 256, D = 64;
    const int64_t ctx = 64;

    for (int64_t W : {-1L, 16L, 8L}) {
        ModernGPT model(V, D, 4, 2, 2, 128, 0, 42, W);
        auto cache = model.make_kv_cache(ctx);
        std::vector<int32_t> prompt;
        for (int32_t i = 0; i < 8; ++i) prompt.push_back(i);

        LongContextConfig cfg;
        cfg.max_new_tokens = 8;
        auto out = generate_cached(model, prompt, cache, cfg);

        const std::string label = (W < 0) ? "full" : std::format("W={}", W);
        std::cout << std::format("  {:10s}: generated {} tokens — OK\n",
                                 label, out.size());
    }
}

// ── §4  RoPE NTK-aware context extension ─────────────────────────────────────

static void demo_rope_ntk_scaling() {
    std::cout << "\n=== §4  RoPE NTK-aware context scaling ===\n"
              << "Scales the RoPE base so high-frequency dims distort less.\n"
              << "Formula: base_scaled = 10000 × alpha^(Dh / (Dh − 2))\n"
              << "         where alpha = target_len / train_len\n\n";

    const int64_t V = 256, D = 64;
    const int64_t train_ctx = 32;
    const int64_t target_ctx = 64;  // 2× extrapolation

    // Model trained for train_ctx context
    ModernGPT model(V, D, 4, 2, 2, 128);

    // Generate at 2× context using NTK scaling
    const float alpha = static_cast<float>(target_ctx) /
                         static_cast<float>(train_ctx);

    auto cache = model.make_kv_cache(target_ctx);
    std::vector<int32_t> prompt;
    for (int32_t i = 0; i < 16; ++i) prompt.push_back(i % static_cast<int32_t>(V));

    LongContextConfig cfg;
    cfg.max_new_tokens = 16;
    cfg.rope_scaling   = alpha;
    auto out = generate_cached(model, prompt, cache, cfg);

    const float Dh = static_cast<float>(D) / 4.0f;  // head_dim
    const float base_scaled = 10000.0f * std::pow(alpha, Dh / (Dh - 2.0f));
    std::cout << std::format("  Train context  : {} tokens\n", train_ctx)
              << std::format("  Target context : {} tokens (alpha={:.1f})\n",
                             target_ctx, alpha)
              << std::format("  Scaled RoPE base: {:.0f} (base was 10000)\n",
                             base_scaled)
              << std::format("  Generated {} new tokens — no SIGFPE\n", out.size());
}

// ── §5  Memory budget table ───────────────────────────────────────────────────

static void demo_memory_budget() {
    std::cout << "\n=== §5  KV cache memory footprint ===\n"
              << "Showing cache size for a 7B-class model (32 layers, 8 KV heads, head_dim=128)\n\n"
              << std::left
              << std::setw(12) << "Ctx length"
              << std::setw(18) << "Cache (MB, fp32)"
              << "\n"
              << std::string(30, '-') << "\n";

    const int layers = 32, kv_heads = 8, head_dim = 128;
    for (int ctx : {2048, 4096, 8192, 32768, 131072}) {
        const double mb = 2.0 * layers * kv_heads * head_dim * ctx * 4.0 / (1024.0 * 1024.0);
        std::cout << std::setw(12) << ctx
                  << std::format("  {:.0f} MB\n", mb);
    }
    std::cout << "\nFor fp16 (bfloat16) cut these figures in half.\n"
              << "For MQA (n_kv_heads=1) cut by the KV head ratio.\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "Chapter 25 — Long-Context Inference\n"
              << "====================================\n";

    demo_quadratic_wall();
    demo_kv_cache_speedup();
    demo_sliding_window();
    demo_rope_ntk_scaling();
    demo_memory_budget();

    std::cout << "\n[ch25] Done.\n";
    return 0;
}
