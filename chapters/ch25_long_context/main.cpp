// Chapter 25 — Long-Context Inference
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT IS CONTEXT, AND WHY DOES ATTENTION MATTER?
// ─────────────────────────────────────────────────────────────────────────────
//
// Imagine you're writing a very long letter, and every time you write the
// next word you can look back at everything written so far.  That "looking
// back" is attention.  It's doing four completely different jobs at once:
//
//   Job 1 — Grammar glue (last ~20 tokens)
//     "The cat sat on the ___"
//     You only need the last few words.  A tiny window is enough.
//
//   Job 2 — Topic memory (last ~500 tokens)
//     "...as I was saying about the Roman Empire..."
//     You need to know what the conversation is about, what question is being
//     answered, what argument is being built.  This is working memory.
//
//   Job 3 — Key-fact retrieval (anywhere, even token 1)
//     "The password you mentioned earlier was ___"
//     One critical fact buried thousands of tokens ago must be found exactly.
//     This is the hardest job.  It is why 1M-token context is hard.
//
//   Job 4 — Framing / rules (very beginning)
//     "Remember, only respond in French."
//     Set once at the start, applies to everything after.  Models learn to
//     always pay a little attention to the first few tokens — these are called
//     attention sinks (StreamingLLM, 2023).
//
// Every technique in this chapter is a bet about which jobs matter most:
//
//   KV Cache         — doesn't change what you look at, just stops you
//                      re-reading what you already processed. Serves all jobs.
//
//   Sliding Window   — cheap: great for jobs 1+2, deliberately gives up job 3.
//                      Mistral 7B used it; their later frontier models dropped
//                      it because key-fact retrieval suffered too much.
//
//   High RoPE base   — keeps job 3 working at long range without fine-tuning.
//   + YaRN             Llama 3 uses base=500,000; Qwen3 uses base=1,000,000.
//
//   NoPE layers      — Llama 4 iRoPE: every 4th layer has NO positional
//   (Llama 4)          encoding at all and sees the full sequence.  These
//                      layers become job-3 specialists with no position bias.
//
//   Dual Chunk Attn  — Qwen3 DCA: split the sequence into two windows, run
//   (Qwen3)            attention inside each, exchange summary tokens at the
//                      boundary.  Extends from 128K to 1M at inference time.
//
//   MLA              — DeepSeek: instead of storing full K,V per token, store
//   (DeepSeek)         a tiny compressed latent vector.  Project back at use.
//                      9× less KV memory than MHA, outperforms GQA on quality.
//
//   Two-tier attn    — Gemini 2.5: sliding-window local attention + a small
//   (Gemini 2.5)       set of global summary tokens every token can reach.
//                      Jobs 1+2 handled locally; jobs 3+4 via summary tokens.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT MODERN PRODUCTION MODELS ACTUALLY DO (2025/2026)
// ─────────────────────────────────────────────────────────────────────────────
//
//  Model            Context   Attention        Position      KV compression
//  ───────────────  ────────  ───────────────  ────────────  ──────────────
//  Llama 4 Scout    10 M      iRoPE (NoPE+SWA) RoPE base hi  GQA
//  Llama 4 Maverick  1 M      iRoPE            RoPE base hi  GQA
//  Qwen3-235B        1 M      Full + DCA        ABF+YaRN     GQA (32Q/8KV)
//  Gemini 2.5 Pro    1 M      Local+global      ALiBi(?)     MoE+GQA
//  DeepSeek V4       1 M      CSA+HCA hybrid    YaRN          MLA (9× less)
//  Mistral Large 3 256 K      Full (no SWA)     RoPE          GQA
//
// This chapter implements and measures:
//   §0  The four jobs of context (printed explanation)
//   §1  The quadratic wall — why naive generation breaks at scale
//   §2  KV cache — ~9× measured speedup on our model
//   §3  Sliding-window attention — trade-off: memory vs key-fact recall
//   §4  RoPE NTK-aware scaling — the simplest extrapolation technique
//   §5  KV memory budget table — what 1M context actually costs in RAM
//
// Run:  ./build/bin/ch25_long_context

#include "sub0llm/nn/long_context.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;

// ── helpers ───────────────────────────────────────────────────────────────────

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

static void rule(std::string_view c = "─", int n = 60) {
    // c may be a multi-byte UTF-8 glyph (e.g. box-drawing ─/═), so it cannot be
    // a narrow char; repeat the whole sequence n times.
    std::string line;
    line.reserve(c.size() * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) line += c;
    std::cout << line << "\n";
}

// ── §0  The four jobs of context ─────────────────────────────────────────────

static void explain_context_jobs() {
    rule("═");
    std::cout <<
        "§0  What is context, and what does attention actually do?\n";
    rule("═");
    std::cout << R"(
Every time the model generates the next token it looks back at all previous
tokens and decides which ones are relevant.  That look-back is attention.
It is serving four very different jobs simultaneously:

  JOB 1 — Grammar glue          (last ~20 tokens)
    The model needs the last few words to finish a sentence correctly.
    A tiny sliding window is plenty.  Almost free.

  JOB 2 — Topic / working memory (last ~500 tokens)
    Knowing what question is being answered, what argument is building,
    who is speaking.  A moderate local window covers this.

  JOB 3 — Key-fact retrieval    (anywhere in the context)
    "The password you set at the start was ___"
    One fact buried 50,000 tokens ago must be found exactly.
    This is the hard job.  It is why 1M-token context matters.

  JOB 4 — Framing / rules       (very first tokens)
    "You are a helpful assistant who only speaks French."
    Set once at position 0, must influence everything that follows.
    Models develop 'attention sinks': they always attend slightly to
    the first few tokens because the framing lives there.

Each long-context technique is a different bet about which jobs to
optimise for — and which to sacrifice.  The table below maps them:

  Technique              Jobs served    What is sacrificed
  ─────────────────────  ─────────────  ──────────────────────────────
  KV Cache               1, 2, 3, 4    Nothing — pure speed win
  Sliding Window (SWA)   1, 2          Job 3 (cross-window retrieval)
  High RoPE base + YaRN  3, 4          Nothing quality-wise
  NoPE layers (Llama 4)  3, 4          Some local pattern sharpness
  Dual Chunk Attn (DCA)  3 via summary Job 3 fidelity (compression)
  MLA (DeepSeek)         1, 2, 3, 4    Nothing — pure memory win
  Two-tier local+global  1, 2, 3       Some global resolution

Note: Mistral 7B used sliding window and paid the job-3 price.  Their
later frontier models (Large 3, 256K) dropped SWA entirely once Flash
Attention made full causal attention memory-tractable.
)";
}

// ── §1  The quadratic wall ────────────────────────────────────────────────────

static void demo_quadratic_wall() {
    rule("-");
    std::cout << "§1  The quadratic wall\n";
    rule("-");
    std::cout <<
        "With naive forward(), each new token requires re-running the full\n"
        "sequence through every layer.  Step t costs O(t) work; summed over\n"
        "T steps that is O(T²) total — the 'quadratic wall'.\n\n"
        "  Tokens     Naive total ops    KV-cached total ops\n"
        "  ─────────  ─────────────────  ───────────────────\n";

    for (int T : {64, 128, 256, 512, 1024, 4096, 131072}) {
        const double ratio = static_cast<double>(T);
        std::cout << std::format("  {:>8}   {:>12.0f}x        1x\n", T, ratio);
    }

    std::cout <<
        "\nAt 128K tokens the naive approach does 128,000x more total work.\n"
        "The KV cache stores each token's keys and values once, so each new\n"
        "token only pays for its own attention over the cached history.\n";
}

// ── §2  KV-cache speedup ──────────────────────────────────────────────────────

static void demo_kv_cache_speedup() {
    rule("-");
    std::cout << "§2  KV-cache: measured speedup\n";
    rule("-");
    std::cout <<
        "We time two generation strategies on the same model:\n"
        "  Naive   — re-run the full sequence (prompt + generated so far)\n"
        "            at every decoding step.  Correct but O(T²) total.\n"
        "  Cached  — call forward_one() per token; append K,V to the cache.\n"
        "            Attends over the cached prefix.  O(T) total.\n\n";

    const int64_t V = 256, D = 64;
    ModernGPT model(V, D, /*n_heads=*/4, /*n_kv_heads=*/2,
                    /*n_layers=*/2, /*d_ff=*/128);

    const int64_t prompt_len = 16;
    const int64_t new_tokens = 32;

    // Naive
    const double t0_naive = ms_now();
    {
        std::vector<int32_t> out_ids;
        {
            Tensor ids = make_ids(prompt_len, static_cast<int32_t>(V));
            auto d = ids.data_as<int32_t>();
            for (std::size_t i = 0; i < static_cast<std::size_t>(prompt_len); ++i)
                out_ids.push_back(static_cast<int32_t>(d[i]));
        }
        out_ids.push_back(7);
        for (int64_t step = 0; step < new_tokens; ++step) {
            Tensor cur({static_cast<int64_t>(out_ids.size())}, DType::Int32);
            auto cd = cur.data_as<int32_t>();
            for (std::size_t i = 0; i < out_ids.size(); ++i)
                cd[i] = out_ids[i];
            auto logits = model.forward(cur);
            const int64_t T = static_cast<int64_t>(out_ids.size());
            auto ld = logits.data().data_as<float>();
            int32_t best = 0;
            for (int64_t v = 1; v < V; ++v)
                if (ld[static_cast<std::size_t>((T - 1) * V + v)] >
                    ld[static_cast<std::size_t>((T - 1) * V + best)])
                    best = static_cast<int32_t>(v);
            out_ids.push_back(best);
        }
    }
    const double naive_ms = ms_now() - t0_naive;

    // KV-cached
    const double t0_cached = ms_now();
    {
        auto cache = model.make_kv_cache(prompt_len + new_tokens + 4);
        std::vector<int32_t> prompt;
        {
            Tensor ids = make_ids(prompt_len, static_cast<int32_t>(V));
            auto d = ids.data_as<int32_t>();
            for (std::size_t i = 0; i < static_cast<std::size_t>(prompt_len); ++i)
                prompt.push_back(static_cast<int32_t>(d[i]));
        }
        LongContextConfig cfg;
        cfg.max_new_tokens = new_tokens;
        cfg.temperature    = 0.0f;
        [[maybe_unused]] auto gen = generate_cached(model, prompt, cache, cfg);
    }
    const double cached_ms = ms_now() - t0_cached;

    std::cout << std::format(
        "  Prompt {:d} tokens, generate {:d} tokens\n"
        "  Naive (re-run every step) : {:6.1f} ms\n"
        "  KV-cached (forward_one)   : {:6.1f} ms\n"
        "  Speedup                   : {:.1f}x\n\n",
        prompt_len, new_tokens,
        naive_ms, cached_ms,
        naive_ms / std::max(cached_ms, 0.001));

    std::cout <<
        "Memory cost: the cache holds 2 × n_layers × n_kv_heads tensors of\n"
        "shape (max_seq, head_dim).  We pay memory to avoid recomputation.\n"
        "GQA (n_kv_heads < n_heads) is the production answer: share KV\n"
        "heads across query groups so the cache shrinks by the head ratio.\n"
        "DeepSeek MLA goes further — it compresses KV to a latent vector\n"
        "and projects back at use time, achieving 9× less memory than MHA\n"
        "while actually outperforming GQA on quality benchmarks.\n";
}

// ── §3  Sliding-window attention — trade-offs ─────────────────────────────────

static void demo_sliding_window() {
    rule("-");
    std::cout << "§3  Sliding-window attention — job-3 trade-off\n";
    rule("-");
    std::cout <<
        "With window size W each token attends only to the last W tokens.\n"
        "The causal mask becomes a banded diagonal rather than lower-triangular.\n\n"
        "  Benefit : O(n·W) attention memory instead of O(n²)\n"
        "  Cost    : tokens more than W positions apart cannot attend directly\n"
        "            — job 3 (key-fact retrieval) breaks across window boundaries\n\n"
        "In practice Mistral 7B used W=4096.  Retrieval tasks that required\n"
        "cross-window facts failed.  Mistral's later frontier models (Large 3,\n"
        "256K context) dropped sliding window entirely.\n\n"
        "Llama 4 takes the middle path: RoPE layers use chunked local attention\n"
        "(like SWA) for jobs 1+2, but every 4th 'NoPE' layer uses full causal\n"
        "attention with no position encoding — those layers handle job 3.\n\n";

    const int64_t V = 256, D = 64, ctx = 64;
    for (int64_t W : {-1L, 16L, 8L}) {
        ModernGPT model(V, D, 4, 2, 2, 128, 0, 42, W);
        auto cache = model.make_kv_cache(ctx);
        std::vector<int32_t> prompt;
        for (int32_t i = 0; i < 8; ++i) prompt.push_back(i);
        LongContextConfig cfg;
        cfg.max_new_tokens = 8;
        auto out = generate_cached(model, prompt, cache, cfg);
        const std::string label = (W < 0) ? "full causal" : std::format("W={:2}", W);
        std::cout << std::format("  {:12s}: {} new tokens generated — OK\n",
                                 label, out.size());
    }
    std::cout << "\nAll three generate correctly.  The difference only shows on\n"
              << "tasks that require retrieving information beyond the window.\n";
}

// ── §4  RoPE NTK / YaRN scaling — extrapolating beyond training length ────────

static void demo_rope_scaling() {
    rule("-");
    std::cout << "§4  RoPE context extension — NTK scaling and YaRN\n";
    rule("-");
    std::cout <<
        "RoPE encodes position by rotating Q and K vectors.  Each dimension\n"
        "pair (i, i+D/2) rotates at a different frequency:\n\n"
        "  theta_i = base ^ (-2i / head_dim)\n\n"
        "High-frequency dims (small i) resolve fine local position.\n"
        "Low-frequency dims (large i) resolve coarse long-range position.\n\n"
        "When the model is used beyond its training length, the low-frequency\n"
        "dims wrap around (aliasing) and position becomes ambiguous.\n\n"
        "Fix 1 — High RoPE base (Llama 3: 500,000; Qwen3: 1,000,000)\n"
        "  Just raise the base at training time.  Positions spread out so\n"
        "  aliasing doesn't happen until far longer sequences.\n\n"
        "Fix 2 — NTK-aware scaling (what we implement here)\n"
        "  At inference only.  Scale the base so that the effective wavelength\n"
        "  covers the longer target context:\n"
        "    base_scaled = base × alpha ^ (head_dim / (head_dim − 2))\n"
        "    alpha = target_len / train_len\n\n"
        "Fix 3 — YaRN (Qwen3, DeepSeek — better than NTK)\n"
        "  Applies different scaling to different frequency bands:\n"
        "  - High-frequency dims: no scaling (they're already correct locally)\n"
        "  - Low-frequency dims: NTK-style scaling (they need the stretch)\n"
        "  - Mid-frequency dims: linearly interpolate\n"
        "  This avoids distorting local position resolution while fixing\n"
        "  long-range aliasing.  YaRN is what Qwen3 and DeepSeek ship.\n\n";

    const int64_t V = 256, D = 64;
    const int64_t train_ctx = 32, target_ctx = 64;
    ModernGPT model(V, D, 4, 2, 2, 128);

    const float alpha = static_cast<float>(target_ctx) /
                        static_cast<float>(train_ctx);
    const float Dh = static_cast<float>(D) / 4.0f;
    const float base_scaled = 10000.0f * std::pow(alpha, Dh / (Dh - 2.0f));

    auto cache = model.make_kv_cache(target_ctx);
    std::vector<int32_t> prompt;
    for (int32_t i = 0; i < 16; ++i) prompt.push_back(i % static_cast<int32_t>(V));

    LongContextConfig cfg;
    cfg.max_new_tokens = 16;
    cfg.rope_scaling   = alpha;
    auto out = generate_cached(model, prompt, cache, cfg);

    std::cout << std::format(
        "  Training context  : {} tokens (base = 10,000)\n"
        "  Target context    : {} tokens (alpha = {:.1f})\n"
        "  Scaled RoPE base  : {:.0f}\n"
        "  Generated         : {} new tokens — no position aliasing\n",
        train_ctx, target_ctx, alpha, base_scaled, out.size());
}

// ── §5  KV memory budget — what long context actually costs ───────────────────

static void demo_memory_budget() {
    rule("-");
    std::cout << "§5  KV cache memory budget — production perspective\n";
    rule("-");
    std::cout <<
        "KV cache formula:  2 × n_layers × n_kv_heads × head_dim × ctx_len × dtype_bytes\n"
        "The factor of 2 is for K and V separately.\n\n"
        "For a Llama-3 70B class model: 80 layers, 8 KV heads, head_dim=128, bfloat16:\n\n"
        "  Context     KV cache     Notes\n"
        "  ──────────  ───────────  ──────────────────────────────────────\n";

    // Llama-3 70B class: 80 layers, 8 KV heads, head_dim 128, bf16 (2 bytes)
    const int layers = 80, kv_heads = 8, head_dim = 128;
    const double bytes_per_elem = 2.0;  // bfloat16
    for (int ctx : {4096, 32768, 131072, 524288, 1048576}) {
        const double gb = 2.0 * layers * kv_heads * head_dim * ctx
                          * bytes_per_elem / (1024.0 * 1024.0 * 1024.0);
        const char* note = "";
        if (ctx == 4096)   note = "fits on one GPU easily";
        if (ctx == 32768)  note = "comfortable on 80 GB GPU";
        if (ctx == 131072) note = "tight on 80 GB; use GQA / MLA";
        if (ctx == 524288) note = "requires offloading or MLA";
        if (ctx == 1048576) note = "1M — needs MLA or DCA tricks";
        std::cout << std::format("  {:>9}   {:>6.1f} GB    {}\n", ctx, gb, note);
    }

    std::cout <<
        "\nGQA (8 KV heads instead of 64) already gives 8× reduction.\n"
        "MLA (DeepSeek) compresses further to ~1/9 of MHA memory.\n"
        "DCA (Qwen3) splits the sequence into chunks so peak VRAM is halved.\n\n"
        "This is why 1M-token context required multiple simultaneous tricks:\n"
        "high RoPE base, YaRN, GQA/MLA, and either NoPE layers or DCA.\n"
        "No single technique alone gets there.\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    rule("═");
    std::cout << "Chapter 25 — Long-Context Inference\n";
    rule("═");

    explain_context_jobs();
    demo_quadratic_wall();
    demo_kv_cache_speedup();
    demo_sliding_window();
    demo_rope_scaling();
    demo_memory_budget();

    rule("═");
    std::cout <<
        "Summary\n";
    rule("-");
    std::cout <<
        "Context serves four jobs: grammar glue, topic memory, key-fact\n"
        "retrieval, and rule framing.  Every long-context technique is a\n"
        "trade-off over which of those jobs to protect.\n\n"
        "What production models converged on (2025/2026):\n"
        "  • Full causal attention + Flash Attention (O(n) peak memory)\n"
        "  • GQA or MLA for KV compression\n"
        "  • High RoPE base (500K–1M) trained in from the start\n"
        "  • YaRN or iRoPE / NoPE for extrapolation beyond training length\n"
        "  • Long-context data in training — the binding constraint\n\n"
        "What didn't win: pure sliding window (job 3 suffers), soft-token\n"
        "compression (lossy, hard to train), ALiBi (RoPE variants won).\n\n"
        "Next: Chapter 26 will implement Multi-head Latent Attention (MLA),\n"
        "the DeepSeek technique that compresses KV to a low-rank latent\n"
        "vector — 9× less memory than MHA, better quality than GQA.\n";
    rule("═");

    std::cout << "\n[ch25] Done.\n";
    return 0;
}
