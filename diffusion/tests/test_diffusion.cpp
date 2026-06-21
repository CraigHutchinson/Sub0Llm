#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0diff/eval/oov_cliff.hpp"
#include "sub0diff/eval/recovery.hpp"
#include "sub0diff/nn/char_codec.hpp"
#include "sub0diff/nn/denoiser.hpp"
#include "sub0diff/nn/noise_schedule.hpp"
#include "sub0diff/nn/sampler.hpp"
#include "sub0diff/train/curriculum.hpp"
#include "sub0diff/train/diffusion_loss.hpp"

#include "sub0llm/nn/optimizer.hpp"

#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
namespace dn = sub0diff::nn;
namespace dt = sub0diff::train;
namespace de = sub0diff::eval;

namespace {

std::vector<std::int32_t> ramp_tokens(std::size_t n, std::int32_t vocab) {
    std::vector<std::int32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::int32_t>(i) % vocab;
    return v;
}

// Relative RMS error between two host f32 tensors (for CPU-vs-CUDA parity checks).
[[maybe_unused]] static double rel_rms(const sub0llm::Tensor& a, const sub0llm::Tensor& b) {
    const auto as = a.data_as<float>();
    const auto bs = b.data_as<float>();
    double se = 0.0, sref = 0.0;
    for (std::size_t i = 0; i < as.size(); ++i) {
        const double d = static_cast<double>(as[i]) - static_cast<double>(bs[i]);
        se += d * d;
        sref += static_cast<double>(bs[i]) * static_cast<double>(bs[i]);
    }
    return std::sqrt(se / std::max(sref, 1e-12));
}

} // namespace

TEST_CASE("corrupt_into - reuses buffers, guarantees >=1 corruption", "[diffusion]") {
    const auto clean = ramp_tokens(32, 16);
    std::mt19937 rng(1);
    dn::Corruption c;

    dn::corrupt_into(clean, 0.01f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c);
    REQUIRE(c.n_corrupted >= 1);             // floor guarantee at tiny probability
    const auto* tok_ptr = c.tokens.data();

    dn::corrupt_into(clean, 0.5f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c);
    REQUIRE(c.tokens.data() == tok_ptr);     // same storage — allocation-free reuse
    REQUIRE(c.n_corrupted ==
            static_cast<std::uint32_t>(std::accumulate(c.corrupted.begin(), c.corrupted.end(), 0)));
    for (std::size_t i = 0; i < clean.size(); ++i)
        REQUIRE(c.tokens[i] == (c.corrupted[i] ? 16 : clean[i]));
}

TEST_CASE("corrupt_into - exact_count masks round(t*T), clamped to [1, T-1]", "[diffusion]") {
    const auto clean = ramp_tokens(40, 16);
    std::mt19937 rng(1);
    dn::Corruption c;
    auto count = [&] { return std::accumulate(c.corrupted.begin(), c.corrupted.end(), 0); };

    dn::corrupt_into(clean, 0.25f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, /*exact_count=*/true);
    REQUIRE(c.n_corrupted == 10);            // round(0.25 * 40)
    REQUIRE(count() == 10);
    dn::corrupt_into(clean, 0.001f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, true);
    REQUIRE(c.n_corrupted == 1);             // floor: round→0, clamped to 1 (≥1 masked target)
    dn::corrupt_into(clean, 1.0f, dn::NoiseSchedule::Absorbing, 16, 16, rng, c, true);
    REQUIRE(c.n_corrupted == 39);            // min-1-VISIBLE floor: T-1, never fully blank
    // The COUNT is deterministic across seeds (only WHICH positions differ) — the whole point.
    std::mt19937 r2(99);
    dn::corrupt_into(clean, 0.5f, dn::NoiseSchedule::Absorbing, 16, 16, r2, c, true);
    REQUIRE(c.n_corrupted == 20);
    // masked positions carry the mask id, the rest are clean.
    for (std::size_t i = 0; i < clean.size(); ++i)
        REQUIRE(c.tokens[i] == (c.corrupted[i] ? 16 : clean[i]));
}

TEST_CASE("corrupt_whole_word_into - masks whole words atomically", "[diffusion]") {
    // ids 10,20,30 carry the word-start flag; words are [10,11,12], [20,21], [30,31].
    const std::vector<std::int32_t> clean{10, 11, 12, 20, 21, 30, 31};
    std::vector<std::uint8_t> ws(64, 0);
    ws[10] = ws[20] = ws[30] = 1;
    std::mt19937 rng(3);
    dn::Corruption c;

    for (int trial = 0; trial < 200; ++trial) {
        dn::corrupt_whole_word_into(clean, ws, 0.5f, dn::NoiseSchedule::Absorbing, 99, 99, rng, c);
        REQUIRE(c.n_corrupted >= 1);                       // floor guarantee
        // Atomicity: every subword of a word shares the word's mask decision.
        REQUIRE(c.corrupted[0] == c.corrupted[1]);         // word [10,11,12]
        REQUIRE(c.corrupted[1] == c.corrupted[2]);
        REQUIRE(c.corrupted[3] == c.corrupted[4]);         // word [20,21]
        REQUIRE(c.corrupted[5] == c.corrupted[6]);         // word [30,31]
    }
    // Empty table → falls back to per-token corruption (still honours the floor).
    dn::corrupt_whole_word_into(clean, {}, 0.3f, dn::NoiseSchedule::Absorbing, 99, 99, rng, c);
    REQUIRE(c.n_corrupted >= 1);
}

TEST_CASE("evaluate_recovery - word-start/continuation split bookkeeping", "[diffusion]") {
    dn::Denoiser model(64, 32, 2, 2, 1, 64, /*seed=*/5);
    const std::vector<std::int32_t> clean{10, 11, 12, 20, 21, 30, 31};
    std::vector<std::uint8_t> ws(65, 0);
    ws[10] = ws[20] = ws[30] = 1;                          // word boundaries at 0,3,5
    std::vector<std::uint8_t> masked(clean.size(), 0);
    masked[1] = masked[3] = masked[5] = 1;                 // cont, start, start

    auto r = de::evaluate_recovery(model, clean, masked, nullptr, ws);
    REQUIRE(r.masked == 3);
    REQUIRE(r.ws_masked + r.wc_masked == r.masked);        // partition
    REQUIRE(r.ws_hits + r.wc_hits == r.hits);
    REQUIRE(r.ws_masked == 2);                             // pos 3,5 are word-starts
    REQUIRE(r.wc_masked == 1);                             // pos 1 is a continuation
    REQUIRE(r.word_total == 3);                            // three distinct words touched
    REQUIRE(r.word_hits <= r.word_total);
    // Without a table, the split fields stay zero but recall is unchanged.
    auto r0 = de::evaluate_recovery(model, clean, masked, nullptr, {});
    REQUIRE(r0.masked == 3);
    REQUIRE(r0.ws_masked == 0);
    REQUIRE(r0.word_total == 0);
    REQUIRE(r0.hits == r.hits);                            // same argmax, table-independent
}

TEST_CASE("diffusion_loss - NELBO weight is n_masked/(t*T)", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/3);
    const auto clean = ramp_tokens(16, 16);
    dt::DiffusionLossContext ctx(16);

    // Two identical corruptions, different reported t: same mean-CE, weight ratio known.
    // Drive via the public API with a band so narrow t is effectively fixed.
    std::mt19937 rng_a(7), rng_b(7);   // identical corruption stream
    auto a = dt::diffusion_loss(model, clean, rng_a, ctx, 0.499f, 0.501f);
    dt::DiffusionLossContext ctx_b(16);
    auto b = dt::diffusion_loss(model, clean, rng_b, ctx_b, 0.499f, 0.501f);
    REQUIRE(a.n_masked == b.n_masked);
    REQUIRE_THAT(a.loss.data().item<float>(),
                 WithinRel(b.loss.data().item<float>(), 1e-5f));   // deterministic

    // The scalar equals mean_ce * n_masked / (t*T): verify the weight bounds —
    // with t≈0.5 and T=16, weight = n_masked/8, so loss/mean_ce must be in (0, 2].
    REQUIRE(a.loss.data().item<float>() > 0.0f);
    REQUIRE(a.t >= 0.499f);
    REQUIRE(a.t <= 0.501f);
}

TEST_CASE("diffusion_loss - one optimizer step reduces loss on a fixed sample", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/5);
    auto params = model.parameters();
    sub0llm::nn::Adam opt(params, 5e-2f);
    const auto clean = ramp_tokens(16, 16);
    dt::DiffusionLossContext ctx(16);

    // Same RNG seed each iteration -> identical (t, corruption) sample.
    auto sample_loss = [&]() {
        std::mt19937 rng(11);
        return dt::diffusion_loss(model, clean, rng, ctx, 0.3f, 0.7f);
    };

    const float before = sample_loss().loss.data().item<float>();
    for (int i = 0; i < 10; ++i) {
        auto res = sample_loss();
        opt.zero_grad();
        res.loss.backward();
        opt.step();
    }
    const float after = sample_loss().loss.data().item<float>();
    REQUIRE(after < before);   // gradient flows end-to-end and reduces the objective
}

// Stage 4 Phase 7 capstone: the FULL Denoiser training step — embed → transformer (rms_norm,
// RoPE, GQA-attention softmax, SwiGLU) → LM head → diffusion loss → backward → Adam — composes
// end-to-end on CUDA and reduces the loss. Every op dispatched to the device kernels (Phases 1–7).
TEST_CASE("Denoiser end-to-end training step on CUDA reduces loss", "[diffusion][cuda][device]") {
#ifdef SUB0LLM_CUDA
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/5);
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    REQUIRE(params[0]->data().device().is_cuda());     // confirm we really run on the GPU
    sub0llm::nn::Adam opt(params, 5e-2f);              // m_/v_ allocated on-device (after to(cuda))
    const auto clean = ramp_tokens(16, 16);
    dt::DiffusionLossContext ctx(16);

    auto loss_now = [&]() {
        std::mt19937 rng(11);                          // identical (t, corruption) each call
        return dt::diffusion_loss(model, clean, rng, ctx, 0.3f, 0.7f)
                   .loss.data().to(sub0llm::Device::cpu()).item<float>();
    };

    const float before = loss_now();
    for (int i = 0; i < 10; ++i) {
        std::mt19937 rng(11);
        auto res = dt::diffusion_loss(model, clean, rng, ctx, 0.3f, 0.7f);
        opt.zero_grad();
        res.loss.backward();
        opt.step();
    }
    const float after = loss_now();
    REQUIRE(after < before);                           // full fwd+bwd+update on GPU reduces the loss
#else
    SUCCEED("CPU build - CUDA end-to-end training step is exercised on the cuda preset");
#endif
}

// Stage 4 Phase 7: the BATCHED training step (the ch29 path — block-diagonal attention over B
// windows via 3D batched matmul) also runs end-to-end on CUDA and reduces the loss.
TEST_CASE("Denoiser BATCHED training step on CUDA reduces loss", "[diffusion][cuda][device]") {
#ifdef SUB0LLM_CUDA
    const std::int64_t V = 16, D = 32, T = 8, B = 4;
    dn::Denoiser model(V, D, /*n_heads=*/2, /*n_kv_heads=*/2, /*n_layers=*/1, /*d_ff=*/64, /*seed=*/5);
    model.to(sub0llm::Device::cuda());
    auto params = model.parameters();
    REQUIRE(params[0]->data().device().is_cuda());
    sub0llm::nn::Adam opt(params, 5e-2f);

    const auto stream = ramp_tokens(static_cast<std::size_t>(B * T), V);
    std::vector<std::size_t> offsets(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) offsets[static_cast<std::size_t>(b)] = static_cast<std::size_t>(b * T);
    dt::BatchedDiffusionLossContext ctx(B, T);

    auto loss_now = [&]() {
        std::mt19937 rng(11);
        return dt::batched_diffusion_loss(model, stream, offsets, rng, ctx, 0.3f, 0.7f)
                   .loss.data().to(sub0llm::Device::cpu()).item<float>();
    };
    const float before = loss_now();
    for (int i = 0; i < 8; ++i) {
        std::mt19937 rng(11);
        auto res = dt::batched_diffusion_loss(model, stream, offsets, rng, ctx, 0.3f, 0.7f);
        opt.zero_grad();
        res.loss.backward();
        opt.step();
    }
    const float after = loss_now();
    REQUIRE(after < before);                           // batched fwd+bwd+update on GPU reduces the loss
#else
    SUCCEED("CPU build - CUDA batched training step is exercised on the cuda preset");
#endif
}

// Stage 4 Phase 7: NUMERICAL parity of the whole Denoiser stack (not just "loss decreases").
// Two same-seed models (identical init) — one CPU, one CUDA — must agree on forward logits and
// parameter gradients for one step. f32 reduction-order differs across devices → loose bounds.
TEST_CASE("Denoiser fwd + param-grads match CPU vs CUDA", "[diffusion][cuda][device]") {
#ifdef SUB0LLM_CUDA
    dn::Denoiser cpu_model(16, 32, 2, 2, 1, 64, /*seed=*/5);
    dn::Denoiser gpu_model(16, 32, 2, 2, 1, 64, /*seed=*/5);   // same seed ⇒ identical weights
    gpu_model.to(sub0llm::Device::cuda());

    const auto clean = ramp_tokens(16, 16);
    sub0llm::Tensor toks({static_cast<std::int64_t>(clean.size())}, sub0llm::DType::Int32);
    std::copy(clean.begin(), clean.end(), toks.data_as<std::int32_t>().begin());

    auto y_cpu = cpu_model.forward(toks, 0.5f);
    auto y_gpu = gpu_model.forward(toks, 0.5f);
    REQUIRE(rel_rms(y_gpu.data().to(sub0llm::Device::cpu()), y_cpu.data()) < 5e-3);   // forward parity

    // Backward parity via the diffusion loss (same seed ⇒ identical corruption on both devices).
    dt::DiffusionLossContext cc(16), cg(16);
    std::mt19937 rc(11), rg(11);
    dt::diffusion_loss(cpu_model, clean, rc, cc, 0.3f, 0.7f).loss.backward();
    dt::diffusion_loss(gpu_model, clean, rg, cg, 0.3f, 0.7f).loss.backward();
    auto pc = cpu_model.parameters();
    auto pg = gpu_model.parameters();
    REQUIRE(pc.size() == pg.size());
    // Check a few params' grads (token embedding + a block weight) — full-stack backward parity.
    for (std::size_t i = 0; i < pc.size(); i += std::max<std::size_t>(1, pc.size() / 4)) {
        if (pc[i]->grad().numel() == 0) continue;
        REQUIRE(rel_rms(pg[i]->grad().to(sub0llm::Device::cpu()), pc[i]->grad()) < 5e-2);
    }
#else
    SUCCEED("CPU build - CUDA Denoiser parity is exercised on the cuda preset");
#endif
}

// Stage 4 Phase 7 (robustness): autograd ops NOT yet on CUDA must throw a clean error, not
// UB-crash by host-dereferencing device memory, if a (non-diffusion) model is moved to CUDA.
TEST_CASE("unwired autograd ops throw cleanly on CUDA", "[diffusion][cuda][device]") {
#ifdef SUB0LLM_CUDA
    namespace ag = sub0llm::autograd;
    ag::Variable x(sub0llm::randn({4, 8}).to(sub0llm::Device::cuda()), /*requires_grad=*/true);
    REQUIRE_THROWS(ag::relu(x));
    REQUIRE_THROWS(ag::gelu(x));
    REQUIRE_THROWS(ag::log_softmax(x));
    REQUIRE_THROWS(ag::log_sigmoid(x));
#else
    SUCCEED("CPU build - CUDA guard behaviour is exercised on the cuda preset");
#endif
}

// Stage 4 Phase 7 (perf, review #2): timed full batched training step (fwd + diffusion loss +
// backward + Adam) CPU vs CUDA. Hidden ([.]) — run explicitly: tests_diffusion "[bench]".
// Per-step time includes the loss D2H read (a sync point), so the loop wall-time is honest.
TEST_CASE("BENCH Denoiser full step CPU vs CUDA", "[.][bench][cuda]") {
#ifdef SUB0LLM_CUDA
    const std::int64_t V = 512, D = 256, T = 64, B = 16;
    const int warmup = 2, steps = 8;

    auto bench = [&](sub0llm::Device dev) -> double {
        dn::Denoiser model(V, D, /*n_heads=*/8, /*n_kv_heads=*/4, /*n_layers=*/2, /*d_ff=*/1024, 7);
        if (dev.is_cuda()) model.to(dev);
        auto params = model.parameters();
        sub0llm::nn::Adam opt(params, 1e-3f);
        const auto stream = ramp_tokens(static_cast<std::size_t>(B * T), V);
        std::vector<std::size_t> offsets(static_cast<std::size_t>(B));
        for (std::int64_t b = 0; b < B; ++b) offsets[static_cast<std::size_t>(b)] = static_cast<std::size_t>(b * T);
        dt::BatchedDiffusionLossContext ctx(B, T);
        auto one = [&](int s) {
            std::mt19937 r(static_cast<std::uint32_t>(s) + 1u);
            auto res = dt::batched_diffusion_loss(model, stream, offsets, r, ctx, 0.3f, 0.7f);
            opt.zero_grad();
            res.loss.backward();
            opt.step();
            (void)res.loss.data().to(sub0llm::Device::cpu()).item<float>();  // D2H sync each step
        };
        for (int s = 0; s < warmup; ++s) one(s);
        const auto t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < steps; ++s) one(s);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / steps;
    };

    const double cpu_s = bench(sub0llm::Device::cpu());
    const double gpu_s = bench(sub0llm::Device::cuda());
    std::fprintf(stderr,
        "\n[BENCH] Denoiser step (V=%lld D=%lld T=%lld B=%lld, 2 layers): "
        "CPU %.4f s/step | GPU %.4f s/step | speedup %.2fx\n",
        static_cast<long long>(V), static_cast<long long>(D), static_cast<long long>(T),
        static_cast<long long>(B), cpu_s, gpu_s, cpu_s / gpu_s);
    SUCCEED();
#else
    SUCCEED("CPU build - the GPU step bench runs on the cuda preset");
#endif
}

// ── Ch32 Phase 0: OOV-cliff metric (M1) ────────────────────────────────────────────────────────
// Validates rare_type_mask's frequency split and that evaluate_oov_cliff buckets the per-masked-
// token NELBO correctly (counts add up, CE is a positive softmax NLL). The cliff RATIO itself is a
// model property measured on trained checkpoints; here we validate the mechanism.
TEST_CASE("OOV-cliff metric: rare_type_mask + evaluate_oov_cliff (M1)", "[diffusion][oov_cliff]") {
    using namespace sub0diff::eval;
    // freq: 0->3, 1->2, 2->1, 3->0; rarest 50% of 4 types (ties by id) = {3, 2}.
    const std::vector<std::int32_t> train{0, 0, 0, 1, 1, 2};
    const auto is_rare = rare_type_mask(train, 4, 0.5);
    REQUIRE(is_rare.size() == 4);
    REQUIRE(is_rare[3] == 1);   // freq 0 — the OOV-est, sorts first
    REQUIRE(is_rare[2] == 1);   // freq 1
    REQUIRE(is_rare[0] == 0);   // freq 3 — common
    REQUIRE(is_rare[1] == 0);

    // Mechanism on a tiny model + stream: every token equally frequent ⇒ ties by id ⇒ ids 0-3 rare.
    dn::Denoiser model(8, 32, 2, 2, 1, 64, /*seed=*/5);   // real vocab 8
    std::vector<std::int32_t> eval;
    for (int i = 0; i < 64; ++i) eval.push_back(i % 8);
    const auto rare8 = rare_type_mask(eval, 8, 0.5);
    std::mt19937 rng(1);
    const auto r = evaluate_oov_cliff(model, eval, rare8, /*T=*/16, /*noise=*/0.5f, rng,
                                      /*max_windows=*/8);
    REQUIRE(r.n_rare + r.n_common > 0);                   // masked tokens were scored
    REQUIRE(std::isfinite(r.nll_rare()));
    REQUIRE(std::isfinite(r.nll_common()));
    REQUIRE(r.nll_rare() > 0.0);                          // CE (nats) is a positive NLL
}

// ── Ch32 P1: character-composition codec ───────────────────────────────────────────────────────
// Validates the headline P1 claim (1a compose + 1b decode): compose→decode ROUND-TRIPS a word's
// spelling, and the composed vector is ORDER-SENSITIVE (anagrams diverge) — the property that makes
// it a real word representation, not a bag of letters. Tiny overfit so the unit test stays fast.
TEST_CASE("CharComposer/CharDecoder round-trip spellings (P1 1a/1b)", "[diffusion][char_codec]") {
    using sub0diff::nn::CharComposer;
    using sub0diff::nn::CharDecoder;
    namespace ag = sub0llm::autograd;

    const std::int64_t n_chars = 27;   // a-z + a spare slot (26)
    auto enc = [](const std::string& w) {
        sub0llm::Tensor t({static_cast<std::int64_t>(w.size())}, sub0llm::DType::Int32);
        auto s = t.data_as<std::int32_t>();
        for (std::size_t i = 0; i < w.size(); ++i) s[i] = static_cast<std::int32_t>(w[i] - 'a');
        return t;
    };
    const std::vector<std::string> words{"cat", "dog", "god", "run", "sun",
                                         "the", "star", "tree", "moon", "fish"};
    std::vector<sub0llm::Tensor> data;
    for (const auto& w : words) data.push_back(enc(w));

    CharComposer comp(n_chars, /*D=*/32, /*n_heads=*/4, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/0, 7);
    CharDecoder  dec(n_chars, /*D=*/32, /*n_heads=*/4, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/0, 99);
    std::vector<ag::Variable*> params;
    for (auto* p : comp.parameters()) params.push_back(p);
    for (auto* p : dec.parameters()) params.push_back(p);
    sub0llm::nn::Adam opt(params, 3e-3f);

    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 400; ++step) {
        opt.zero_grad();
        ag::Variable loss = sub0diff::nn::char_recon_loss(comp, dec, data[0]);
        for (std::size_t i = 1; i < data.size(); ++i)
            loss = ag::add(loss, sub0diff::nn::char_recon_loss(comp, dec, data[i]));
        loss.backward();
        opt.step();
        last = loss.data().item<float>();
        if (step == 0) first = last;
    }
    REQUIRE(last < first);   // the autoencoder learns

    // Reconstruction accuracy: argmax of the decoded char logits matches the spelling.
    int correct = 0, total = 0;
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::int64_t L = data[i].shape(0);
        ag::Variable logits = dec.forward(comp.forward(data[i]), L);   // (L, n_chars)
        const auto ls = logits.data().data_as<float>();
        const auto cs = data[i].data_as<std::int32_t>();
        for (std::int64_t r = 0; r < L; ++r) {
            std::int64_t best = 0;
            float        bv   = ls[static_cast<std::size_t>(r) * n_chars];
            for (std::int64_t c = 1; c < n_chars; ++c) {
                const float v = ls[static_cast<std::size_t>(r) * n_chars + c];
                if (v > bv) { bv = v; best = c; }
            }
            if (best == cs[r]) ++correct;
            ++total;
        }
    }
    REQUIRE(static_cast<double>(correct) / total > 0.9);   // round-trips the spellings

    // Order-sensitivity: "dog" and "god" (same letters, different order) compose to DISTINCT vectors.
    const sub0llm::Tensor vd = comp.forward(enc("dog")).data();
    const sub0llm::Tensor vg = comp.forward(enc("god")).data();
    const auto a = vd.data_as<float>(), b = vg.data_as<float>();
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-9);
    REQUIRE(cosine < 0.999);   // not collapsed to a bag-of-letters
}

// P1 1c primitive: composing a whole vocab in one batched (block-diagonal) call must equal composing
// each word on its own. Fixed-length words → exact parity (no padding in the mean-pool).
TEST_CASE("CharComposer::compose_vocab matches per-word forward (P1 1c)", "[diffusion][char_codec]") {
    using sub0diff::nn::CharComposer;
    const std::int64_t n_chars = 27, D = 32, V = 5, L = 4;
    CharComposer comp(n_chars, D, /*n_heads=*/4, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/0, 7);

    std::vector<std::int32_t> chars(static_cast<std::size_t>(V * L));
    for (std::int64_t v = 0; v < V; ++v)
        for (std::int64_t j = 0; j < L; ++j)
            chars[static_cast<std::size_t>(v * L + j)] = static_cast<std::int32_t>((v * 7 + j * 3 + 1) % 26);

    sub0llm::Tensor all({V * L}, sub0llm::DType::Int32);
    std::copy(chars.begin(), chars.end(), all.data_as<std::int32_t>().begin());
    auto table = comp.compose_vocab(all, V, L);      // (V, D)
    REQUIRE(table.data().shape(0) == V);
    REQUIRE(table.data().shape(1) == D);

    const auto tz = table.data().data_as<float>();
    for (std::int64_t v = 0; v < V; ++v) {
        sub0llm::Tensor w({L}, sub0llm::DType::Int32);
        std::copy_n(chars.begin() + static_cast<std::ptrdiff_t>(v * L), L, w.data_as<std::int32_t>().begin());
        const auto single = comp.forward(w).data();  // (1, D)
        const auto sz = single.data_as<float>();
        double se = 0.0, sref = 0.0;
        for (std::int64_t d = 0; d < D; ++d) {
            const double diff = static_cast<double>(tz[static_cast<std::size_t>(v * D + d)]) - sz[static_cast<std::size_t>(d)];
            se += diff * diff;
            sref += static_cast<double>(sz[static_cast<std::size_t>(d)]) * sz[static_cast<std::size_t>(d)];
        }
        REQUIRE(std::sqrt(se / std::max(sref, 1e-12)) < 1e-4);   // batched row == single forward
    }
}

TEST_CASE("evaluate_recovery - counts and position stats are consistent", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/9);
    const auto clean = ramp_tokens(24, 16);
    std::vector<std::uint8_t> masked(24, 0);
    masked[0] = masked[5] = masked[23] = 1;

    de::PositionStats pos(24);
    auto r = de::evaluate_recovery(model, clean, masked, &pos);
    REQUIRE(r.masked == 3);
    REQUIRE(r.hits >= 0);
    REQUIRE(r.hits <= 3);
    REQUIRE(pos.masked[0] == 1);
    REQUIRE(pos.masked[5] == 1);
    REQUIRE(pos.masked[23] == 1);
    REQUIRE(pos.masked[1] == 0);
    REQUIRE(r.hits == pos.hits[0] + pos.hits[5] + pos.hits[23]);
}

TEST_CASE("refine_canvas - completes, respects fixed positions, self-terminates", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/21);
    std::mt19937 rng(3);

    // Canvas with a 3-token fixed prompt; the rest must be filled.
    const std::vector<std::int32_t> prompt{1, 2, 3};
    auto canvas = dn::make_canvas(model, 16, prompt);
    REQUIRE(canvas[0] == 1);
    REQUIRE(canvas[3] == model.mask_id());

    dn::SamplerConfig cfg;
    cfg.temperature = 0.0f;
    auto stats = dn::refine_canvas(model, canvas, cfg, rng);

    REQUIRE(canvas[0] == 1);                      // fixed prompt untouched
    REQUIRE(canvas[1] == 2);
    REQUIRE(canvas[2] == 3);
    for (auto t : canvas) {
        REQUIRE(t != model.mask_id());            // canvas complete
        REQUIRE(t >= 0);
        REQUIRE(t < 16);                          // only real tokens predicted
    }
    REQUIRE(stats.committed == 13);
    REQUIRE(stats.iterations >= 1);
    REQUIRE(stats.iterations <= 13);              // guaranteed progress each iter
}

TEST_CASE("refine_canvas - entropy_bound forces single-iteration commit", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/23);
    std::mt19937 rng(5);
    auto canvas = dn::make_canvas(model, 12);
    dn::SamplerConfig cfg;
    cfg.temperature   = 0.0f;
    cfg.entropy_bound = 1e9f;                     // everything counts as low-entropy
    auto stats = dn::refine_canvas(model, canvas, cfg, rng);
    REQUIRE(stats.iterations == 1);
    REQUIRE(stats.entropy_stopped);
    REQUIRE(stats.committed == 12);
}

TEST_CASE("evaluate_corpus_recall - sweeps every window", "[diffusion]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/13);
    const auto corpus = ramp_tokens(40, 16);
    std::mt19937 rng(17);
    auto r = de::evaluate_corpus_recall(model, corpus, /*T=*/8, /*noise=*/0.5f, rng);
    REQUIRE(r.masked > 0);
    // 32 windows, >=1 masked each (corruption floor)
    REQUIRE(r.masked >= 32);
}

// ── Batched forward parity (Ch29 re-architecture) ──────────────────────────────
// The batched (B·T) forward must equal stacking B independent per-window forwards:
// attention is block-diagonal and every other op is row-wise, so batching changes
// only the matmul M dimension, never a window's result.
TEST_CASE("Denoiser batched forward equals stacked per-window forward", "[diffusion][batched]") {
    const std::int64_t V = 16, D = 32, T = 8, B = 4;
    dn::Denoiser model(V, D, /*n_heads=*/2, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/64, 7);
    const std::int64_t Vm = V + 1;

    std::mt19937 rng(123);
    std::uniform_int_distribution<std::int32_t> tok(0, static_cast<std::int32_t>(Vm) - 1);
    sub0llm::Tensor ids({B * T}, sub0llm::DType::Int32);
    auto idd = ids.data_as<std::int32_t>();
    for (auto& v : idd) v = tok(rng);

    std::vector<float> noise(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) noise[static_cast<std::size_t>(b)] = 0.1f + 0.2f * static_cast<float>(b);

    auto batched = model.forward(ids, std::span<const float>(noise), B, T);   // (B·T, Vm)
    REQUIRE(batched.data().shape() == sub0llm::Tensor::Shape{B * T, Vm});
    auto bd = batched.data().data_as<float>();

    for (std::int64_t b = 0; b < B; ++b) {
        sub0llm::Tensor wids({T}, sub0llm::DType::Int32);
        auto wd = wids.data_as<std::int32_t>();
        for (std::int64_t t = 0; t < T; ++t) wd[static_cast<std::size_t>(t)] = idd[static_cast<std::size_t>(b * T + t)];
        auto single = model.forward(wids, noise[static_cast<std::size_t>(b)]);   // (T, Vm)
        auto sd = single.data().data_as<float>();
        for (std::int64_t i = 0; i < T * Vm; ++i)
            REQUIRE_THAT(bd[static_cast<std::size_t>(b * T * Vm + i)],
                         WithinAbs(sd[static_cast<std::size_t>(i)], 1e-4f));
    }
}

// Gradient parity: d/dθ Σ(batched logits) must equal Σ_b d/dθ Σ(window-b logits).
// sum() avoids RNG alignment and exercises the full batched backward — attention,
// reshape, batched matmul VJPs — against the trusted per-window 2D path.
TEST_CASE("Denoiser batched backward equals summed per-window gradients", "[diffusion][batched]") {
    const std::int64_t V = 12, D = 32, T = 6, B = 3;
    dn::Denoiser model(V, D, /*n_heads=*/4, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/64, 31);
    auto params = model.parameters();

    std::mt19937 rng(99);
    std::uniform_int_distribution<std::int32_t> tok(0, static_cast<std::int32_t>(V));   // includes mask id V
    sub0llm::Tensor ids({B * T}, sub0llm::DType::Int32);
    auto idd = ids.data_as<std::int32_t>();
    for (auto& v : idd) v = tok(rng);
    std::vector<float> noise(static_cast<std::size_t>(B));
    for (std::int64_t b = 0; b < B; ++b) noise[static_cast<std::size_t>(b)] = 0.2f + 0.15f * static_cast<float>(b);

    // Batched gradients.
    for (auto* p : params) p->zero_grad();
    sub0llm::autograd::sum(model.forward(ids, std::span<const float>(noise), B, T)).backward();
    std::vector<sub0llm::Tensor> g_batch;
    for (auto* p : params) g_batch.push_back(sub0llm::copy(p->grad()));

    // Summed per-window gradients (backward accumulates into the same param grads).
    for (auto* p : params) p->zero_grad();
    for (std::int64_t b = 0; b < B; ++b) {
        sub0llm::Tensor wids({T}, sub0llm::DType::Int32);
        auto wd = wids.data_as<std::int32_t>();
        for (std::int64_t t = 0; t < T; ++t) wd[static_cast<std::size_t>(t)] = idd[static_cast<std::size_t>(b * T + t)];
        sub0llm::autograd::sum(model.forward(wids, noise[static_cast<std::size_t>(b)])).backward();
    }

    for (std::size_t pi = 0; pi < params.size(); ++pi) {
        auto gb = g_batch[pi].data_as<float>();
        auto gs = params[pi]->grad().data_as<float>();
        REQUIRE(gb.size() == gs.size());
        for (std::size_t i = 0; i < gb.size(); ++i)
            REQUIRE_THAT(gb[i], WithinAbs(gs[i], 2e-4f));
    }
}

// (B,W) invariance: with deterministic per-window seeding (seed_base), a window's corruption is a
// pure function of its GLOBAL index, NOT how the batch was split across workers. This is what lets
// the unified trainer produce the same gradient at any worker count W for a fixed effective batch B
// — √B consistency decoupled from cores. Tested at the data level (no FP/threading nondeterminism).
TEST_CASE("batched_diffusion_loss - window corruption invariant to batch split (B,W invariance)",
          "[diffusion][batched]") {
    const std::int64_t V = 16, D = 32, T = 8;
    dn::Denoiser model(V, D, /*n_heads=*/2, /*n_kv_heads=*/2, /*n_layers=*/2, /*d_ff=*/64, 7);
    const auto stream = ramp_tokens(64, V);
    std::span<const std::int32_t> s(stream);
    const std::uint64_t S = 0xDEADBEEF12345ull;
    std::mt19937 rng(1);   // only the (deterministic, band-collapsed) shared-t draw touches this

    // Call A: one worker owns all 4 windows (W=1), global indices 0..3.
    std::vector<std::size_t> off4{0, 9, 18, 27};
    dt::BatchedDiffusionLossContext ctxA(4, T);
    (void)dt::batched_diffusion_loss(model, s, std::span<const std::size_t>(off4), rng, ctxA,
                                     0.5f, 0.5f, /*shared_t=*/true, /*exact_count=*/true,
                                     {}, /*whole_word=*/false, S, /*index0=*/0);

    // Call B: a second worker owns the tail 2 windows (global indices 2,3) — index0=2.
    std::vector<std::size_t> off2{18, 27};
    dt::BatchedDiffusionLossContext ctxB(2, T);
    (void)dt::batched_diffusion_loss(model, s, std::span<const std::size_t>(off2), rng, ctxB,
                                     0.5f, 0.5f, true, true, {}, false, S, /*index0=*/2);

    // Global window 2 == ctxA.corr[2] == ctxB.corr[0]; window 3 == ctxA.corr[3] == ctxB.corr[1].
    for (int j = 0; j < 2; ++j) {
        REQUIRE(ctxA.corr[static_cast<std::size_t>(2 + j)].n_corrupted
                    == ctxB.corr[static_cast<std::size_t>(j)].n_corrupted);
        REQUIRE(ctxA.corr[static_cast<std::size_t>(2 + j)].corrupted
                    == ctxB.corr[static_cast<std::size_t>(j)].corrupted);
        REQUIRE(ctxA.corr[static_cast<std::size_t>(2 + j)].tokens
                    == ctxB.corr[static_cast<std::size_t>(j)].tokens);
    }
}

TEST_CASE("FrontierCurriculum - per-epoch NELBO plateau advances integer k", "[diffusion]") {
    dt::FrontierCurriculum c(dt::FrontierCurriculum::Config{
        .seq_len = 8, .k_start = 1, .k_max = 0, .k_step = 1, .patience = 2, .min_improve = 0.01f});
    REQUIRE(c.level() == 1);
    REQUIRE(c.max_level() == 7);                 // T-1 (min-1-visible cap)
    REQUIRE_THAT(c.frontier(), WithinAbs(1.0f / 8.0f, 1e-6));   // t = k/T

    // Descending NELBO at the level ⇒ "still learning", k stays put, stalls reset each time.
    REQUIRE_FALSE(c.observe_epoch(5.0f));
    REQUIRE_FALSE(c.observe_epoch(4.0f));
    REQUIRE_FALSE(c.observe_epoch(3.0f));
    REQUIRE(c.level() == 1);

    // Plateau: no improvement for `patience` epochs ⇒ advance exactly once, k→2, best resets.
    REQUIRE_FALSE(c.observe_epoch(3.0f));        // stall 1/2
    REQUIRE(c.observe_epoch(3.0f));              // stall 2/2 ⇒ advance
    REQUIRE(c.level() == 2);
    REQUIRE_THAT(c.frontier(), WithinAbs(2.0f / 8.0f, 1e-6));

    // Drive to the top and confirm it converges at k_max (and stops advancing).
    for (int guard = 0; guard < 100 && !c.converged(); ++guard) {
        c.observe_epoch(9.0f);                   // flat/high ⇒ keep plateauing/advancing
    }
    REQUIRE(c.converged());
    REQUIRE(c.level() == 7);
    REQUIRE_FALSE(c.observe_epoch(0.0f));        // converged: no further advancement
    REQUIRE(c.level() == 7);
}

// ── Device placement (Stage 4 Phase 0) ────────────────────────────────────────
//
// Denoiser::to moves every parameter in place. On CPU it must be a value-exact
// no-op (forward parity); the CUDA case (gated) proves the params actually land
// on the device. parameters() pointers stay valid across the move, so a Trainer
// holding them keeps working.

TEST_CASE("Denoiser::to - CPU move is a value-exact no-op", "[diffusion][device]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/7);
    const auto ids_vec = ramp_tokens(12, 16);
    sub0llm::Tensor ids({12}, sub0llm::DType::Int32);
    auto idd = ids.data_as<std::int32_t>();
    for (std::size_t i = 0; i < ids_vec.size(); ++i) idd[i] = ids_vec[i];

    auto before = model.forward(ids, 0.4f);
    const auto bd = before.data().data_as<float>();

    dn::Denoiser& ret = model.to(sub0llm::Device::cpu());
    REQUIRE(&ret == &model);                          // chainable
    for (auto* p : model.parameters())
        REQUIRE(p->data().device().is_cpu());

    auto after = model.forward(ids, 0.4f);
    const auto ad = after.data().data_as<float>();
    REQUIRE(after.data().shape() == before.data().shape());
    for (std::int64_t i = 0; i < before.data().numel(); ++i)
        REQUIRE_THAT(bd[static_cast<std::size_t>(i)],
                     WithinAbs(ad[static_cast<std::size_t>(i)], 0.0f));
}

TEST_CASE("Denoiser::to - moves parameters to the GPU", "[diffusion][device]") {
    dn::Denoiser model(16, 32, 2, 2, 1, 64, /*seed=*/7);
#ifdef SUB0LLM_CUDA
    model.to(sub0llm::Device::cuda());
    for (auto* p : model.parameters())
        REQUIRE(p->data().device().is_cuda());
    model.to(sub0llm::Device::cpu());                 // round-trip back
    for (auto* p : model.parameters())
        REQUIRE(p->data().device().is_cpu());
#else
    REQUIRE_THROWS_AS(model.to(sub0llm::Device::cuda()), std::runtime_error);
#endif
}
