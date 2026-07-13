// chainspike_engine_tests.cpp -- the CHAINED / DYNAMIC-TURN delegation spike (sub0/chainspike.hpp): does
// delegation COMPOSE over a DYNAMIC number of node calls, and does the model terminate correctly? Train on
// list-sums of length k in [2,6]; the model reduces `n1+...+nk` one `+` per node call and stops when a
// single value remains. Eval per-length, INCLUDING held-out lengths (7,8) -- because the per-step decision
// is local (reduce again iff the injected state still holds a `+`), correct iteration should length-
// generalise. Tagged "[.chainspike]" (hidden). Needs a wide context (states are re-emitted) -> run on seq128.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/chainspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ch  = sub0::chainspike;
namespace cas = sub0::casing;

constexpr int   kBatch    = 16;
constexpr int   kDigits   = 2;                 // small numbers: the test is DYNAMIC ITERATION, not big-int
constexpr int   kTrainMin = 2, kTrainMax = 6;  // trained list lengths
constexpr int   kTestMax  = 8;                 // eval up to here -> 7,8 are HELD-OUT lengths
constexpr float kLr       = 0.003f * (128.0f / static_cast<float>(D_MODEL));

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

void train_steps(const ch::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng, int window) {
    std::vector<std::size_t> starts(kBatch);
    std::vector<int>         lens(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, window, ds.tokens.size(),
                                                       std::span<const std::uint64_t>(ds.doc_starts));
            starts[static_cast<std::size_t>(b)] = w.start;
            lens[static_cast<std::size_t>(b)]   = w.len;
        }
        opt.zero_grad();
        (void)sub0::train_batch(ds.tokens.data(), starts.data(), kBatch, window, lens.data(), ds.mask.data(), nullptr);
        opt.step();
    }
}

struct Acc { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };

// Held-out eval at a fixed list length k: fresh random numbers; the model drives a dynamic number of reduce
// calls; success = the final reduced value equals the true sum.
Acc eval_k(std::mt19937_64& rng, int k, int n_tasks) {
    Acc a;
    const auto reduce = [](const std::vector<int>& c) { return ch::reduce_node(c); };
    for (int i = 0; i < n_tasks; ++i) {
        const ch::Task t = ch::make_task(ch::gen_list(rng, k, kDigits, kDigits));
        std::vector<int> ctx = t.prompt;
        std::mt19937 grng(0);
        sub0::kv_decode_generate(ctx, /*n=*/48, /*temp=*/1.f, /*topk=*/1, grng, cas::TOK_EOS,
                                 /*use_gpu=*/false, /*on_token=*/{}, /*expand=*/{}, /*combine=*/{},
                                 reduce, ch::EXEC);
        a.ok += (ch::extract(ctx) == t.sum); ++a.n;
    }
    return a;
}

}  // namespace

// Does dynamic-turn delegation work and length-generalise? Train k in [2,6]; eval each length 2..8.
TEST_CASE("chainspike: dynamic-turn chained reduction (variable length, held-out lengths)", "[.chainspike]") {
    const int window = SEQ_LEN - 1;
    REQUIRE(window >= 96);   // re-emitted states are O(k^2); run this on seq128 (a short-context build skips)

    std::mt19937_64 dsrng(2027);
    const ch::Dataset ds = ch::build_dataset(dsrng, /*n_tasks=*/5000, kTrainMin, kTrainMax, kDigits, kDigits);
    REQUIRE(ds.tokens.size() > static_cast<std::size_t>(window));

    sub0::build_model();
    reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    std::string report = "\n=== chainspike: reduce n1+...+nk via a DYNAMIC number of node calls (d" +
                         std::to_string(D_MODEL) + ", train k=" + std::to_string(kTrainMin) + ".." +
                         std::to_string(kTrainMax) + ") ===\n"
                         "  (held-out per length; 7,8 are UNSEEN lengths -- does the dynamic loop generalise?)\n";
    {   // trace length per k vs SEQ_LEN -- an O(k^2) re-emit means longer lists eventually overflow the context
        std::mt19937_64 dr(1);
        report += "  sample trace length: ";
        for (int k = 2; k <= kTestMax; ++k) {
            const std::size_t len = ch::make_task(ch::gen_list(dr, k, kDigits, kDigits)).trace.size();
            report += "k" + std::to_string(k) + "=" + std::to_string(len) + (len >= static_cast<std::size_t>(SEQ_LEN) ? "!" : "") + " ";
        }
        report += "(SEQ_LEN=" + std::to_string(SEQ_LEN) + "; ! = overflows)\n";
    }
    for (int r = 0; r < 12; ++r) {
        train_steps(ds, opt, 300, rng, window);
        std::mt19937_64 ev(4242ULL);
        std::string line = "  step " + std::to_string((r + 1) * 300) + " |";
        for (int k = 2; k <= kTestMax; ++k) {
            char c[24];
            std::snprintf(c, sizeof c, " k%d=%.2f%s", k, eval_k(ev, k, 40).rate(), k > kTrainMax ? "*" : "");
            line += c;
        }
        report += line + "\n";
    }
    report += "  (* = held-out length, never trained)\n";
    WARN(report);
    REQUIRE(true);
}
