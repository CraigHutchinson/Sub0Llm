// nodespike_engine_tests.cpp -- the REGION-FRAME + NODE-REGISTRY spike (sub0/nodespike.hpp): stage 1 of the
// deterministic-mechanism plan (docs/DETERMINISTIC_MECHANISMS.md). Proves, with data, the design claim that
// naming compute ops in ORDINARY WORD TOKENS (zero tokenizer budget) routes as accurately as DEDICATED
// per-op marker tokens (one reserved id each). If WORD ~= TOKEN, the frame design is strictly better.
//
// One model learns to route `A <op-symbol> B =` to `{ <op-name> }`, and a registry dispatches on the emitted
// name over FOUR ops (add/sub/max/min -- arithmetic AND comparison), injecting the exact result. Held-out =
// fresh random numbers. Tagged "[.nodespike]" (hidden): trains.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/nodespike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <array>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ns  = sub0::nodespike;
namespace cas = sub0::casing;

constexpr int   kBatch     = 16;
constexpr int   kMinDigits = 6, kMaxDigits = 12;
constexpr float kLr        = 0.003f * (128.0f / static_cast<float>(D_MODEL));

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

void train_steps(const ns::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng, int window) {
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
struct OpAcc { std::array<Acc, 4> per; Acc all; };

// Held-out eval: fresh random operands per op; the model generates the frame, the registry injects the
// result; success = the injected digits equal the true answer (so a MISROUTE -> wrong op -> wrong digits).
OpAcc eval(std::mt19937_64& rng, int n_per_op, bool word) {
    OpAcc acc;
    for (int oi = 0; oi < static_cast<int>(ns::ops().size()); ++oi) {
        const ns::Op& op = ns::ops()[static_cast<std::size_t>(oi)];
        for (int i = 0; i < n_per_op; ++i) {
            std::string a, b;
            ns::gen_operands(rng, op, kMinDigits, kMaxDigits, a, b);
            const ns::Task k = ns::make_task(op, a, b, word);
            std::vector<int> ctx = k.prompt;
            std::mt19937 grng(0);
            const auto comp = [word](const std::vector<int>& c) { return ns::dispatch(c, word); };
            sub0::kv_decode_generate(ctx, /*n=*/40, /*temp=*/1.f, /*topk=*/1, grng, cas::TOK_EOS,
                                     /*use_gpu=*/false, /*on_token=*/{}, /*expand=*/{}, /*combine=*/{},
                                     comp, ns::EXEC);
            const bool ok = (ns::extract_result(ctx) == k.sum);
            acc.per[static_cast<std::size_t>(oi)].ok += ok; ++acc.per[static_cast<std::size_t>(oi)].n;
            acc.all.ok += ok; ++acc.all.n;
        }
    }
    return acc;
}

}  // namespace

// THE A/B: one region frame, ops named by WORD tokens (free) vs a DEDICATED per-op token (a reserved id
// each). Same model/task/budget. Expect both to clear ~1.0 held-out (routing generalises; nodes are exact),
// so word-naming carries no penalty -> free, unbounded extensibility. Also a within-arm check that adding
// ops (4 here, incl. comparison) all route correctly.
TEST_CASE("nodespike: region-frame node registry -- WORD op-names vs DEDICATED tokens (A/B)", "[.nodespike]") {
    const int window = std::min(52, SEQ_LEN - 1);   // max trace ~= 2*maxDigits + result + frame < 52
    REQUIRE(3 * kMaxDigits + 12 <= window);

    auto run = [&](bool word, std::string& log) {
        std::mt19937_64 dsrng(word ? 11ULL : 22ULL);
        const ns::Dataset ds = ns::build_dataset(dsrng, /*n_tasks=*/4000, kMinDigits, kMaxDigits, word);
        REQUIRE(ds.tokens.size() > static_cast<std::size_t>(window));
        sub0::build_model();
        reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        char head[80];
        std::snprintf(head, sizeof head, "\n--- %s op-names ---\n", word ? "WORD (free)" : "DEDICATED-TOKEN");
        log += head;
        double best = 0.0;
        for (int r = 0; r < 10; ++r) {
            train_steps(ds, opt, 300, rng, window);
            std::mt19937_64 ev(9999ULL);
            const OpAcc a = eval(ev, /*n_per_op=*/40, word);
            best = std::max(best, a.all.rate());
            char line[160];
            std::snprintf(line, sizeof line,
                          "  step %5d | HELD-OUT overall=%.3f  [add %.2f sub %.2f max %.2f min %.2f]\n",
                          (r + 1) * 300, a.all.rate(),
                          a.per[0].rate(), a.per[1].rate(), a.per[2].rate(), a.per[3].rate());
            log += line;
        }
        return best;
    };

    std::string report = "\n=== nodespike: one region frame, 4 ops (add/sub/max/min) -- WORD vs TOKEN op-names (d" +
                         std::to_string(D_MODEL) + ") ===\n"
                         "  (does naming ops in ORDINARY WORDS -- zero token cost -- route as well as a dedicated token per op?)\n";
    const double w_best = run(true,  report);
    const double t_best = run(false, report);
    char verdict[160];
    std::snprintf(verdict, sizeof verdict,
                  "\n  => BEST held-out overall:  WORD %.3f   vs   DEDICATED-TOKEN %.3f\n", w_best, t_best);
    WARN(report + verdict);
    REQUIRE(std::isfinite(w_best));
    REQUIRE(std::isfinite(t_best));
}
