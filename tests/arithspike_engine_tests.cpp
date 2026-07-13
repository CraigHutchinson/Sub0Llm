// arithspike_engine_tests.cpp -- the ARITHMETIC-DELEGATION spike (sub0/arithspike.hpp): the crispest test
// of the deterministic-mechanism thesis (docs/DETERMINISTIC_MECHANISMS.md). A/B two ways to answer
// `A + B = ?` over big integers on the SAME tiny model:
//   * DELEGATION -- the model emits a COMPUTE marker; a deterministic add node injects the EXACT sum. It
//     learns only the routing, so it must generalise perfectly to HELD-OUT (never-trained) numbers.
//   * FUZZY -- the model must produce the sum digits itself (internal arithmetic), which should NOT
//     generalise to held-out numbers.
// Held-out is automatic: eval draws FRESH random numbers (a different rng), essentially never seen among the
// 8..14-digit training pairs. Tagged "[.arithspike]" (hidden): trains, so not in the default sweep.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/arithspike.hpp"
#include "sub0/window.hpp"

#include <algorithm>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace as  = sub0::arithspike;
namespace cas = sub0::casing;

constexpr int   kBatch     = 16;
constexpr int   kMinDigits = 8, kMaxDigits = 14;   // big enough that fuzzy internal arithmetic can't generalise
constexpr float kLr        = 0.003f * (128.0f / static_cast<float>(D_MODEL));   // width-scaled (see scratchspike)

// Each TEST_CASE starts from a COLD optimizer -- build_model re-inits weights but the AdamW moment arenas are
// GLOBAL and persist across cases (see scratchspike's reset_opt_state).
void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

void train_steps(const as::Dataset& ds, sub0::AdamW& opt, int steps, std::mt19937& rng, int window) {
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

struct Acc { int ok = 0, n = 0; long dig_ok = 0, dig_n = 0;
    double rate()     const { return n ? static_cast<double>(ok) / n : 0.0; }
    double dig_rate() const { return dig_n ? static_cast<double>(dig_ok) / static_cast<double>(dig_n) : 0.0; } };

// Right-aligned digit-match count between a produced answer and the truth (nuance: fuzzy tends to nail the
// low-order digits and miss the rest).
void tally_digits(Acc& a, const std::string& got, const std::string& want) {
    const int m = static_cast<int>(std::max(got.size(), want.size()));
    for (int k = 0; k < m; ++k) {
        const char g = k < static_cast<int>(got.size())  ? got[got.size()  - 1 - k]  : '0';
        const char w = k < static_cast<int>(want.size()) ? want[want.size() - 1 - k] : '0';
        a.dig_ok += (g == w); ++a.dig_n;
    }
}

// Eval on FRESH random numbers (held-out). Delegation wires the add node into decode; fuzzy does not.
Acc eval(std::mt19937_64& rng, int n_tasks, bool delegate) {
    Acc a;
    std::uniform_int_distribution<int> dd(kMinDigits, kMaxDigits);
    for (int i = 0; i < n_tasks; ++i) {
        const as::Task k = as::make_task(as::gen_int(rng, dd(rng)), as::gen_int(rng, dd(rng)), delegate);
        std::vector<int> ctx = k.prompt;
        std::mt19937 grng(0);
        if (delegate)
            sub0::kv_decode_generate(ctx, /*n=*/32, /*temp=*/1.f, /*topk=*/1, grng, cas::TOK_EOS,
                                     /*use_gpu=*/false, /*on_token=*/{}, /*expand=*/{}, /*combine=*/{},
                                     as::add_node, as::COMPUTE);
        else
            sub0::kv_decode_generate(ctx, /*n=*/32, /*temp=*/1.f, /*topk=*/1, grng, cas::TOK_EOS, /*use_gpu=*/false);
        const std::string got = as::extract_answer(ctx);
        a.ok += (got == k.sum); ++a.n;
        tally_digits(a, got, k.sum);
    }
    return a;
}

}  // namespace

// THE THESIS TEST. Same model, same task, same budget -- only the mechanism differs. Expect delegation to
// clear ~1.0 on held-out (routing generalises; the node is exact) and fuzzy to stay low (internal big-int
// arithmetic doesn't generalise). If so: teach the model to USE the scalar, not to BE one.
TEST_CASE("arithspike: big-number addition -- delegation vs fuzzy (deterministic-node thesis)", "[.arithspike]") {
    const int window = std::min(56, SEQ_LEN - 1);   // max trace ~= 2*maxDigits + sum + markers < 56
    REQUIRE(2 * kMaxDigits + kMaxDigits + 8 <= window);

    auto run = [&](bool delegate, std::string& log) {
        std::mt19937_64 dsrng(delegate ? 111ULL : 222ULL);
        const as::Dataset ds = as::build_dataset(dsrng, /*n_tasks=*/4000, kMinDigits, kMaxDigits, delegate);
        REQUIRE(ds.tokens.size() > static_cast<std::size_t>(window));
        sub0::build_model();
        reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(1);
        double best = 0.0;
        char head[96];
        std::snprintf(head, sizeof head, "\n--- %s (held-out %d-%d digit sums) ---\n",
                      delegate ? "DELEGATION" : "FUZZY", kMinDigits, kMaxDigits);
        log += head;
        for (int r = 0; r < 10; ++r) {
            train_steps(ds, opt, 300, rng, window);
            std::mt19937_64 ev(9999ULL);          // held-out: fresh numbers, distinct rng
            const Acc a = eval(ev, /*n_tasks=*/120, delegate);
            best = std::max(best, a.rate());
            char line[96];
            std::snprintf(line, sizeof line, "  step %5d | HELD-OUT exact=%.3f  per-digit=%.3f\n",
                          (r + 1) * 300, a.rate(), a.dig_rate());
            log += line;
        }
        return best;
    };

    std::string report = "\n=== arithspike A + B over big integers: DELEGATION vs FUZZY (same d" +
                         std::to_string(D_MODEL) + " model) ===\n"
                         "  (delegation: model emits COMPUTE, exact add node injects the sum; fuzzy: model must produce it)\n";
    const double d_best = run(true,  report);
    const double f_best = run(false, report);
    char verdict[160];
    std::snprintf(verdict, sizeof verdict,
                  "\n  => BEST held-out exact-match:  DELEGATION %.3f   vs   FUZZY %.3f\n", d_best, f_best);
    WARN(report + verdict);
    REQUIRE(std::isfinite(d_best));
    REQUIRE(std::isfinite(f_best));
}
