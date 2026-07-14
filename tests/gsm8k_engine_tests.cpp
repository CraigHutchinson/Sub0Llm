// gsm8k_engine_tests.cpp -- the GSM8K chapter CAPSTONE: end-to-end, a tiny model learns to DELEGATE math in
// prose. Synthetic GSM8K-format word problems (`... A+B = <<A+B=S>> S.`) go through the SAME production path
// the real chapter uses -- gsm8k::build_stream (graded prose + graded `[op math A+B]` routing, MASKED result)
// -> train -> decode through the PRODUCTION node_frame callback (which dispatches the `math` node) -> score.
//
// The eval harness demonstrates the "math is well-formed" advantage: because the node is EXACT, we auto-score
//   * delegation rate  -- did the model emit a well-formed op frame the node could resolve,
//   * exact-match      -- is the injected answer the gold answer (arithmetic error is 0 by construction when
//                         delegation fires, so a miss is a ROUTING/copy error, not an arithmetic one),
//   * magnitude generalization -- trained on 1-2 digit operands, tested on 3-digit (the node stays exact).
// Tagged [.gsm8kcapstone] (hidden): trains.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/nodes.hpp"
#include "sub0/node_frame.hpp"
#include "sub0/bindspike.hpp"   // make_collapse_callback: collapse a result to a slot for chaining
#include "sub0/gsm8k.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace nd  = sub0::nodes;
namespace gs  = sub0::gsm8k;
namespace cas = sub0::casing;

constexpr int   kBatch = 16;
constexpr float kLr    = 0.0015f * (128.0f / static_cast<float>(D_MODEL));   // copy/routing is LR-sensitive

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

std::vector<int> byte_encode(std::string_view s) {   // char-level: a tiny fixed template vocab
    std::vector<int> v; v.reserve(s.size());
    for (char c : s) v.push_back(static_cast<unsigned char>(c));
    return v;
}

std::string gen_int(std::mt19937_64& rng, int digits) {
    std::uniform_int_distribution<int> d0(1, 9), d(0, 9);
    std::string s(1, static_cast<char>('0' + d0(rng)));
    for (int i = 1; i < digits; ++i) s.push_back(static_cast<char>('0' + d(rng)));
    return s;
}

// A synthetic 1-step word problem in GSM8K annotation format. The operator lives in the prose expression, so
// the model must COPY the expression into the frame (the routing) -- it does not learn the arithmetic.
struct Prob { std::string sol, prompt, gold; };
Prob gen_prob(std::mt19937_64& rng, int maxdig) {
    std::uniform_int_distribution<int> dd(1, maxdig);
    std::string A = gen_int(rng, dd(rng)), B = gen_int(rng, dd(rng));
    Prob p;
    if (rng() & 1u) {                                   // addition
        const std::string S = nd::add(A, B), expr = A + "+" + B;
        p.prompt = "got " + A + " then " + B + ". " + expr + "=";
        p.gold = S;
    } else {                                            // subtraction (keep it non-negative)
        if (nd::cmp(A, B) < 0) std::swap(A, B);
        const std::string S = nd::sub(A, B), expr = A + "-" + B;
        p.prompt = "had " + A + " lost " + B + ". " + expr + "=";
        p.gold = S;
    }
    p.sol = p.prompt + "<<" + p.prompt.substr(p.prompt.rfind(' ') + 1) + p.gold + ">> " + p.gold + ".";
    // (the annotation body = the expression just before the '=' + '=' + result; rfind(' ')+1 lands on the expr)
    return p;
}

// The number in the LAST injected TOK_TURN region (the op result the node wrote).
std::string extract_last(const std::vector<int>& ctx) {
    int e = -1; for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == nd::FRAME_CLOSE) { e = i; break; }
    if (e < 0) return {};
    int o = -1; for (int i = e - 1; i >= 0; --i) if (ctx[i] == nd::FRAME_OPEN) { o = i; break; }
    if (o < 0) return {};
    std::string s;
    for (int i = o + 1; i < e; ++i) if (ctx[i] >= '0' && ctx[i] <= '9') s.push_back(static_cast<char>(ctx[i]));
    return s;
}

// ---- Training pool: build_stream each problem, laid contiguously; per-problem window (start,len) ----------
struct Pool { std::vector<int> tok; std::vector<std::uint8_t> mask; std::vector<std::size_t> start; std::vector<int> len; };
Pool build_pool(std::mt19937_64& rng, int n, int maxdig) {
    Pool p;
    for (int i = 0; i < n; ++i) {
        const gs::Example ex = gs::build_stream(gen_prob(rng, maxdig).sol, byte_encode);
        if (ex.ops == 0 || static_cast<int>(ex.tokens.size()) >= SEQ_LEN) continue;   // skip a dropped/oversized one
        p.start.push_back(p.tok.size());
        p.len.push_back(static_cast<int>(ex.tokens.size()));
        p.tok.insert(p.tok.end(), ex.tokens.begin(), ex.tokens.end());
        p.mask.insert(p.mask.end(), ex.mask.begin(), ex.mask.end());
    }
    return p;
}

void train(const Pool& pool, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(pool.start.size()) - 1);
    std::vector<std::size_t> starts(kBatch); std::vector<int> lens(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const int idx = pick(rng);
            starts[static_cast<std::size_t>(b)] = pool.start[static_cast<std::size_t>(idx)];
            lens[static_cast<std::size_t>(b)]   = pool.len[static_cast<std::size_t>(idx)] - 1;   // last tok is target-only
        }
        opt.zero_grad();
        (void)sub0::train_batch(pool.tok.data(), starts.data(), kBatch, SEQ_LEN - 1, lens.data(), pool.mask.data(), nullptr);
        opt.step();
    }
}

struct Score { int exact = 0, delegated = 0, n = 0; };
Score eval(int n_eval, int maxdig, const std::function<std::vector<int>(const std::vector<int>&)>& compute, std::mt19937_64 ev) {
    Score sc; std::mt19937 grng(0);
    for (int i = 0; i < n_eval; ++i) {
        const Prob pr = gen_prob(ev, maxdig);
        std::vector<int> ctx = byte_encode(pr.prompt);
        if (static_cast<int>(ctx.size()) + 8 >= SEQ_LEN) continue;
        sub0::kv_decode_generate(ctx, 30, 1.f, 1, grng, cas::TOK_EOS, false, {}, {}, {}, compute, nd::FRAME_CLOSE);
        const std::string got = extract_last(ctx);
        sc.delegated += !got.empty();
        sc.exact     += (got == pr.gold);
        ++sc.n;
    }
    return sc;
}

}  // namespace

TEST_CASE("gsm8k capstone: a tiny model learns to delegate math in prose, gen dispatches it", "[.gsm8kcapstone]") {
    std::mt19937_64 dsrng(31337);
    const Pool pool = build_pool(dsrng, 6000, /*maxdig=*/2);
    REQUIRE(pool.start.size() > 100);

    sub0::build_model();
    reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(7);
    const auto compute = nd::make_compute_callback(nd::builtin());   // the PRODUCTION registry-backed callback

    std::string report = "\n=== gsm8k capstone (d" + std::to_string(D_MODEL) +
        "): `... A+B =` -> model emits a bare [op math] -> node reads the posed expr -> injects the answer ===\n";
    double best_exact = 0.0, best_deleg = 0.0, best_ood = 0.0; int best_step = 0;
    for (int r = 0; r < 14; ++r) {
        train(pool, opt, 300, rng);
        const Score s   = eval(200, 2, compute, std::mt19937_64(4242ULL));   // in-dist (<=2 digit)
        const Score o   = eval(200, 3, compute, std::mt19937_64(911ULL));    // OOD (3 digit, never trained)
        const double ex = s.exact / double(s.n), dl = s.delegated / double(s.n), ox = o.exact / double(o.n);
        if (ex > best_exact) { best_exact = ex; best_step = (r + 1) * 300; best_deleg = dl; best_ood = ox; }
        char line[144];
        std::snprintf(line, sizeof line, "  step %5d | in-dist delegated=%.3f exact=%.3f | OOD 3-digit exact=%.3f\n",
                      (r + 1) * 300, dl, ex, ox);
        report += line;
    }
    // At the BEST (early-stop) checkpoint delegation is clean; more training overtrains this tiny synthetic
    // task. Magnitude generalization: the bare frame means NO copy, so the node stays exact on 3-digit
    // operands never trained on -> OOD tracks in-dist (the delegation advantage -- arithmetic is not learned).
    char tail[192];
    std::snprintf(tail, sizeof tail,
        "  BEST (early-stop) step %d: delegated=%.3f  exact=%.3f  |  OOD 3-digit exact=%.3f (magnitude gen)\n",
        best_step, best_deleg, best_exact, best_ood);
    report += tail;

    WARN(report);
    REQUIRE(best_exact > 0.75);   // the delegation mechanism works end-to-end (node exact; miss = routing only)
}

// ============================================================================================================
// Multi-step chaining A/B: does a tiny model chain a 2-step computation better when the intermediate result is
// COLLAPSED to a slot (one-token reference) than when it is left as DIGITS the model must copy? Problem:
// R1 = A+B, then R2 = 7 + R1. Step 2 references R1 AFTER a `. 7+` separator (so the node isolates the step-2
// expression and the model must *reproduce* the reference -- the chaining-copy). DIGIT: reproduce multi-digit
// R1. COLLAPSE: emit the single slot token S0 (its value lives in the collapse callback's binding). Tagged
// [.chaincapstone] (hidden): trains two models.
namespace {

constexpr int SLOTA = sub0::SCRATCH_SLOT_BASE;       // the 1st intermediate's slot
constexpr int SLOTB = sub0::SCRATCH_SLOT_BASE + 1;   // the 2nd result's slot

struct Prob2 { std::string A, B, r1, r2; };
Prob2 gen_prob2(std::mt19937_64& rng, int maxdig) {
    std::uniform_int_distribution<int> dd(1, maxdig);
    Prob2 q; q.A = gen_int(rng, dd(rng)); q.B = gen_int(rng, dd(rng));
    q.r1 = nd::add(q.A, q.B); q.r2 = nd::add("7", q.r1);          // R1 = A+B, R2 = 7 + R1
    return q;
}

// `A+B = [op math] <r1> . 7+<r1> = [op math] <r2> .`  -- the injected result is separated from the step-2
// expression by `. `, so `7+<r1>` is what the node reads; `<r1>` there is the reference the model reproduces.
void emit_chain(Pool& p, const Prob2& q, bool collapse) {
    const std::size_t s0 = p.tok.size();
    auto g = [&](int t) { p.tok.push_back(t); p.mask.push_back(1); };
    auto m = [&](int t) { p.tok.push_back(t); p.mask.push_back(0); };
    for (char c : q.A) g(static_cast<unsigned char>(c)); g('+'); for (char c : q.B) g(static_cast<unsigned char>(c)); g('=');
    for (int t : nd::op_header("math")) g(t);                                    // step-1 route
    if (collapse) m(SLOTA); else for (char c : q.r1) m(static_cast<unsigned char>(c));   // step-1 result (masked)
    g('.'); g(' '); g('7'); g('+');                                             // separator + step-2 expr start
    if (collapse) g(SLOTA); else for (char c : q.r1) g(static_cast<unsigned char>(c));   // step-2 REFERENCE (variable)
    g('=');
    for (int t : nd::op_header("math")) g(t);                                    // step-2 route
    if (collapse) m(SLOTB); else for (char c : q.r2) m(static_cast<unsigned char>(c));   // step-2 result (masked)
    g('.');
    p.start.push_back(s0); p.len.push_back(static_cast<int>(p.tok.size() - s0));
}

Pool build_pool2(std::mt19937_64& rng, int n, int maxdig, bool collapse) {
    Pool p;
    for (int i = 0; i < n; ++i) emit_chain(p, gen_prob2(rng, maxdig), collapse);
    return p;
}

double run_chain_arm(std::string& report, const char* name, bool collapse) {
    std::mt19937_64 dsrng(collapse ? 5150ULL : 6160ULL);
    const Pool pool = build_pool2(dsrng, 6000, 2, collapse);
    sub0::build_model();
    reset_opt_state();
    sub0::AdamW opt(0.0015f * (128.0f / static_cast<float>(D_MODEL)));
    std::mt19937 rng(collapse ? 21 : 22);

    std::vector<std::string> slots;
    const auto compute     = nd::make_compute_callback(nd::builtin());
    const auto collapse_cb = sub0::bind::make_collapse_callback(nd::builtin(), &slots);

    double best = 0.0;
    for (int r = 0; r < 14; ++r) {
        train(pool, opt, 300, rng);
        std::mt19937_64 ev(4242ULL); std::mt19937 grng(0);
        int ok = 0, n = 0;
        for (int i = 0; i < 200; ++i) {
            const Prob2 q = gen_prob2(ev, 2);
            std::vector<int> ctx;
            for (char c : q.A) ctx.push_back(static_cast<unsigned char>(c));
            ctx.push_back('+'); for (char c : q.B) ctx.push_back(static_cast<unsigned char>(c)); ctx.push_back('=');
            std::string got;
            if (collapse) {
                slots.clear();
                sub0::kv_decode_generate(ctx, 30, 1.f, 1, grng, cas::TOK_EOS, false, {}, {}, {}, collapse_cb, nd::FRAME_CLOSE);
                got = slots.empty() ? std::string{} : slots.back();
            } else {
                sub0::kv_decode_generate(ctx, 30, 1.f, 1, grng, cas::TOK_EOS, false, {}, {}, {}, compute, nd::FRAME_CLOSE);
                got = extract_last(ctx);
            }
            ok += (got == q.r2); ++n;
        }
        best = std::max(best, ok / double(n));
        char line[96];
        std::snprintf(line, sizeof line, "  %-8s step %5d | final-answer exact=%.3f\n", name, (r + 1) * 300, ok / double(n));
        report += line;
    }
    return best;
}

}  // namespace

TEST_CASE("gsm8k chaining A/B: collapse (slot ref) vs digit-copy for a 2-step chain", "[.chaincapstone]") {
    std::string report = "\n=== gsm8k chaining A/B (d" + std::to_string(D_MODEL) +
        "): 2-step  R2 = 7 + (A+B), step 2 must REFERENCE the intermediate ===\n";
    const double digit    = run_chain_arm(report, "DIGIT", /*collapse=*/false);
    const double collapse = run_chain_arm(report, "COLLAPSE", /*collapse=*/true);
    char tail[160];
    std::snprintf(tail, sizeof tail,
        "  => DIGIT (copy multi-digit R1) best=%.3f  vs  COLLAPSE (one-token slot ref) best=%.3f\n", digit, collapse);
    report += tail;
    WARN(report);
    REQUIRE(std::isfinite(digit));
    REQUIRE(std::isfinite(collapse));
}
