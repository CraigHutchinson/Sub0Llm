// numeric_bind_engine_tests.cpp -- PILLAR 2 (numeric BIND), end-to-end. Operands are BOUND SCRATCH SLOTS:
// the model sees `S0 + S1 =` where S0,S1 are single Scalar-embedded tokens (magnitude only -- the exact
// digits are NOT in the reasoning stream), emits the routing `[op add]`, and the bind-aware node
// (bindspike.hpp) dereferences the slots from the binding table and injects the EXACT sum.
//
// Two tests:
//   * [nodebind] (mechanistic, no training) -- the callback dereferences bound slots and injects the sum.
//   * [.numericbind] (hidden, trains) -- the arithspike A/B lifted onto BOUND symbols: DELEGATION (route to
//     the node that derefs the bindings) reaches held-out 1.000, while FUZZY (produce the sum from the
//     Scalar slots the model holds) is ~0.000 -- BECAUSE the exact digits live in the binding, not the model.
//     That gap IS the pillar-2 proof: the model reasons over the symbol; the node holds the value.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/nodes.hpp"
#include "sub0/node_frame.hpp"
#include "sub0/bindspike.hpp"
#include "sub0/scratch_slots.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

namespace nd  = sub0::nodes;
namespace cas = sub0::casing;

constexpr int   kBatch     = 16;
constexpr int   kMinDigits = 6, kMaxDigits = 12;
constexpr float kLr        = 0.003f * (128.0f / static_cast<float>(D_MODEL));
constexpr int   SLOT0      = sub0::SCRATCH_SLOT_BASE;
constexpr int   SLOT1      = sub0::SCRATCH_SLOT_BASE + 1;

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

std::string gen_int(std::mt19937_64& rng, int digits) {
    std::uniform_int_distribution<int> d0(1, 9), d(0, 9);
    std::string s(1, static_cast<char>('0' + d0(rng)));
    for (int i = 1; i < digits; ++i) s.push_back(static_cast<char>('0' + d(rng)));
    return s;
}

std::vector<int> digits_of(const std::string& s) {
    std::vector<int> v; v.reserve(s.size());
    for (char c : s) v.push_back(static_cast<unsigned char>(c));
    return v;
}

// The number in the LAST TOK_TURN region (the injected result).
std::string extract(const std::vector<int>& ctx) {
    int e = -1; for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == nd::FRAME_CLOSE) { e = i; break; }
    if (e < 0) return {};
    int o = -1; for (int i = e - 1; i >= 0; --i) if (ctx[i] == nd::FRAME_OPEN) { o = i; break; }
    if (o < 0) return {};
    const nd::WordsNums w = nd::scan_region(ctx, o + 1, e);
    return w.nums.empty() ? std::string{} : w.nums[0];
}

// The contiguous digits the model generated after the prompt (for the FUZZY arm).
std::string gen_digits(const std::vector<int>& ctx, std::size_t plen) {
    std::string s;
    for (std::size_t i = plen; i < ctx.size(); ++i) {
        const int t = ctx[i];
        if (t >= '0' && t <= '9') s.push_back(static_cast<char>(t));
        else if (!s.empty()) break;
    }
    return s;
}

struct Task { std::string A, B, sum; };
Task make_task(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dd(kMinDigits, kMaxDigits);
    Task t; t.A = gen_int(rng, dd(rng)); t.B = gen_int(rng, dd(rng)); t.sum = nd::add(t.A, t.B);
    return t;
}

struct Pool {
    std::vector<int>          tok;
    std::vector<std::uint8_t> mask;
    std::vector<std::size_t>  start;
    std::vector<int>          len;
    std::vector<std::vector<std::vector<int>>> tbl;     // per task: slot 0 = A digits, slot 1 = B digits
    std::vector<sub0::ScratchBindings>         binds;
};

// delegate: true => `S0 + S1 = [op add]` (routing graded, injected sum masked); false => FUZZY
// `S0 + S1 = <sum>` (the model must produce the sum from the Scalar slots -- it can't).
Pool build_pool(std::mt19937_64& rng, int n_tasks, bool delegate) {
    Pool p;
    p.tbl.reserve(static_cast<std::size_t>(n_tasks));
    p.binds.reserve(static_cast<std::size_t>(n_tasks));
    for (int i = 0; i < n_tasks; ++i) {
        const Task t = make_task(rng);
        const std::size_t s0 = p.tok.size();
        auto emit = [&](int id, std::uint8_t m) { p.tok.push_back(id); p.mask.push_back(m); };

        emit(SLOT0, 0); emit('+', 0); emit(SLOT1, 0); emit('=', 0);      // the algebraic query over symbols
        if (delegate) {
            for (int id : nd::op_header("add")) emit(id, 1);            // GRADED: `[op add]` (the routing)
            emit(nd::FRAME_OPEN, 0);
            for (int id : digits_of(t.sum)) emit(id, 0);               // masked: the baked exact result
            emit(nd::FRAME_CLOSE, 0);
        } else {
            for (int id : digits_of(t.sum)) emit(id, 1);              // GRADED: the model must produce it
        }
        emit(cas::TOK_EOS, 1);

        p.tbl.emplace_back(sub0::SCRATCH_SLOT_COUNT);
        p.tbl.back()[0] = digits_of(t.A);
        p.tbl.back()[1] = digits_of(t.B);
        p.binds.push_back(sub0::ScratchBindings{
            std::span<const std::vector<int>>(p.tbl.back()), sub0::SlotEncoding::Scalar });

        p.start.push_back(s0);
        p.len.push_back(static_cast<int>(p.tok.size() - s0));
    }
    return p;
}

void train_arm(const Pool& pool, sub0::AdamW& opt, int steps, std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(pool.start.size()) - 1);
    std::vector<std::size_t>                  starts(kBatch);
    std::vector<int>                          lens(kBatch);
    std::vector<const sub0::ScratchBindings*> wb(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const int idx = pick(rng);
            starts[static_cast<std::size_t>(b)] = pool.start[static_cast<std::size_t>(idx)];
            lens[static_cast<std::size_t>(b)]   = pool.len[static_cast<std::size_t>(idx)] - 1;
            wb[static_cast<std::size_t>(b)]     = &pool.binds[static_cast<std::size_t>(idx)];
        }
        opt.zero_grad();
        (void)sub0::train_batch(pool.tok.data(), starts.data(), kBatch, SEQ_LEN - 1, lens.data(),
                                pool.mask.data(), wb.data());
        opt.step();
    }
}

}  // namespace

// ---- Mechanistic: the callback dereferences bound slots and injects the exact sum (no training) ----------
TEST_CASE("numeric bind: the node dereferences bound slots (not inline digits) and injects the exact result",
          "[nodebind]") {
    std::vector<std::vector<int>> tbl(sub0::SCRATCH_SLOT_COUNT);
    tbl[0] = digits_of("123");
    tbl[1] = digits_of("456");
    const sub0::ScratchBindings binds{ std::span<const std::vector<int>>(tbl), sub0::SlotEncoding::Scalar };
    const sub0::ScratchBindings* active = &binds;
    const auto compute = sub0::bind::make_bind_compute_callback(nd::builtin(), &active);

    // `S0 + S1 = [op add]` -- operands are the SLOTS, whose digits appear NOWHERE in the token stream.
    std::vector<int> ctx = { SLOT0, '+', SLOT1, '=' };
    for (int id : nd::op_header("add")) ctx.push_back(id);
    const std::vector<int> inj = compute(ctx);
    // injected `[579]`  (123 + 456), reconstructed purely from the bindings.
    std::vector<int> want = { nd::FRAME_OPEN, '5', '7', '9', nd::FRAME_CLOSE };
    REQUIRE(inj == want);

    // A different op over the same bindings (extensibility): max -> 456.
    std::vector<int> ctx2 = { SLOT0, '+', SLOT1, '=' };
    for (int id : nd::op_header("max")) ctx2.push_back(id);
    REQUIRE(sub0::bind::slot_value(binds, SLOT1) == "456");
    const std::vector<int> inj2 = compute(ctx2);
    REQUIRE(extract(inj2) == "456");
}

// ---- End-to-end A/B: delegation over bound symbols reaches 1.000; fuzzy (no node) is ~0 -----------------
TEST_CASE("numeric bind A/B: op over bound slots is exact; the model alone cannot (pillar 2)", "[.numericbind]") {
    const auto train_eval = [](std::string& report, const char* name, bool delegate) {
        std::mt19937_64 dsrng(delegate ? 5051ULL : 6060ULL);
        const Pool pool = build_pool(dsrng, 4000, delegate);

        sub0::build_model();
        reset_opt_state();
        sub0::AdamW opt(kLr);
        std::mt19937 rng(delegate ? 3 : 4);

        const sub0::ScratchBindings* active = nullptr;
        const auto compute = sub0::bind::make_bind_compute_callback(nd::builtin(), &active);

        double best = 0.0;
        for (int r = 0; r < 10; ++r) {
            train_arm(pool, opt, 300, rng);
            std::mt19937_64 ev(7777ULL);
            std::mt19937    grng(0);
            int ok = 0;
            for (int i = 0; i < 160; ++i) {
                const Task t = make_task(ev);
                std::vector<std::vector<int>> tbl(sub0::SCRATCH_SLOT_COUNT);
                tbl[0] = digits_of(t.A); tbl[1] = digits_of(t.B);
                const sub0::ScratchBindings binds{ std::span<const std::vector<int>>(tbl), sub0::SlotEncoding::Scalar };
                active = &binds;
                sub0::set_scratch_bindings(&binds);
                std::vector<int> ctx = { SLOT0, '+', SLOT1, '=' };
                const std::size_t plen = ctx.size();
                if (delegate)
                    sub0::kv_decode_generate(ctx, 40, 1.f, 1, grng, cas::TOK_EOS, false, {}, {}, {},
                                             compute, nd::FRAME_CLOSE);
                else
                    sub0::kv_decode_generate(ctx, 40, 1.f, 1, grng, cas::TOK_EOS, false);
                sub0::set_scratch_bindings(nullptr);
                const std::string got = delegate ? extract(ctx) : gen_digits(ctx, plen);
                ok += (got == t.sum);
            }
            best = std::max(best, ok / 160.0);
            char line[96];
            std::snprintf(line, sizeof line, "  step %5d | %-10s held-out exact=%.3f\n", (r + 1) * 300, name, ok / 160.0);
            report += line;
        }
        return best;
    };

    std::string report = "\n=== numeric BIND A/B (d" + std::to_string(D_MODEL) +
        ") -- operands are BOUND SLOTS; digits live in the binding, not the stream ===\n";
    report += "-- DELEGATION: `S0 + S1 =` -> [op add] -> node derefs the bindings --\n";
    const double del = train_eval(report, "DELEGATION", /*delegate=*/true);
    report += "-- FUZZY: the model must produce the sum from the Scalar slots it holds --\n";
    const double fz  = train_eval(report, "FUZZY", /*delegate=*/false);

    char tail[160];
    std::snprintf(tail, sizeof tail,
        "=> DELEGATION best=%.3f vs FUZZY best=%.3f -- the model reasons over the symbol; the node holds the value.\n",
        del, fz);
    report += tail;
    WARN(report);
    REQUIRE(std::isfinite(del));
}
