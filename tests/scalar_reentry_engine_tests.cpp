// scalar_reentry_engine_tests.cpp -- THE "brain-swap" A/B. A value the classical computer resolved
// re-enters the model either (SCALAR) as ONE bounded (sign,exp,mantissa) vector via SlotEncoding::Scalar,
// or (DIGIT) as the N literal digit tokens it does today. Can the model reason with a re-entered number when
// it arrives as a single scalar? We use a single-number READOUT (no cross-number alignment, so the baseline
// is strong and the test isolates the re-entry channel): the model sees P re-entered, a query token, and must
// output a digit of P:
//   * query 'H' -> P's LEADING digit  -- lives in the Scalar mantissa, so SCALAR should HOLD.
//   * query 'T' -> P's TRAILING digit -- BELOW the mantissa, absent from the Scalar vector, so SCALAR must
//                                         break down to chance while DIGIT (which carries every digit) holds.
// The two queries are the discriminator: they read the same value through the two channels and expose exactly
// what the one-vector re-entry preserves vs loses -- the real answer, from data. Tagged [.scalarreentry].

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/nodes.hpp"          // nd::add -- exact big-decimal sum (P the model cannot compute in-model)
#include "sub0/scratch_slots.hpp"  // SlotEncoding::Scalar + ScratchBindings + set_scratch_bindings

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
constexpr int   kMinDigits = 6, kMaxDigits = 12;          // a,b digit counts -> sum P is 6..13 digits
constexpr float kLr        = 0.003f * (128.0f / static_cast<float>(D_MODEL));
constexpr int   MARK       = ':';                          // separates the value from the query
constexpr int   Q_LEAD     = 'H';                          // "head": read P's leading digit
constexpr int   Q_LAST     = 'T';                          // "tail": read P's trailing digit
constexpr int   SLOT       = sub0::SCRATCH_SLOT_BASE;      // the one scratch slot we bind P into

void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}

std::string gen_int(std::mt19937_64& rng, int digits) {   // no leading zero
    std::uniform_int_distribution<int> d0(1, 9), d(0, 9);
    std::string s(1, static_cast<char>('0' + d0(rng)));
    for (int i = 1; i < digits; ++i) s.push_back(static_cast<char>('0' + d(rng)));
    return s;
}

std::vector<int> digits_of(const std::string& s) {         // "462" -> {'4','6','2'}
    std::vector<int> v; v.reserve(s.size());
    for (char c : s) v.push_back(static_cast<unsigned char>(c));
    return v;
}

struct Task { std::string P; int q = 0, ans = 0; };
Task make_task(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dd(kMinDigits, kMaxDigits);
    const std::string a = gen_int(rng, dd(rng)), b = gen_int(rng, dd(rng));
    Task t; t.P = nd::add(a, b);
    t.q   = (rng() & 1u) ? Q_LEAD : Q_LAST;
    t.ans = (t.q == Q_LEAD) ? static_cast<unsigned char>(t.P.front())
                            : static_cast<unsigned char>(t.P.back());
    return t;
}

// ---- Training pool: tasks laid contiguously, per-task window (start,len) + SCALAR per-task bindings ----
struct Pool {
    std::vector<int>          tok;
    std::vector<std::uint8_t> mask;
    std::vector<std::size_t>  start;
    std::vector<int>          len;
    std::vector<std::vector<std::vector<int>>> tbl;    // SCALAR: per task, a SCRATCH_SLOT_COUNT-wide table
    std::vector<sub0::ScratchBindings>         binds;  // SCALAR: a binding into tbl[i] (slot 0 = P's digits)
};

// mode: true => SCALAR (P is one bound slot token); false => DIGIT (P is its literal digit tokens).
Pool build_pool(std::mt19937_64& rng, int n_tasks, bool scalar) {
    Pool p;
    p.tbl.reserve(static_cast<std::size_t>(n_tasks));      // reserve so binds' spans into tbl[i] stay valid
    p.binds.reserve(static_cast<std::size_t>(n_tasks));
    for (int i = 0; i < n_tasks; ++i) {
        const Task t = make_task(rng);
        const std::size_t s0 = p.tok.size();
        auto emit = [&](int id, std::uint8_t m) { p.tok.push_back(id); p.mask.push_back(m); };

        if (scalar) {
            emit(SLOT, 0);                                 // P re-enters as ONE scalar-embedded slot token
            p.tbl.emplace_back(sub0::SCRATCH_SLOT_COUNT);
            p.tbl.back()[0] = digits_of(t.P);
            p.binds.push_back(sub0::ScratchBindings{
                std::span<const std::vector<int>>(p.tbl.back()), sub0::SlotEncoding::Scalar });
        } else {
            for (int id : digits_of(t.P)) emit(id, 0);     // P re-enters as its literal digits
        }
        emit(MARK, 0);
        emit(t.q, 0);
        emit(t.ans, 1);                                    // GRADED: predict the queried digit
        emit(cas::TOK_EOS, 1);                             // GRADED: and then stop

        p.start.push_back(s0);
        p.len.push_back(static_cast<int>(p.tok.size() - s0));
    }
    return p;
}

void train_arm(const Pool& pool, sub0::AdamW& opt, int steps, std::mt19937& rng, bool scalar) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(pool.start.size()) - 1);
    std::vector<std::size_t>                  starts(kBatch);
    std::vector<int>                          lens(kBatch);
    std::vector<const sub0::ScratchBindings*> wb(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const int idx = pick(rng);
            starts[static_cast<std::size_t>(b)] = pool.start[static_cast<std::size_t>(idx)];
            lens[static_cast<std::size_t>(b)]   = pool.len[static_cast<std::size_t>(idx)] - 1;   // Tb: last tok target-only
            wb[static_cast<std::size_t>(b)]     = scalar ? &pool.binds[static_cast<std::size_t>(idx)] : nullptr;
        }
        opt.zero_grad();
        (void)sub0::train_batch(pool.tok.data(), starts.data(), kBatch, SEQ_LEN - 1, lens.data(),
                                pool.mask.data(), scalar ? wb.data() : nullptr);
        opt.step();
    }
}

// Greedy next-token at the last prompt position IS the model's answer (no decode machinery needed).
int predict(const std::vector<int>& prompt, const sub0::ScratchBindings* binds) {
    if (binds) sub0::set_scratch_bindings(binds);
    sub0::graph_reset();
    sub0::Node* lg = sub0::forward(prompt.data(), static_cast<int>(prompt.size()));
    if (binds) sub0::set_scratch_bindings(nullptr);
    const std::size_t row = (prompt.size() - 1) * static_cast<std::size_t>(VOCAB);
    int best = 0;
    for (int v = 1; v < VOCAB; ++v) if (lg->data[row + v] > lg->data[row + static_cast<std::size_t>(best)]) best = v;
    return best;
}

struct Bucket { int ok = 0, n = 0; double rate() const { return n ? static_cast<double>(ok) / n : 0.0; } };

void eval_arm(std::string& report, const char* name, bool scalar, int n_eval, std::mt19937_64 ev) {
    Bucket lead, last;
    for (int i = 0; i < n_eval; ++i) {
        const Task t = make_task(ev);
        std::vector<int> prompt;
        std::vector<std::vector<int>> tbl;                 // kept alive across predict() for SCALAR
        sub0::ScratchBindings binds;
        if (scalar) {
            prompt.push_back(SLOT);
            tbl.assign(sub0::SCRATCH_SLOT_COUNT, {});
            tbl[0] = digits_of(t.P);
            binds = sub0::ScratchBindings{ std::span<const std::vector<int>>(tbl), sub0::SlotEncoding::Scalar };
        } else {
            for (int id : digits_of(t.P)) prompt.push_back(id);
        }
        prompt.push_back(MARK);
        prompt.push_back(t.q);

        const bool ok = (predict(prompt, scalar ? &binds : nullptr) == t.ans);
        Bucket& bk = (t.q == Q_LEAD) ? lead : last;
        bk.ok += ok; bk.n += 1;
    }
    char line[192];
    std::snprintf(line, sizeof line, "  %-7s | leading-digit=%.3f (n=%d)   trailing-digit=%.3f (n=%d)\n",
                  name, lead.rate(), lead.n, last.rate(), last.n);
    report += line;
}

void train_and_eval(std::string& report, const char* name, bool scalar) {
    std::mt19937_64 dsrng(scalar ? 20281ULL : 90833ULL);
    const Pool pool = build_pool(dsrng, 6000, scalar);

    sub0::build_model();
    reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(scalar ? 11 : 22);

    for (int r = 0; r < 12; ++r) {
        train_arm(pool, opt, 300, rng, scalar);
        report += "  step " + std::to_string((r + 1) * 300) + " ";
        eval_arm(report, name, scalar, 240, std::mt19937_64(7777ULL));
    }
    eval_arm(report, name, scalar, 600, std::mt19937_64(4242ULL));   // final, larger eval
}

}  // namespace

TEST_CASE("scalar re-entry A/B: one scalar vector vs N digit tokens (single-number readout)", "[.scalarreentry]") {
    std::string report = "\n=== scalar re-entry A/B (d" + std::to_string(D_MODEL) +
        ", mantissa=" + std::to_string(sub0::SCALAR_MANT_DIGITS) + " leading digits) ===\n"
        "  Task: read a queried digit of a re-entered value P.  'H' -> leading digit, 'T' -> trailing digit.\n"
        "  Prediction: SCALAR holds on leading (it's in the mantissa) and collapses to chance on trailing;\n"
        "              DIGIT holds on both (it carries every digit).\n";

    report += "-- DIGIT arm (baseline: P as N digit tokens) --\n";
    train_and_eval(report, "DIGIT", /*scalar=*/false);
    report += "-- SCALAR arm (P as ONE bounded scalar vector) --\n";
    train_and_eval(report, "SCALAR", /*scalar=*/true);

    WARN(report);
    REQUIRE(true);   // reporting test: the numbers ARE the result (compare the two arms' two buckets)
}
