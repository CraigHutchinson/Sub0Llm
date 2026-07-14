// op_curriculum_tests.cpp -- the production op-delegation curriculum (sub0/op_curriculum.hpp), engine-free:
// structure of the masked dataset, the FILTER guarantee (a single-step frame dispatched through the
// production callback reproduces its masked result), and the collapse-chain layout.

#include <catch2/catch_test_macros.hpp>

#include "sub0/op_curriculum.hpp"
#include "sub0/tokenizer.hpp"
#include "sub0/node_frame.hpp"
#include "sub0/scratch_slots.hpp"
#include "sub0/casing.hpp"

#include <string>
#include <vector>

namespace oc  = sub0::op_curriculum;
namespace nd  = sub0::nodes;
namespace cas = sub0::casing;

namespace {
// A tokenizer that has learned the curriculum's prose words (digits/operators stay byte tokens regardless).
sub0::tok::Tokenizer make_tok() {
    return sub0::tok::learn("total so then we get count the answer is sum plus minus times over and more");
}
// The masked digits inside doc [lo,hi) -- the delegated results.
std::string masked_digits(const oc::Dataset& ds, std::size_t lo, std::size_t hi) {
    std::string s;
    for (std::size_t i = lo; i < hi; ++i)
        if (ds.mask[i] == 0 && ds.tokens[i] >= '0' && ds.tokens[i] <= '9') s.push_back(static_cast<char>(ds.tokens[i]));
    return s;
}
}  // namespace

TEST_CASE("op_curriculum: dataset structure is well-formed", "[opcurric]") {
    const auto tk = make_tok();
    oc::Options o; o.seed = 1; o.n_examples = 400; o.chain_frac = 0.4; o.max_digits = 3;
    const oc::Dataset ds = oc::build_dataset(tk, o);

    REQUIRE(ds.tokens.size() == ds.mask.size());
    REQUIRE(ds.doc_starts.front() == 0);
    REQUIRE(ds.doc_starts.back() == ds.tokens.size());
    REQUIRE(ds.doc_starts.size() > 100);                 // most examples produced a document
    for (std::size_t i = 1; i < ds.doc_starts.size(); ++i) REQUIRE(ds.doc_starts[i] > ds.doc_starts[i - 1]);

    int frames = 0, masked = 0;
    for (int t : ds.tokens) if (t == cas::TOK_TURN_START) ++frames;
    for (std::uint8_t m : ds.mask) if (m == 0) ++masked;
    REQUIRE(frames >= static_cast<int>(ds.doc_starts.size()) - 1);   // >= 1 op frame per example
    REQUIRE(masked > 0);                                             // results are masked (delegated)
}

TEST_CASE("op_curriculum: single-step frames dispatch to their masked result (FILTER guarantee)", "[opcurric]") {
    const auto tk = make_tok();
    oc::Options o; o.seed = 7; o.n_examples = 300; o.chain_frac = 0.0; o.max_digits = 3;   // single-step only
    const oc::Dataset ds = oc::build_dataset(tk, o);
    const auto compute = nd::make_compute_callback(nd::builtin());

    int checked = 0;
    for (std::size_t d = 0; d + 1 < ds.doc_starts.size(); ++d) {
        const std::size_t lo = ds.doc_starts[d], hi = ds.doc_starts[d + 1];
        // Find the op frame's TOK_TURN_END; run the callback over the doc prefix ending there.
        std::size_t close = lo;
        for (std::size_t i = lo; i < hi; ++i) if (ds.tokens[i] == cas::TOK_TURN_END) { close = i; break; }
        REQUIRE(close > lo);
        const std::vector<int> ctx(ds.tokens.begin() + static_cast<std::ptrdiff_t>(lo),
                                   ds.tokens.begin() + static_cast<std::ptrdiff_t>(close) + 1);
        const std::vector<int> inj = compute(ctx);
        REQUIRE_FALSE(inj.empty());                                  // the frame is a resolvable op
        std::string got; for (int t : inj) if (t >= '0' && t <= '9') got.push_back(static_cast<char>(t));
        REQUIRE(got == masked_digits(ds, lo, hi));                   // node result == the masked span
        ++checked;
    }
    REQUIRE(checked > 100);
}

TEST_CASE("op_curriculum: collapse-chain examples have two frames and reference the slot", "[opcurric]") {
    const auto tk = make_tok();
    oc::Options o; o.seed = 3; o.n_examples = 120; o.chain_frac = 1.0; o.max_digits = 3;   // chains only
    const oc::Dataset ds = oc::build_dataset(tk, o);
    const int S0 = sub0::SCRATCH_SLOT_BASE, S1 = sub0::SCRATCH_SLOT_BASE + 1;

    int chains = 0;
    for (std::size_t d = 0; d + 1 < ds.doc_starts.size(); ++d) {
        const std::size_t lo = ds.doc_starts[d], hi = ds.doc_starts[d + 1];
        int frames = 0, s0 = 0, s1 = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            if (ds.tokens[i] == cas::TOK_TURN_START) ++frames;
            if (ds.tokens[i] == S0) ++s0;
            if (ds.tokens[i] == S1) ++s1;
        }
        REQUIRE(frames == 2);       // two routed steps
        REQUIRE(s0 == 2);           // S0 appears masked (step-1 result) AND as the step-2 reference
        REQUIRE(s1 == 1);           // S1 is the (masked) step-2 result
        ++chains;
    }
    REQUIRE(chains > 50);
}
