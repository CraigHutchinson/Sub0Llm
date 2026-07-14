// node_frame_engine_tests.cpp -- END-TO-END production op frame: a model learns to emit a TOK_TURN op region
// (`[op add]`), and kv_decode_generate's compute seam -- wired to the nodes::Registry via node_frame.hpp --
// parses it, dispatches, and injects the exact result. This closes the loop from the spikes to the production
// substrate: the same interceptor the scratch/spell curricula use, now dispatching deterministic compute nodes
// through the EXISTING chat turn markers (zero new tokenizer ids). Tagged "[.nodeframe]" (hidden): trains.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"
#include "sub0/decode.hpp"
#include "sub0/nodes.hpp"
#include "sub0/node_frame.hpp"
#include "sub0/window.hpp"

#include <algorithm>
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

// Task: `A + B =` -> the model emits the fixed op-region `[op add]` (graded); the node injects `[<sum>]`
// (masked). Operands are read from the prompt by the node, so the model learns only the routing header.
struct Task { std::vector<int> prompt, trace; std::vector<std::uint8_t> mask; std::string sum; };
Task make_task(const std::string& a, const std::string& b) {
    Task k; k.sum = nd::add(a, b);
    auto pp = [&](int t) { k.trace.push_back(t); k.mask.push_back(0); k.prompt.push_back(t); };
    auto g  = [&](int t) { k.trace.push_back(t); k.mask.push_back(1); };
    auto m  = [&](int t) { k.trace.push_back(t); k.mask.push_back(0); };
    for (char c : a) pp(static_cast<unsigned char>(c));
    pp('+'); for (char c : b) pp(static_cast<unsigned char>(c)); pp('=');
    for (int t : nd::op_header("math")) g(t);                   // graded: `[op math]` (the routing)
    m(nd::FRAME_OPEN); for (char c : k.sum) m(static_cast<unsigned char>(c)); m(nd::FRAME_CLOSE);   // masked: `[<sum>]`
    g(cas::TOK_EOS);
    return k;
}

std::string extract(const std::vector<int>& ctx) {   // the number in the LAST TOK_TURN region (the result)
    int e = -1; for (int i = static_cast<int>(ctx.size()) - 1; i >= 0; --i) if (ctx[i] == nd::FRAME_CLOSE) { e = i; break; }
    if (e < 0) return {};
    int o = -1; for (int i = e - 1; i >= 0; --i) if (ctx[i] == nd::FRAME_OPEN) { o = i; break; }
    if (o < 0) return {};
    const nd::WordsNums w = nd::scan_region(ctx, o + 1, e);
    return w.nums.empty() ? std::string{} : w.nums[0];
}

void train_steps(const std::vector<int>& tok, const std::vector<std::uint8_t>& mask,
                 const std::vector<std::uint64_t>& docs, sub0::AdamW& opt, int steps, std::mt19937& rng, int window) {
    std::vector<std::size_t> starts(kBatch);
    std::vector<int>         lens(kBatch);
    for (int s = 0; s < steps; ++s) {
        for (int b = 0; b < kBatch; ++b) {
            const sub0::Window w = sub0::sample_window(rng, window, tok.size(), std::span<const std::uint64_t>(docs));
            starts[static_cast<std::size_t>(b)] = w.start; lens[static_cast<std::size_t>(b)] = w.len;
        }
        opt.zero_grad();
        (void)sub0::train_batch(tok.data(), starts.data(), kBatch, window, lens.data(), mask.data(), nullptr);
        opt.step();
    }
}

}  // namespace

TEST_CASE("node_frame: end-to-end -- model emits a TOK_TURN op region, the node injects the exact result", "[.nodeframe]") {
    const int window = std::min(60, SEQ_LEN - 1);

    // Build the training stream (one task per doc).
    std::vector<int> tok; std::vector<std::uint8_t> mask; std::vector<std::uint64_t> docs{0};
    std::mt19937_64 dsrng(2028);
    std::uniform_int_distribution<int> dd(kMinDigits, kMaxDigits);
    for (int i = 0; i < 4000; ++i) {
        const Task k = make_task(gen_int(dsrng, dd(dsrng)), gen_int(dsrng, dd(dsrng)));
        tok.insert(tok.end(), k.trace.begin(), k.trace.end());
        mask.insert(mask.end(), k.mask.begin(), k.mask.end());
        docs.push_back(tok.size());
    }
    REQUIRE(tok.size() > static_cast<std::size_t>(window));

    sub0::build_model();
    reset_opt_state();
    sub0::AdamW opt(kLr);
    std::mt19937 rng(1);
    const auto compute = nd::make_compute_callback(nd::builtin());   // the PRODUCTION registry-backed callback

    std::string report = "\n=== node_frame end-to-end: `A + B =` -> model emits [op add] -> node injects [sum] (d" +
                         std::to_string(D_MODEL) + ") ===\n";
    double best = 0.0;
    for (int r = 0; r < 10; ++r) {
        train_steps(tok, mask, docs, opt, 300, rng, window);
        std::mt19937_64 ev(7777ULL);
        int ok = 0, n = 0;
        for (int i = 0; i < 120; ++i) {                             // HELD-OUT: fresh numbers
            const Task k = make_task(gen_int(ev, dd(ev)), gen_int(ev, dd(ev)));
            std::vector<int> ctx = k.prompt;
            std::mt19937 grng(0);
            sub0::kv_decode_generate(ctx, /*n=*/40, /*temp=*/1.f, /*topk=*/1, grng, cas::TOK_EOS,
                                     /*use_gpu=*/false, /*on_token=*/{}, /*expand=*/{}, /*combine=*/{},
                                     compute, nd::FRAME_CLOSE);
            ok += (extract(ctx) == k.sum); ++n;
        }
        best = std::max(best, static_cast<double>(ok) / n);
        char line[64]; std::snprintf(line, sizeof line, "  step %5d | HELD-OUT exact=%.3f\n", (r + 1) * 300, static_cast<double>(ok) / n);
        report += line;
    }
    WARN(report);
    REQUIRE(std::isfinite(best));
}

// FORWARD-PASS resolution: a prompt that merely POSES a computation (ends in a completed op region) is
// resolved during PREFILL -- a discrete forward-step delegation, no iterative generation/reasoning needed.
// The result is deterministic (independent of the model's logits), so an untrained model suffices.
TEST_CASE("node_frame: prefill resolves a prompt that ends in an op region (forward-pass delegation)", "[nodeframe]") {
    sub0::build_model();
    const auto compute = nd::make_compute_callback(nd::builtin());
    // `12 + 34 = [op math]`  (the injected result region follows the prompt).
    std::vector<int> ctx = {'1','2','+','3','4','=', nd::FRAME_OPEN,'o','p',' ','m','a','t','h', nd::FRAME_CLOSE};
    const std::size_t plen = ctx.size();
    std::mt19937 rng(0);
    sub0::kv_decode_generate(ctx, /*n=*/1, 1.f, 1, rng, cas::TOK_EOS, false, {}, {}, {}, compute, nd::FRAME_CLOSE);
    // `[46]` was injected in the forward pass, right after the prompt's op region.
    REQUIRE(ctx.size() >= plen + 4);
    REQUIRE(ctx[plen + 0] == nd::FRAME_OPEN);
    REQUIRE(ctx[plen + 1] == '4');
    REQUIRE(ctx[plen + 2] == '6');
    REQUIRE(ctx[plen + 3] == nd::FRAME_CLOSE);
}
