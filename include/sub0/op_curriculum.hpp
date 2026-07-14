// sub0/op_curriculum.hpp -- the PRODUCTION op-delegation curriculum (a --op-mix blend source for
// train_stage, mirroring spellspike/scratchspike). Synthetic, self-verifying arithmetic problems teach the
// model to ROUTE to the `math` node rather than compute -- the two mechanisms the GSM8K chapter proved:
//
//   * SINGLE-STEP (contextual): `... A+B = [op math]` -- the model emits a bare op frame and the node reads
//     the expression posed in the prose (node_frame::preceding_expr). Proved 0.935 held-out (gsm8k capstone).
//   * MULTI-STEP (collapse chain): `A+B = [op math] <S0> . S0-C = [op math] <S1>` -- the step-1 result is a
//     SLOT SYMBOL the next step references (one token, not a multi-digit copy). Proved 1.000 vs digit-copy
//     0.095 (the chaining A/B). At train time the slots are baked/masked tokens the model learns to emit;
//     at gen time the collapse callback binds the live result and the node dereferences it.
//
// Every arithmetic span is MASKED (the node supplies it -- the FILTER pillar), so the model NEVER learns the
// fuzzy arithmetic. Digits + operators are byte tokens: the Unigram word-encoder's Viterbi runs only over
// word-byte runs (casing::is_word_byte = alpha/high-byte, which EXCLUDES digits + punctuation), so it never
// absorbs them into a piece -- the node reads numbers at byte granularity, consistent train<->infer.
//
// Engine-free (tokenizer + nodes + std). One Dataset per run; borrowed by the blend source, so it must
// outlive the training loop.

#pragma once

#include "sub0/gsm8k.hpp"          // build_stream (single-step contextual), op_frame markers
#include "sub0/nodes.hpp"          // eval / add / cmp -- exact verification
#include "sub0/node_frame.hpp"     // op_header, FRAME markers
#include "sub0/scratch_slots.hpp"  // SCRATCH_SLOT_BASE (the collapse slots)
#include "sub0/tokenizer.hpp"      // tok::encode (prose -> ids)

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace sub0::op_curriculum {

// A masked token dataset in the shape the blender consumes (parallel tokens/mask + a document index).
struct Dataset {
    std::vector<int>          tokens;
    std::vector<std::uint8_t> mask;        // 1 = graded (prose + routing), 0 = masked (delegated result)
    std::vector<std::uint64_t> doc_starts; // [0, len0, len0+len1, ...]  -- one entry per example + the total
};

struct Options {
    std::uint64_t seed        = 0;
    int           n_examples  = 3000;
    double        chain_frac  = 0.4;   // fraction that are 2-step collapse chains (rest single-step)
    int           max_digits  = 3;
};

namespace detail {
inline std::string gen_int(std::mt19937_64& rng, int digits) {   // no leading zero
    std::uniform_int_distribution<int> d0(1, 9), d(0, 9);
    std::string s(1, static_cast<char>('0' + d0(rng)));
    for (int i = 1; i < digits; ++i) s.push_back(static_cast<char>('0' + d(rng)));
    return s;
}

// A single-step problem as a GSM8K-style solution: `<prose> EXPR = <<EXPR=R>> R .` (EXPR in the prose right
// before the annotation, so the bare frame's preceding_expr reads it). op in {+,-,*}, exact integer result.
inline std::string gen_single(std::mt19937_64& rng, int maxdig) {
    std::uniform_int_distribution<int> dd(1, maxdig), pick(0, 2);
    std::string A = gen_int(rng, dd(rng)), B = gen_int(rng, dd(rng));
    char op = '+';
    switch (pick(rng)) {
        case 1: op = '-'; if (nodes::cmp(A, B) < 0) std::swap(A, B); break;               // non-negative
        case 2: op = '*'; A = gen_int(rng, std::min(2, maxdig)); B = gen_int(rng, std::min(2, maxdig)); break;
        default: break;
    }
    const std::string expr = A + std::string(1, op) + B, res = nodes::eval(expr);
    if (res.empty()) return gen_single(rng, maxdig);   // (shouldn't happen for + - *)
    static const char* pre[] = { "total ", "so ", "we get ", "count ", "then " };
    return std::string(pre[std::uniform_int_distribution<int>(0, 4)(rng)]) +
           expr + "=<<" + expr + "=" + res + ">> " + res + ".";
}
}  // namespace detail

// Append a 2-step COLLAPSE chain directly (it needs slot tokens, which build_stream doesn't emit):
// `A+B = [op math] <S0 masked> . S0<op2>C = [op math] <S1 masked> .`  -- the step-1 result is slot S0, which
// the step-2 expression references as one symbol. The `. ` separates the injected result from the step-2
// expression so preceding_expr isolates `S0<op2>C`.
inline void emit_chain(Dataset& ds, const tok::Tokenizer& tk, std::mt19937_64& rng, int maxdig) {
    std::uniform_int_distribution<int> dd(1, maxdig);
    const std::string A = detail::gen_int(rng, dd(rng)), B = detail::gen_int(rng, dd(rng));
    const std::string r1 = nodes::add(A, B);
    char op2 = (rng() & 1u) ? '+' : '-';
    const std::string C = detail::gen_int(rng, std::min(2, maxdig));
    if (op2 == '-' && nodes::cmp(r1, C) < 0) op2 = '+';                 // keep step 2 non-negative
    const std::string r2 = nodes::eval(r1 + std::string(1, op2) + C);
    if (r2.empty()) return;

    auto g     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(1); };
    auto m     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(0); };
    auto gtext = [&](const std::string& s) { for (int t : tok::encode(tk, s)) g(t); };
    const int S0 = SCRATCH_SLOT_BASE, S1 = SCRATCH_SLOT_BASE + 1;

    gtext(A + "+" + B + "=");
    for (int t : nodes::op_header("math")) g(t);   // step-1 route
    m(S0);                                          // step-1 result COLLAPSED to slot S0 (masked)
    gtext(". ");                                    // separator
    g(S0);                                          // step-2 REFERENCE (the slot symbol, one token)
    gtext(std::string(1, op2) + C + "=");
    for (int t : nodes::op_header("math")) g(t);   // step-2 route
    m(S1);                                          // step-2 result collapsed to S1 (masked)
    gtext(".");
    ds.doc_starts.push_back(ds.tokens.size());
}

// Build the full curriculum: `n_examples` problems, `chain_frac` of them 2-step collapse chains.
inline Dataset build_dataset(const tok::Tokenizer& tk, const Options& opt) {
    Dataset ds; ds.doc_starts.push_back(0);
    std::mt19937_64 rng(opt.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto enc = [&tk](std::string_view s) { return tok::encode(tk, std::string(s)); };
    for (int i = 0; i < opt.n_examples; ++i) {
        if (u(rng) < opt.chain_frac) {
            emit_chain(ds, tk, rng, opt.max_digits);
        } else {
            const gsm8k::Example ex = gsm8k::build_stream(detail::gen_single(rng, opt.max_digits), enc);
            if (ex.ops == 0) continue;
            ds.tokens.insert(ds.tokens.end(), ex.tokens.begin(), ex.tokens.end());
            ds.mask.insert(ds.mask.end(), ex.mask.begin(), ex.mask.end());
            ds.doc_starts.push_back(ds.tokens.size());
        }
    }
    return ds;
}

}  // namespace sub0::op_curriculum
