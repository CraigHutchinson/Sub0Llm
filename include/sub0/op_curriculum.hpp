// sub0/op_curriculum.hpp -- the PRODUCTION op-delegation curriculum (a --op-mix blend source for
// train_stage, mirroring spellspike/scratchspike). Synthetic, self-verifying arithmetic problems teach the
// model to ROUTE to the `math` node rather than compute -- the two mechanisms the GSM8K chapter proved:
//
//   * SINGLE-STEP (contextual): `... A+B = [op math] <S0>` -- the model emits a bare op frame and the node
//     reads the expression posed in the prose (node_frame::preceding_expr). Proved 0.935 held-out.
//   * MULTI-STEP (collapse chain): `A+B = [op math] <S0> . S0-C = [op math] <S1>` -- the step-1 result is a
//     SLOT SYMBOL the next step references (one token, not a multi-digit copy). Proved 1.000 vs digit-copy
//     0.095 (the chaining A/B).
//
// EVERY op result COLLAPSES to a scratch slot (uniform), so gen's behaviour is uniform (always collapse: bind
// the live result to a slot, inject that token, expand it for display) and a chained reference is always a
// one-token symbol. At train time the slots are baked/masked tokens the model learns to emit; at gen time the
// collapse callback binds the live result and the node dereferences it (gen_stage).
//
// Every arithmetic span is MASKED (the node supplies it -- the FILTER pillar), so the model NEVER learns the
// fuzzy arithmetic. Digits + operators are byte tokens: the Unigram bars all-digit pieces (unigram.cpp), so a
// number is never absorbed into a piece -- the node reads it at byte granularity, consistent train<->infer.
// NOTE (v2 schemeV4): digits are now word bytes (casing::is_word_byte), so a number forms one UNIT with no
// per-digit JOIN -- the byte-granularity guarantee holds, but the operand parser's dependence on the old
// digit spacing/JOIN framing still needs a rework (deferred; this curriculum is a spike/POC).
//
// Engine-free (tokenizer + nodes + std). One Dataset per run; borrowed by the blend source, so it must
// outlive the training loop.

#pragma once

#include "sub0/nodes.hpp"          // eval / add / cmp -- exact verification
#include "sub0/node_frame.hpp"     // op_header, FRAME markers
#include "sub0/scratch_slots.hpp"  // SCRATCH_SLOT_BASE (the collapse slots)
#include "sub0/gsm8k.hpp"          // build_stream: convert REAL GSM8K annotations to delegated op-frames
#include "sub0/tokenizer.hpp"      // tok::encode (prose -> ids)

#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace sub0::op_curriculum {

// A masked token dataset in the shape the blender consumes (parallel tokens/mask + a document index).
// `doc_bindings[d]` mirrors scratchspike::Dataset's shape (slot i -> SCRATCH_SLOT_BASE+i's fragment byte
// ids for document d) -- what content-embed needs to bind a collapsed op result to a real vector instead of
// a plain reserved-id row. One entry per document (parallel to doc_starts.size()-1); empty per-doc vector if
// that document collapsed nothing (should not happen here -- every emitted example has >=1 op result).
struct Dataset {
    std::vector<int>          tokens;
    std::vector<std::uint8_t> mask;        // 1 = graded (prose + routing), 0 = masked (delegated result)
    std::vector<std::uint64_t> doc_starts; // [0, len0, len0+len1, ...]  -- one entry per example + the total
    std::vector<std::vector<std::vector<int>>> doc_bindings;  // per doc: slot -> fragment token ids
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

// A binary op over exact-integer operands: {+,-,*}, non-negative result. Returns the expression.
inline std::string gen_expr(std::mt19937_64& rng, int maxdig) {
    std::uniform_int_distribution<int> dd(1, maxdig), pick(0, 2);
    std::string A = gen_int(rng, dd(rng)), B = gen_int(rng, dd(rng));
    char op = '+';
    switch (pick(rng)) {
        case 1: op = '-'; if (nodes::cmp(A, B) < 0) std::swap(A, B); break;               // non-negative
        case 2: op = '*'; A = gen_int(rng, std::min(2, maxdig)); B = gen_int(rng, std::min(2, maxdig)); break;
        default: break;
    }
    return A + std::string(1, op) + B;
}
inline const char* gen_prose(std::mt19937_64& rng) {
    static const char* pre[] = { "total ", "so ", "we get ", "count ", "then ", "answer " };
    return pre[std::uniform_int_distribution<int>(0, 5)(rng)];
}
}  // namespace detail

// Append a SINGLE-STEP contextual example: `<prose> EXPR = [op math] <S0 masked> .`. The model emits the bare
// frame; the node reads EXPR from the prose; the result COLLAPSES to slot S0 (masked) -- uniform with chains.
inline void emit_single(Dataset& ds, const tok::Tokenizer& tk, std::mt19937_64& rng, int maxdig) {
    const std::string expr = detail::gen_expr(rng, maxdig);
    const std::string result = nodes::eval(expr);
    if (result.empty()) return;
    auto g     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(1); };
    auto m     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(0); };
    auto gtext = [&](const std::string& s) { for (int t : tok::encode(tk, s)) g(t); };
    gtext(std::string(detail::gen_prose(rng)) + expr + "=");
    for (int t : nodes::op_header("math")) g(t);   // route
    m(SCRATCH_SLOT_BASE);                          // result collapsed to slot S0 (masked)
    gtext(".");
    ds.doc_starts.push_back(ds.tokens.size());
    std::vector<int> frags; for (char c : result) frags.push_back(static_cast<unsigned char>(c));
    ds.doc_bindings.push_back({ std::move(frags) });   // slot 0 -> the result's byte fragments
}

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
    std::vector<int> f1; for (char c : r1) f1.push_back(static_cast<unsigned char>(c));
    std::vector<int> f2; for (char c : r2) f2.push_back(static_cast<unsigned char>(c));
    ds.doc_bindings.push_back({ std::move(f1), std::move(f2) });   // slot 0 -> r1, slot 1 -> r2
}

// An EVAL problem: the prompt (prose + expression, up to the `=`) and the gold answer the node should
// produce. A trained op model should emit a bare `[op math]` the collapse callback resolves to `gold`.
struct EvalProblem { std::string prompt, gold; };
inline EvalProblem gen_eval(std::mt19937_64& rng, int maxdig) {
    const std::string expr = detail::gen_expr(rng, maxdig);
    return { std::string(detail::gen_prose(rng)) + expr + "=", nodes::eval(expr) };
}

// Build the full curriculum: `n_examples` problems, `chain_frac` of them 2-step collapse chains, the rest
// single-step. Every op result collapses to a slot (uniform).
inline Dataset build_dataset(const tok::Tokenizer& tk, const Options& opt) {
    Dataset ds; ds.doc_starts.push_back(0);
    std::mt19937_64 rng(opt.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < opt.n_examples; ++i) {
        if (u(rng) < opt.chain_frac) emit_chain(ds, tk, rng, opt.max_digits);
        else                         emit_single(ds, tk, rng, opt.max_digits);
    }
    return ds;
}

// Convert a REAL GSM8K corpus (get_gsm8k.py's output: <|endoftext|>-separated problems, calc-annotations
// intact) into the delegated dataset: each `<<EXPR=R>>` becomes a graded `[op math]` with R collapsed to a
// masked slot (via gsm8k::build_stream), so GSM8K's OWN arithmetic delegates instead of being learned as
// text. Prose (question + reasoning) is graded. Documents that yield no verifiable op are skipped.
inline Dataset gsm8k_file_dataset(const std::string& path, const tok::Tokenizer& tk) {
    Dataset ds; ds.doc_starts.push_back(0);
    std::ifstream is(path, std::ios::binary);
    if (!is) return ds;
    const std::string all((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    auto enc = [&tk](std::string_view s) { return tok::encode(tk, std::string(s)); };

    const std::string sep = "<|endoftext|>";
    for (std::size_t pos = 0; pos < all.size();) {
        const std::size_t e = all.find(sep, pos);
        const std::string_view doc(all.data() + pos, (e == std::string::npos ? all.size() : e) - pos);
        if (!doc.empty()) {
            const gsm8k::Example ex = gsm8k::build_stream(doc, enc, /*collapse=*/true);
            if (ex.ops > 0 && !ex.tokens.empty()) {
                ds.tokens.insert(ds.tokens.end(), ex.tokens.begin(), ex.tokens.end());
                ds.mask.insert(ds.mask.end(), ex.mask.begin(), ex.mask.end());
                ds.doc_starts.push_back(ds.tokens.size());
                ds.doc_bindings.push_back(ex.bindings);   // slot i -> annotation i's result fragments
            }
        }
        if (e == std::string::npos) break;
        pos = e + sep.size();
    }
    return ds;
}

}  // namespace sub0::op_curriculum
