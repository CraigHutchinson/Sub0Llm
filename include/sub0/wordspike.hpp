// sub0/wordspike.hpp -- SPIKE: does the scratch-slot mechanism generalize from repeatspike's synthetic
// control-symbol probe to a slot sitting mid-sentence in REAL, natural GSM8K-style prose, and does it
// compose with op_curriculum's arithmetic-collapse in the SAME document?
//
// repeatspike.hpp already proved HARNESS-DRIVEN collapse of a repeated OOV mention beats fuzzy re-spelling
// (0.95 vs 0.13 held-out) -- but over a synthetic `W , <slot> , <slot> # n <slot>` control-symbol passage,
// not real text. op_curriculum.hpp already proves computed arithmetic results collapse cleanly into
// scratch slots inside real GSM8K-shaped prose (`... EXPR = [op math] <result>`). This spike asks whether
// the SAME slot-carrying skill repeatspike proved survives when: (1) the surrounding text is genuinely
// natural (built via tok::encode, so real TOK_CAP/TOK_SPELL_START/END/TOK_JOIN markers appear exactly as
// they would at real inference time -- see src/tokenizer.cpp's encode_join), and (2) a NAME slot and an
// arithmetic-RESULT slot share the same 6-slot pool in one document (proven collision-free at the engine
// level by tests/blended_capstone_engine_tests.cpp's "[blended]" case; this spike is the TRAINED, not just
// engine-level, half of that same question).
//
// Trace shape (one document): "<Name1> has <A> eggs. <Name2> gives <B> eggs to <Name1SLOT>. <A>+<B>=
// [op math] <ResultSLOT> ." -- Name1's first mention is real encoded text (masked: harness-observed
// context, not something to predict from nothing, matching repeatspike's own mention-1 convention);
// Name2 (mentioned once, never repeated) is likewise masked; Name1's SECOND mention is the bound scratch
// slot token directly (masked: harness-substituted, matching repeatspike's mentions 2/3). BOTH operands
// (A and B) appear in the narrative BEFORE the name-collapse point, so by the time the literal "<A>+<B>="
// expression is needed the model only ever COPIES digits already in context -- never conjures one it was
// never shown (an earlier version put B after the collapse point, making the "gold" answer depend on a
// digit generation could never see; not a training-quality issue, an unsatisfiable eval target). The
// "<A>+<B>=" expression sits DIRECTLY adjacent to "[op math]" -- node_frame.hpp's preceding_expr scans
// backward over pure expression characters and stops dead at the first prose letter, so connective words
// between the posed expression and the frame would starve op-delegation entirely (exactly
// op_curriculum.hpp's own "A+B=[op math]" shape / gsm8k_engine_tests.cpp's "the model must copy the
// expression, it does not learn the arithmetic" discipline). Ordinary connective prose + digits + the
// "[op math]" routing header are GRADED (ordinary language modeling + the routing DECISION, never the
// fuzzy arithmetic or the unpredictable name spellings -- same masking philosophy as op_curriculum.hpp's
// own header comment and repeatspike.hpp's COLLAPSE arm).
//
// FUZZY control arm (no collapse: Name1's second mention is ALSO spelled out in full) isolates whether the
// slot mechanism itself is doing the work, mirroring repeatspike's own FUZZY-vs-COLLAPSE A/B exactly.
//
// EVAL METHODOLOGY NOTE (a real design error this now documents): the collapse decision at Name1's second
// mention is masked (mask=0) in BOTH arms during training, so neither arm ever carries gradient signal
// about what to emit AT that position -- exactly mirroring repeatspike.hpp's own repeat_task_collapse,
// where mentions 2/3 are fed as already-resolved slot tokens, never something free generation is asked to
// produce. An eval that starts free generation right before the collapse point and hopes the model either
// emits the slot or byte-exactly re-spells an arbitrary invented name is therefore testing an unsupported
// claim -- see tests/wordspike_engine_tests.cpp's EvalProblem/eval_arm for the fix (resolve the collapse
// via the harness before generation starts, matching training's own teacher-forced shape, and grade only
// the downstream op-delegation continuation that training actually taught).
//
// Engine-free (tokenizer + nodes + std only), matching every other spike header in this codebase.

#pragma once

#include "sub0/scratchspike.hpp"   // oov_bytes-equivalent discipline, SCRATCH_SLOT_BASE/scratch_slot
#include "sub0/op_curriculum.hpp"  // Dataset shape, nodes::add, node_frame::op_header/FRAME_*
#include "sub0/tokenizer.hpp"      // tok::encode

#include <random>
#include <string>
#include <vector>

namespace sub0::wordspike {

namespace ss = sub0::scratchspike;
namespace nd = sub0::nodes;

// A short digit string (no leading zero), matching op_curriculum::detail::gen_int's own shape/discipline
// (kept as a local copy -- op_curriculum::detail is an implementation-private namespace of that header,
// not meant to be reused across spike files, same convention scratchspike/repeatspike already follow for
// their own small helpers).
inline std::string gen_digits(std::mt19937_64& rng, int digits) {
    std::uniform_int_distribution<int> d0(1, 9), d(0, 9);
    std::string s(1, static_cast<char>('0' + d0(rng)));
    for (int i = 1; i < digits; ++i) s.push_back(static_cast<char>('0' + d(rng)));
    return s;
}

// A capitalized OOV proper-noun-shaped name: genuinely not a single vocab piece under `t` (same
// discipline as scratchspike::gen_oov -- tested against the LOWERCASE spelling, since that's what the
// tokenizer's own truecasing reduces a capitalized word to before vocab lookup; see casing.hpp), long
// enough (5-9 letters) to read as a name rather than a short common word.
inline std::string gen_name(std::mt19937_64& rng, const tok::Tokenizer& t) {
    std::uniform_int_distribution<int> len_d(5, 9), ch(0, 25);
    for (int tries = 0; tries < 64; ++tries) {
        std::string lower(1, static_cast<char>('a' + ch(rng)));
        for (int i = 1; i < len_d(rng); ++i) lower.push_back(static_cast<char>('a' + ch(rng)));
        if (t.piece_index.find(lower) == t.piece_index.end()) {
            std::string name = lower;
            name[0] = static_cast<char>(name[0] - 'a' + 'A');
            return name;
        }
    }
    return "Qxzjvwyk";   // fallback: essentially never a single vocab piece
}

// Same Dataset shape as scratchspike/op_curriculum -- what the production blend source consumes.
struct Dataset {
    std::vector<int>                             tokens;
    std::vector<std::uint8_t>                    mask;
    std::vector<std::uint64_t>                   doc_starts;
    std::vector<std::vector<std::vector<int>>>   doc_bindings;   // per doc: slot -> fragment token ids
};

enum class Arm { Collapse, Fuzzy };   // Collapse: Name1's 2nd mention is the slot. Fuzzy: spelled out again.

// Append one document. `arm` selects COLLAPSE (the mechanism under test) vs FUZZY (the control, matching
// repeatspike's own A/B). Returns the byte spelling of Name1 (for eval prompt construction) and whether a
// name slot was actually bound this doc (COLLAPSE only).
struct DocInfo { std::string name1; bool bound = false; };

inline DocInfo emit_doc(Dataset& ds, const tok::Tokenizer& tk, std::mt19937_64& rng, int maxdig, Arm arm) {
    const std::string name1 = gen_name(rng, tk), name2 = gen_name(rng, tk);
    const std::string A = gen_digits(rng, maxdig), B = gen_digits(rng, maxdig);
    const std::string result = nd::add(A, B);
    DocInfo info{ name1, false };
    if (result.empty()) return info;   // overflow guard, same discipline as op_curriculum::emit_single

    auto g     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(1); };
    auto m     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(0); };
    auto gtext = [&](const std::string& s) { for (int t : tok::encode(tk, s)) g(t); };
    auto mtext = [&](const std::string& s) { for (int t : tok::encode(tk, s)) m(t); };

    // node_frame.hpp's preceding_expr scans BACKWARD from [op math] over pure expression characters
    // (digits/operators/=/spaces/slots) and stops dead at the first PROSE letter -- so the literal
    // "A+B=" must sit DIRECTLY adjacent to the frame, no connective words in between (exactly
    // op_curriculum.hpp's own "A+B=[op math]" shape). CRITICAL (a real bug this comment now documents):
    // BOTH A and B must be shown in the narrative BEFORE the name-collapse point, not after -- an
    // earlier version introduced B only after the collapse point, so at EVAL time (no teacher forcing)
    // the "gold" A+B answer depended on a digit the model was never shown and had no way to reproduce;
    // it wasn't a training failure, the target was unsatisfiable by construction. op_curriculum's own
    // "the model must COPY the expression into the frame, it does not learn the arithmetic" discipline
    // (gsm8k_engine_tests.cpp's comment) requires every operand to already be IN CONTEXT by the time
    // it's copied -- exactly like the collapsed name reference itself.
    mtext(name1 + " has " ); gtext(A + " eggs. ");
    mtext(name2 + " gives " + B + " eggs to ");
    const int slot1 = sub0::SCRATCH_SLOT_BASE;
    if (arm == Arm::Collapse) { m(slot1); info.bound = true; }
    else                      mtext(name1);
    gtext(". " + A + "+" + B + "=");
    for (int t : nd::op_header("math")) g(t);          // routing: graded (the model must learn to invoke it)
    const int slot2 = sub0::SCRATCH_SLOT_BASE + 1;
    m(slot2);                                            // op result: harness-computed, masked
    gtext(".");

    ds.doc_starts.push_back(ds.tokens.size());
    std::vector<int> f1; for (char c : name1) f1.push_back(static_cast<unsigned char>(c));
    std::vector<int> f2; for (char c : result) f2.push_back(static_cast<unsigned char>(c));
    ds.doc_bindings.push_back({ std::move(f1), std::move(f2) });   // slot 0 -> name1, slot 1 -> result
    return info;
}

struct Options { std::uint64_t seed = 0; int n_examples = 3000; int max_digits = 2; };

inline Dataset build_dataset(const tok::Tokenizer& tk, const Options& opt, Arm arm) {
    Dataset ds; ds.doc_starts.push_back(0);
    std::mt19937_64 rng(opt.seed);
    for (int i = 0; i < opt.n_examples; ++i) emit_doc(ds, tk, rng, opt.max_digits, arm);
    return ds;
}

// Raw ingredients for an eval problem -- deliberately NOT a single flat prompt string. The name-collapse
// decision is never something training asks the model to produce from scratch: emit_doc masks (mask=0)
// BOTH the bound slot (COLLAPSE arm) and the re-spelled name (FUZZY arm), so at that exact position
// NEITHER arm carries any gradient signal about what to emit there -- matching repeatspike.hpp's own
// design exactly (repeat_task_collapse's mentions 2/3 are fed as already-resolved slot tokens, never
// something the model is asked to freely generate; only the QUERY AFTER the collapse point is graded).
// An earlier version of this eval built a single prompt string ending right before the collapse point and
// asked free generation to either emit the slot or re-spell the name byte-exact -- an unsupported claim
// that scored 0.0 for a structural reason, not a training-quality one. The fix: resolve the collapse via
// the harness BEFORE generation starts (mirrors prefill_collapse's real use over known prompt text), and
// grade only what training actually taught -- the op-delegation continuation.
struct EvalProblem { std::string name1, name2, A, B, gold; };

inline EvalProblem gen_eval(std::mt19937_64& rng, const tok::Tokenizer& tk, int maxdig) {
    EvalProblem pr;
    pr.name1 = gen_name(rng, tk);
    pr.name2 = gen_name(rng, tk);
    pr.A     = gen_digits(rng, maxdig);
    pr.B     = gen_digits(rng, maxdig);
    pr.gold  = nd::add(pr.A, pr.B);
    return pr;
}

}  // namespace sub0::wordspike
