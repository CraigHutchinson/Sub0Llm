// sub0/factspike.hpp -- does a factual association survive PIECE-EMBEDDING TRANSFER into a scratch slot?
//
// Every scratch mechanism proven so far (scratchspike/repeatspike/wordspike/op_curriculum) binds a slot to
// a fragment's BYTE-DECOMPOSED spelling (ScratchTable::expand() always resolves a piece down to individual
// base-byte codes before content_embed's MeanPool/HRR composes over them). This header tests something
// different: does the model's OWN already-trained embedding for a word's REAL vocab piece(s) -- literally
// the weights that would sit in context if the word were spelled out normally -- still carry a learned
// factual association when moved to a scratch-slot token instead? No engine change is needed for this:
// ScratchTable::bind() stores whatever ids it's given verbatim, and encode_slot() (scratch_slots.hpp)
// doesn't know or care whether bound ids are byte codes or real vocab-piece ids -- it just indexes
// tok_emb[id] and composes. Binding a subject's OWN tokenized piece ids directly (via
// sub0::detail::word_span(tok::encode(tk, subject), 0).second -- skipping expand()'s byte decomposition
// entirely) makes the slot's embedding a literal composition of the word's own already-trained piece rows.
// See docs/FACTSPIKE.md for the full design record, the staged verification plan, and why this needs a
// small dedicated model with a vocab forced to be multi-piece (so "compose across a few piece embeddings"
// is the NORMAL way the model represents any word, not a rare OOV special case).
//
// Engine-free (tokenizer + std only), matching every other spike header in this codebase.

#pragma once

#include "sub0/tokenizer.hpp"
#include "sub0/scratch.hpp"   // ScratchTable / detail::word_span / SCRATCH_SLOT_BASE -- reused as-is,
                              // no engine change: content_embed doesn't care whether bound ids are
                              // byte codes or real vocab-piece ids (see this header's own top comment)

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sub0::factspike {

// A small, fixed vocabulary of common color words -- the "fact" side. Deliberately common/frequent
// (unlike the invented subjects), since what matters for this experiment is the SUBJECT's forced
// multi-piece composition, not the fact word's -- the fact word just needs to be a short, checkable
// completion.
inline const std::vector<std::string>& fact_vocab() {
    static const std::vector<std::string> v = {
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "black"
    };
    return v;
}

// An invented, OOV-shaped proper noun (capitalized, 5-9 letters) -- mirrors wordspike::gen_name's own
// random-letter generation, WITHOUT a piece_index existence check: the corpus this name goes into is
// what DEFINES the vocab in the first place (Phase A), so there is no tokenizer to check against yet.
// A small vocab_target (Phase A) is what forces these to stay multi-piece, not this generator.
inline std::string gen_subject(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> len_d(5, 9), ch(0, 25);
    std::string name(1, static_cast<char>('A' + ch(rng)));
    for (int i = 1; i < len_d(rng); ++i) name.push_back(static_cast<char>('a' + ch(rng)));
    return name;
}

struct FactPair { std::string subject, fact; };
struct FactSplit { std::vector<FactPair> drilled, held_out; };

// n_total distinct invented subjects, each assigned a UNIFORMLY RANDOM fact color, independent of the
// subject's own spelling -- a genuinely arbitrary association (the fact must not be guessable from the
// subject's bytes, only from learned exposure), matching scratchspike's own OOV<->meaning discipline.
// Split drilled/held-out by drilled_frac, mirroring scratchspike::make_oov_split exactly
// (scratchspike.hpp:330-345) -- held-out subjects get NO fact-teaching exposure at all (the negative
// control: a slot bound to a held-out subject's bytes should not reliably predict any particular fact).
inline FactSplit make_fact_split(std::mt19937_64& rng, int n_total, double drilled_frac) {
    FactSplit split;
    std::uniform_int_distribution<std::size_t> color_d(0, fact_vocab().size() - 1);
    std::vector<std::string> seen;
    seen.reserve(static_cast<std::size_t>(n_total));
    for (int i = 0; i < n_total; ++i) {
        std::string subj;
        for (int tries = 0; tries < 64; ++tries) {
            subj = gen_subject(rng);
            if (std::find(seen.begin(), seen.end(), subj) == seen.end()) break;
        }
        seen.push_back(subj);
        FactPair fp{ subj, fact_vocab()[color_d(rng)] };
        (static_cast<double>(i) < drilled_frac * n_total ? split.drilled : split.held_out).push_back(std::move(fp));
    }
    return split;
}

// A handful of varied sentence templates stating the SAME fact -- so training has to learn the
// underlying (subject, fact) association rather than memorize one exact sentence shape, matching how
// real language models generalize facts from varied restatement, not verbatim repetition. `{S}`/`{F}`
// are substituted with the subject/fact text.
inline const std::vector<std::string>& fact_templates() {
    static const std::vector<std::string> v = {
        "{S} loves the color {F}. {S} plays outside every day.\n",
        "Everyone knows that {S}'s favorite color is {F}.\n",
        "{S} always wears {F}. It matches {S}'s favorite color.\n",
        "When you think of {S}, you think of the color {F}.\n",
        "{S} picked {F} as a favorite color a long time ago.\n",
        "The color {F} makes {S} very happy.\n",
    };
    return v;
}

inline std::string fill_template(const std::string& tmpl, const std::string& subject, const std::string& fact) {
    std::string out;
    out.reserve(tmpl.size() + subject.size() * 4 + fact.size() * 2);
    for (std::size_t i = 0; i < tmpl.size(); ) {
        if (tmpl.compare(i, 3, "{S}") == 0) { out += subject; i += 3; }
        else if (tmpl.compare(i, 3, "{F}") == 0) { out += fact; i += 3; }
        else { out.push_back(tmpl[i]); ++i; }
    }
    return out;
}

// Phase A: raw fact-teaching TEXT (not tokenized) for a real sub0-configure run -- docs_per_fact
// documents per drilled subject, template chosen at random each time. Held-out subjects appear NOWHERE
// in this text (they must get zero fact-teaching exposure).
inline std::string build_corpus_text(const std::vector<FactPair>& drilled, int docs_per_fact,
                                     std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> tmpl_d(0, fact_templates().size() - 1);
    std::string text;
    for (const FactPair& fp : drilled)
        for (int i = 0; i < docs_per_fact; ++i)
            text += fill_template(fact_templates()[tmpl_d(rng)], fp.subject, fp.fact) + "\n";
    return text;
}

// Phase B: the SAME fact-teaching documents as build_corpus_text, but tokenized in-memory as a Dataset
// matching every other curriculum's shape -- ORDINARY graded text (mask=1 throughout), NO scratch slots
// anywhere. This is what teaches the (subject, fact) association via sheer repetition; whether it
// actually sticks at all is Phase B's own open question (docs/FACTSPIKE.md).
struct Dataset {
    std::vector<int>           tokens;
    std::vector<std::uint8_t>  mask;
    std::vector<std::uint64_t> doc_starts;
};

inline Dataset build_dataset(const tok::Tokenizer& tk, const std::vector<FactPair>& drilled,
                             int docs_per_fact, std::uint64_t seed) {
    Dataset ds;
    ds.doc_starts.push_back(0);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> tmpl_d(0, fact_templates().size() - 1);
    for (const FactPair& fp : drilled) {
        for (int i = 0; i < docs_per_fact; ++i) {
            const std::string doc = fill_template(fact_templates()[tmpl_d(rng)], fp.subject, fp.fact);
            for (int t : tok::encode(tk, doc)) { ds.tokens.push_back(t); ds.mask.push_back(1); }
            ds.doc_starts.push_back(ds.tokens.size());
        }
    }
    return ds;
}

// Phase C: slot-reading exposure documents -- teaches the model that a scratch slot, once bound to a
// subject's OWN tokenized piece ids (not byte-decomposed -- see this header's own top comment), should
// be treated as equivalent to that subject for fact-recall purposes. `exposure_subjects` MUST be
// disjoint from the drilled/held-out subjects used in the real eval (docs/FACTSPIKE.md's "avoid an
// in-document shortcut" requirement) -- the fact stated here is real (taught via full text, same as
// Phase B) so the model learns the SAME retrieval skill the eval needs, just over different
// subjects/facts than the ones under test.
struct SlotDataset {
    std::vector<int>                           tokens;
    std::vector<std::uint8_t>                  mask;
    std::vector<std::uint64_t>                 doc_starts;
    std::vector<std::vector<std::vector<int>>> doc_bindings;   // per doc: slot 0 -> subject's piece ids
};

inline SlotDataset build_slot_exposure_dataset(const tok::Tokenizer& tk,
                                               const std::vector<FactPair>& exposure_subjects,
                                               int docs_per_fact, std::uint64_t seed) {
    SlotDataset ds;
    ds.doc_starts.push_back(0);
    std::mt19937_64 rng(seed);
    (void)rng;   // reserved for future template variation; single fixed shape for now
    for (const FactPair& fp : exposure_subjects) {
        const std::vector<int> subj_ctx = tok::encode(tk, fp.subject);
        std::vector<int> pieces;
        for (std::size_t k = 0; k < subj_ctx.size(); ) {
            const auto [span_len, ids] = detail::word_span(subj_ctx, k);
            pieces.insert(pieces.end(), ids.begin(), ids.end());
            k += span_len;
        }
        for (int i = 0; i < docs_per_fact; ++i) {
            auto g     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(1); };
            auto m     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(0); };
            auto gtext = [&](const std::string& s) { for (int t : tok::encode(tk, s)) g(t); };

            // Mention 1: the fact, stated in full (teaches the association, same as Phase B).
            gtext(fp.subject + " loves the color " + fp.fact + ". ");
            // Mention 2: the SAME subject, but as a slot bound to its own piece ids -- teaches the
            // model that reading this slot is equivalent to reading the subject itself.
            m(SCRATCH_SLOT_BASE);
            gtext(" plays outside every day.");

            ds.doc_starts.push_back(ds.tokens.size());
            ds.doc_bindings.push_back({ pieces });   // slot 0 -> this subject's piece ids
        }
    }
    return ds;
}

// Phase H (docs/FACTSPIKE.md's "SPELL marker finding"): the SAME slot-exposure shape as
// build_slot_exposure_dataset above, but `doc_bindings` binds the slot to the FULL, marker-INCLUSIVE
// tokenization (`tok::encode(tk, subject)` directly -- SPELL_START/pieces/SPELL_END for a multi-piece
// subject) instead of running it through detail::word_span's piece extraction. Direct test of whether
// composing from the marker-inclusive span (what a real forward pass over the subject actually processes)
// instead of the marker-stripped one narrows the baseline-vs-scratch gap -- see FACTSPIKE.md's Phase G
// entry for the controlled KV-trace-splice experiment that founded this hypothesis. Training exposure
// MUST match what eval binds to (a train/eval mismatch on the binding convention itself would test
// generalization to a novel convention, not "does marker-inclusive composition help") -- so this is a
// genuinely separate dataset/training run, not just an eval-time swap.
inline SlotDataset build_slot_exposure_dataset_markers(const tok::Tokenizer& tk,
                                                       const std::vector<FactPair>& exposure_subjects,
                                                       int docs_per_fact, std::uint64_t seed) {
    SlotDataset ds;
    ds.doc_starts.push_back(0);
    std::mt19937_64 rng(seed);
    (void)rng;   // reserved for future template variation; single fixed shape for now
    for (const FactPair& fp : exposure_subjects) {
        const std::vector<int> full_span = tok::encode(tk, fp.subject);   // marker-INCLUSIVE, no word_span
        for (int i = 0; i < docs_per_fact; ++i) {
            auto g     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(1); };
            auto m     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(0); };
            auto gtext = [&](const std::string& s) { for (int t : tok::encode(tk, s)) g(t); };

            gtext(fp.subject + " loves the color " + fp.fact + ". ");
            m(SCRATCH_SLOT_BASE);
            gtext(" plays outside every day.");

            ds.doc_starts.push_back(ds.tokens.size());
            ds.doc_bindings.push_back({ full_span });
        }
    }
    return ds;
}

// Phase D ("Pack-Aware Training", docs/FACTSPIKE.md): slot-RETRIEVAL documents -- unlike
// build_slot_exposure_dataset above (which grades generic filler text after the slot, so gradient
// reaching the packed vector is only weakly, indirectly informative), these documents grade the FACT
// ITSELF immediately after the slot. The fact is NOT restated via full text anywhere in THIS document --
// a separate build_dataset() call over the same subjects supplies that (mirroring how the eval only
// ever sees the subject via the packed slot) -- so predicting the correct color has no source in this
// document's own context other than the packed vector. This makes the gradient through encode_slot_bwd
// (HRR: correlation-by-role, hrr_unbind's structural counterpart) directly task-contingent, the same way
// QAT computes its real task loss through the dequantized forward pass rather than a decoupled auxiliary
// objective. Token shape after the slot exactly matches eval_scratch's own prompt ("<slot> loves the
// color <fact>"), minimizing train/eval distribution shift.
inline SlotDataset build_slot_retrieval_dataset(const tok::Tokenizer& tk,
                                                const std::vector<FactPair>& exposure_subjects,
                                                int docs_per_fact, std::uint64_t seed) {
    SlotDataset ds;
    ds.doc_starts.push_back(0);
    std::mt19937_64 rng(seed);
    (void)rng;   // reserved for future template variation; single fixed shape for now (matches eval)
    for (const FactPair& fp : exposure_subjects) {
        const std::vector<int> subj_ctx = tok::encode(tk, fp.subject);
        std::vector<int> pieces;
        for (std::size_t k = 0; k < subj_ctx.size(); ) {
            const auto [span_len, ids] = detail::word_span(subj_ctx, k);
            pieces.insert(pieces.end(), ids.begin(), ids.end());
            k += span_len;
        }
        for (int i = 0; i < docs_per_fact; ++i) {
            auto g     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(1); };
            auto m     = [&](int t) { ds.tokens.push_back(t); ds.mask.push_back(0); };
            auto gtext = [&](const std::string& s) { for (int t : tok::encode(tk, s)) g(t); };

            m(SCRATCH_SLOT_BASE);
            gtext(" loves the color " + fp.fact + ".");   // graded target IS the fact -- task-contingent

            ds.doc_starts.push_back(ds.tokens.size());
            ds.doc_bindings.push_back({ pieces });
        }
    }
    return ds;
}

}  // namespace sub0::factspike
