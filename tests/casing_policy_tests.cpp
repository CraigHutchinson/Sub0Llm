// casing_policy_tests.cpp — measurement harness for the TRUECASING NAME POLICY.
//
// Why this exists. Generated text repeats capitalised words at 6.7x their base rate (measured, see
// docs/REPETITION.md): 92.7% of repeats are capitalised against a 13.8% base rate. The mechanism is
// representational, not a sampler artifact -- proper nouns are kept VERBATIM by the truecaser, so BPE
// must spell them from short cased fragments (`Al` `Ma` `Ch` `Am` `Sa`), and 9.1% of the learned vocab
// (1469 of 16220 merges) is spent on that duplicate, sparsely-trained cased inventory.
//
// The policy under measurement is derive_attested()'s rule: a lowercase form is eligible for
// CAP/UP-marker collapse unless its mid-sentence-capital uses outnumber its lowercase uses
// (`mid > lc` -> withheld -> the capitalised form stays verbatim).
//
// The question this harness answers FIRST, before any tuning: the withheld population has two
// disjoint halves, and only ONE of them a ratio knob can reach.
//
//   * AMBIGUOUS  (0 < lc < mid) -- "Mark"/"mark", "Rose"/"rose", "Will"/"will". These are what a
//     `mid > lc * K` threshold moves.
//   * PURE NAME  (lc == 0)      -- "Bronx", "Lily". These never appear lowercase, so they are not in
//     lower_count at all and derive_attested never even considers them. **No value of K reaches
//     them.** Collapsing these requires changing the RULE (always collapse a Capitalized word),
//     not tuning its threshold.
//
// If pure names dominate, tuning the threshold is theatre and the real lever is the rule itself. That
// is a cheap thing to know before writing either change, and it needs only corpus pass 1 -- no BPE
// learn, no training.
//
// Hidden ([.casing]): reads a multi-hundred-MB real corpus from the source tree, so it is neither fast
// nor available on every checkout. Run explicitly:
//     sub0_frontend_tests "[.casing]" --reporter compact

#include <catch2/catch_test_macros.hpp>

#include "sub0/casing.hpp"
#include "sub0/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef SUB0_SOURCE_DIR

namespace {

// Read up to `max_mb` from `path`, truncated at the last newline so a word unit is never split
// (the same alignment invariant Scan's chunking relies on).
std::string read_slice(const std::filesystem::path& path, std::size_t max_mb) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::string buf(max_mb * 1024u * 1024u, '\0');
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    buf.resize(static_cast<std::size_t>(in.gcount()));
    const std::size_t nl = buf.rfind('\n');
    if (nl != std::string::npos) buf.resize(nl + 1);
    return buf;
}

struct Bucket {
    long long forms = 0;        // distinct word forms
    long long occurrences = 0;  // capitalised occurrences they account for
};

// The corpora a production tokenizer is actually learned over. minipile carries the CODE, and code is
// where this policy could plausibly behave differently from prose: ALL_CAPS constants (MAX_SIZE),
// CamelCase identifiers (HttpClient), and identifiers that recur often enough that a verbatim merge
// would pay for itself. Measuring only prose would average that away -- hence a per-corpus breakdown
// AND a combined run, since the combined stream is what the vocabulary is really fitted to.
struct Source { const char* label; const char* file; };
constexpr Source kSources[] = {
    {"cosmopedia (prose)", "cosmopedia.txt"},
    {"minipile (code+web)", "minipile.txt"},
    {"fineweb_edu (web)", "fineweb_edu.txt"},
};

}  // namespace

namespace {

// Measurement 1 for one text: partition every form that ever appears capitalised mid-sentence, and
// report how much of the withheld population a `mid > lc*K` threshold could possibly reach.
std::string reachability_report(const char* label, const std::string& text) {
    sub0::tok::Scan s;
    s.add_names(text);                       // pass 1 only: fills lower_count / midcap_count

    // Partition every form that EVER appears capitalised mid-sentence.
    Bucket collapsed, ambiguous, pure;
    for (const auto& [w, mid] : s.midcap_count) {
        if (mid <= 0) continue;
        const auto it = s.lower_count.find(w);
        const long long lc = (it == s.lower_count.end()) ? 0 : it->second;
        Bucket& b = (lc == 0) ? pure : (mid > lc ? ambiguous : collapsed);
        ++b.forms;
        b.occurrences += mid;
    }

    const long long withheld_forms = ambiguous.forms + pure.forms;
    const long long withheld_occ   = ambiguous.occurrences + pure.occurrences;
    if (withheld_occ == 0) return std::string("\n=== ") + label + ": no capitalised population ===\n";

    const double pure_form_share = 100.0 * static_cast<double>(pure.forms) / static_cast<double>(withheld_forms);
    const double pure_occ_share  = 100.0 * static_cast<double>(pure.occurrences) / static_cast<double>(withheld_occ);

    std::ostringstream r;
    r << "\n=== " << label << ", " << (text.size() / (1024 * 1024)) << " MB ===\n";
    auto row = [&](const char* name, const Bucket& b) {
        r << "  " << name << "  forms=" << b.forms << "  capitalised-occurrences=" << b.occurrences << "\n";
    };
    row("COLLAPSED (attested, mid<=lc) ", collapsed);
    row("WITHHELD ambiguous (0<lc<mid) ", ambiguous);
    row("WITHHELD pure name  (lc==0)   ", pure);
    r << "  -> pure names are " << pure_form_share << "% of withheld FORMS, "
      << pure_occ_share << "% of withheld OCCURRENCES\n"
      << "     (a `mid > lc*K` threshold CANNOT reach these -- only the rule change can)\n";

    // What a threshold sweep would actually buy: forms/occurrences moved out of `withheld` as K grows.
    // K=1 is today's rule. K=inf still leaves every pure name behind, which is the point.
    r << "  threshold sweep (ambiguous half only):\n";
    for (const double K : {1.0, 2.0, 4.0, 8.0, 1e9}) {
        long long moved_forms = 0, moved_occ = 0;
        for (const auto& [w, mid] : s.midcap_count) {
            const auto it = s.lower_count.find(w);
            if (it == s.lower_count.end()) continue;             // pure name: unreachable, skip
            const long long lc = it->second;
            if (mid > lc && !(static_cast<double>(mid) > static_cast<double>(lc) * K)) {
                ++moved_forms;
                moved_occ += mid;
            }
        }
        const double occ_recovered = 100.0 * static_cast<double>(moved_occ) / static_cast<double>(withheld_occ);
        r << "    K=" << K << "  collapses " << moved_forms << " more forms, "
          << moved_occ << " occurrences (" << occ_recovered << "% of all withheld)\n";
    }
    return r.str();
}

}  // namespace

TEST_CASE("casing policy: how much of the withheld-name population is even reachable by a threshold",
          "[.casing]") {
    std::string all, report;
    int found = 0;
    for (const Source& src : kSources) {
        const std::filesystem::path p = std::filesystem::path(SUB0_SOURCE_DIR) / "data" / src.file;
        if (!std::filesystem::exists(p)) { report += std::string("\n(missing: ") + src.file + ")\n"; continue; }
        const std::string text = read_slice(p, 128);
        if (text.size() < 1024u * 1024u) continue;
        ++found;
        report += reachability_report(src.label, text);
        all += text;
    }
    REQUIRE(found > 0);
    // The combined stream is the one the production vocabulary is actually fitted to; the per-corpus
    // rows above only exist to show whether code behaves differently from prose.
    if (found > 1) report += reachability_report("COMBINED (what the tokenizer learns on)", all);
    WARN(report);
    CHECK(!all.empty());
}

// Having established that a threshold cannot reach 92.4% of the withheld forms, this is the A/B for
// the RULE change itself: keep names verbatim (today) vs always collapse a Capitalized/ALL-UPPER word
// to a marker + its lowercase form.
//
// Collapsing does NOT lose the name/word distinction -- `<|cap|>bronx` and `bronx` remain different
// token sequences, so the information is factored out, not discarded. What changes is WHERE a name's
// spelling comes from: the shared, densely-trained lowercase piece inventory instead of a duplicate
// cased one that only capitalised words ever use.
//
// Three pillars, per standing policy -- the win must show up as vocabulary AND token count, not as one
// number in isolation. A rule that halves cased pieces but inflates the stream is not a win.
// Learned over the BLEND, not one corpus: a production tokenizer sees prose and code together, and the
// interesting risk is code-specific -- ALL_CAPS constants, CamelCase identifiers, and identifiers that
// recur often enough that a verbatim cased merge might pay for itself. Held-out text is drawn from each
// source in the same proportion so the token-cost number is not silently a prose-only measurement.
TEST_CASE("casing policy A/B: verbatim names vs always-collapse (vocab composition + token cost)",
          "[.casing]") {
    std::string text, heldout;
    std::vector<std::string> present;
    for (const Source& src : kSources) {
        const std::filesystem::path p = std::filesystem::path(SUB0_SOURCE_DIR) / "data" / src.file;
        if (!std::filesystem::exists(p)) continue;
        const std::string s = read_slice(p, 32);            // learn slice, per source
        if (s.size() < 1024u * 1024u) continue;
        text += s;
        present.emplace_back(src.label);
        // Held out from a disjoint region of the SAME source (the tail of a learn slice is not held out).
        std::ifstream in(p, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(96) * 1024 * 1024);
        std::string buf(4u * 1024u * 1024u, '\0');
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        buf.resize(static_cast<std::size_t>(in.gcount()));
        const std::size_t lo = buf.find('\n'), hi = buf.rfind('\n');
        if (lo != std::string::npos && hi != std::string::npos && hi > lo) heldout += buf.substr(lo + 1, hi - lo);
    }
    REQUIRE(text.size() > 1024u * 1024u);
    REQUIRE(heldout.size() > 1024u * 1024u);

    sub0::tok::LearnOptions opts;
    opts.vocab_target = 16384;          // the production scale where the 9.1% cased figure was measured

    // Build one tokenizer under a given attested set and report its composition + held-out cost.
    struct Result { long long cased_pieces = 0, pieces = 0; long long tokens = 0; };
    auto build = [&](const std::unordered_set<std::string>& attested) {
        sub0::tok::Scan s;
        s.add_names(text);
        s.add_words(text, attested);
        const sub0::tok::Tokenizer t = sub0::tok::learn(s, attested, opts);
        Result r;
        for (const auto& [piece, id] : t.piece_index) {
            if (id < t.n_base) continue;               // base bytes/markers are not learned pieces
            ++r.pieces;
            if (std::any_of(piece.begin(), piece.end(),
                            [](char c) { return sub0::casing::is_upper(static_cast<unsigned char>(c)); }))
                ++r.cased_pieces;
        }
        r.tokens = static_cast<long long>(sub0::tok::encode(t, heldout).size());
        return r;
    };

    sub0::tok::Scan pass1;
    pass1.add_names(text);

    const std::unordered_set<std::string> attested_today = sub0::tok::derive_attested(pass1);

    // Always-collapse, simulated without touching production code: every lowercase form that any
    // capitalised word could reduce to is declared attested, so emit_word()'s `attested.contains(lw)`
    // is true for all of them.
    std::unordered_set<std::string> attested_all = attested_today;
    for (const auto& [w, mid] : pass1.midcap_count) { (void)mid; attested_all.insert(w); }
    for (const auto& [w, lc] : pass1.lower_count)   { (void)lc;  attested_all.insert(w); }

    const Result a = build(attested_today);
    const Result b = build(attested_all);

    auto pct = [](long long num, long long den) {
        return den > 0 ? 100.0 * static_cast<double>(num) / static_cast<double>(den) : 0.0;
    };
    std::ostringstream r;
    r << "\n=== name policy A/B, " << (text.size() / (1024 * 1024)) << " MB learn / "
      << (heldout.size() / (1024 * 1024)) << " MB held out, vocab_target " << opts.vocab_target << " ===\n"
      << "  sources:";
    for (const std::string& p : present) r << " [" << p << "]";
    r << "\n"
      << "  verbatim names (today) : pieces=" << a.pieces << "  cased=" << a.cased_pieces
      << " (" << pct(a.cased_pieces, a.pieces) << "%)  held-out tokens=" << a.tokens << "\n"
      << "  always-collapse        : pieces=" << b.pieces << "  cased=" << b.cased_pieces
      << " (" << pct(b.cased_pieces, b.pieces) << "%)  held-out tokens=" << b.tokens << "\n"
      << "  delta: cased pieces " << (pct(b.cased_pieces, b.pieces) - pct(a.cased_pieces, a.pieces))
      << " pp,  token count " << pct(b.tokens - a.tokens, a.tokens) << "%\n";
    WARN(r.str());

    CHECK(a.pieces > 0);
    CHECK(b.pieces > 0);

    // THE ARMS MUST DIFFER, or this harness is measuring nothing.
    //
    // It simulates the old policy by handing emit_word a different `attested` set. That only works
    // while emit_word still CONSULTS attested. Once schemeV5 made collapse unconditional, both arms
    // silently became the same policy and this case reported cased=0 vs cased=0, delta 0% -- a
    // confident-looking null result that was pure artifact.
    //
    // So: to re-run the A/B, the `collapse` line in casing.hpp emit_word must be temporarily restored
    // to `force_collapse || attested.contains(lw)`. This assertion is what tells you that, instead of
    // letting a vacuous zero pass for a measurement.
    const bool arms_identical = (a.cased_pieces == b.cased_pieces) && (a.tokens == b.tokens);
    INFO("Both arms produced an IDENTICAL tokenizer, so this A/B measured nothing. Expected under the "
         "shipped schemeV5 policy: emit_word no longer consults `attested`, which is the only lever "
         "this harness has. To re-measure, temporarily restore casing.hpp's emit_word to "
         "`collapse = force_collapse || attested.contains(lw)`, run, then put it back. Failing here "
         "deliberately -- a vacuous 0-vs-0 delta once passed for a result.");
    REQUIRE_FALSE(arms_identical);
}

#endif  // SUB0_SOURCE_DIR
