// sub0llm-tutorspike -- builds the SPLICED corpus docs/TUTOR.md's experiment is defined against, plus
// the manifest that labels which population each document came from.
//
// SPIKE TOOL. Retires with the spike (AGENTS.md 11): nothing in the engine depends on it, and the
// artefacts it writes are plain data the ordinary configure->train path consumes with no special case.
//
// WHY A SPLICED CORPUS RATHER THAN A SYNTHETIC ONE
//
// The mastery surface's central claim is that learning VELOCITY separates "not yet learnt" from
// "unlearnable", where a LEVEL-based rule cannot. Testing that needs a corpus containing all three
// states at once, with the answer known in advance so the result is falsifiable:
//
//   tinystories  -- simple, templated, low-entropy. Should master FAST: level falls, then velocity -> 0.
//   cosmopedia   -- substantially more complex real prose. Should retain velocity far longer.
//   shuffled     -- the unlearnable slice. See below.
//
// Two of the three are real text, so the result speaks to real training rather than to an artefact of
// generated data, and the ordering prediction (cosmopedia outlasts tinystories) is registered before
// the run rather than read off it afterwards.
//
// WHY THE GARBAGE SLICE IS SHUFFLED WORDS, NOT RANDOM BYTES
//
// Random bytes are unlearnable in an uninteresting way -- they miss the vocabulary entirely, so a high
// score could just mean "out of distribution". Shuffling the WORDS of real tinystories text holds the
// vocabulary and the unigram distribution fixed and destroys only the sequential structure. That makes
// it the sharp test: the model CAN still learn the unigram statistics, so this slice's loss falls a
// little early and then stops -- high level, velocity -> 0 -- which is precisely the state a level-based
// rule mistakes for "still has information to give" and up-weights forever.
//
// SAMPLING IS DISTRIBUTED, NOT A PREFIX
//
// Documents are drawn by seeking to random offsets across the WHOLE source file and taking the next
// complete document, never by reading a prefix. Same reasoning include/sub0/window.hpp gives for
// doc_in_subset: these corpora are not shuffled, so their first N bytes are whatever the crawl or the
// generator happened to order first, not N bytes of the distribution.
//
// THE DOCUMENT BOUNDARY IS <|endoftext|>, NOT A BLANK LINE
//
// Load-bearing, and checked against the pipeline rather than assumed. src/tokenizer.cpp's
// scan_doc_boundaries recognises exactly ONE boundary: the TOK_EOS token. Blank lines are ordinary
// paragraph breaks INSIDE a document -- all three sources here contain them freely. So:
//
//   * splitting the sources on blank lines would cut documents mid-story and, worse, let one sampled
//     "document" span a real boundary -- exactly the cross-document contamination window.hpp exists to
//     prevent;
//   * emitting documents separated by blank lines would leave the configurator seeing ONE document,
//     and every ordinal in the manifest would be meaningless;
//   * an <|endoftext|> surviving inside a document body would introduce a spurious boundary, which in
//     the shuffled population is not hypothetical -- shuffling scatters the marker into the middle of
//     the text, and the first version of this tool did exactly that.
//
// Hence: split on the marker, strip any occurrence from a document body, and re-emit the marker as the
// separator. Then document i in this file is document i in corpus.tok, which is the join the manifest
// depends on.

#include "sub0/log.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string_view>
#include <string>
#include <vector>

namespace {

// splitmix64, as used by window.hpp's doc_in_subset -- avalanches well enough that neighbouring seeds
// give independent streams.
std::uint64_t mix64(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

constexpr const char* kEosMarker = "<|endoftext|>";

// True when a line IS the document separator (the marker alone on its line, CRLF-tolerant).
bool is_eos_line(const std::string& line) {
    std::string_view v{ line };
    if (!v.empty() && v.back() == '\r') v.remove_suffix(1);
    return v == kEosMarker;
}

// One document read from `f` at a random offset: seek, discard the partial document we landed inside,
// then take everything up to the next <|endoftext|> line. Returns empty on a short/degenerate read,
// which the caller retries -- rejection sampling is fine here because failures are rare and this runs
// once. Any marker occurring INSIDE the body (not alone on its line) is stripped: it would otherwise
// become a spurious document boundary once the configurator tokenizes this file.
std::string sample_document(std::ifstream& f, std::uintmax_t size, std::mt19937_64& rng,
                            std::size_t min_bytes, std::size_t max_bytes) {
    if (size < 4096) return {};
    std::uniform_int_distribution<std::uintmax_t> off(0, size - 4096);
    f.clear();
    f.seekg(static_cast<std::streamoff>(off(rng)));

    std::string line, doc;
    bool at_boundary = false;
    // Discard up to the first boundary: we landed mid-document, and its head is missing.
    while (std::getline(f, line)) {
        if (is_eos_line(line)) { at_boundary = true; break; }
    }
    if (!at_boundary) return {};
    while (std::getline(f, line)) {
        if (is_eos_line(line)) break;                      // end of this document
        if (!doc.empty()) doc += '\n';
        doc += line;
        if (doc.size() > max_bytes) break;                 // long document: take a prefix, still one doc
    }
    // Strip any embedded marker (a source line like "text <|endoftext|> more") -- rare, but one
    // survivor silently splits a document in two and desynchronises every ordinal after it.
    for (std::size_t p = doc.find(kEosMarker); p != std::string::npos; p = doc.find(kEosMarker, p))
        doc.erase(p, std::char_traits<char>::length(kEosMarker));
    if (doc.size() < min_bytes) return {};
    return doc;
}

// Whitespace-split, shuffle, rejoin. Holds the vocabulary and unigram distribution of `src` and destroys
// its sequential structure -- see the header comment on why that is the sharp form of "unlearnable".
std::string shuffle_words(const std::string& src, std::mt19937_64& rng) {
    std::vector<std::string> words;
    std::string w;
    for (const char c : src) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!w.empty()) { words.push_back(w); w.clear(); }
        } else {
            w += c;
        }
    }
    if (!w.empty()) words.push_back(w);
    std::shuffle(words.begin(), words.end(), rng);
    std::string out;
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (i) out += ' ';
        out += words[i];
    }
    return out;
}

struct Population {
    std::string name;
    int         want = 0;      // documents requested
    int         got  = 0;      // documents actually written
};

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"sub0-tutorspike: splice a 3-population corpus + its population manifest (docs/TUTOR.md)"};

    std::string tinystories = "data/tinystories.txt";
    std::string cosmopedia  = "data/cosmopedia.txt";
    std::string out_corpus  = "data/tutorspike_corpus.txt";
    std::string out_manifest = "data/tutorspike_manifest.json";
    int    n_simple = 6000, n_complex = 2000, n_garbage = 1000;
    int    chunk    = 64;
    std::uint64_t seed = 20260730;
    std::size_t min_bytes = 256, max_bytes = 8192;

    app.add_option("--tinystories", tinystories, "Simple/low-entropy source")->check(CLI::ExistingFile);
    app.add_option("--cosmopedia", cosmopedia, "Complex source")->check(CLI::ExistingFile);
    app.add_option("--out", out_corpus, "Spliced corpus path (feed this to sub0llm-configure)");
    app.add_option("--manifest", out_manifest, "Population manifest path (JSON)");
    app.add_option("--n-simple", n_simple, "Documents drawn from the simple source");
    app.add_option("--n-complex", n_complex, "Documents drawn from the complex source");
    app.add_option("--n-garbage", n_garbage, "Word-shuffled (unlearnable) documents");
    app.add_option("--chunk", chunk,
                   "Documents per population run in the interleave -- keeps document ORDINAL from "
                   "coinciding with population, so nothing position-dependent can masquerade as a "
                   "population effect");
    app.add_option("--seed", seed, "Sampling seed (the splice is fully reproducible from it)");
    app.add_option("--min-doc-bytes", min_bytes, "Reject sampled documents shorter than this");
    app.add_option("--max-doc-bytes", max_bytes, "Truncate sampled documents longer than this");
    CLI11_PARSE(app, argc, argv);

    if (chunk < 1) { sub0::log::error("tutorspike: --chunk must be >= 1"); return 1; }

    std::ifstream fs(tinystories, std::ios::binary);
    std::ifstream fc(cosmopedia,  std::ios::binary);
    if (!fs || !fc) { sub0::log::error("tutorspike: cannot open a source corpus"); return 1; }
    const std::uintmax_t size_s = std::filesystem::file_size(tinystories);
    const std::uintmax_t size_c = std::filesystem::file_size(cosmopedia);

    std::ofstream out(out_corpus, std::ios::binary);
    if (!out) { sub0::log::error("tutorspike: cannot write '{}'", out_corpus); return 1; }

    // Population 2 (shuffled) is derived from FRESH tinystories draws rather than reusing the ones
    // written as population 0 -- otherwise the garbage slice would be a word-shuffled copy of documents
    // the model also sees intact, and any transfer between the two would confound the reading.
    std::vector<Population> pops = { {"tinystories", n_simple, 0},
                                     {"cosmopedia",  n_complex, 0},
                                     {"shuffled",    n_garbage, 0} };
    std::mt19937_64 rng(mix64(seed));

    // Round-robin in runs of `chunk`, so the file interleaves populations instead of blocking them.
    struct Run { int pop; int count; };
    std::vector<Run> runs;
    int total_docs = 0;
    std::vector<int> remaining{ n_simple, n_complex, n_garbage };
    std::string doc;
    while (remaining[0] > 0 || remaining[1] > 0 || remaining[2] > 0) {
        for (int p = 0; p < 3; ++p) {
            if (remaining[static_cast<std::size_t>(p)] <= 0) continue;
            const int take = std::min(chunk, remaining[static_cast<std::size_t>(p)]);
            int written = 0;
            for (int k = 0; k < take; ++k) {
                int attempts = 0;
                doc.clear();
                while (doc.empty() && attempts++ < 64) {
                    if (p == 0)      doc = sample_document(fs, size_s, rng, min_bytes, max_bytes);
                    else if (p == 1) doc = sample_document(fc, size_c, rng, min_bytes, max_bytes);
                    else {
                        const std::string src = sample_document(fs, size_s, rng, min_bytes, max_bytes);
                        if (!src.empty()) doc = shuffle_words(src, rng);
                    }
                }
                if (doc.empty()) continue;              // source exhausted of usable documents
                // The marker on its own line is the ONLY boundary scan_doc_boundaries recognises.
                out << doc << "\n" << kEosMarker << "\n";
                ++written;
                ++pops[static_cast<std::size_t>(p)].got;
            }
            remaining[static_cast<std::size_t>(p)] -= take;
            if (written > 0) {
                // Merge with the previous run when it is the same population (a short/failed draw can
                // otherwise split one logical run in two).
                if (!runs.empty() && runs.back().pop == p) runs.back().count += written;
                else runs.push_back(Run{ p, written });
                total_docs += written;
            }
        }
    }
    out.close();

    // The manifest maps DOCUMENT ORDINAL -> population, as run-lengths. Ordinal is the only join key
    // available: the configurator tokenizes this file into corpus.tok and records document starts in the
    // same order it read them, so document i here is document i there. `total_docs` exists so a consumer
    // can VERIFY that assumption instead of trusting it -- if the configurator ever drops or merges a
    // document, the counts disagree and the mismatch is loud rather than a silently mislabelled surface.
    std::ofstream mf(out_manifest);
    if (!mf) { sub0::log::error("tutorspike: cannot write '{}'", out_manifest); return 1; }
    mf << "{\n  \"corpus\": \"" << out_corpus << "\",\n";
    mf << "  \"seed\": " << seed << ",\n";
    mf << "  \"total_docs\": " << total_docs << ",\n";
    mf << "  \"populations\": [";
    for (std::size_t i = 0; i < pops.size(); ++i)
        mf << (i ? ", " : "") << '"' << pops[i].name << '"';
    mf << "],\n";
    mf << "  \"runs\": [";
    for (std::size_t i = 0; i < runs.size(); ++i)
        mf << (i ? ", " : "") << '[' << runs[i].pop << ", " << runs[i].count << ']';
    mf << "]\n}\n";
    mf.close();

    sub0::log::line("tutorspike: wrote {} ({} documents, {} runs)", out_corpus, total_docs, runs.size());
    for (const Population& p : pops)
        sub0::log::line("  {:<12} {} documents{}", p.name, p.got,
                        p.got < p.want ? " (SHORT -- source ran out of usable draws)" : "");
    sub0::log::line("tutorspike: manifest -> {}", out_manifest);
    sub0::log::line("next: sub0llm-configure --corpus {}", out_corpus);
    return 0;
}
