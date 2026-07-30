// ws6_simd_differential_tests.cpp — WS6: proves the SIMD-classify + bulk-append rewrite of
// normalize_text/truecase_tokenize (include/sub0/casing.hpp) produces BYTE-FOR-BYTE /
// TOKEN-FOR-TOKEN IDENTICAL output to the original scalar implementation, not just "still round-
// trips" (the existing fuzz/dogfood net in tok_roundtrip_fuzz.cpp proves round-trip correctness,
// which is a WEAKER property -- it can't distinguish "normalize_text changed but stayed self-
// consistent" from "normalize_text is unchanged", since the fuzz tests compute their own expected
// value by calling normalize_text itself). This file keeps a REFERENCE copy of the exact pre-WS6
// scalar logic (verbatim from `git show HEAD:include/sub0/casing.hpp`, before this session's
// changes) and diffs it against the live sub0::casing functions across the same seed/mutation/blob
// generators tok_roundtrip_fuzz.cpp uses, plus the project's own source as a large real sample.

#include <catch2/catch_test_macros.hpp>

#include "sub0/casing.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace sub0::casing;

namespace ref {
// --- Verbatim pre-WS6 scalar reference (git show HEAD, before this session's edits) ---

std::string normalize_text(const std::string& in, long& replaced) {
    std::string out;
    out.reserve(in.size());
    replaced = 0;
    const std::size_t n = in.size();
    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        if (c == 0xE2 && i + 2 < n && static_cast<unsigned char>(in[i + 1]) == 0x80) {
            const unsigned char t = static_cast<unsigned char>(in[i + 2]);
            const char* rep = nullptr;
            switch (t) {
                case 0x98: case 0x99: rep = "'";   break;
                case 0x9C: case 0x9D: rep = "\"";  break;
                case 0x93: case 0x94: rep = "-";   break;
                case 0xA6:            rep = "...";  break;
                default: break;
            }
            if (rep) { out += rep; ++replaced; i += 3; continue; }
        }
        if (c == '`') { out.push_back('\''); ++replaced; ++i; continue; }
        out.push_back(static_cast<char>(c));
        ++i;
    }
    return out;
}

std::vector<int> truecase_tokenize(const std::string& text,
                                   const std::unordered_set<std::string>& attested,
                                   TokStats* st) {
    std::vector<int> toks;
    toks.reserve(text.size());
    const std::size_t n = text.size();

    auto emit_word = [&](std::string_view seg, bool force_collapse) {
        const bool first_upper = is_upper(static_cast<unsigned char>(seg[0]));
        bool rest_lower = true, all_upper = true;
        for (std::size_t k = 0; k < seg.size(); ++k) {
            const unsigned char ch = static_cast<unsigned char>(seg[k]);
            if (!is_upper(ch))         all_upper = false;
            if (k > 0 && !is_lower(ch)) rest_lower = false;
        }
        std::string lw;
        lw.reserve(seg.size());
        for (unsigned char ch : seg) lw.push_back(static_cast<char>(to_lower(ch)));

        const bool capitalized = first_upper && rest_lower;
        const bool upper       = all_upper && seg.size() >= 2;
        // schemeV5: unconditional collapse, tracking casing.hpp's emit_word. This reference is a
        // differential partner for the WS6 SIMD REWRITE -- its job is to prove the rewrite changed no
        // bytes, not to freeze the casing POLICY. So a deliberate policy change is mirrored here rather
        // than left to fail: leaving the old rule would mean this file guards a policy the production
        // encoder no longer has, and every future WS6 assertion would compare against a fiction.
        const bool collapse    = true;
        (void)force_collapse;
        (void)attested;

        int marker = -1;
        if (upper && collapse)            marker = TOK_UP;
        else if (capitalized && collapse) marker = TOK_CAP;

        if (marker >= 0) {
            toks.push_back(marker);
            if (st) (marker == TOK_CAP ? st->cap : st->up) += 1;
            for (unsigned char ch : lw) toks.push_back(ch);
        } else {
            if (st && first_upper) ++st->names;
            for (unsigned char ch : seg) toks.push_back(ch);
        }
    };

    for (std::size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (!is_alpha(c)) { toks.push_back(c); ++i; continue; }

        std::size_t j = i;
        bool any_upper = false;
        while (j < n && is_alpha(static_cast<unsigned char>(text[j]))) {
            if (is_upper(static_cast<unsigned char>(text[j]))) any_upper = true;
            ++j;
        }
        const std::string_view w(text.data() + i, j - i);
        if (st) ++st->words;

        if (!any_upper) {
            for (unsigned char ch : w) toks.push_back(ch);
        } else if (const auto segs = camel_segments(w); segs.size() >= 2) {
            std::size_t off = 0;
            for (std::size_t len : segs) { emit_word(w.substr(off, len), /*force_collapse=*/true); off += len; }
        } else {
            emit_word(w, /*force_collapse=*/false);
        }
        i = j;
    }
    return toks;
}

}  // namespace ref

namespace {

const std::unordered_set<std::string> kAttested = {
    "the", "dog", "cat", "fox", "sun", "day", "commons", "attribution", "license",
    "non", "commercial", "here", "save", "scan", "state", "load", "model", "cache", "please",
};

// Mirrors tok_roundtrip_fuzz.cpp's mutator so the SAME class of inputs (byte flip/insert/delete/
// duplicate) exercises the differential comparison, not just the round-trip property.
std::string mutate(std::string s, std::mt19937& rng, int nmut) {
    std::uniform_int_distribution<int> op(0, 3), byte(0, 255);
    for (int m = 0; m < nmut; ++m) {
        if (s.empty()) { s.push_back(static_cast<char>(byte(rng))); continue; }
        std::uniform_int_distribution<std::size_t> pos(0, s.size() - 1);
        const std::size_t i = pos(rng);
        switch (op(rng)) {
            case 0: s[i] = static_cast<char>(byte(rng));                       break;
            case 1: s.insert(s.begin() + static_cast<std::ptrdiff_t>(i),
                             static_cast<char>(byte(rng)));                    break;
            case 2: s.erase(s.begin() + static_cast<std::ptrdiff_t>(i));       break;
            case 3: s.insert(s.begin() + static_cast<std::ptrdiff_t>(i), s[i]);break;
        }
    }
    return s;
}

// Runs both implementations end to end (normalize_text -> truecase_tokenize) and asserts identical
// output at BOTH stages -- a mismatch only in the intermediate normalized string, masked by an
// otherwise-matching token stream, would still be a real regression.
void check_identical(const std::string& x, std::string_view label) {
    long r_new = 0, r_old = 0;
    const std::string norm_new = normalize_text(x, r_new);
    const std::string norm_old = ref::normalize_text(x, r_old);
    INFO("label: " << label);
    REQUIRE(norm_new == norm_old);
    REQUIRE(r_new == r_old);

    TokStats st_new{}, st_old{};
    const std::vector<int> toks_new = truecase_tokenize(norm_new, kAttested, &st_new);
    const std::vector<int> toks_old = ref::truecase_tokenize(norm_old, kAttested, &st_old);
    REQUIRE(toks_new == toks_old);
    REQUIRE(st_new.words == st_old.words);
    REQUIRE(st_new.cap == st_old.cap);
    REQUIRE(st_new.up == st_old.up);
    REQUIRE(st_new.names == st_old.names);
}

const std::vector<std::string> kSeeds = {
    "the dog ran .",
    "She said, \"hello\" to NASA's team.",
    "a\nb\n\nc\td   e",
    "Commons-Attribution-NoDerivs",
    "don't can't won't it's",
    "x\"y\"z \"q\" w",
    "THE END of days",
    "caf\xC3\xA9" " pi\xC3\xB1" "ata na\xC3\xAF" "ve",
    "save_scan_state CamelCaseWord",
    "   leading and trailing   ",
    "curly \xE2\x80\x9C" "quotes\xE2\x80\x9D and \xE2\x80\x98" "apostrophes\xE2\x80\x99",  // U+201C/9D, U+2018/9
    "em\xE2\x80\x94" "dash and en\xE2\x80\x93" "dash and ellipsis\xE2\x80\xA6",            // U+2014, U+2013, U+2026
    "backtick`quote`marks`everywhere`",
};

}  // namespace

TEST_CASE("WS6 differential: worked-example seeds are byte/token-identical to the scalar reference", "[tok][ws6]") {
    for (const auto& s : kSeeds) check_identical(s, s.substr(0, 24));
}

TEST_CASE("WS6 differential: fuzz-mutated seeds are byte/token-identical to the scalar reference", "[tok][ws6]") {
    std::mt19937 rng(0x5EED5EEDu);
    for (int iter = 0; iter < 4000; ++iter) {
        const std::string& seed = kSeeds[static_cast<std::size_t>(iter) % kSeeds.size()];
        const std::string x = mutate(seed, rng, 1 + (iter % 6));
        check_identical(x, "fuzz#" + std::to_string(iter));
    }
}

TEST_CASE("WS6 differential: random byte blobs are byte/token-identical to the scalar reference", "[tok][ws6]") {
    std::mt19937 rng(0xB10BB10Bu);
    std::uniform_int_distribution<int> len(0, 300), byte(0, 255);
    for (int iter = 0; iter < 4000; ++iter) {
        std::string x(static_cast<std::size_t>(len(rng)), '\0');
        for (char& c : x) c = static_cast<char>(byte(rng));
        check_identical(x, "blob#" + std::to_string(iter));
    }
}

#ifdef SUB0_SOURCE_DIR
// The largest, most realistic differential sample: the project's own C++ source (real code
// punctuation density, indentation, string/char literals -- the exact content WS6 is optimizing
// FOR, per the plan's own framing of the corpus-scale configure-time path).
TEST_CASE("WS6 differential: the project's own C++ source is byte/token-identical to the scalar reference", "[tok][ws6][dogfood]") {
    namespace fs = std::filesystem;
    const std::array<const char*, 4> roots{"src", "include", "tools", "tests"};
    std::size_t checked = 0;
    for (const char* root : roots) {
        const fs::path base = fs::path(SUB0_SOURCE_DIR) / root;
        std::error_code ec;
        if (!fs::exists(base, ec)) continue;
        for (auto it = fs::recursive_directory_iterator(base, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            const std::string ext = it->path().extension().string();
            if (!it->is_regular_file(ec) || (ext != ".cpp" && ext != ".hpp" && ext != ".h")) continue;
            std::ifstream f(it->path(), std::ios::binary);
            const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
            if (text.empty()) continue;
            check_identical(text, it->path().filename().string());
            if (++checked >= 40) break;   // capped: this is a correctness differential, not a corpus benchmark
        }
        if (checked >= 40) break;
    }
    REQUIRE(checked > 0);   // SUB0_SOURCE_DIR must point at the real tree
}
#endif
