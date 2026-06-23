#include "sub0llm/tokenizer/bpe.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace sub0llm {

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace {

// Split text into "words" on whitespace boundaries, prepending a space
// marker Ġ (U+0120) to every word that follows a space.  This matches the
// GPT-2 convention where "hello" and " hello" are different tokens.
std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> words;
    bool at_start = true;
    std::string cur;

    for (auto raw_c : text) {
        const auto c = static_cast<unsigned char>(raw_c);
        const bool is_ws = (c == ' ' || c == '\n' || c == '\t' || c == '\r');
        if (is_ws) {
            if (!cur.empty()) { words.push_back(std::move(cur)); cur.clear(); }
            at_start = false;
        } else {
            if (!at_start && cur.empty()) {
                // Prepend the GPT-2 space marker: UTF-8 encoding of U+0120
                cur += '\xC4';
                cur += '\xA0';
            }
            cur += static_cast<char>(c);
        }
    }
    if (!cur.empty()) words.push_back(std::move(cur));
    return words;
}

// Represent a word as individual Unicode code-point tokens (not raw bytes).
// Keeping multi-byte UTF-8 sequences together ensures every token is a
// valid UTF-8 string, which is required for JSON serialisation.
std::vector<std::string> word_to_chars(std::string_view word) {
    std::vector<std::string> chars;
    const auto* p   = reinterpret_cast<const unsigned char*>(word.data());
    const auto* end = p + word.size();

    while (p < end) {
        int len;
        if      (*p < 0x80) len = 1;
        else if (*p < 0xC0) len = 1;   // unexpected continuation — treat as 1
        else if (*p < 0xE0) len = 2;
        else if (*p < 0xF0) len = 3;
        else                len = 4;

        len = std::min(len, static_cast<int>(end - p));
        chars.emplace_back(reinterpret_cast<const char*>(p),
                           static_cast<std::size_t>(len));
        p += len;
    }
    return chars;
}

// TODO(tokenizer-normalization, later): two orthographic-noise sources still split a word's statistics
// across tokens — (1) curly vs straight quotes (U+2019 ’ vs ' → "Billy’s" and "Billy's" are distinct
// ids); (2) TRAILING apostrophes from closing quotes ("Monster'", "Nibbles'") that this splitter keeps
// welded (the rule deliberately keeps th'/dogs'). A normalization pass (NFKC-style ’→', drop ambiguous
// trailing quotes) is the quick win but it BREAKS the lossless decode∘encode invariant this file
// guarantees — so it's a deliberate design change, not a free fix. The deeper lever is FACTORED tokens:
// one id per lemma + separable attribute axes (case / number / possession / tense) collapsed during
// learning — see the `tokenizer-normalization-and-factored-tokens` memory. Don't fold either into a
// model mid-training (it changes the vocab → requires a retrain).
//
// Split raw text into word-level tokens for word_level(): a "word" is a maximal run of
// ASCII letters and apostrophes (ASCII ' or the curly U+2019, so "I'll"/"th'" stay whole);
// a "number" is a run of ASCII digits; every whitespace char and every other code point
// (punctuation, non-ASCII) is its own token. Lossless — concatenating the tokens reproduces
// the text exactly — so it round-trips and preserves verse/line structure.
std::vector<std::string> split_words(std::string_view text) {
    static const std::string apostrophe = "\xE2\x80\x99";  // U+2019 RIGHT SINGLE QUOTATION
    // Decode the single UTF-8 code point in `c` (word_to_chars yields one code point per element).
    auto cp_value = [](const std::string& c) -> std::uint32_t {
        const auto b0 = static_cast<unsigned char>(c[0]);
        if (b0 < 0x80) return b0;
        std::uint32_t u; int n;
        if (b0 < 0xE0) { u = b0 & 0x1Fu; n = 1; }
        else if (b0 < 0xF0) { u = b0 & 0x0Fu; n = 2; }
        else { u = b0 & 0x07u; n = 3; }
        for (int i = 1; i <= n && i < static_cast<int>(c.size()); ++i)
            u = (u << 6) | (static_cast<unsigned char>(c[i]) & 0x3Fu);
        return u;
    };
    // A "letter" is ASCII alpha OR an accented Latin letter (Latin-1 Supplement + Latin Extended-A/B:
    // é ñ ü ç ø … — excludes × U+00D7 and ÷ U+00F7), so piñata / café / naïve stay ONE token instead of
    // splitting on the multi-byte accent. (Earlier the size==1 ASCII-only test dropped every accent.)
    auto is_letter = [&](const std::string& c) {
        if (c.empty()) return false;
        if (c.size() == 1) return std::isalpha(static_cast<unsigned char>(c[0])) != 0;
        const std::uint32_t u = cp_value(c);
        return u >= 0x00C0u && u <= 0x024Fu && u != 0x00D7u && u != 0x00F7u;
    };
    auto is_apos = [&](const std::string& c) { return c == "'" || c == apostrophe; };
    auto is_digit = [](const std::string& c) {
        return c.size() == 1 && (std::isdigit(static_cast<unsigned char>(c[0])) != 0);
    };

    std::vector<std::string> out;
    std::string buf;
    int kind = 0;  // 0 none, 1 word, 2 number
    auto flush = [&] { if (!buf.empty()) { out.push_back(std::move(buf)); buf.clear(); } kind = 0; };

    for (auto& cp : word_to_chars(text)) {
        if (is_letter(cp)) {
            if (kind == 2) flush();
            buf += cp; kind = 1;
        } else if (is_apos(cp) && kind == 1) {
            // Apostrophe joins only when ALREADY inside a letter run — a contraction/possessive
            // (What's, I'll, th'). A LEADING apostrophe (TinyStories dialogue: 'And, ''What's) is NOT a
            // word char: it falls through to its own punctuation token, so quotes stop welding to words
            // (which otherwise spawned 1000s of rare 'Word / ''Word variants and bloated the vocab).
            buf += cp;
        } else if (is_digit(cp)) {
            if (kind == 1) flush();
            buf += cp; kind = 2;
        } else {
            flush();
            out.push_back(cp);  // whitespace / punctuation / non-ASCII — its own token
        }
    }
    flush();
    return out;
}

// ── Truecasing (case-marker tokens) ─────────────────────────────────────────────
// A word_level(truecase=true) tokenizer lowercases each word's LEMMA and prepends a marker token for the
// case it stripped, so "Need"/"need"/"NEED" share ONE lemma id `need` (and the most frequent words —
// the/The, he/He — stop splitting their statistics across two ids). Round-trip is GUARANTEED by
// construction: a marker is assigned ONLY when the ASCII-only inverse of the lowercased form reproduces
// the original word EXACTLY; anything else (mixed case, leading accented capital) is emitted verbatim.
constexpr const char* kCapMarker = "<|cap|>";   // first letter upper:  Need  → <|cap|> need
constexpr const char* kUpMarker  = "<|up|>";    // all letters upper:   NEED  → <|up|>  need

std::string ascii_lower(std::string s) {
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}
std::string ascii_upper(std::string s) {
    for (char& c : s) if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
    return s;
}
std::string ascii_cap(std::string s) {   // uppercase only the FIRST byte (if ASCII lower)
    if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') s[0] = static_cast<char>(s[0] - 32);
    return s;
}
[[nodiscard]] bool is_ascii_word_char(unsigned char b) {
    return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z');
}

// Transform one split_words unit into its truecased form: returns {marker, lemma} for a capitalised /
// all-caps word, or {unit} unchanged otherwise. The marker is assigned only if the ASCII-only inverse
// round-trips (so accents and mixed case stay verbatim). Length-1 words are left alone (e.g. "I").
void truecase_unit(const std::string& u, std::vector<std::string>& out) {
    std::size_t letters = 0;
    for (unsigned char b : u) letters += is_ascii_word_char(b);
    if (letters < 2) { out.push_back(u); return; }          // single-letter words: keep ("I", "A")
    const std::string lw = ascii_lower(u);
    if (u == lw) { out.push_back(u); return; }               // already lowercase → unmarked
    if (u == ascii_cap(lw))   { out.emplace_back(kCapMarker); out.push_back(lw); return; }
    if (u == ascii_upper(lw)) { out.emplace_back(kUpMarker);  out.push_back(lw); return; }
    out.push_back(u);                                         // mixed case (McDonald) → verbatim
}

std::vector<std::string> truecase_units(const std::vector<std::string>& words) {
    std::vector<std::string> out;
    out.reserve(words.size() + words.size() / 4);
    for (const auto& u : words) truecase_unit(u, out);
    return out;
}

// Count frequency of adjacent pairs in a list of tokenised words.
using WordList = std::vector<std::vector<std::string>>;
using PairFreq = std::map<std::pair<std::string, std::string>, std::size_t>;

PairFreq count_pairs(const WordList& words) {
    PairFreq freq;
    for (const auto& word : words) {
        for (std::size_t i = 0; i + 1 < word.size(); ++i)
            ++freq[{word[i], word[i + 1]}];
    }
    return freq;
}

// Merge every occurrence of (a, b) into "ab" in all words.
void merge_pair(WordList& words,
                const std::string& a, const std::string& b) {
    const std::string merged = a + b;
    for (auto& word : words) {
        std::vector<std::string> next;
        next.reserve(word.size());
        for (std::size_t i = 0; i < word.size(); ) {
            if (i + 1 < word.size() && word[i] == a && word[i + 1] == b) {
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(word[i]);
                ++i;
            }
        }
        word = std::move(next);
    }
}

// GPT-2 byte-level decoder: maps each remapped Unicode code point back to the
// original byte.  GPT-2/Qwen vocabularies encode every byte as a printable code
// point (printable ASCII maps to itself; other bytes map to U+0100+n), so to
// recover text we reverse that mapping.  See Radford et al. bytes_to_unicode().
const std::unordered_map<uint32_t, unsigned char>& gpt2_byte_decoder() {
    static const std::unordered_map<uint32_t, unsigned char> dec = [] {
        std::vector<int> bs;
        for (int b = '!';   b <= '~';   ++b) bs.push_back(b);   // 0x21..0x7E
        for (int b = 0xA1;  b <= 0xAC;  ++b) bs.push_back(b);
        for (int b = 0xAE;  b <= 0xFF;  ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        std::unordered_map<uint32_t, unsigned char> d;
        for (std::size_t i = 0; i < bs.size(); ++i)
            d[static_cast<uint32_t>(cs[i])] = static_cast<unsigned char>(bs[i]);
        return d;
    }();
    return dec;
}

// Decode the next UTF-8 code point from [p, end); advances p. Returns the code
// point, or 0xFFFD on a malformed sequence (consuming one byte).
uint32_t next_codepoint(const unsigned char*& p, const unsigned char* end) {
    const unsigned char c = *p;
    int len; uint32_t cp;
    if      (c < 0x80) { len = 1; cp = c; }
    else if (c < 0xE0) { len = 2; cp = c & 0x1Fu; }
    else if (c < 0xF0) { len = 3; cp = c & 0x0Fu; }
    else               { len = 4; cp = c & 0x07u; }
    if (p + len > end) { ++p; return 0xFFFDu; }
    for (int i = 1; i < len; ++i)
        cp = (cp << 6) | (static_cast<uint32_t>(p[i]) & 0x3Fu);
    p += len;
    return cp;
}

} // anonymous namespace

// ── add_token / add_special_token ──────────────────────────────────────────────

BPETokenizer::TokenId BPETokenizer::add_token(std::string token) {
    if (const auto it = vocab_.find(token); it != vocab_.end())
        return it->second;
    const auto id = static_cast<TokenId>(id_to_token_.size());
    vocab_[token] = id;
    id_to_token_.push_back(std::move(token));
    return id;
}

BPETokenizer::TokenId BPETokenizer::add_special_token(std::string token) {
    return add_token(std::move(token));
}

// ── train ──────────────────────────────────────────────────────────────────────

BPETokenizer BPETokenizer::train(
    const std::vector<std::string>& corpus,
    std::size_t vocab_size)
{
    BPETokenizer tok;

    // Step 1: seed vocabulary with every unique Unicode code point found in the
    // pre-tokenized corpus (using code points, not raw bytes, keeps tokens
    // as valid UTF-8 strings and makes JSON serialisation safe).
    for (const auto& text : corpus) {
        for (const auto& word : pre_tokenize(text)) {
            for (const auto& ch : word_to_chars(word)) {
                tok.add_token(ch);
            }
        }
    }

    // Step 2: pre-tokenize corpus and represent each word as individual chars.
    WordList words;
    for (const auto& text : corpus) {
        for (auto& w : pre_tokenize(text)) {
            words.push_back(word_to_chars(w));
        }
    }

    // Step 3: BPE merge loop.
    while (tok.id_to_token_.size() < vocab_size) {
        const PairFreq freq = count_pairs(words);
        if (freq.empty()) break;

        // Find the most frequent pair (break ties lexicographically for determinism).
        const auto best = std::max_element(freq.begin(), freq.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second ||
                       (a.second == b.second && a.first > b.first);
            });

        if (best->second == 0) break;

        const auto& [a, b] = best->first;
        const std::string merged = a + b;

        tok.merges_.emplace_back(a, b);
        tok.add_token(merged);
        merge_pair(words, a, b);
    }

    // Add standard GPT-2 special tokens.
    tok.eos_id_ = tok.add_special_token("<|endoftext|>");
    tok.bos_id_ = tok.eos_id_; // GPT-2 uses same token for both
    tok.unk_id_ = tok.add_special_token("<|unk|>");

    return tok;
}

// ── char_level ────────────────────────────────────────────────────────────────
//
// One token per Unicode code point over the RAW text — spaces, tabs, and newlines
// included, byte-for-byte, with no GPT-2 space remapping and no merges. This is the
// graceful-degradation tokenizer: at small scale a diffusion model that emits whole
// characters produces real words and keeps line/verse structure, where a BPE-512 vocab
// forced subword emission and degraded to non-words (TRAINING_DESIGN §13.6).
BPETokenizer BPETokenizer::char_level(const std::vector<std::string>& corpus) {
    BPETokenizer tok;
    tok.char_level_ = true;

    // Seed the vocabulary with every code point appearing anywhere in the corpus,
    // INCLUDING whitespace. No pre_tokenize: walk the raw text so ' ', '\n', '\t'
    // become first-class tokens (BPE's pre_tokenize would drop or remap them).
    for (const auto& text : corpus)
        for (auto& ch : word_to_chars(text))
            tok.add_token(ch);

    tok.eos_id_ = tok.add_special_token("<|endoftext|>");
    tok.bos_id_ = tok.eos_id_;  // GPT-2 convention: shared BOS/EOS
    tok.unk_id_ = tok.add_special_token("<|unk|>");
    return tok;
}

// ── word_level ────────────────────────────────────────────────────────────────
//
// One token per whole word (+ each punctuation/whitespace char), no merges — the far
// end of the granularity spectrum from char_level(). Every token IS a real word, so the
// model literally cannot emit a non-word; it isolates whether BPE-512's salad came from
// mis-assembled subword fragments under diffusion's parallel denoising (§13.6). Vocab is
// every unique word/punct/whitespace token in the corpus (no frequency cap → large vocab).
BPETokenizer BPETokenizer::word_level(const std::vector<std::string>& corpus, bool truecase) {
    BPETokenizer tok;
    tok.word_level_ = true;
    tok.truecased_  = truecase;

    if (truecase) {                       // markers get low, stable ids; lemmas follow
        tok.add_token(kCapMarker);
        tok.add_token(kUpMarker);
    }
    for (const auto& text : corpus) {
        const auto words = split_words(text);
        for (const auto& w : (truecase ? truecase_units(words) : words))
            tok.add_token(w);
    }

    tok.eos_id_ = tok.add_special_token("<|endoftext|>");
    tok.bos_id_ = tok.eos_id_;
    tok.unk_id_ = tok.add_special_token("<|unk|>");
    return tok;
}

// ── load ───────────────────────────────────────────────────────────────────────

BPETokenizer BPETokenizer::load(
    const std::filesystem::path& vocab_json,
    const std::filesystem::path& merges_txt)
{
    BPETokenizer tok;

    // ── vocab.json: { "token_string": id, ... } ───────────────────────────────
    {
        std::ifstream f(vocab_json);
        if (!f) throw std::runtime_error("BPETokenizer::load: cannot open " + vocab_json.string());

        const auto j = nlohmann::json::parse(f);
        if (!j.is_object())
            throw std::runtime_error("vocab.json: expected a JSON object");

        // Build id→token mapping; vocab may not be in insertion order.
        std::vector<std::pair<TokenId, std::string>> entries;
        entries.reserve(j.size());
        for (auto& [k, v] : j.items())
            entries.emplace_back(v.get<TokenId>(), k);
        std::sort(entries.begin(), entries.end());

        tok.id_to_token_.resize(static_cast<std::size_t>(entries.back().first) + 1);
        for (auto& [id, str] : entries) {
            tok.vocab_[str]                              = id;
            tok.id_to_token_[static_cast<std::size_t>(id)] = str;
        }
    }

    // ── merges.txt: one "a b" merge pair per line, first line is a header ──────
    {
        std::ifstream f(merges_txt);
        if (!f) throw std::runtime_error("BPETokenizer::load: cannot open " + merges_txt.string());

        std::string line;
        std::getline(f, line); // skip header (#version: ...)

        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            const auto sp = line.find(' ');
            if (sp == std::string::npos)
                throw std::runtime_error("merges.txt: malformed line: " + line);
            tok.merges_.emplace_back(line.substr(0, sp), line.substr(sp + 1));
        }
    }

    // Locate special token ids from the loaded vocabulary.
    if (auto it = tok.vocab_.find("<|endoftext|>"); it != tok.vocab_.end())
        tok.eos_id_ = tok.bos_id_ = it->second;
    if (auto it = tok.vocab_.find("<|unk|>"); it != tok.vocab_.end())
        tok.unk_id_ = it->second;

    // A char_level()/word_level() tokenizer round-trips as an EMPTY merges.txt plus a vocab
    // holding a literal space/newline token — neither of which a trained BPE vocab ever has
    // (its spaces are the Ġ marker and it always has merges). Restore the right flag so
    // encode uses the matching splitter on a resumed/eval run. char_level has ONLY single
    // code-point tokens; word_level additionally has multi-character word tokens — so the
    // presence of any multi-byte non-special token distinguishes the two.
    const bool literal = tok.merges_.empty() &&
        (tok.vocab_.find(" ") != tok.vocab_.end() || tok.vocab_.find("\n") != tok.vocab_.end());
    if (literal) {
        bool has_word = false;
        for (const auto& t : tok.id_to_token_)
            if (!t.empty() && t.front() != '<' && word_to_chars(t).size() >= 2) { has_word = true; break; }
        tok.word_level_ = has_word;
        tok.char_level_ = !has_word;
        // Truecasing leaves its markers in the vocab — restore the flag so a resumed/eval run re-applies
        // the lowercasing transform on encode and the case restoration on decode.
        tok.truecased_  = has_word && tok.vocab_.find(kCapMarker) != tok.vocab_.end();
    }

    return tok;
}

// ── from_vocab ────────────────────────────────────────────────────────────────

BPETokenizer BPETokenizer::from_vocab(
    const std::vector<std::string>& id_to_token,
    const std::vector<std::string>& merges_list)
{
    BPETokenizer tok;
    tok.byte_level_  = true;   // GGUF vocabs use GPT-2 byte-level encoding
    tok.id_to_token_ = id_to_token;
    tok.vocab_.reserve(id_to_token.size());
    for (TokenId i = 0; i < static_cast<TokenId>(id_to_token.size()); ++i)
        tok.vocab_[id_to_token[static_cast<std::size_t>(i)]] = i;

    tok.merges_.reserve(merges_list.size());
    for (const auto& entry : merges_list) {
        const auto sp = entry.find(' ');
        if (sp == std::string::npos) continue;
        tok.merges_.emplace_back(entry.substr(0, sp), entry.substr(sp + 1));
    }

    // Detect common special tokens (names vary by model family).
    for (const std::string& eos_str : {"<|endoftext|>", "<eos>", "</s>", "<|im_end|>", "<EOS>"}) {
        if (auto it = tok.vocab_.find(eos_str); it != tok.vocab_.end()) {
            tok.eos_id_ = it->second;
            if (tok.bos_id_ < 0) tok.bos_id_ = it->second;
            break;
        }
    }
    for (const std::string& bos_str : {"<|startoftext|>", "<bos>", "<s>", "<|im_start|>"}) {
        if (auto it = tok.vocab_.find(bos_str); it != tok.vocab_.end()) {
            tok.bos_id_ = it->second;
            break;
        }
    }
    for (const std::string& unk_str : {"<unk>", "<|unk|>", "[UNK]", "<UNK>"}) {
        if (auto it = tok.vocab_.find(unk_str); it != tok.vocab_.end()) {
            tok.unk_id_ = it->second;
            break;
        }
    }
    for (const char* name : {"<pad>", "<PAD>"}) {
        if (auto it = tok.vocab_.find(name); it != tok.vocab_.end()) {
            tok.pad_id_ = it->second;
            break;
        }
    }

    return tok;
}

// ── encode_word ────────────────────────────────────────────────────────────────

std::vector<BPETokenizer::TokenId>
BPETokenizer::encode_word(std::string_view word) const {
    // Start from individual bytes.
    std::vector<std::string> tokens = word_to_chars(word);

    // Apply merge rules in their ranked order.
    // Build a rank map: pair_string → rank for O(1) lookup.
    // (Constructed once per encode call; pre-cache in production via Ch08.)
    std::unordered_map<std::string, int> merge_rank;
    merge_rank.reserve(merges_.size());
    for (std::size_t i = 0; i < merges_.size(); ++i)
        merge_rank[merges_[i].first + " " + merges_[i].second] = static_cast<int>(i);

    // Iteratively apply the lowest-ranked (earliest) applicable merge.
    while (tokens.size() > 1) {
        int    best_rank = std::numeric_limits<int>::max();
        std::size_t best_pos  = 0;
        bool   found = false;

        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            const std::string key = tokens[i] + " " + tokens[i + 1];
            if (const auto it = merge_rank.find(key); it != merge_rank.end()) {
                if (it->second < best_rank) {
                    best_rank = it->second;
                    best_pos  = i;
                    found     = true;
                }
            }
        }
        if (!found) break;

        // Apply the merge at best_pos.
        tokens[best_pos] += tokens[best_pos + 1];
        tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_pos) + 1);
    }

    // Map token strings → ids.
    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (const auto& t : tokens) {
        if (const auto it = vocab_.find(t); it != vocab_.end())
            ids.push_back(it->second);
        else if (unk_id_ >= 0)
            ids.push_back(unk_id_);
        else
            throw std::runtime_error("BPETokenizer::encode: unknown token: " + t);
    }
    return ids;
}

// ── encode ────────────────────────────────────────────────────────────────────

std::vector<BPETokenizer::TokenId>
BPETokenizer::encode(std::string_view text) const {
    std::vector<TokenId> result;
    // Char-level / word-level: one token per raw code point (char) or per whole word
    // (word), whitespace and newlines preserved — skip the GPT-2 pre-tokenizer that would
    // collapse whitespace and remap spaces.
    if (char_level_ || word_level_) {
        std::vector<std::string> units;
        if (!word_level_)        units = word_to_chars(text);
        else if (truecased_)     units = truecase_units(split_words(text));
        else                     units = split_words(text);
        for (const auto& u : units) {
            if (const auto it = vocab_.find(u); it != vocab_.end())
                result.push_back(it->second);
            else if (unk_id_ >= 0)
                result.push_back(unk_id_);
            else
                throw std::runtime_error("BPETokenizer::encode: unknown token: " + u);
        }
        return result;
    }
    for (const auto& word : pre_tokenize(text)) {
        auto word_ids = encode_word(word);
        result.insert(result.end(), word_ids.begin(), word_ids.end());
    }
    return result;
}

// ── decode ────────────────────────────────────────────────────────────────────

std::string BPETokenizer::decode(std::span<const TokenId> ids) const {
    std::string out;

    // Char-level / word-level: tokens are literal text (code points or whole words, with
    // whitespace/newlines included) — concatenate as-is, no Ġ→space rewrite or byte remap.
    if (char_level_ || word_level_) {
        int pending = 0;   // truecasing: 1 = capitalise next word, 2 = uppercase next word
        for (const TokenId id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
                throw std::runtime_error(std::format(
                    "BPETokenizer::decode: id {} out of range [0,{})", id, id_to_token_.size()));
            const std::string& t = id_to_token_[static_cast<std::size_t>(id)];
            if (truecased_) {
                if (t == kCapMarker) { pending = 1; continue; }
                if (t == kUpMarker)  { pending = 2; continue; }
                if (pending == 1) { out += ascii_cap(t);  pending = 0; continue; }
                if (pending == 2) { out += ascii_upper(t); pending = 0; continue; }
            }
            out += t;
        }
        return out;
    }

    // Byte-level (GPT-2/Qwen) vocab: every token char is a remapped byte, so map
    // each code point back to its byte; the assembled bytes form the UTF-8 text.
    if (byte_level_) {
        const auto& dec = gpt2_byte_decoder();
        for (const TokenId id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
                throw std::runtime_error(
                    std::format("BPETokenizer::decode: id {} out of range [0,{})",
                        id, id_to_token_.size()));
            const std::string& tok = id_to_token_[static_cast<std::size_t>(id)];
            const auto* p   = reinterpret_cast<const unsigned char*>(tok.data());
            const auto* end = p + tok.size();
            while (p < end) {
                const uint32_t cp = next_codepoint(p, end);
                if (const auto it = dec.find(cp); it != dec.end())
                    out += static_cast<char>(it->second);   // recovered byte
                // Special tokens (e.g. <|endoftext|>) aren't byte-mapped — skip.
            }
        }
        return out;
    }

    for (const TokenId id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
            throw std::runtime_error(
                std::format("BPETokenizer::decode: id {} out of range [0,{})",
                    id, id_to_token_.size()));
        const std::string_view tok = id_to_token_[static_cast<std::size_t>(id)];

        // Replace the GPT-2 space marker (UTF-8: C4 A0) with an ASCII space.
        if (tok.size() >= 2 &&
            static_cast<unsigned char>(tok[0]) == 0xC4 &&
            static_cast<unsigned char>(tok[1]) == 0xA0) {
            out += ' ';
            out += tok.substr(2);
        } else {
            out += tok;
        }
    }
    return out;
}

// ── token_str / token_id ──────────────────────────────────────────────────────

std::string_view BPETokenizer::token_str(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
        throw std::out_of_range(
            std::format("BPETokenizer::token_str: id {} out of range", id));
    return id_to_token_[static_cast<std::size_t>(id)];
}

BPETokenizer::TokenId BPETokenizer::token_id(std::string_view tok) const {
    const auto it = vocab_.find(std::string{tok});
    return (it != vocab_.end()) ? it->second : TokenId{-1};
}

// ── save ──────────────────────────────────────────────────────────────────────

void BPETokenizer::save(const std::filesystem::path& dir) const {
    std::filesystem::create_directories(dir);

    // vocab.json
    {
        nlohmann::json j;
        for (const auto& [tok, id] : vocab_) j[tok] = id;
        std::ofstream f(dir / "vocab.json");
        f << j.dump(2);
    }

    // merges.txt
    {
        std::ofstream f(dir / "merges.txt");
        f << "#version: sub0llm-bpe\n";
        for (const auto& [a, b] : merges_)
            f << a << ' ' << b << '\n';
    }
}

} // namespace sub0llm
