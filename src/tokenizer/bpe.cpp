#include "sub0llm/tokenizer/bpe.hpp"

#include <algorithm>
#include <cassert>
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

    // A char_level() tokenizer round-trips as an EMPTY merges.txt plus a vocab holding a
    // literal space/newline token — neither of which a trained BPE vocab ever has (its
    // spaces are the Ġ marker and it always has merges). Detect that and restore the flag
    // so encode bypasses the whitespace-collapsing pre-tokenizer on a resumed/eval run.
    if (tok.merges_.empty() &&
        (tok.vocab_.find(" ") != tok.vocab_.end() || tok.vocab_.find("\n") != tok.vocab_.end()))
        tok.char_level_ = true;

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
    // Char-level: one token per raw code point, whitespace and newlines preserved —
    // skip the GPT-2 pre-tokenizer that would collapse whitespace and remap spaces.
    if (char_level_) {
        for (const auto& ch : word_to_chars(text)) {
            if (const auto it = vocab_.find(ch); it != vocab_.end())
                result.push_back(it->second);
            else if (unk_id_ >= 0)
                result.push_back(unk_id_);
            else
                throw std::runtime_error("BPETokenizer::encode: unknown char: " + ch);
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

    // Char-level: tokens are literal code points (whitespace/newlines included) —
    // concatenate them as-is, no Ġ→space rewrite and no byte remapping.
    if (char_level_) {
        for (const TokenId id : ids) {
            if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
                throw std::runtime_error(std::format(
                    "BPETokenizer::decode: id {} out of range [0,{})", id, id_to_token_.size()));
            out += id_to_token_[static_cast<std::size_t>(id)];
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
