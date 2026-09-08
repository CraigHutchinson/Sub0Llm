// qwen_tokenizer.cpp -- the out-of-line half of sub0/qwen_tokenizer.hpp: the hand-transliterated
// pre-tokenization regex, the BPE merge loop, and the three real-file readers. See the header for
// what this component is and docs/QWEN_TOKENIZER.md for how each piece was verified against the
// real reference.

#include "sub0/qwen_tokenizer.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <utility>

namespace sub0::qwen_tok {

namespace {

using qwen_uni::Cp;
using qwen_uni::decode_utf8;

constexpr std::size_t kNo = static_cast<std::size_t>(-1);   // "this alternative did not match"

// A codepoint eligible for `[\p{L}\p{M}]`.
inline bool lm(std::uint32_t cp) { return qwen_uni::is_letter(cp) || qwen_uni::is_mark(cp); }
// A codepoint eligible for `[^\s\p{L}\p{M}\p{N}]`.
inline bool sym(std::uint32_t cp) {
    return !qwen_uni::is_space(cp) && !lm(cp) && !qwen_uni::is_number(cp);
}
// End (exclusive) of the maximal `[\p{L}\p{M}]+` run starting at `i`, which must be non-empty.
inline std::size_t run_lm(std::string_view s, std::size_t i) {
    std::size_t e = i;
    while (e < s.size()) {
        const Cp c = decode_utf8(s, e);
        if (!lm(c.cp)) break;
        e += c.size;
    }
    return e;
}

// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`. Unicode simple case folding, so each ASCII letter also matches its
// upper case -- plus U+017F LATIN SMALL LETTER LONG S for 's', the one non-ASCII fold in the set
// (both facts confirmed against the reference by sweeping every codepoint that folds to s/t/r/e/v/
// m/l/d; nothing else in Unicode does). The alternatives are pairwise distinguishable by the letter
// after the apostrophe, so leftmost-first order costs nothing here, but it is preserved anyway.
inline bool fold_is(std::uint32_t cp, char lower) {
    return cp == static_cast<std::uint32_t>(static_cast<unsigned char>(lower)) ||
           cp == static_cast<std::uint32_t>(static_cast<unsigned char>(lower) - 32) ||
           (lower == 's' && cp == 0x017F);
}
std::size_t m_contraction(std::string_view s, std::size_t i) {
    if (s[i] != '\'' || i + 1 >= s.size()) return kNo;
    const Cp a = decode_utf8(s, i + 1);
    const std::size_t after_a = i + 1 + a.size;
    if (fold_is(a.cp, 's') || fold_is(a.cp, 't') || fold_is(a.cp, 'm') || fold_is(a.cp, 'd'))
        return after_a;
    const bool two = fold_is(a.cp, 'r') || fold_is(a.cp, 'v') || fold_is(a.cp, 'l');
    if (!two || after_a >= s.size()) return kNo;
    const Cp b = decode_utf8(s, after_a);
    const char want = fold_is(a.cp, 'l') ? 'l' : 'e';       // 're / 've take 'e'; 'll takes 'l'
    return fold_is(b.cp, want) ? after_a + b.size : kNo;
}

// `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`. The optional leading character is tried FIRST (greedy `?`) and
// only then dropped, which is exactly the backtracking order the reference uses; both orders reach
// the same extent whenever the leading character is itself a Mark, but coding it literally means
// that equivalence never has to be relied on.
std::size_t m_word(std::string_view s, std::size_t i) {
    const Cp c0 = decode_utf8(s, i);
    const bool can_prefix =
        c0.cp != '\r' && c0.cp != '\n' && !qwen_uni::is_letter(c0.cp) && !qwen_uni::is_number(c0.cp);
    if (can_prefix && i + c0.size < s.size() && lm(decode_utf8(s, i + c0.size).cp))
        return run_lm(s, i + c0.size);
    return lm(c0.cp) ? run_lm(s, i) : kNo;
}

// `\p{N}` -- one character, which is why every digit of "123456" becomes its own token.
std::size_t m_number(std::string_view s, std::size_t i) {
    const Cp c = decode_utf8(s, i);
    return qwen_uni::is_number(c.cp) ? i + c.size : kNo;
}

// ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`. The optional leading space is an ASCII space specifically, not
// any whitespace; `[\r\n]*` needs no backtracking because CR/LF are whitespace and so can never be
// swallowed by the `+` before it.
std::size_t m_symbols(std::string_view s, std::size_t i) {
    std::size_t start = kNo;
    if (s[i] == ' ' && i + 1 < s.size() && sym(decode_utf8(s, i + 1).cp)) start = i + 1;
    else if (sym(decode_utf8(s, i).cp))                                  start = i;
    if (start == kNo) return kNo;
    std::size_t e = start;
    while (e < s.size()) {
        const Cp c = decode_utf8(s, e);
        if (!sym(c.cp)) break;
        e += c.size;
    }
    while (e < s.size() && (s[e] == '\r' || s[e] == '\n')) ++e;
    return e;
}

// `\s*[\r\n]+`. Backtracking makes the greedy `\s*` give back exactly as much as it must for a CR/LF
// to follow, so the match ends immediately after the LAST CR/LF inside the whitespace run at `i`.
std::size_t m_newlines(std::string_view s, std::size_t i) {
    std::size_t e = i, best = kNo;
    while (e < s.size()) {
        const Cp c = decode_utf8(s, e);
        if (!qwen_uni::is_space(c.cp)) break;
        e += c.size;
        if (c.cp == '\r' || c.cp == '\n') best = e;
    }
    return best;
}

// `\s+(?!\S)` -- a whitespace run that is not followed by a non-space. The lookahead can only be
// satisfied at end-of-input or with whitespace next, so after backtracking this is: the whole run if
// it ends the input, otherwise the run minus its final character (which then IS the whitespace the
// lookahead sees), and no match at all for a single-character run in mid-text. This is the clause
// that makes "  a" split as [" ", " a"].
std::size_t m_trailing_space(std::string_view s, std::size_t i) {
    std::size_t e = i, prev = i, n = 0;
    while (e < s.size()) {
        const Cp c = decode_utf8(s, e);
        if (!qwen_uni::is_space(c.cp)) break;
        prev = e;
        e += c.size;
        ++n;
    }
    if (n == 0) return kNo;
    if (e == s.size()) return e;
    return n >= 2 ? prev : kNo;
}

// `\s+`
std::size_t m_space(std::string_view s, std::size_t i) {
    std::size_t e = i;
    while (e < s.size()) {
        const Cp c = decode_utf8(s, e);
        if (!qwen_uni::is_space(c.cp)) break;
        e += c.size;
    }
    return e == i ? kNo : e;
}

}  // namespace

void pretokenize(std::string_view text, std::vector<std::string_view>& out) {
    out.clear();
    std::size_t i = 0;
    while (i < text.size()) {
        std::size_t e = m_contraction(text, i);
        if (e == kNo) e = m_word(text, i);
        if (e == kNo) e = m_number(text, i);
        if (e == kNo) e = m_symbols(text, i);
        if (e == kNo) e = m_newlines(text, i);
        if (e == kNo) e = m_trailing_space(text, i);
        if (e == kNo) e = m_space(text, i);
        // Unreachable for well-formed UTF-8: every codepoint is whitespace, a number, a letter/mark,
        // or a symbol, so one of the clauses above always matches. Kept as a guaranteed-progress
        // guard so a malformed byte can never spin this loop.
        if (e == kNo || e <= i) e = i + decode_utf8(text, i).size;
        out.push_back(text.substr(i, e - i));
        i = e;
    }
}

// --- BPE ----------------------------------------------------------------------------------------
// One chunk at a time; merges never cross a pre-token boundary. Symbols live in a doubly linked list
// so a merge is O(1), and candidate merges live in a min-heap keyed by (rank, position) -- the same
// shape the reference uses, and the reason a pathological chunk (a megabyte of one repeated symbol)
// stays O(n log n) instead of the O(n^2) a rescan-the-whole-chunk loop would cost.
void Tokenizer::bpe_chunk(std::string_view chunk, std::vector<int>& out) const {
    struct Sym { int id; std::size_t prev, next; };
    static thread_local std::vector<Sym> syms;     // reused across chunks and across calls
    syms.clear();
    syms.reserve(chunk.size());
    for (std::size_t k = 0; k < chunk.size(); ++k)
        syms.push_back({byte_id_[static_cast<unsigned char>(chunk[k])], k - 1, k + 1});
    if (syms.empty()) return;
    const std::size_t n = syms.size();

    using Cand = std::pair<int, std::size_t>;      // (rank, left position)
    static thread_local std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> heap;
    while (!heap.empty()) heap.pop();              // priority_queue has no clear()

    auto rule = [&](std::size_t a, std::size_t b) -> const Merge* {
        const auto it = merges_.find((static_cast<std::uint64_t>(static_cast<std::uint32_t>(syms[a].id)) << 32) |
                                     static_cast<std::uint32_t>(syms[b].id));
        return it == merges_.end() ? nullptr : &it->second;
    };
    for (std::size_t k = 0; k + 1 < n; ++k)
        if (const Merge* m = rule(k, k + 1)) heap.emplace(m->rank, k);

    while (!heap.empty()) {
        const auto [rank, a] = heap.top();
        heap.pop();
        if (syms[a].id < 0) continue;                       // left symbol was absorbed
        const std::size_t b = syms[a].next;
        if (b >= n || syms[b].id < 0) continue;
        const Merge* m = rule(a, b);
        // A stale entry names a pair this position no longer holds. Comparing the RANK is enough to
        // detect that: ranks are unique per pair, and a position's pair can never repeat, because
        // every merge strictly extends the left symbol.
        if (!m || m->rank != rank) continue;
        syms[a].id   = m->id;
        syms[a].next = syms[b].next;
        if (syms[b].next < n) syms[syms[b].next].prev = a;
        syms[b].id = -1;
        if (syms[a].prev < n)
            if (const Merge* p = rule(syms[a].prev, a)) heap.emplace(p->rank, syms[a].prev);
        if (syms[a].next < n)
            if (const Merge* q = rule(a, syms[a].next)) heap.emplace(q->rank, a);
    }
    for (std::size_t k = 0; k < n; k = syms[k].next)
        out.push_back(syms[k].id);
}

// Everything after the added-token split: normalize, pre-tokenize, merge.
//
// The byte-level alphabet does NOT appear here, deliberately. Its whole job is to give each of the
// 256 byte values a distinct vocabulary entry; `byte_id_` already holds the resulting byte -> id
// table, so BPE can start from the chunk's RAW bytes. Materialising the alphabet string first and
// then re-reading its bytes would encode the input twice -- and does not fail loudly, it just quietly
// tokenizes the UTF-8 of the alphabet characters instead of the text (a space came out as two tokens
// rather than one).
void Tokenizer::encode_span(std::string_view text, std::vector<int>& out) const {
    if (text.empty()) return;
    const std::string norm = qwen_uni::nfc(text);
    static thread_local std::vector<std::string_view> chunks;
    pretokenize(norm, chunks);
    for (std::string_view c : chunks) bpe_chunk(c, out);
}

void Tokenizer::encode(std::string_view text, std::vector<int>& out) const {
    out.clear();
    // Added tokens are matched on the RAW bytes, BEFORE normalization -- that is the reference's own
    // order (its added-vocabulary split runs first, and every one of these tokens is declared
    // `normalized: false`). Leftmost-longest, so `<|im_end|>` wins over any shorter prefix of it.
    std::size_t i = 0, span = 0;
    while (i < text.size()) {
        if (added_first_[static_cast<unsigned char>(text[i])]) {
            const Added* hit = nullptr;
            for (const Added& a : added_)                    // sorted longest-first
                if (text.compare(i, a.content.size(), a.content) == 0) { hit = &a; break; }
            if (hit) {
                encode_span(text.substr(span, i - span), out);
                out.push_back(hit->id);
                i += hit->content.size();
                span = i;
                continue;
            }
        }
        ++i;
    }
    encode_span(text.substr(span), out);
}

std::vector<int> Tokenizer::encode(std::string_view text) const {
    std::vector<int> out;
    encode(text, out);
    return out;
}

std::string Tokenizer::decode(std::span<const int> ids) const {
    std::string bytes;
    for (int id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) + 1 >= dec_off_.size()) continue;
        const auto lo = dec_off_[static_cast<std::size_t>(id)], hi = dec_off_[static_cast<std::size_t>(id) + 1];
        bytes.append(dec_blob_, lo, hi - lo);
    }
    // The reference decodes the assembled bytes with errors="replace" (tokenizer_config.json's
    // `errors` field), so a sequence cut off mid-character yields U+FFFD rather than a raw fragment.
    return qwen_uni::decode_lossy(bytes);
}

// --- loading -------------------------------------------------------------------------------------
namespace {
bool fail(std::string* err, std::string msg) {
    if (err) *err = std::move(msg);
    return false;
}
}  // namespace

bool Tokenizer::load(const std::string& vocab_json_path,
                     const std::string& merges_txt_path,
                     const std::string& tokenizer_config_json_path,
                     std::string* error) {
    *this = Tokenizer{};

    // --- vocab.json: a flat {"token": id} object, 248,044 entries. Read in file order and placed by
    // id, so the table does not depend on the file happening to be sorted.
    simdjson::padded_string vjson;
    if (simdjson::padded_string::load(vocab_json_path).get(vjson))
        return fail(error, "cannot read vocab file: " + vocab_json_path);
    std::vector<std::pair<std::string, int>> vocab;
    int max_id = -1;
    {
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc;
        if (parser.iterate(vjson).get(doc)) return fail(error, "vocab.json: not parseable as JSON");
        simdjson::ondemand::object obj;
        if (doc.get_object().get(obj)) return fail(error, "vocab.json: top level is not an object");
        for (auto field : obj) {
            std::string_view key;
            std::int64_t     id;
            if (field.unescaped_key().get(key)) return fail(error, "vocab.json: bad key");
            std::string k(key);
            if (field.value().get_int64().get(id)) return fail(error, "vocab.json: value for '" + k + "' is not an integer");
            max_id = std::max(max_id, static_cast<int>(id));
            vocab.emplace_back(std::move(k), static_cast<int>(id));
        }
    }
    if (vocab.empty()) return fail(error, "vocab.json: empty");
    base_n_ = max_id + 1;
    if (static_cast<std::size_t>(base_n_) != vocab.size())
        return fail(error, "vocab.json: ids are not a dense 0..N-1 range (" + std::to_string(vocab.size()) +
                           " entries, max id " + std::to_string(max_id) + ")");

    // --- added tokens + eos/pad, from tokenizer_config.json's added_tokens_decoder. Loaded before
    // the blobs are built so their text lands in the same blob and gets the same decode treatment.
    std::vector<std::pair<std::string, int>> added;
    std::string eos_text, pad_text;
    {
        simdjson::padded_string cjson;
        if (simdjson::padded_string::load(tokenizer_config_json_path).get(cjson))
            return fail(error, "cannot read tokenizer config: " + tokenizer_config_json_path);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc;
        if (parser.iterate(cjson).get(doc)) return fail(error, "tokenizer_config.json: not parseable as JSON");
        simdjson::ondemand::object obj;
        if (doc.get_object().get(obj)) return fail(error, "tokenizer_config.json: top level is not an object");
        for (auto field : obj) {
            std::string_view key;
            if (field.unescaped_key().get(key)) continue;
            if (key == "eos_token" || key == "pad_token") {
                std::string_view v;
                if (!field.value().get_string().get(v)) (key == "eos_token" ? eos_text : pad_text).assign(v);
                continue;
            }
            if (key != "added_tokens_decoder") continue;
            simdjson::ondemand::object table;
            if (field.value().get_object().get(table))
                return fail(error, "tokenizer_config.json: added_tokens_decoder is not an object");
            for (auto entry : table) {
                std::string_view id_key;
                if (entry.unescaped_key().get(id_key)) continue;
                const int id = std::atoi(std::string(id_key).c_str());
                simdjson::ondemand::object rec;
                if (entry.value().get_object().get(rec)) continue;
                for (auto f2 : rec) {
                    std::string_view k2;
                    if (f2.unescaped_key().get(k2) || k2 != "content") continue;
                    std::string_view content;
                    if (!f2.value().get_string().get(content)) added.emplace_back(std::string(content), id);
                }
            }
        }
    }
    added_n_ = static_cast<int>(added.size());

    // --- build the blobs. Index by id so lookup is O(1) and decode is a slice.
    const std::size_t total = static_cast<std::size_t>(base_n_) + added.size();
    std::vector<std::string> text_by_id(total);
    for (auto& [t, id] : vocab) text_by_id[static_cast<std::size_t>(id)] = std::move(t);
    for (auto& [t, id] : added) {
        if (id < 0 || static_cast<std::size_t>(id) >= total)
            return fail(error, "tokenizer_config.json: added token id " + std::to_string(id) + " out of range");
        text_by_id[static_cast<std::size_t>(id)] = t;
    }

    const ByteAlphabet& ab = byte_alphabet();
    tok_off_.reserve(total + 1);
    dec_off_.reserve(total + 1);
    tok_off_.push_back(0);
    dec_off_.push_back(0);
    for (const std::string& t : text_by_id) {
        tok_blob_ += t;
        tok_off_.push_back(static_cast<std::uint32_t>(tok_blob_.size()));
        // Decode a token's text back to raw bytes through the byte alphabet. If ANY character is
        // outside the alphabet the reference falls back to the token's own UTF-8 -- which is what
        // keeps an added token like "<|im_end|>" printing as itself. (For these files every added
        // token is printable ASCII, so it maps character-for-character anyway; the fallback is kept
        // because the reference has it, not because this data needs it.)
        const std::size_t mark = dec_blob_.size();
        bool ok = true;
        for (std::size_t k = 0; k < t.size() && ok;) {
            const qwen_uni::Cp c = qwen_uni::decode_utf8(t, k);
            k += c.size;
            const int b = c.cp < 0x180 ? ab.to_byte[c.cp] : -1;
            if (b < 0) ok = false;
            else       dec_blob_.push_back(static_cast<char>(static_cast<unsigned char>(b)));
        }
        if (!ok) { dec_blob_.resize(mark); dec_blob_ += t; }
        dec_off_.push_back(static_cast<std::uint32_t>(dec_blob_.size()));
    }
    id_of_.reserve(total * 2);
    for (std::size_t id = 0; id < total; ++id)
        id_of_.emplace(std::string_view(tok_blob_).substr(tok_off_[id], tok_off_[id + 1] - tok_off_[id]),
                       static_cast<int>(id));

    // --- the 256 single-byte tokens. Byte-level BPE guarantees all of them exist; if one does not,
    // the file is not a byte-level vocabulary and every later step would be silently wrong.
    for (int b = 0; b < 256; ++b) {
        std::string s;
        qwen_uni::append_utf8(s, ab.to_cp[static_cast<std::size_t>(b)]);
        const auto it = id_of_.find(std::string_view(s));
        if (it == id_of_.end())
            return fail(error, "vocab.json: not a byte-level vocabulary -- no token for byte " + std::to_string(b));
        byte_id_[static_cast<std::size_t>(b)] = it->second;
    }

    // --- merges.txt: one "left right" rule per line, rank == line number. The leading "#version:"
    // comment the GPT-2-era files carry is skipped exactly as the reference does -- by matching that
    // literal prefix on the FIRST line only, never by "starts with #", because '#' is itself a
    // byte-alphabet character and "# #" is a legitimate merge rule.
    {
        std::ifstream in(merges_txt_path, std::ios::binary);
        if (!in) return fail(error, "cannot read merges file: " + merges_txt_path);
        std::string line;
        int rank = 0;
        for (int lineno = 0; std::getline(in, line); ++lineno) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (lineno == 0 && line.rfind("#version:", 0) == 0) continue;
            const std::size_t sp = line.find(' ');
            if (sp == std::string::npos)
                return fail(error, "merges.txt line " + std::to_string(lineno + 1) + ": no separator");
            const std::string_view a(line.data(), sp), b(line.data() + sp + 1, line.size() - sp - 1);
            const auto ia = id_of_.find(a), ib = id_of_.find(b);
            if (ia == id_of_.end() || ib == id_of_.end()) { ++rank; continue; }
            std::string joined;
            joined.reserve(a.size() + b.size());
            joined += a;
            joined += b;
            const auto ij = id_of_.find(std::string_view(joined));
            if (ij == id_of_.end()) { ++rank; continue; }
            merges_.emplace((static_cast<std::uint64_t>(static_cast<std::uint32_t>(ia->second)) << 32) |
                                static_cast<std::uint32_t>(ib->second),
                            Merge{rank, ij->second});
            ++rank;
        }
        if (rank == 0) return fail(error, "merges.txt: no merge rules");
    }

    // --- added-token matching order + the first-byte reject table.
    added_.reserve(added.size());
    for (const auto& [t, id] : added) added_.push_back({t, id});
    std::sort(added_.begin(), added_.end(),
              [](const Added& x, const Added& y) { return x.content.size() > y.content.size(); });
    for (const Added& a : added_)
        if (!a.content.empty()) added_first_[static_cast<unsigned char>(a.content[0])] = true;

    const auto find_special = [&](const std::string& text) {
        const auto it = text.empty() ? id_of_.end() : id_of_.find(std::string_view(text));
        return it == id_of_.end() ? -1 : it->second;
    };
    eos_ = find_special(eos_text);
    pad_ = find_special(pad_text);
    return true;
}

}  // namespace sub0::qwen_tok
