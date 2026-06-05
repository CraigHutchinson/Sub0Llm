#include "sub0llm/tokenizer/sentencepiece.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>

namespace sub0llm {

namespace {
constexpr std::string_view kSpace = "\xe2\x96\x81";  // ▁ U+2581

// UTF-8 sequence length from a leading byte.
std::size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xe) return 3;
    if ((c >> 3) == 0x1e) return 4;
    return 1;  // invalid lead → treat as single byte
}
} // namespace

SPTokenizer SPTokenizer::from_vocab(const std::vector<std::string>& tokens,
                                    const std::vector<float>& scores,
                                    int32_t bos_id, int32_t eos_id, bool add_bos) {
    SPTokenizer t;
    t.id_to_piece_ = tokens;
    t.scores_      = scores;
    t.scores_.resize(tokens.size(), 0.0f);
    t.bos_id_  = bos_id;
    t.eos_id_  = eos_id;
    t.add_bos_ = add_bos;
    t.piece_to_id_.reserve(tokens.size() * 2);
    for (int32_t id = 0; id < static_cast<int32_t>(tokens.size()); ++id)
        t.piece_to_id_.emplace(tokens[static_cast<std::size_t>(id)], id);
    return t;
}

std::vector<int32_t> SPTokenizer::encode(std::string_view text) const {
    // Normalize: spaces → ▁ (no dummy prefix — matches Gemma's GGUF tokenizer).
    std::string norm;
    norm.reserve(text.size() + text.size() / 4);
    for (char c : text) {
        if (c == ' ') norm += kSpace;
        else          norm.push_back(c);
    }

    // Doubly-linked symbol list over UTF-8 characters of `norm`.
    struct Symbol { int prev, next; const char* text; std::size_t n; };
    std::vector<Symbol> syms;
    for (std::size_t off = 0; off < norm.size();) {
        const std::size_t len = std::min(utf8_len(static_cast<unsigned char>(norm[off])),
                                         norm.size() - off);
        const int idx = static_cast<int>(syms.size());
        syms.push_back({idx - 1, -1, norm.data() + off, len});
        off += len;
    }
    for (std::size_t i = 0; i + 1 < syms.size(); ++i)
        syms[i].next = static_cast<int>(i + 1);

    // Max-heap of merge candidates by score (tie-break: earlier left index).
    struct Bigram {
        int left, right; float score; std::size_t size;
        bool operator<(const Bigram& o) const {
            return score != o.score ? score < o.score : left > o.left;
        }
    };
    std::priority_queue<Bigram> pq;
    auto try_add = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        const std::string piece(syms[static_cast<std::size_t>(left)].text,
                                syms[static_cast<std::size_t>(left)].n
                                + syms[static_cast<std::size_t>(right)].n);
        auto it = piece_to_id_.find(piece);
        if (it == piece_to_id_.end()) return;
        pq.push({left, right, scores_[static_cast<std::size_t>(it->second)], piece.size()});
    };
    for (std::size_t i = 0; i + 1 < syms.size(); ++i)
        try_add(static_cast<int>(i), static_cast<int>(i + 1));

    while (!pq.empty()) {
        const Bigram b = pq.top(); pq.pop();
        Symbol& l = syms[static_cast<std::size_t>(b.left)];
        Symbol& r = syms[static_cast<std::size_t>(b.right)];
        if (l.n == 0 || r.n == 0 || l.n + r.n != b.size) continue;  // stale
        l.n += r.n; r.n = 0;
        l.next = r.next;
        if (r.next >= 0) syms[static_cast<std::size_t>(r.next)].prev = b.left;
        try_add(l.prev, b.left);
        try_add(b.left, l.next);
    }

    std::vector<int32_t> out;
    if (add_bos_ && bos_id_ >= 0) out.push_back(bos_id_);
    for (int i = 0; i >= 0; i = syms[static_cast<std::size_t>(i)].next) {
        const Symbol& s = syms[static_cast<std::size_t>(i)];
        if (s.n == 0) continue;
        const std::string piece(s.text, s.n);
        if (auto it = piece_to_id_.find(piece); it != piece_to_id_.end()) {
            out.push_back(it->second);
        } else {
            // Byte fallback: emit each byte as its <0xNN> token.
            for (std::size_t k = 0; k < s.n; ++k) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>",
                              static_cast<unsigned char>(s.text[k]));
                if (auto bit = piece_to_id_.find(buf); bit != piece_to_id_.end())
                    out.push_back(bit->second);
            }
        }
    }
    return out;
}

std::string SPTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(id_to_piece_.size())) continue;
        if (id == bos_id_ || id == eos_id_) continue;
        const std::string& p = id_to_piece_[static_cast<std::size_t>(id)];
        // Byte token <0xNN> → raw byte.
        if (p.size() == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
            out.push_back(static_cast<char>(std::strtol(p.substr(3, 2).c_str(), nullptr, 16)));
            continue;
        }
        // ▁ → space.
        for (std::size_t i = 0; i < p.size();) {
            if (p.compare(i, kSpace.size(), kSpace) == 0) { out.push_back(' '); i += kSpace.size(); }
            else { out.push_back(p[i]); ++i; }
        }
    }
    return out;
}

} // namespace sub0llm
