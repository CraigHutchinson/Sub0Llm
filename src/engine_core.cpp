// engine_core.cpp — backend-agnostic part of the engine core (part of libsub0_core).
//
// Holds everything that does NOT depend on how the math is executed: model
// serialization (the checkpoint header + the flat parameter blob), the runtime BPE
// tokenizer, the logits sampler, and the baked-in artifact paths. The differentiable
// compute (forward / backward / train_batch / AdamW) and the parameter arenas live
// in a compute backend translation unit (backend_cpu.cpp today; a CUDA backend
// later). This file reaches the parameters only through the backend's public
// accessors — params_ptr() / trainable_floats() — bracketed by the host/device
// sync hooks, so the same serialization works whether the live weights are in host
// memory (CPU) or device memory (GPU).

#include "sub0/core.hpp"
#include "sub0/casing.hpp"
#include "sub0/layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <print>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sub0 {

// ============================================================================
//  Serialization (header validates the generated config)
// ============================================================================
// The on-disk format is one backend-agnostic blob: a Header that pins the constexpr
// config, then PARAM_FLOATS floats in PARAM_LAYOUT order. Reading/writing goes
// through params_ptr() so a device backend can stage the weights via the sync hooks
// without changing the file format.

namespace {
struct Header {
    char magic[4] = {'S', '0', 'L', '4'};
    int d_model = D_MODEL, n_layers = N_LAYERS, n_heads = N_HEADS;
    int d_ff = D_FF, seq_len = SEQ_LEN, vocab = VOCAB, ternary = USE_TERNARY;
    uint64_t param_floats = PARAM_FLOATS;
};
}

void save_model(const char* path) {
    sync_params_to_host();                       // pull live weights into the host staging buffer
    std::ofstream os(path, std::ios::binary);
    Header h;
    os.write((const char*)&h, sizeof(h));
    os.write((const char*)params_ptr(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
}

bool load_model(const char* path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    Header h, ref;
    is.read((char*)&h, sizeof(h));
    if (std::memcmp(h.magic, ref.magic, 4) != 0 ||
        h.d_model != ref.d_model || h.n_layers != ref.n_layers || h.n_heads != ref.n_heads ||
        h.d_ff != ref.d_ff || h.seq_len != ref.seq_len || h.vocab != ref.vocab ||
        h.ternary != ref.ternary || h.param_floats != ref.param_floats) {
        std::println(stderr, "error: model was built with a different (constexpr) config");
        return false;
    }
    is.read((char*)params_ptr(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
    if (!is) return false;
    sync_params_to_device();                     // push the loaded weights to the live (device) copy
    return true;
}

const char* default_corpus()     { return DEFAULT_CORPUS; }
const char* default_corpus_tok() { return DEFAULT_CORPUS_TOK; }
const char* default_tokenizer()  { return DEFAULT_TOKENIZER; }

// ============================================================================
//  Runtime BPE tokenizer (loaded from tokenizer.bin)
// ============================================================================

namespace {

struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(static_cast<std::uint32_t>(p.first)) << 32) ^
               static_cast<std::uint32_t>(p.second);
    }
};

struct Tokenizer {
    bool loaded = false;
    int  vocab  = 0;
    int  n_base = 0;
    std::array<int, 256> byte_base{};  // byte value -> base id (-1 if unused)
    int  cap_id = -1, up_id = -1;      // base ids of the case markers
    std::vector<std::vector<int>> expansion;  // id -> base symbol codes (0-255, 256, 257)
    std::unordered_map<std::pair<int, int>, int, PairHash> merge_rank;  // (left,right) -> merge index
    std::unordered_set<std::string> attested;  // lowercase words eligible for case collapse

    int sym_to_base(int code) const {
        if (code == casing::TOK_CAP) return cap_id;
        if (code == casing::TOK_UP)  return up_id;
        return byte_base[static_cast<unsigned char>(code)];
    }
};

Tokenizer g_tok;

template <class T> T read_pod(std::ifstream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof v);
    return v;
}

// Apply learned merges to one pre-token word (sequence of base ids), lowest rank
// first, then append the resulting ids to `out`. This reproduces the corpus
// tokenization for the same word, keeping prompts in-distribution.
void bpe_encode_word(std::vector<int>& seq, std::vector<int>& out) {
    while (seq.size() >= 2) {
        int best_rank = std::numeric_limits<int>::max(), best_pos = -1;
        for (std::size_t k = 0; k + 1 < seq.size(); ++k) {
            auto it = g_tok.merge_rank.find({seq[k], seq[k + 1]});
            if (it != g_tok.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = static_cast<int>(k);
            }
        }
        if (best_pos < 0) break;
        seq[static_cast<std::size_t>(best_pos)] = g_tok.n_base + best_rank;
        seq.erase(seq.begin() + best_pos + 1);
    }
    for (int id : seq) out.push_back(id);
}

}  // namespace

bool load_tokenizer(const char* path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::println(stderr, "tokenizer: cannot open '{}'", path); return false; }
    if (read_pod<std::uint32_t>(is) != 0x5A543053u) {  // "S0TZ"
        std::println(stderr, "tokenizer: bad magic in '{}'", path);
        return false;
    }
    Tokenizer t;
    t.vocab  = static_cast<int>(read_pod<std::uint32_t>(is));
    t.n_base = static_cast<int>(read_pod<std::uint32_t>(is));
    t.byte_base.fill(-1);
    t.expansion.resize(static_cast<std::size_t>(t.n_base));
    for (int i = 0; i < t.n_base; ++i) {
        const int code = static_cast<int>(read_pod<std::uint16_t>(is));
        t.expansion[static_cast<std::size_t>(i)] = {code};
        if (code == casing::TOK_CAP)     t.cap_id = i;
        else if (code == casing::TOK_UP) t.up_id  = i;
        else                             t.byte_base[static_cast<unsigned char>(code)] = i;
    }
    const int n_merges = static_cast<int>(read_pod<std::uint32_t>(is));
    t.expansion.reserve(static_cast<std::size_t>(t.n_base + n_merges));
    for (int i = 0; i < n_merges; ++i) {
        const int a = static_cast<int>(read_pod<std::uint32_t>(is));
        const int b = static_cast<int>(read_pod<std::uint32_t>(is));
        t.merge_rank.emplace(std::pair{a, b}, i);
        std::vector<int> exp = t.expansion[static_cast<std::size_t>(a)];
        exp.insert(exp.end(), t.expansion[static_cast<std::size_t>(b)].begin(),
                   t.expansion[static_cast<std::size_t>(b)].end());
        t.expansion.push_back(std::move(exp));
    }
    const int n_words = static_cast<int>(read_pod<std::uint32_t>(is));
    for (int i = 0; i < n_words; ++i) {
        const int len = static_cast<int>(read_pod<std::uint16_t>(is));
        std::string w(static_cast<std::size_t>(len), '\0');
        is.read(w.data(), len);
        t.attested.insert(std::move(w));
    }
    if (!is) { std::println(stderr, "tokenizer: truncated '{}'", path); return false; }
    if (t.vocab != VOCAB)
        std::println(stderr, "tokenizer: vocab {} != built-in VOCAB {} (stale artifact?)", t.vocab, VOCAB);
    t.loaded = true;
    g_tok = std::move(t);
    return true;
}

std::vector<int> encode(const std::string& text) {
    std::vector<int> out;
    if (!g_tok.loaded) return out;
    long replaced = 0;
    const std::string norm = casing::normalize_text(text, replaced);
    const std::vector<int> stream = casing::truecase_tokenize(norm, g_tok.attested, nullptr);

    // Mirror the configurator's pre-tokenization exactly (casing::word_unit_end):
    // word units are letter runs incl. accented UTF-8 and interior apostrophes, the
    // case markers stay atomic. A prompt's "They" thus encodes as <|cap|> + the
    // shared `they` token, in-distribution with training.
    out.reserve(stream.size());
    for (std::size_t i = 0, n = stream.size(); i < n;) {
        const std::size_t end = casing::word_unit_end(stream, i);
        if (end == i) {
            const int id = g_tok.sym_to_base(stream[i]);
            if (id >= 0) out.push_back(id);
            ++i;
            continue;
        }
        std::vector<int> seq;
        for (std::size_t k = i; k < end; ++k) {
            const int id = g_tok.sym_to_base(stream[k]);
            if (id >= 0) seq.push_back(id);
        }
        bpe_encode_word(seq, out);
        i = end;
    }
    return out;
}

std::string detokenize(const std::vector<int>& ids) {
    std::vector<int> stream;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(g_tok.expansion.size())) continue;
        const std::vector<int>& e = g_tok.expansion[static_cast<std::size_t>(id)];
        stream.insert(stream.end(), e.begin(), e.end());
    }
    return casing::detokenize(stream);
}

namespace {
// Render a token's expansion (base-symbol codes) into a printable string: case
// markers become <|cap|>/<|up|>, control and high bytes are escaped, so the table
// stays one line per token regardless of the bytes a merge happens to contain.
std::string render_expansion(const std::vector<int>& codes) {
    std::string s;
    for (int code : codes) {
        if (code == casing::TOK_CAP)      { s += "<|cap|>"; continue; }
        if (code == casing::TOK_UP)       { s += "<|up|>";  continue; }
        const unsigned char c = static_cast<unsigned char>(code);
        switch (c) {
            case '\n': s += "\\n"; break;
            case '\t': s += "\\t"; break;
            case '\r': s += "\\r"; break;
            default:
                if (c >= 0x20 && c < 0x7F) s += static_cast<char>(c);
                else { char b[5]; std::snprintf(b, sizeof b, "\\x%02X", c); s += b; }
        }
    }
    return s;
}
}  // namespace

std::vector<TokenEntry> vocab_entries() {
    std::vector<TokenEntry> rows;
    if (!g_tok.loaded) return rows;
    rows.reserve(g_tok.expansion.size());
    for (std::size_t id = 0; id < g_tok.expansion.size(); ++id) {
        const std::vector<int>& exp = g_tok.expansion[id];
        TokenEntry e;
        e.id = static_cast<int>(id);
        if (static_cast<int>(id) >= g_tok.n_base)        e.kind = TokenEntry::Kind::Merge;
        else if (!exp.empty() && exp[0] == casing::TOK_CAP) e.kind = TokenEntry::Kind::CapMarker;
        else if (!exp.empty() && exp[0] == casing::TOK_UP)  e.kind = TokenEntry::Kind::UpMarker;
        else                                                e.kind = TokenEntry::Kind::Byte;
        e.expansion_len = static_cast<int>(exp.size());
        e.text = render_expansion(exp);
        rows.push_back(std::move(e));
    }
    return rows;
}

int sample_token(const float* logits, float temp, int topk, std::mt19937& rng) {
    std::array<float, VOCAB> l;
    const float invT = 1.f / std::max(1e-6f, temp);
    for (int j = 0; j < VOCAB; ++j) l[j] = logits[j] * invT;
    if (topk > 0 && topk < VOCAB) {                       // keep only the top-k logits
        std::array<int, VOCAB> idx;
        for (int j = 0; j < VOCAB; ++j) idx[j] = j;
        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                          [&](int a, int b) { return l[a] > l[b]; });
        std::array<float, VOCAB> keep;
        keep.fill(-1e30f);
        for (int t = 0; t < topk; ++t) keep[idx[t]] = l[idx[t]];
        l = keep;
    }
    float mx = -1e30f;
    for (float x : l) mx = std::max(mx, x);
    float Z = 0.f;
    for (float& x : l) { x = std::exp(x - mx); Z += x; }  // softmax (unnormalized; scale r by Z)
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    float r = ud(rng) * Z, acc = 0.f;
    for (int j = 0; j < VOCAB; ++j) { acc += l[j]; if (r <= acc) return j; }
    return VOCAB - 1;
}

}  // namespace sub0
