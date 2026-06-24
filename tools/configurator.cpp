// configurator.cpp — stage 1 of the build chain ("sub0-configure").
//
// This tool is intentionally INDEPENDENT of the engine: it is the program that
// *produces* the compile-time configuration the engine is built against, so it
// cannot itself depend on that configuration. It is the single source of truth
// for tokenization: it reads the corpus, applies corpus-aware truecasing, learns
// a BPE vocabulary, tokenizes the whole corpus, and writes three artifacts:
//
//   - <out>            the constexpr header (model dims + final VOCAB + paths)
//   - corpus.tok       the tokenized corpus (int32 stream) consumed by training
//   - tokenizer.bin    the runtime tokenizer (base alphabet + merges + attested
//                      words) used only to encode prompts and detokenize output
//
// Training and generation therefore consume integer tokens directly; only
// generation maps token ids back to text for the user.
//
// Usage:
//   sub0-configure --corpus <path> -o <generated_config.hpp>
//                  [--dmodel N --layers N --heads N --seq N --ternary 0|1]
//                  [--vocab N --min-merge N]

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>

#include "sub0/casing.hpp"

namespace {

// Hash for an adjacent-symbol pair, used by the BPE merge counter.
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(static_cast<std::uint32_t>(p.first)) << 32) ^
               static_cast<std::uint32_t>(p.second);
    }
};

// Serialize a base-id sequence to a string key (two bytes per id; ids < 65536),
// so word units can index an unordered_map.
inline std::string seq_key(const std::vector<int>& s) {
    std::string k(s.size() * 2, '\0');
    for (std::size_t i = 0; i < s.size(); ++i) {
        k[2 * i]     = static_cast<char>(s[i] & 0xFF);
        k[2 * i + 1] = static_cast<char>((s[i] >> 8) & 0xFF);
    }
    return k;
}

}  // namespace

using namespace sub0::casing;

int main(int argc, char** argv) {
    CLI::App app{"sub0-configure: truecase + BPE a corpus, emit the tokenized corpus and a constexpr config header"};

    std::string corpus;
    std::string out;
    int d_model      = 96;
    int n_layers     = 3;
    int n_heads      = 4;
    int seq_len      = 64;
    int ternary      = 0;
    int vocab_target = 2048;
    int min_merge    = 2;

    app.add_option("--corpus", corpus,  "Training corpus path (drives vocabulary derivation)")
       ->required()->check(CLI::ExistingFile);
    app.add_option("-o",       out,     "Output generated header path")->required();
    app.add_option("--dmodel", d_model, "Embedding / residual width")->capture_default_str();
    app.add_option("--layers", n_layers,"Transformer block count")->capture_default_str();
    app.add_option("--heads",  n_heads, "Attention head count")->capture_default_str();
    app.add_option("--seq",    seq_len, "Context window length")->capture_default_str();
    app.add_option("--ternary",ternary, "1 = BitNet-style ternary block weights")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--vocab",  vocab_target, "Target BPE vocabulary size (base symbols + markers + merges)")
       ->capture_default_str();
    app.add_option("--min-merge", min_merge, "Stop merging once the best pair occurs fewer than this many times")
       ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (d_model % n_heads != 0) {
        std::println(stderr, "configure error: dmodel ({}) not divisible by heads ({})",
                     d_model, n_heads);
        return 1;
    }

    // ----------------------------------------------------------------------
    // Read the corpus.
    // ----------------------------------------------------------------------
    std::ifstream is(corpus, std::ios::binary);
    if (!is) {
        std::println(stderr, "configure error: cannot read corpus '{}'", corpus);
        return 1;
    }
    const std::string text{std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()};
    if (text.empty()) {
        std::println(stderr, "configure error: empty corpus");
        return 1;
    }
    const std::string abspath = std::filesystem::absolute(corpus).string();

    // 1. Fold fancy typographic glyphs (curly quotes/apostrophes, dashes, ellipsis)
    //    to ASCII so they neither fragment into multibyte tokens nor split words.
    long quote_repl = 0;
    const std::string norm = normalize_text(text, quote_repl);

    // 2. Attested lowercase words license Capitalized/UPPER collapses; genuine
    //    names (whose lowercase never appears) stay verbatim.
    std::unordered_set<std::string> attested;
    for (std::size_t i = 0, n = norm.size(); i < n;) {
        const unsigned char c = static_cast<unsigned char>(norm[i]);
        if (!is_alpha(c)) { ++i; continue; }
        std::size_t j = i;
        bool all_lower = true;
        while (j < n && is_alpha(static_cast<unsigned char>(norm[j]))) {
            if (!is_lower(static_cast<unsigned char>(norm[j]))) all_lower = false;
            ++j;
        }
        if (all_lower) attested.insert(norm.substr(i, j - i));
        i = j;
    }

    // 3. Truecase into a base-symbol stream (bytes + <|cap|>/<|up|> markers).
    TokStats st;
    const std::vector<int> stream = truecase_tokenize(norm, attested, &st);
    if (detokenize(stream) != norm) {
        std::println(stderr, "configure error: truecasing round-trip failed");
        return 1;
    }

    // 4. Fix the base alphabet: distinct byte values used, then the markers.
    std::array<int, 256> byte_base;
    byte_base.fill(-1);
    bool used_cap = false, used_up = false;
    for (int s : stream) {
        if (s == TOK_CAP)      used_cap = true;
        else if (s == TOK_UP)  used_up = true;
        else                   byte_base[s] = 1;
    }
    std::vector<int> base_symbol;  // base id -> symbol code (0..255 byte, 256 cap, 257 up)
    for (int b = 0; b < 256; ++b)
        if (byte_base[b] == 1) { byte_base[b] = static_cast<int>(base_symbol.size()); base_symbol.push_back(b); }
    int cap_id = -1, up_id = -1;
    if (used_cap) { cap_id = static_cast<int>(base_symbol.size()); base_symbol.push_back(TOK_CAP); }
    if (used_up)  { up_id  = static_cast<int>(base_symbol.size()); base_symbol.push_back(TOK_UP); }
    const int n_base = static_cast<int>(base_symbol.size());
    auto sym_to_base = [&](int s) {
        return s == TOK_CAP ? cap_id : s == TOK_UP ? up_id : byte_base[s];
    };

    // 5. Pre-tokenize into word units and standalone symbols. A unit is a run of
    //    word bytes (letters, accented UTF-8 letters, interior apostrophes) per
    //    casing::word_unit_end; each becomes one entry in the unique-word table.
    //    Everything else -- punctuation, spaces, AND the case markers -- is a
    //    standalone base token that never participates in merges. Keeping the
    //    markers atomic is the whole point of truecasing: BPE merges the lowercase
    //    letters only, so "They" tokenizes as <|cap|> + the same `they` token as
    //    lowercase "they" rather than a separate <|cap|>they merge.
    std::vector<int> items;                   // >=0 standalone base id; <0 -> -(word index + 1)
    items.reserve(stream.size());
    std::vector<std::vector<int>> word_syms;  // unique word -> base-id sequence (mutated by BPE)
    std::vector<long> word_freq;
    std::unordered_map<std::string, int> word_index;
    for (std::size_t i = 0, n = stream.size(); i < n;) {
        const std::size_t end = word_unit_end(stream, i);
        if (end == i) { items.push_back(sym_to_base(stream[i])); ++i; continue; }
        std::vector<int> seq;
        for (std::size_t k = i; k < end; ++k) seq.push_back(sym_to_base(stream[k]));
        const std::string key = seq_key(seq);
        auto it = word_index.find(key);
        int idx;
        if (it == word_index.end()) {
            idx = static_cast<int>(word_syms.size());
            word_index.emplace(key, idx);
            word_syms.push_back(std::move(seq));
            word_freq.push_back(0);
        } else {
            idx = it->second;
        }
        word_freq[idx] += 1;
        items.push_back(-(idx + 1));
        i = end;
    }

    // 6. BPE: greedily merge the most frequent adjacent pair (weighted by word
    //    frequency) until the target vocabulary size or the floor is reached.
    std::vector<std::vector<int>> expansion(static_cast<std::size_t>(n_base));
    for (int id = 0; id < n_base; ++id)
        expansion[static_cast<std::size_t>(id)] = {base_symbol[static_cast<std::size_t>(id)]};
    std::vector<std::pair<int, int>> merges;
    int vocab = n_base;
    while (vocab < vocab_target) {
        std::unordered_map<std::pair<int, int>, long, PairHash> pc;
        for (std::size_t w = 0; w < word_syms.size(); ++w) {
            const std::vector<int>& s = word_syms[w];
            const long f = word_freq[w];
            for (std::size_t k = 0; k + 1 < s.size(); ++k) pc[{s[k], s[k + 1]}] += f;
        }
        if (pc.empty()) break;
        std::pair<int, int> best{0, 0};
        long best_c = -1;
        for (const auto& [p, c] : pc)
            if (c > best_c || (c == best_c && p < best)) { best_c = c; best = p; }
        if (best_c < min_merge) break;

        const int new_id = vocab++;
        merges.push_back(best);
        std::vector<int> exp = expansion[static_cast<std::size_t>(best.first)];
        exp.insert(exp.end(), expansion[static_cast<std::size_t>(best.second)].begin(),
                   expansion[static_cast<std::size_t>(best.second)].end());
        expansion.push_back(std::move(exp));

        for (std::vector<int>& s : word_syms) {
            if (s.size() < 2) continue;
            std::vector<int> ns;
            ns.reserve(s.size());
            for (std::size_t k = 0; k < s.size();) {
                if (k + 1 < s.size() && s[k] == best.first && s[k + 1] == best.second) {
                    ns.push_back(new_id);
                    k += 2;
                } else {
                    ns.push_back(s[k]);
                    ++k;
                }
            }
            s.swap(ns);
        }
    }

    // 7. Emit the final token stream: standalone bytes plus each word's merged id
    //    sequence. Reconstruct the base stream to verify lossless tokenization.
    std::vector<std::int32_t> tokens;
    tokens.reserve(items.size());
    for (int v : items) {
        if (v >= 0) tokens.push_back(v);
        else for (int id : word_syms[static_cast<std::size_t>(-(v + 1))]) tokens.push_back(id);
    }
    std::vector<int> recon;
    recon.reserve(stream.size());
    for (std::int32_t id : tokens)
        recon.insert(recon.end(), expansion[static_cast<std::size_t>(id)].begin(),
                     expansion[static_cast<std::size_t>(id)].end());
    const bool tok_rt = (recon == stream);

    // 8. Write the tokenized corpus next to the header.
    const std::filesystem::path gen_dir  = std::filesystem::path(out).parent_path();
    const std::filesystem::path tok_path = gen_dir / "corpus.tok";
    const std::filesystem::path tkz_path = gen_dir / "tokenizer.bin";
    {
        std::ofstream ts(tok_path, std::ios::binary);
        if (!ts) { std::println(stderr, "configure error: cannot write '{}'", tok_path.string()); return 1; }
        const std::uint32_t magic  = 0x4B543053;  // "S0TK"
        const std::uint32_t vfield = static_cast<std::uint32_t>(vocab);
        const std::uint32_t nfield = static_cast<std::uint32_t>(tokens.size());
        ts.write(reinterpret_cast<const char*>(&magic),  sizeof magic);
        ts.write(reinterpret_cast<const char*>(&vfield), sizeof vfield);
        ts.write(reinterpret_cast<const char*>(&nfield), sizeof nfield);
        ts.write(reinterpret_cast<const char*>(tokens.data()),
                 static_cast<std::streamsize>(tokens.size() * sizeof(std::int32_t)));
    }

    // 9. Write the runtime tokenizer (base alphabet + merges + attested words).
    //    Generation uses this to encode prompts and detokenize output; training
    //    needs only corpus.tok.
    {
        std::ofstream tz(tkz_path, std::ios::binary);
        if (!tz) { std::println(stderr, "configure error: cannot write '{}'", tkz_path.string()); return 1; }
        auto wu32 = [&](std::uint32_t v) { tz.write(reinterpret_cast<const char*>(&v), sizeof v); };
        auto wu16 = [&](std::uint16_t v) { tz.write(reinterpret_cast<const char*>(&v), sizeof v); };
        wu32(0x5A543053);  // "S0TZ"
        wu32(static_cast<std::uint32_t>(vocab));
        wu32(static_cast<std::uint32_t>(n_base));
        for (int code : base_symbol) wu16(static_cast<std::uint16_t>(code));
        wu32(static_cast<std::uint32_t>(merges.size()));
        for (const auto& [a, b] : merges) { wu32(static_cast<std::uint32_t>(a)); wu32(static_cast<std::uint32_t>(b)); }
        wu32(static_cast<std::uint32_t>(attested.size()));
        for (const std::string& w : attested) {
            wu16(static_cast<std::uint16_t>(w.size()));
            tz.write(w.data(), static_cast<std::streamsize>(w.size()));
        }
    }

    // 10. Emit the constexpr header (model dims + VOCAB + artifact paths).
    std::ofstream os(out, std::ios::binary);
    if (!os) { std::println(stderr, "configure error: cannot write '{}'", out); return 1; }
    auto emit_path = [&](const char* name, const std::string& p) {
        os << "inline constexpr char " << name << "[] = \"";
        for (char c : p) { if (c == '\\' || c == '"') os << '\\'; os << c; }
        os << "\";\n";
    };
    os << "// AUTO-GENERATED by sub0-configure. Do not edit.\n";
    os << "// Model dimensions and the BPE vocabulary size are baked in at build time.\n";
    os << "#pragma once\n\n";
    os << "constexpr int  D_MODEL     = " << d_model  << ";\n";
    os << "constexpr int  N_LAYERS    = " << n_layers << ";\n";
    os << "constexpr int  N_HEADS     = " << n_heads  << ";\n";
    os << "constexpr int  SEQ_LEN     = " << seq_len  << ";\n";
    os << "constexpr int  D_FF        = 4 * D_MODEL;\n";
    os << "constexpr int  D_HEAD      = D_MODEL / N_HEADS;\n";
    os << "constexpr bool USE_TERNARY = " << (ternary ? "true" : "false") << ";\n";
    os << "constexpr int  VOCAB       = " << vocab << ";\n\n";
    emit_path("DEFAULT_CORPUS",     abspath);
    emit_path("DEFAULT_CORPUS_TOK", std::filesystem::absolute(tok_path).string());
    emit_path("DEFAULT_TOKENIZER",  std::filesystem::absolute(tkz_path).string());

    // 11. Report the real numbers.
    const double bytes_per_tok =
        tokens.empty() ? 0.0 : static_cast<double>(norm.size()) / static_cast<double>(tokens.size());
    std::size_t wordset_bytes = 0;
    for (const std::string& w : attested) wordset_bytes += w.size() + 1;

    std::println(stderr, "--- BPE truecasing tokenizer (real numbers) ---");
    std::println(stderr, "corpus bytes (raw / normalized): {} / {}", text.size(), norm.size());
    std::println(stderr, "quote glyphs collapsed:          {}", quote_repl);
    std::println(stderr, "alpha words (total / unique):    {} / {}", st.words, word_syms.size());
    std::println(stderr, "  collapsed <|cap|> / <|up|>:    {} / {}", st.cap, st.up);
    std::println(stderr, "  kept verbatim (names/mixed):   {}", st.names);
    std::println(stderr, "base symbols / merges / vocab:   {} / {} / {}", n_base, merges.size(), vocab);
    std::println(stderr, "total tokens:                    {}", tokens.size());
    std::println(stderr, "compression (bytes/token):       {:.3f}", bytes_per_tok);
    std::println(stderr, "attested words / table size:     {} / {} bytes ({:.1f} KB)",
                 attested.size(), wordset_bytes, wordset_bytes / 1024.0);
    std::println(stderr, "round-trip truecase / tokenize:  OK / {}", tok_rt ? "OK" : "FAIL");
    std::println(stderr, "corpus.tok / tokenizer.bin:      {} | {}", tok_path.string(), tkz_path.string());
    std::println(stderr, "-----------------------------------------------");
    if (!tok_rt) { std::println(stderr, "configure error: tokenization round-trip failed"); return 1; }

    std::println("sub0-configure: vocab={} (base {} + {} merges), d={} L={} H={} seq={}{} -> {}",
                 vocab, n_base, merges.size(), d_model, n_layers, n_heads, seq_len,
                 ternary ? " (ternary)" : "", out);
    return 0;
}
