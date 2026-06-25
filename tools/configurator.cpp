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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
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

constexpr std::size_t CHUNK_BYTES = 32u << 20;  // ~32 MB per streamed corpus chunk

// Stream the corpus to `fn` one newline-aligned chunk at a time, so it is never held in
// memory all at once -- the out-of-core path for a corpus larger than RAM. Newlines are
// safe split points: they never fall inside a UTF-8 glyph, a word unit, or the look-back
// the name detector and truecaser use (which already treats a newline as a line start),
// so processing the corpus chunk-by-chunk is byte-identical to processing it whole. The
// trailing partial line is carried into the next chunk; the final remainder is flushed at
// EOF. Returns false only if the file cannot be opened.
template <class Fn>
bool for_each_chunk(const std::string& path, Fn&& fn) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    std::string buf, block(CHUNK_BYTES, '\0');
    for (;;) {
        is.read(block.data(), static_cast<std::streamsize>(block.size()));
        const std::streamsize got = is.gcount();
        if (got > 0) buf.append(block.data(), static_cast<std::size_t>(got));
        if (got <= 0 || is.eof()) {                       // last read: flush the remainder
            if (!buf.empty()) fn(std::string_view{buf});
            break;
        }
        const std::size_t nl = buf.rfind('\n');           // split at the last full line
        if (nl == std::string::npos) continue;            // no boundary yet: read more
        fn(std::string_view{buf.data(), nl + 1});
        buf.erase(0, nl + 1);
    }
    return true;
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
    int emit_tok     = 1;
    std::string tune_cache;

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
    app.add_option("--corpus-tok", emit_tok,
                   "1 = pre-tokenize the whole corpus to corpus.tok; 0 = skip it and let training tokenize "
                   "on demand (avoids the ~2x-on-disk token copy for a huge corpus)")
       ->capture_default_str()->check(CLI::Range(0, 1));
    app.add_option("--tune-cache", tune_cache,
                   "Persisted tuned-defaults cache; read to bake DEFAULT_THREADS / DEFAULT_WINDOWS_PER_THREAD")
       ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (d_model % n_heads != 0) {
        std::println(stderr, "configure error: dmodel ({}) not divisible by heads ({})",
                     d_model, n_heads);
        return 1;
    }

    const std::string abspath = std::filesystem::absolute(corpus).string();

    // ----------------------------------------------------------------------
    // Out-of-core tokenization: the corpus is STREAMED in newline-aligned chunks (never
    // held whole in memory) across three sequential passes that keep only bounded tables
    // resident -- the unique-word frequency table (saturates by Heaps' law), the base
    // alphabet, the case-marker stats and the BPE merge list. corpus.tok is written
    // incrementally. This is byte-identical to whole-corpus processing because a newline
    // never splits a glyph, a word unit, or the truecaser/name-detector look-back.
    // ----------------------------------------------------------------------

    // --- Pass 1: decide which lowercase forms license a Capitalized/UPPER -> marker
    //     collapse. A form qualifies if it appears lowercase AND is not really a proper
    //     noun. The tell is *position*: sentence-initial capitals are positional
    //     ("The ..."), but a capital following a lowercase word mid-sentence ("...to Spot")
    //     is a name. Per lowercase form we count lowercase uses vs mid-sentence-capital
    //     uses; a form whose name uses dominate is withheld from `attested`, keeping e.g.
    //     the dog "Spot" verbatim instead of folding it onto the noun <|cap|>spot.
    std::size_t raw_bytes = 0, norm_bytes = 0;
    long quote_repl = 0;
    std::unordered_map<std::string, long> lower_count, midcap_count;
    const bool open_ok = for_each_chunk(corpus, [&](std::string_view chunk) {
        raw_bytes += chunk.size();
        long qr = 0;
        const std::string norm = normalize_text(std::string(chunk), qr);
        quote_repl += qr;
        norm_bytes += norm.size();
        // Chunk-initial words follow a newline, hence are line starts -- matching the
        // whole-corpus rule (newline is a line start), so no cross-chunk look-back needed.
        auto preceded_by_lowercase = [&](std::size_t start) {
            std::size_t k = start;
            while (k > 0 && (norm[k - 1] == ' ' || norm[k - 1] == '\t')) --k;
            return k > 0 && is_lower(static_cast<unsigned char>(norm[k - 1]));
        };
        for (std::size_t i = 0, n = norm.size(); i < n;) {
            const unsigned char c = static_cast<unsigned char>(norm[i]);
            if (!is_alpha(c)) { ++i; continue; }
            std::size_t j = i;
            bool all_lower = true, rest_lower = true;
            while (j < n && is_alpha(static_cast<unsigned char>(norm[j]))) {
                const unsigned char ch = static_cast<unsigned char>(norm[j]);
                if (!is_lower(ch)) all_lower = false;
                if (j > i && !is_lower(ch)) rest_lower = false;
                ++j;
            }
            const std::string w = norm.substr(i, j - i);
            if (all_lower) {
                lower_count[w] += 1;
            } else if (is_upper(static_cast<unsigned char>(w[0])) && rest_lower) {  // "Spot", "The"
                if (preceded_by_lowercase(i)) {
                    std::string lw = w;
                    lw[0] = static_cast<char>(to_lower(static_cast<unsigned char>(lw[0])));
                    midcap_count[lw] += 1;
                }
            }
            i = j;
        }
    });
    if (!open_ok)      { std::println(stderr, "configure error: cannot read corpus '{}'", corpus); return 1; }
    if (raw_bytes == 0) { std::println(stderr, "configure error: empty corpus"); return 1; }

    std::unordered_set<std::string> attested;
    long names_withheld = 0;
    for (const auto& [w, lc] : lower_count) {
        const auto it = midcap_count.find(w);
        const long mid = (it == midcap_count.end()) ? 0 : it->second;
        if (mid > lc) { ++names_withheld; continue; }  // name sense dominates -> keep verbatim
        attested.insert(w);
    }
    lower_count = {}; midcap_count = {};   // free; only `attested` is needed onward

    // --- Pass 2: truecase each chunk into a base-symbol stream (bytes + <|cap|>/<|up|>
    //     markers), then pre-tokenize. A word unit (run of word bytes per word_unit_end)
    //     becomes one entry in the unique-word table; everything else -- punctuation,
    //     spaces, and the markers -- is a standalone symbol that never participates in
    //     merges (keeping markers atomic is the point of truecasing). The table is keyed by
    //     the RAW byte-symbol sequence so it is independent of the not-yet-fixed base
    //     alphabet; we also record which byte values / markers actually occur.
    TokStats st;
    std::array<int, 256> byte_used; byte_used.fill(0);
    bool used_cap = false, used_up = false;
    std::vector<std::vector<int>> word_syms;   // unique word -> symbol sequence (raw byte values)
    std::vector<long> word_freq;
    std::unordered_map<std::string, int> word_index;
    for_each_chunk(corpus, [&](std::string_view chunk) {
        long qr = 0;
        const std::string  norm   = normalize_text(std::string(chunk), qr);
        const std::vector<int> stream = truecase_tokenize(norm, attested, &st);
        for (std::size_t i = 0, n = stream.size(); i < n;) {
            const std::size_t end = word_unit_end(stream, i);
            if (end == i) {                                  // standalone symbol
                const int s = stream[i];
                if (s == TOK_CAP)      used_cap = true;
                else if (s == TOK_UP)  used_up = true;
                else                   byte_used[s] = 1;
                ++i; continue;
            }
            std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                 stream.begin() + static_cast<std::ptrdiff_t>(end));  // raw byte values
            for (int b : seq) byte_used[b] = 1;
            const std::string key = seq_key(seq);
            auto it = word_index.find(key);
            if (it == word_index.end()) {
                word_index.emplace(key, static_cast<int>(word_syms.size()));
                word_syms.push_back(std::move(seq));
                word_freq.push_back(1);
            } else {
                word_freq[static_cast<std::size_t>(it->second)] += 1;
            }
            i = end;
        }
    });

    // Fix the base alphabet: distinct byte values used (in byte order) then the markers --
    // the same ordering as before, so base ids match and BPE tie-breaking is unchanged.
    std::array<int, 256> byte_base; byte_base.fill(-1);
    std::vector<int> base_symbol;  // base id -> symbol code (0..255 byte, 256 cap, 257 up)
    for (int b = 0; b < 256; ++b)
        if (byte_used[b]) { byte_base[b] = static_cast<int>(base_symbol.size()); base_symbol.push_back(b); }
    int cap_id = -1, up_id = -1;
    if (used_cap) { cap_id = static_cast<int>(base_symbol.size()); base_symbol.push_back(TOK_CAP); }
    if (used_up)  { up_id  = static_cast<int>(base_symbol.size()); base_symbol.push_back(TOK_UP); }
    const int n_base = static_cast<int>(base_symbol.size());
    auto sym_to_base = [&](int s) { return s == TOK_CAP ? cap_id : s == TOK_UP ? up_id : byte_base[s]; };

    // Remap the word table from raw byte symbols to base ids. A bijection on used bytes, so
    // it preserves pair frequencies AND tie-breaking -> identical merges.
    for (std::vector<int>& w : word_syms)
        for (int& s : w) s = byte_base[s];

    // --- BPE: greedily merge the most frequent adjacent pair (weighted by word frequency)
    //     until the target vocabulary size or the per-pair floor is reached. Operates only
    //     on the bounded unique-word table -- no corpus-length state.
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

    const std::filesystem::path gen_dir  = std::filesystem::path(out).parent_path();
    const std::filesystem::path tok_path = gen_dir / "corpus.tok";
    const std::filesystem::path tkz_path = gen_dir / "tokenizer.bin";

    // --- Pass 3 (optional): stream the corpus once more and emit the merged token stream to
    //     corpus.tok incrementally (never materialised in memory). Skipped when --corpus-tok
    //     0: training then tokenizes windows on demand from the raw corpus + tokenizer.bin,
    //     avoiding the ~2x-on-disk token copy for a huge corpus. Each word unit looks up its
    //     post-BPE id sequence in the table; standalone symbols map straight to base ids.
    //     Losslessness is verified per chunk by reconstructing the base symbols; the id
    //     array's length is back-patched after the count is known.
    std::size_t token_count = 0;
    bool tok_rt = true;
    if (emit_tok) {
        std::ofstream ts(tok_path, std::ios::binary);
        if (!ts) { std::println(stderr, "configure error: cannot write '{}'", tok_path.string()); return 1; }
        {
            const std::uint32_t magic = 0x4B543053u, vfield = static_cast<std::uint32_t>(vocab), nfield = 0u;
            ts.write(reinterpret_cast<const char*>(&magic),  sizeof magic);
            ts.write(reinterpret_cast<const char*>(&vfield), sizeof vfield);
            ts.write(reinterpret_cast<const char*>(&nfield), sizeof nfield);  // patched below
        }
        std::vector<std::int32_t> out_tokens;     // per-chunk emit buffer (bounded by the chunk)
        std::vector<int> recon;                   // per-chunk reconstruction (bounded)
        for_each_chunk(corpus, [&](std::string_view chunk) {
            long qr = 0;
            const std::string norm = normalize_text(std::string(chunk), qr);
            const std::vector<int> stream = truecase_tokenize(norm, attested, nullptr);
            if (detokenize(stream) != norm) tok_rt = false;     // round-trip 1: truecasing
            out_tokens.clear();
            for (std::size_t i = 0, n = stream.size(); i < n;) {
                const std::size_t end = word_unit_end(stream, i);
                if (end == i) { out_tokens.push_back(sym_to_base(stream[i])); ++i; continue; }
                std::vector<int> seq(stream.begin() + static_cast<std::ptrdiff_t>(i),
                                     stream.begin() + static_cast<std::ptrdiff_t>(end));
                auto it = word_index.find(seq_key(seq));
                if (it == word_index.end()) {                   // unreachable if passes agree
                    for (int b : seq) out_tokens.push_back(byte_base[b]);
                    tok_rt = false;
                } else {
                    for (int id : word_syms[static_cast<std::size_t>(it->second)]) out_tokens.push_back(id);
                }
                i = end;
            }
            recon.clear();                                       // round-trip 2: tokenization
            for (std::int32_t id : out_tokens)
                recon.insert(recon.end(), expansion[static_cast<std::size_t>(id)].begin(),
                             expansion[static_cast<std::size_t>(id)].end());
            if (recon != stream) tok_rt = false;
            ts.write(reinterpret_cast<const char*>(out_tokens.data()),
                     static_cast<std::streamsize>(out_tokens.size() * sizeof(std::int32_t)));
            token_count += out_tokens.size();
        });
        {
            const std::uint32_t nfield = static_cast<std::uint32_t>(token_count);  // back-patch ntok
            ts.seekp(8, std::ios::beg);
            ts.write(reinterpret_cast<const char*>(&nfield), sizeof nfield);
        }
        ts.close();
    } else {
        // Drop any stale corpus.tok from a previous emit so training reliably falls back to
        // on-demand tokenization instead of mmapping an out-of-date token copy.
        std::error_code ec;
        std::filesystem::remove(tok_path, ec);
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

    // Cache the machine's logical-core count so the engine sizes its statically
    // allocated worker pool to the hardware instead of a fixed guess. If a previous
    // `tune` run persisted better runtime defaults, fold those in here too; otherwise
    // fall back to the hardware core count and a conservative windows/thread of 4.
    const unsigned hw_concurrency = std::max(1u, std::thread::hardware_concurrency());
    const int      max_workers    = static_cast<int>(hw_concurrency);
    int            default_threads = static_cast<int>(hw_concurrency);
    int            default_wpt     = 4;
    std::string    tune_cache_abs;
    if (!tune_cache.empty()) {
        tune_cache_abs = std::filesystem::absolute(tune_cache).string();
        std::ifstream tc(tune_cache);
        for (std::string line; std::getline(tc, line); ) {
            const auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = line.substr(0, eq);
            const int         val = std::atoi(line.substr(eq + 1).c_str());
            if      (key == "threads"            && val > 0) default_threads = val;
            else if (key == "windows_per_thread" && val > 0) default_wpt     = val;
        }
    }

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
    os << "// --- Cached hardware facts + persisted tuned runtime defaults -----------\n";
    os << "constexpr int  HW_CONCURRENCY            = " << hw_concurrency  << ";\n";
    os << "constexpr int  MAX_WORKERS               = " << max_workers     << ";\n";
    os << "constexpr int  DEFAULT_THREADS           = " << default_threads << ";\n";
    os << "constexpr int  DEFAULT_WINDOWS_PER_THREAD = " << default_wpt    << ";\n\n";
    emit_path("DEFAULT_CORPUS",     abspath);
    emit_path("DEFAULT_CORPUS_TOK", std::filesystem::absolute(tok_path).string());
    emit_path("DEFAULT_TOKENIZER",  std::filesystem::absolute(tkz_path).string());
    emit_path("DEFAULT_TUNE_CACHE", tune_cache_abs);

    // 11. Report the real numbers.
    const double bytes_per_tok =
        token_count == 0 ? 0.0 : static_cast<double>(norm_bytes) / static_cast<double>(token_count);
    std::size_t wordset_bytes = 0;
    for (const std::string& w : attested) wordset_bytes += w.size() + 1;

    std::println(stderr, "--- BPE truecasing tokenizer (real numbers) ---");
    std::println(stderr, "corpus bytes (raw / normalized): {} / {}", raw_bytes, norm_bytes);
    std::println(stderr, "quote glyphs collapsed:          {}", quote_repl);
    std::println(stderr, "alpha words (total / unique):    {} / {}", st.words, word_syms.size());
    std::println(stderr, "  collapsed <|cap|> / <|up|>:    {} / {}", st.cap, st.up);
    std::println(stderr, "  kept verbatim (names/mixed):   {}", st.names);
    std::println(stderr, "  names withheld (mid-sent cap): {}", names_withheld);
    std::println(stderr, "base symbols / merges / vocab:   {} / {} / {}", n_base, merges.size(), vocab);
    if (emit_tok) {
        std::println(stderr, "total tokens:                    {}", token_count);
        std::println(stderr, "compression (bytes/token):       {:.3f}", bytes_per_tok);
    } else {
        std::println(stderr, "corpus.tok:                      skipped (--corpus-tok 0; on-demand)");
    }
    std::println(stderr, "attested words / table size:     {} / {} bytes ({:.1f} KB)",
                 attested.size(), wordset_bytes, wordset_bytes / 1024.0);
    std::println(stderr, "round-trip truecase / tokenize:  OK / {}", tok_rt ? "OK" : "FAIL");
    std::println(stderr, "corpus.tok / tokenizer.bin:      {} | {}",
                 emit_tok ? tok_path.string() : std::string("(on-demand)"), tkz_path.string());
    std::println(stderr, "-----------------------------------------------");
    if (!tok_rt) { std::println(stderr, "configure error: tokenization round-trip failed"); return 1; }

    std::println("sub0-configure: vocab={} (base {} + {} merges), d={} L={} H={} seq={}{} -> {}",
                 vocab, n_base, merges.size(), d_model, n_layers, n_heads, seq_len,
                 ternary ? " (ternary)" : "", out);
    return 0;
}
