// sub0llm-tokenizer — an engine-free frontend tool over a built tokenizer.tok: WS7's interchange
// export (see docs/TOKENIZER_REVIEW.md's WS7 section and docs/WORKFLOW_ARCHITECTURE.md's
// "Tokenizer / vocab as engine-free frontend tools" section). Links sub0_frontend only, never the
// engine -- inspecting/exporting a tokenizer needs nothing more. `encode`/`decode`/`roundtrip`/
// `vocab` are explicitly out of scope for this pass (see WORKFLOW_ARCHITECTURE.md's own staged
// sequencing); this tool currently has exactly one subcommand, `export`.

#include "sub0/tok_interchange.hpp"
#include "sub0/modality.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <string>
#include <utility>
#include <vector>

using sub0::tok::Interchange;
using sub0::tok::Tokenizer;
using sub0::tok::TokenKind;

namespace {

// Minimal, always-valid JSON string escaping. `text` holds RAW BYTES (not guaranteed valid UTF-8 --
// see TokInterchange.hpp's own doc comment), so every non-ASCII-printable byte is escaped as \u00XX
// rather than passed through, keeping the output valid JSON regardless of what a piece's bytes are.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    static const char* hex = "0123456789abcdef";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20 || c >= 0x7F) {
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xF];
                    out += hex[c & 0xF];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

const char* kind_name(TokenKind k) {
    switch (k) {
        case TokenKind::Byte:   return "byte";
        case TokenKind::Marker: return "marker";
        case TokenKind::Piece:  return "piece";
    }
    return "?";
}

void write_json(const Interchange& ic, std::ostream& os) {
    os << "{\n";
    os << "  \"eos_id\": " << ic.eos_id << ",\n";
    os << "  \"bos_id\": " << ic.bos_id << ",\n";
    os << "  \"pad_id\": " << ic.pad_id << ",\n";
    os << "  \"unk_id\": " << ic.unk_id << ",\n";
    os << "  \"tokens\": [\n";
    for (std::size_t i = 0; i < ic.tokens.size(); ++i) {
        const auto& r = ic.tokens[i];
        os << "    {\"id\": " << r.id << ", \"text\": \"" << json_escape(r.text)
           << "\", \"score\": " << r.score << ", \"kind\": \"" << kind_name(r.kind) << "\"}"
           << (i + 1 < ic.tokens.size() ? "," : "") << "\n";
    }
    os << "  ]\n}\n";
}

// TSV: one row per token, backslash-escaping the same control bytes JSON escapes (so a row never
// spans multiple lines and never contains a literal tab inside the text field, either of which
// would desync a naive column split).
std::string tsv_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += static_cast<char>(c);
        }
    }
    return out;
}

void write_tsv(const Interchange& ic, std::ostream& os) {
    os << "id\ttext\tscore\tkind\n";
    for (const auto& r : ic.tokens)
        os << r.id << '\t' << tsv_escape(r.text) << '\t' << r.score << '\t' << kind_name(r.kind) << '\n';
}

namespace mod = sub0::modality;

// Stream up to `max_bytes` of a corpus in chunks, normalize each (the tokenizer's real view), and
// accumulate per-codepoint modality. A codepoint split across a 16 MB chunk boundary is a negligible
// miscount over a real corpus.
mod::ModalityStats scan_corpus(const std::string& path, std::size_t max_bytes) {
    mod::ModalityStats st;
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::println(stderr, "  cannot open {}", path); return st; }
    std::string buf(16u << 20, '\0'), norm;
    std::size_t done = 0;
    long rep = 0;
    while (done < max_bytes) {
        const std::size_t want = std::min(buf.size(), max_bytes - done);
        f.read(buf.data(), static_cast<std::streamsize>(want));
        const std::size_t got = static_cast<std::size_t>(f.gcount());
        if (!got) break;
        done += got;
        sub0::casing::normalize_text(std::string_view(buf.data(), got), rep, norm);
        mod::add_modality(st, norm);
    }
    return st;
}

// Dominant-modality table, descending by count.
void print_modality_table(const mod::ModalityStats& st, int top) {
    std::vector<std::pair<std::uint32_t, mod::CharModality>> rows(st.chars.begin(), st.chars.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.total() > b.second.total(); });
    std::println("{:<14} {:>13}  {:>5} {:>5} {:>5} {:>5}  {:<12} {}", "char", "count", "SS%", "SG%", "GS%", "GG%", "dominant", "shape");
    int shown = 0;
    for (const auto& [cp, cm] : rows) {
        if (cm.total() < 100) break;
        if (top && shown++ >= top) break;
        const double t = static_cast<double>(cm.total());
        const std::string label = (cp >= 0x21 && cp <= 0x7E)
            ? std::format("'{}' U+{:04X}", static_cast<char>(cp), cp)
            : std::format("U+{:04X}", cp);
        std::println("{:<14} {:>13}  {:>5.1f} {:>5.1f} {:>5.1f} {:>5.1f}  {:<12} {}",
                     label, cm.total(), 100.0 * cm.n[0] / t, 100.0 * cm.n[1] / t, 100.0 * cm.n[2] / t,
                     100.0 * cm.n[3] / t, mod::combo_name(cm.dominant()), cm.bimodal() ? "BIMODAL" : "unimodal");
    }
}

// calibrate: scan corpora for per-character spacing modality, collate into one ledger, and flag any
// corpus whose dominant modality for a character contradicts the accumulated finding.
int run_calibrate(const std::vector<std::string>& corpora, const std::string& load_path,
                  const std::string& out_path, std::size_t max_mb) {
    mod::ModalityStats acc;
    if (!load_path.empty()) {
        std::ifstream in(load_path);
        if (in) {
            mod::deserialize(acc, in);
            std::println(stderr, "loaded ledger '{}' ({} codepoints, {} bytes)", load_path, acc.chars.size(), acc.scanned_bytes);
        }
    }
    const std::size_t cap = max_mb * (1u << 20);
    for (const std::string& c : corpora) {
        std::println(stderr, "scanning {} (<= {} MB)...", c, max_mb);
        const mod::ModalityStats fresh = scan_corpus(c, cap);
        if (!acc.chars.empty()) {
            const auto bad = mod::find_contradictions(acc, fresh);
            if (bad.empty()) std::println(stderr, "  OK: no dominant-modality contradictions vs the ledger");
            for (const auto& x : bad)
                std::println(stderr, "  MISMATCH U+{:04X}: ledger says {} (n={}) but this corpus says {} (n={})",
                             x.cp, mod::combo_name(x.prior_dom), x.prior_n, mod::combo_name(x.fresh_dom), x.fresh_n);
        }
        mod::merge(acc, fresh);
    }
    std::println(stderr, "\n=== collated modality over {:.2f} GB scanned, {} codepoints ===", acc.scanned_bytes / 1e9, acc.chars.size());
    print_modality_table(acc, 48);
    if (!out_path.empty()) {
        std::ofstream o(out_path);
        if (!o) { std::println(stderr, "cannot write '{}'", out_path); return 1; }
        mod::serialize(acc, o);
        std::println(stderr, "wrote ledger '{}'", out_path);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"sub0llm-tokenizer — engine-free tokenizer inspection/interchange"};
    app.require_subcommand(1);

    CLI::App* exp = app.add_subcommand("export", "Export the full vocabulary as JSON or TSV");
    std::string tok_path, out_path, format = "json";
    exp->add_option("--tokenizer", tok_path, "Path to a built tokenizer.tok")->required()->check(CLI::ExistingFile);
    exp->add_option("-o,--out", out_path, "Output path (default: stdout)");
    exp->add_option("--format", format, "json (default) | tsv")
       ->check(CLI::IsMember({"json", "tsv"}));

    CLI::App* cal = app.add_subcommand("calibrate",
        "Scan corpora for per-character spacing modality; collate into one ledger and flag mismatches");
    std::vector<std::string> corpora;
    std::string cal_load, cal_out;
    std::size_t cal_max_mb = 512;
    cal->add_option("corpus", corpora, "Corpus file(s) to scan")->required()->check(CLI::ExistingFile);
    cal->add_option("--load", cal_load, "Existing modality ledger to accumulate into / compare against")->check(CLI::ExistingFile);
    cal->add_option("-o,--out", cal_out, "Write the merged ledger here");
    cal->add_option("--max-mb", cal_max_mb, "Cap bytes scanned per corpus (default 512)");

    CLI11_PARSE(app, argc, argv);

    if (*cal) return run_calibrate(corpora, cal_load, cal_out, cal_max_mb);

    std::ifstream is(tok_path, std::ios::binary);
    Tokenizer t;
    if (!is || !sub0::tok::deserialize(t, is)) {
        std::println(stderr, "sub0llm-tokenizer: cannot read tokenizer '{}'", tok_path);
        return 1;
    }
    const Interchange ic = sub0::tok::to_interchange(t);

    std::ofstream ofs;
    if (!out_path.empty()) {
        ofs.open(out_path, std::ios::binary);
        if (!ofs) {
            std::println(stderr, "sub0llm-tokenizer: cannot write '{}'", out_path);
            return 1;
        }
    }
    std::ostream& os = out_path.empty() ? std::cout : ofs;
    if (format == "tsv") write_tsv(ic, os); else write_json(ic, os);
    return 0;
}
