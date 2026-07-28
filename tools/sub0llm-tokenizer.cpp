// sub0llm-tokenizer — an engine-free frontend tool over a built tokenizer.tok: WS7's interchange
// export (see docs/TOKENIZER_REVIEW.md's WS7 section and docs/WORKFLOW_ARCHITECTURE.md's
// "Tokenizer / vocab as engine-free frontend tools" section). Links sub0_frontend only, never the
// engine -- inspecting/exporting a tokenizer needs nothing more. `encode`/`decode`/`roundtrip`/
// `vocab` are explicitly out of scope for this pass (see WORKFLOW_ARCHITECTURE.md's own staged
// sequencing); this tool currently has exactly one subcommand, `export`.

#include "sub0/tok_interchange.hpp"

#include <CLI/CLI.hpp>

#include <fstream>
#include <iostream>
#include <print>
#include <string>

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

    CLI11_PARSE(app, argc, argv);

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
