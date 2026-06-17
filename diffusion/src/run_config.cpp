// run_config.cpp — layered resolution (defaults → run_config.json → CLI) for Ch29.

#include "sub0diff/config/run_config.hpp"

#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "sub0llm/config/json.hpp"
#include "sub0llm/config/schema.hpp"

namespace sub0diff::config {

namespace cfgns = sub0llm::config;
namespace fs    = std::filesystem;

void apply_founded(Model& m) {
    m.embed_dim  = 256;
    m.n_layers   = 6;
    m.n_heads    = 8;     // head_dim = 256/8 = 32
    m.n_kv_heads = 4;     // 2:1 GQA
    m.d_ff       = 1024;  // 4·D
}

std::uint64_t config_sha(const RunConfig& cfg) {
    return cfgns::build_sha(cfg);
}

// Flag aliases → canonical field name. The flag→field mapping is otherwise mechanical
// (kebab→snake), so only genuine synonyms live here.
[[nodiscard]] static std::string canonical_field(std::string field) {
    if (field == "workers")    return "threads";
    if (field == "batch_size") return "batch";
    return field;
}

// Apply command-line overrides onto an already-defaulted-and-json-loaded config.
static void apply_cli(RunConfig& cfg, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg.size() < 2 || arg[0] != '-')
            throw std::runtime_error(std::format("unexpected argument: {}", arg));

        std::string field = canonical_field(cfgns::flag_to_field(arg));

        // --founded: re-assert the founded model proportions (later flags still override).
        if (field == "founded") {
            cfgns::set_bool(cfg, "founded", true);
            apply_founded(cfg.model);
            continue;
        }

        const cfgns::FieldKind kind = cfgns::field_kind(cfg, field);

        if (kind == cfgns::FieldKind::Bool) {            // presence flag → true
            cfgns::set_bool(cfg, field, true);
            continue;
        }
        if (kind == cfgns::FieldKind::None) {            // maybe a --no-<bool> negation
            if (field.rfind("no_", 0) == 0 &&
                cfgns::field_kind(cfg, field.substr(3)) == cfgns::FieldKind::Bool) {
                cfgns::set_bool(cfg, field.substr(3), false);
                continue;
            }
            throw std::runtime_error(std::format("unknown argument: {}", arg));
        }

        // Value field: consume the next token.
        if (i + 1 >= argc)
            throw std::runtime_error(std::format("missing value for {}", arg));
        const std::string value = argv[++i];
        if (!cfgns::set_field(cfg, field, value))
            throw std::runtime_error(std::format("invalid value '{}' for {}", value, arg));

        if (field == "lr") cfgns::set_bool(cfg, "lr_explicit", true);
    }
}

// Pre-scan only for --ckpt-dir, so we know where to look for run_config.json BEFORE the
// full layered apply (the CLI location wins over any default/saved value for the lookup).
[[nodiscard]] static std::string scan_ckpt_dir(int argc, char** argv, std::string fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string_view(argv[i]) == "--ckpt-dir") return argv[i + 1];
    return fallback;
}

RunConfig resolve(int argc, char** argv) {
    RunConfig cfg;   // 1. compiled-in defaults

    // 2. load run_config.json from the resolved checkpoint dir, if present
    const std::string ckpt_dir = scan_ckpt_dir(argc, argv, cfg.data.ckpt_dir);
    if (auto doc = cfgns::JsonDoc::parse_file(fs::path(ckpt_dir) / "run_config.json")) {
        cfgns::load_json(cfg, *doc);
        cfg.data.ckpt_dir = ckpt_dir;   // the dir we loaded from is authoritative
    }

    // 3. command-line overrides (each explicit flag wins over default and saved value)
    apply_cli(cfg, argc, argv);

    // Validation — the one combination the trainer cannot honour.
    if (cfg.curriculum.curriculum && cfg.curriculum.curriculum_converge)
        throw std::runtime_error("--curriculum and --curriculum-converge are mutually exclusive");

    return cfg;
}

void write_run_config(const fs::path& dir, const RunConfig& cfg, std::string_view code_sha) {
    fs::create_directories(dir);
    std::ofstream f(dir / "run_config.json");
    if (!f) return;
    // Meta header (code_sha + config_sha tag this run/build) followed by the flat field
    // object. We hand-stitch the meta keys ahead of the schema's object body so the file
    // is one well-formed JSON object that the schema loader reads back (it ignores the
    // unknown meta keys).
    std::string body = cfgns::to_json(cfg);   // "{\n  \"field\": value,\n ... \n}\n"
    // Insert the meta lines right after the opening brace.
    const std::string meta = std::format(
        "  \"_code_sha\": \"{}\",\n  \"_config_sha\": \"{:016x}\",\n",
        code_sha, config_sha(cfg));
    const auto brace = body.find('{');
    if (brace != std::string::npos)
        body.insert(brace + 2, meta);   // after "{\n"
    f << body;
}

}  // namespace sub0diff::config
