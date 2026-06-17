// test_config.cpp — the reflected config module: CLI override, JSON round-trip,
// layered resolution (defaults → file → CLI), the build/run scope split, and the SHA.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sub0diff/config/run_config.hpp"
#include "sub0llm/config/json.hpp"
#include "sub0llm/config/schema.hpp"

namespace cfg  = sub0diff::config;
namespace scfg = sub0llm::config;

namespace {
// resolve() takes (argc, argv); build a mutable argv from string literals.
cfg::RunConfig resolve_args(std::vector<std::string> args) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("ch29"));
    for (auto& a : args) argv.push_back(a.data());
    return cfg::resolve(static_cast<int>(argv.size()), argv.data());
}
}  // namespace

TEST_CASE("config defaults are the founded proportions", "[config]") {
    const cfg::RunConfig c = resolve_args({});
    CHECK(c.model.embed_dim == 256);
    CHECK(c.model.n_layers == 6);
    CHECK(c.model.n_heads == 8);
    CHECK(c.model.d_ff == 1024);
    CHECK(c.optim.optimizer == "adamw");
    CHECK(c.optim.shared_t == true);
    CHECK(c.optim.exact_noise == true);
}

TEST_CASE("CLI flags auto-derive from the field table", "[config]") {
    const cfg::RunConfig c = resolve_args(
        {"--embed-dim", "128", "--n-layers", "4", "--lr", "5e-4",
         "--corpus", "data/x.txt", "--curriculum-k-step", "2"});
    CHECK(c.model.embed_dim == 128);
    CHECK(c.model.n_layers == 4);
    CHECK(c.optim.lr == 5e-4f);
    CHECK(c.optim.lr_explicit == true);          // --lr sets the explicit marker
    CHECK(c.data.corpus == "data/x.txt");
    CHECK(c.curriculum.curriculum_k_step == 2);
}

TEST_CASE("presence-bool flags and --no- negation", "[config]") {
    SECTION("presence sets true") {
        const cfg::RunConfig c = resolve_args({"--track-recall", "--per-t"});
        CHECK(c.diag.track_recall == true);
        CHECK(c.diag.per_t == true);
    }
    SECTION("--no- negates a default-true bool") {
        const cfg::RunConfig c = resolve_args({"--no-shared-t", "--no-exact-noise"});
        CHECK(c.optim.shared_t == false);
        CHECK(c.optim.exact_noise == false);
    }
}

TEST_CASE("flag aliases resolve to the canonical field", "[config]") {
    const cfg::RunConfig c = resolve_args({"--workers", "8", "--batch", "16"});
    CHECK(c.optim.threads == 8);
    CHECK(c.optim.batch == 16);
}

TEST_CASE("founded flag re-asserts proportions; later flags still override", "[config]") {
    const cfg::RunConfig c = resolve_args({"--embed-dim", "64", "--founded", "--n-layers", "12"});
    CHECK(c.model.embed_dim == 256);   // founded overrode the earlier --embed-dim 64
    CHECK(c.model.n_layers == 12);     // but the later --n-layers wins over founded
}

TEST_CASE("unknown flag and missing value are rejected", "[config]") {
    CHECK_THROWS(resolve_args({"--not-a-flag", "1"}));
    CHECK_THROWS(resolve_args({"--embed-dim"}));   // missing value
}

TEST_CASE("mutually-exclusive curricula are rejected", "[config]") {
    CHECK_THROWS(resolve_args({"--curriculum", "--curriculum-converge"}));
}

TEST_CASE("JSON round-trip preserves every field", "[config]") {
    cfg::RunConfig a;
    a.model.embed_dim = 192;
    a.model.n_layers = 5;
    a.data.corpus = "data/round\"trip.txt";   // embedded quote exercises escaping
    a.optim.lr = 3.25e-4f;
    a.optim.batch = 24;
    a.optim.shared_t = false;
    a.train.eval_factor = 0.75;
    a.curriculum.curriculum_converge = true;
    a.curriculum.curriculum_k_step = 3;
    a.diag.overfit = 16;

    const std::string json = scfg::to_json(a);
    auto doc = scfg::JsonDoc::parse(json);
    REQUIRE(doc.has_value());

    cfg::RunConfig b;   // fresh defaults
    scfg::load_json(b, *doc);

    CHECK(b.model.embed_dim == 192);
    CHECK(b.model.n_layers == 5);
    CHECK(b.data.corpus == "data/round\"trip.txt");
    CHECK(b.optim.lr == 3.25e-4f);
    CHECK(b.optim.batch == 24);
    CHECK(b.optim.shared_t == false);
    CHECK(b.train.eval_factor == 0.75);
    CHECK(b.curriculum.curriculum_converge == true);
    CHECK(b.curriculum.curriculum_k_step == 3);
    CHECK(b.diag.overfit == 16);
}

TEST_CASE("layered resolution: file fills, CLI overrides", "[config]") {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "ch29_cfg_test";
    fs::create_directories(dir);

    cfg::RunConfig saved;
    saved.model.embed_dim = 320;
    saved.data.corpus = "data/saved_corpus.txt";
    saved.optim.batch = 32;
    cfg::write_run_config(dir, saved, "deadbeef");

    const std::string d = dir.string();
    // Bare --ckpt-dir X recovers the saved settings …
    cfg::RunConfig loaded = resolve_args({"--ckpt-dir", d});
    CHECK(loaded.model.embed_dim == 320);
    CHECK(loaded.data.corpus == "data/saved_corpus.txt");
    CHECK(loaded.optim.batch == 32);

    // … and a flag overrides just that field, keeping the rest from the file.
    cfg::RunConfig overridden = resolve_args({"--ckpt-dir", d, "--batch", "8"});
    CHECK(overridden.model.embed_dim == 320);   // still from file
    CHECK(overridden.optim.batch == 8);         // CLI wins

    fs::remove_all(dir);
}

TEST_CASE("config_sha tracks the model shape only", "[config]") {
    cfg::RunConfig a, b;
    CHECK(cfg::config_sha(a) == cfg::config_sha(b));

    b.optim.lr = 9.9e-3f;          // a Runtime field …
    b.diag.track_recall = true;
    CHECK(cfg::config_sha(a) == cfg::config_sha(b));   // … does NOT change the sha

    b.model.n_layers = 12;          // a BuildTime field …
    CHECK(cfg::config_sha(a) != cfg::config_sha(b));   // … does
}

TEST_CASE("config_sha is stable across a JSON round-trip", "[config]") {
    cfg::RunConfig a;
    a.model.embed_dim = 384;
    a.model.n_heads = 6;
    a.optim.lr = 1.23e-4f;          // Runtime change must not perturb the (BuildTime) sha
    const auto doc = scfg::JsonDoc::parse(scfg::to_json(a));
    REQUIRE(doc.has_value());
    cfg::RunConfig b;
    scfg::load_json(b, *doc);
    CHECK(cfg::config_sha(a) == cfg::config_sha(b));
}

TEST_CASE("to_json scope filter emits only the requested scope", "[config]") {
    cfg::Model m;
    const std::string build = scfg::to_json(m, /*all=*/false, scfg::Scope::BuildTime);
    const std::string runtime = scfg::to_json(m, /*all=*/false, scfg::Scope::Runtime);
    // Model fields are all BuildTime: present under BuildTime, absent under Runtime.
    CHECK(build.find("embed_dim") != std::string::npos);
    CHECK(build.find("n_layers") != std::string::npos);
    CHECK(runtime.find("embed_dim") == std::string::npos);
}

TEST_CASE("write_run_config stamps meta and stays loadable", "[config]") {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "ch29_cfg_meta_test";
    fs::create_directories(dir);
    cfg::RunConfig c;
    c.model.embed_dim = 200;
    cfg::write_run_config(dir, c, "abc1234");

    auto doc = scfg::JsonDoc::parse_file(dir / "run_config.json");
    REQUIRE(doc.has_value());
    CHECK(doc->str("_code_sha").value_or("") == "abc1234");   // meta present
    CHECK(doc->has("_config_sha"));
    // The schema loader ignores the meta keys and still restores the fields.
    cfg::RunConfig back;
    scfg::load_json(back, *doc);
    CHECK(back.model.embed_dim == 200);
    fs::remove_all(dir);
}

TEST_CASE("JsonDoc scalar coercion, missing keys, and bad input", "[config][json]") {
    auto d = scfg::JsonDoc::parse(R"({"i":42,"f":1.5,"neg":-3,"b":true,"s":"hi","fi":64.0})");
    REQUIRE(d.has_value());
    CHECK(d->i64("i").value() == 42);
    CHECK(d->i64("fi").value() == 64);            // float with no fraction → int
    CHECK(d->u64("i").value() == 42u);
    CHECK_FALSE(d->u64("neg").has_value());       // negative rejected as unsigned
    CHECK(d->f64("i").value() == 42.0);           // int widened to double
    CHECK(d->boolean("b").value() == true);
    CHECK(d->str("s").value() == "hi");
    CHECK_FALSE(d->i64("missing").has_value());   // absent key
    CHECK_FALSE(d->boolean("i").has_value());     // type mismatch
    CHECK(d->has("f"));
    CHECK_FALSE(d->has("nope"));

    CHECK_FALSE(scfg::JsonDoc::parse("{ not json").has_value());   // malformed → nullopt
    auto arr = scfg::JsonDoc::parse("[1,2,3]");                    // non-object root
    REQUIRE(arr.has_value());
    CHECK_FALSE(arr->i64("x").has_value());                       // no keys on an array
}
