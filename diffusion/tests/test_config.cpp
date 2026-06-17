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
