// run_config_tests.cpp — sub0::registry::RunConfig round-trip (config.json write/read via
// simdjson::ondemand). Born from a real incident: a GPU-fault auto-resume once silently switched
// Muon->AdamW mid-training because the optimizer choice had no persisted, read-back-on-resume home
// (see registry.hpp's RunConfig doc comment). These tests prove the write/read round-trip itself is
// correct; train_stage.cpp's own resume path is where the fix is actually wired in (not covered
// here -- this target links sub0_core/sub0_frontend only, not sub0_train).

#include <catch2/catch_test_macros.hpp>

#include "sub0/registry.hpp"

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

using sub0::registry::RunConfig;

namespace {
// A scratch directory unique to this ScratchDir INSTANCE, cleaned up on scope exit -- config.json
// writes are real file I/O (deliberately: this is exactly the path train_stage.cpp uses).
// `catch_discover_tests` registers each TEST_CASE as its own ctest entry, invoked as a SEPARATE
// `sub0_tests.exe --test-case=...` process -- `ctest -j` runs several of those concurrently. A fixed
// literal path here (the original form) let two concurrently-running RunConfig test PROCESSES race on
// the same file, an observed flake (a test fails only under -j, passes in isolation or serially). A
// random suffix per instance closes it without needing a platform-specific process-id API.
struct ScratchDir {
    std::filesystem::path path;
    ScratchDir() {
        std::random_device rd;
        path = std::filesystem::temp_directory_path()
             / ("sub0_run_config_test_" + std::to_string(rd()) + "_" + std::to_string(rd()));
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    ~ScratchDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
}  // namespace

TEST_CASE("RunConfig: write_config_json/read_config_json round-trips every field", "[registry][run_config]") {
    ScratchDir scratch;
    RunConfig c;
    c.corpus = "tinystories";
    c.d_model = 512; c.n_layers = 12; c.n_heads = 8; c.seq_len = 256; c.vocab = 16517;
    c.ternary = 0; c.pos_encoding = 1; c.gated_ffn = 1; c.tied_embeddings = 1; c.qk_norm = 1;
    c.optimizer = 1;      // muon -- the field this whole struct exists to protect
    c.batch = 385;
    c.lr = 0.00693722;
    c.seed = 42;
    c.config_schema = 2;

    sub0::registry::write_config_json(c, scratch.path);
    REQUIRE(std::filesystem::exists(scratch.path / "config.json"));

    RunConfig r;
    REQUIRE(sub0::registry::read_config_json(r, scratch.path));
    CHECK(r.corpus == c.corpus);
    CHECK(r.d_model == c.d_model);
    CHECK(r.n_layers == c.n_layers);
    CHECK(r.n_heads == c.n_heads);
    CHECK(r.seq_len == c.seq_len);
    CHECK(r.vocab == c.vocab);
    CHECK(r.ternary == c.ternary);
    CHECK(r.pos_encoding == c.pos_encoding);
    CHECK(r.gated_ffn == c.gated_ffn);
    CHECK(r.tied_embeddings == c.tied_embeddings);
    CHECK(r.qk_norm == c.qk_norm);
    CHECK(r.optimizer == c.optimizer);
    CHECK(r.batch == c.batch);
    CHECK(r.lr == c.lr);
    CHECK(r.seed == c.seed);
    CHECK(r.config_schema == c.config_schema);
}

TEST_CASE("RunConfig: enum fields write as NAMES; numeric config.json still reads (back-compat)", "[registry][run_config]") {
    ScratchDir scratch;
    RunConfig c;
    c.d_model = 512; c.pos_encoding = 1; c.optimizer = 1; c.gated_ffn = 1; c.tied_embeddings = 0; c.qk_norm = 1; c.ternary = 0;
    c.config_schema = 2;
    sub0::registry::write_config_json(c, scratch.path);

    // The written file uses readable names for the enum fields, plain numbers for the rest.
    std::string text;
    { std::ifstream is(scratch.path / "config.json"); text.assign(std::istreambuf_iterator<char>(is), {}); }
    CHECK(text.find("\"pos_encoding\": \"rope\"")    != std::string::npos);
    CHECK(text.find("\"optimizer\": \"muon\"")       != std::string::npos);
    CHECK(text.find("\"gated_ffn\": \"on\"")         != std::string::npos);
    CHECK(text.find("\"tied_embeddings\": \"off\"")  != std::string::npos);
    CHECK(text.find("\"qk_norm\": \"on\"")           != std::string::npos);
    CHECK(text.find("\"d_model\": 512")              != std::string::npos);   // numeric field unchanged
    CHECK(text.find("\"config_schema\": 2")          != std::string::npos);

    // A hand-written OLD-STYLE config.json (predating the blend-schedule redesign, so it still carries
    // the now-removed spell_mix/scratch_mix/op_mix/content_embed/content_embed_kind keys and never wrote
    // config_schema at all) still loads without failing -- the unrecognized keys are silently ignored
    // (the same forward-compatible tolerance proven generically below), and config_schema lands on its
    // struct default (1), the load-bearing signal train_stage.cpp/gen_stage.cpp use to refuse operating
    // on a model this old rather than silently reinterpreting its removed fields as absent/off.
    { std::ofstream os(scratch.path / "config.json");
      os << "{\n  \"pos_encoding\": 1,\n  \"optimizer\": 1,\n  \"gated_ffn\": 0,\n  \"qk_norm\": 1,\n"
            "  \"d_model\": 448,\n  \"scratch_mix\": 0.3,\n  \"content_embed\": \"on\",\n"
            "  \"content_embed_kind\": \"hrr\"\n}\n"; }
    RunConfig r;
    REQUIRE(sub0::registry::read_config_json(r, scratch.path));
    CHECK(r.pos_encoding == 1); CHECK(r.optimizer == 1); CHECK(r.gated_ffn == 0); CHECK(r.qk_norm == 1); CHECK(r.d_model == 448);
    CHECK(r.config_schema == 1);   // absent key -> struct default -- "this model predates the redesign"
}

TEST_CASE("RunConfig: read_config_json returns false for a missing file, doesn't crash", "[registry][run_config]") {
    ScratchDir scratch;   // never written to -- no config.json exists in it
    RunConfig r;
    REQUIRE_FALSE(sub0::registry::read_config_json(r, scratch.path));
}

TEST_CASE("RunConfig: read_config_json on a missing directory returns false, doesn't crash", "[registry][run_config]") {
    RunConfig r;
    REQUIRE_FALSE(sub0::registry::read_config_json(r, "Z:/definitely/does/not/exist/anywhere"));
}

TEST_CASE("RunConfig: read_config_json tolerates a corrupt file (not valid JSON)", "[registry][run_config]") {
    ScratchDir scratch;
    std::error_code ec;
    std::filesystem::create_directories(scratch.path, ec);
    {
        std::ofstream os(scratch.path / "config.json");
        os << "{ this is not valid json ";
    }
    RunConfig r;
    REQUIRE_FALSE(sub0::registry::read_config_json(r, scratch.path));
}

TEST_CASE("RunConfig: read_config_json is forward-compatible with an unrecognized extra key", "[registry][run_config]") {
    ScratchDir scratch;
    std::error_code ec;
    std::filesystem::create_directories(scratch.path, ec);
    {
        // A hand-written file with a field this build's RunConfig doesn't know about (simulating a
        // FUTURE build having added a new SUB0_RUN_CONFIG_FIELDS row) mixed in with known ones --
        // must not fail the whole parse, matching read_state's existing forward-compatible tolerance.
        std::ofstream os(scratch.path / "config.json");
        os << "{\n  \"optimizer\": 1,\n  \"some_future_field\": \"unknown to this build\",\n  \"batch\": 96\n}\n";
    }
    RunConfig r;
    REQUIRE(sub0::registry::read_config_json(r, scratch.path));
    CHECK(r.optimizer == 1);
    CHECK(r.batch == 96);
}

TEST_CASE("RunConfig: corpus text with characters needing JSON escaping round-trips correctly", "[registry][run_config]") {
    ScratchDir scratch;
    RunConfig c;
    c.corpus = "weird \"corpus\" name\\with\nbackslashes\tand\rcontrol chars";
    sub0::registry::write_config_json(c, scratch.path);
    RunConfig r;
    REQUIRE(sub0::registry::read_config_json(r, scratch.path));
    CHECK(r.corpus == c.corpus);
}

// The concrete incident this struct exists to prevent, expressed as a data-level regression test
// (the actual resume-time RECONCILIATION logic lives in train_stage.cpp/sub0_train, not tested
// here -- this proves the persisted value a resume would read back is exactly what was written,
// i.e. the round-trip itself introduces no silent value drift for the field that mattered).
TEST_CASE("RunConfig: optimizer choice survives the round-trip distinctly from the default", "[registry][run_config]") {
    ScratchDir scratch;
    RunConfig muon;
    muon.optimizer = 1;
    sub0::registry::write_config_json(muon, scratch.path);

    RunConfig readback;
    REQUIRE(sub0::registry::read_config_json(readback, scratch.path));
    REQUIRE(readback.optimizer == 1);              // NOT silently defaulted back to 0 (AdamW)
    REQUIRE(readback.optimizer != RunConfig{}.optimizer);   // distinct from a fresh default instance
}
