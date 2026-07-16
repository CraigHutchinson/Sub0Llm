// blend_schedule_tests.cpp -- parse/validation tests for sub0::parse_blend_schedule_json
// (src/blend_schedule.cpp, sub0/blend_schedule.hpp's ScheduleSpec). Real file I/O (deliberately: this is
// exactly the path train_stage.cpp uses), a scratch dir unique per test instance -- see run_config_tests.cpp's
// own ScratchDir comment for why a fixed shared path is a real, previously-hit flake under `ctest -j`.

#include <catch2/catch_test_macros.hpp>

#include "sub0/blend_schedule.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using sub0::ScheduleSpec;

namespace {
struct ScratchDir {
    std::filesystem::path path;
    ScratchDir() {
        std::random_device rd;
        path = std::filesystem::temp_directory_path()
             / ("sub0_blend_schedule_test_" + std::to_string(rd()) + "_" + std::to_string(rd()));
        std::filesystem::create_directories(path);
    }
    ~ScratchDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

std::filesystem::path write_json(const ScratchDir& dir, const std::string& content) {
    const std::filesystem::path p = dir.path / "schedule.json";
    std::ofstream os(p);
    os << content;
    return p;
}
}  // namespace

TEST_CASE("blend_schedule: a valid multi-source, multi-stage schedule parses cleanly", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources": [
        { "name": "base",    "corpus": "data/tinystories.txt" },
        { "name": "scratch", "generator": "scratchspike", "params": { "n_oov": 400, "tasks_per_oov": 12, "contains_k": 4 } },
        { "name": "op",      "generator": "op_curriculum", "gsm8k": "data/gsm8k.txt" }
      ],
      "schedule": [
        { "until_epoch": 0.5,   "weights": { "base": 1.0 } },
        { "until_epoch": 0.9,   "weights": { "base": 0.75, "scratch": 0.25 } },
        { "until_epoch": "end", "weights": { "base": 0.6,  "scratch": 0.25, "op": 0.15 } }
      ],
      "content_embed": "hrr",
      "expected_plateau_epoch": 2.0
    })");

    ScheduleSpec spec;
    std::string error;
    std::vector<std::string> warnings;
    REQUIRE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.empty());
    CHECK(warnings.empty());

    REQUIRE(spec.sources.size() == 3);
    CHECK(spec.sources[0].name == "base");
    CHECK(spec.sources[0].corpus == "data/tinystories.txt");
    CHECK(spec.sources[1].name == "scratch");
    CHECK(spec.sources[1].generator == "scratchspike");
    CHECK(spec.sources[1].n_oov == 400);
    CHECK(spec.sources[1].tasks_per_oov == 12);
    CHECK(spec.sources[1].contains_k == 4);
    CHECK(spec.sources[2].name == "op");
    CHECK(spec.sources[2].generator == "op_curriculum");
    CHECK(spec.sources[2].gsm8k_path == "data/gsm8k.txt");

    REQUIRE(spec.stages.size() == 3);
    CHECK(spec.stages[0].until_epoch == 0.5);
    CHECK(spec.stages[2].until_epoch == std::numeric_limits<double>::infinity());

    REQUIRE(spec.content_embed.has_value());
    CHECK(*spec.content_embed == sub0::ContentEmbedKind::HRR);
    REQUIRE(spec.expected_plateau_epoch.has_value());
    CHECK(*spec.expected_plateau_epoch == 2.0);
}

TEST_CASE("blend_schedule: the trivial single-source, single-stage case parses", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "data/tinystories.txt" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ]
    })");
    ScheduleSpec spec;
    std::string error;
    std::vector<std::string> warnings;
    REQUIRE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK_FALSE(spec.content_embed.has_value());          // absent field -> off
    CHECK_FALSE(spec.expected_plateau_epoch.has_value()); // absent field -> nullopt
}

TEST_CASE("blend_schedule: missing file fails cleanly", "[blend_schedule]") {
    ScheduleSpec spec;
    std::string error;
    std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json("Z:/definitely/does/not/exist.json", spec, error, warnings));
    CHECK_FALSE(error.empty());
}

TEST_CASE("blend_schedule: malformed JSON fails cleanly", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, "{ this is not valid json ");
    ScheduleSpec spec;
    std::string error;
    std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK_FALSE(error.empty());
}

TEST_CASE("blend_schedule: a source with NEITHER corpus nor generator is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("EXACTLY ONE") != std::string::npos);
}

TEST_CASE("blend_schedule: a source with BOTH corpus and generator is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "x.txt", "generator": "scratchspike" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("EXACTLY ONE") != std::string::npos);
}

TEST_CASE("blend_schedule: duplicate source names are an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources": [
        { "name": "base", "corpus": "a.txt" },
        { "name": "base", "generator": "scratchspike" }
      ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("duplicate") != std::string::npos);
}

TEST_CASE("blend_schedule: zero corpus-typed sources is an error (epoch needs a base corpus to be "
         "defined against)", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "scratch", "generator": "scratchspike" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "scratch": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("exactly one") != std::string::npos);
}

TEST_CASE("blend_schedule: more than one corpus-typed source is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources": [
        { "name": "a", "corpus": "a.txt" },
        { "name": "b", "corpus": "b.txt" }
      ],
      "schedule": [ { "until_epoch": "end", "weights": { "a": 1.0, "b": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("at most one") != std::string::npos);
}

TEST_CASE("blend_schedule: a stage referencing an undeclared source is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "nope": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("undeclared source") != std::string::npos);
}

TEST_CASE("blend_schedule: a stage with no positive weight is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 0.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("positive weight") != std::string::npos);
}

TEST_CASE("blend_schedule: an empty weights object is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [ { "until_epoch": "end", "weights": {} } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
}

TEST_CASE("blend_schedule: the \"end\" stage must be last", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [
        { "until_epoch": "end", "weights": { "base": 1.0 } },
        { "until_epoch": 0.9,   "weights": { "base": 1.0 } }
      ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
}

TEST_CASE("blend_schedule: the last stage must be \"end\" (schedule must cover the whole run)",
         "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [ { "until_epoch": 0.9, "weights": { "base": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("\"end\"") != std::string::npos);
}

TEST_CASE("blend_schedule: non-ascending until_epoch across stages is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [
        { "until_epoch": 0.9, "weights": { "base": 1.0 } },
        { "until_epoch": 0.5, "weights": { "base": 1.0 } },
        { "until_epoch": "end", "weights": { "base": 1.0 } }
      ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK(error.find("ascending") != std::string::npos);
}

TEST_CASE("blend_schedule: a source never given a positive weight anywhere WARNS, doesn't fail",
         "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources": [
        { "name": "base",   "corpus": "a.txt" },
        { "name": "unused", "generator": "scratchspike" }
      ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ]
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    REQUIRE(warnings.size() == 1);
    CHECK(warnings[0].find("unused") != std::string::npos);
}

TEST_CASE("blend_schedule: an unrecognized content_embed value is an error", "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ],
      "content_embed": "banana"
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
}

TEST_CASE("blend_schedule: content_embed \"off\" resolves to no encoding, same as an absent field",
         "[blend_schedule]") {
    ScratchDir dir;
    const auto p = write_json(dir, R"({
      "sources":  [ { "name": "base", "corpus": "a.txt" } ],
      "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ],
      "content_embed": "off"
    })");
    ScheduleSpec spec; std::string error; std::vector<std::string> warnings;
    REQUIRE(sub0::parse_blend_schedule_json(p, spec, error, warnings));
    CHECK_FALSE(spec.content_embed.has_value());
}

TEST_CASE("blend_schedule: missing sources or schedule array is an error", "[blend_schedule]") {
    ScratchDir dir1;
    const auto p1 = write_json(dir1, R"({ "schedule": [ { "until_epoch": "end", "weights": { "base": 1.0 } } ] })");
    ScheduleSpec spec1; std::string error1; std::vector<std::string> warnings1;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p1, spec1, error1, warnings1));

    ScratchDir dir2;
    const auto p2 = write_json(dir2, R"({ "sources": [ { "name": "base", "corpus": "a.txt" } ] })");
    ScheduleSpec spec2; std::string error2; std::vector<std::string> warnings2;
    REQUIRE_FALSE(sub0::parse_blend_schedule_json(p2, spec2, error2, warnings2));
}
