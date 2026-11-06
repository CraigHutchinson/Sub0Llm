// profile_tests.cpp -- sub0/profile.hpp: the profile-file parser and directory discovery.
// Engine-free (no sub0_config.hpp). The configurator owns the key -> variable mapping, so unknown-key
// rejection is exercised by running the tool, not here; this file pins the file format itself.

#include "sub0/profile.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>

using namespace sub0::profile;

TEST_CASE("profile: parses keys, name, aliases, comments and blank lines", "[profile]") {
    const auto r = parse("# header\nname = Foo\nalias = a, B ,c\n\ndmodel = 64   # trailing\nrope-theta=1e7\n", "t");
    REQUIRE(r.ok);
    CHECK(r.profile.name == "Foo");
    REQUIRE(r.profile.aliases.size() == 3);
    CHECK(r.profile.aliases[1] == "B");
    REQUIRE(r.profile.values.size() == 2);
    CHECK(r.profile.values[0] == std::pair<std::string, std::string>{"dmodel", "64"});
    CHECK(r.profile.values[1].second == "1e7");   // '#' only starts a comment after whitespace
    CHECK(parse("x = a#b\n", "t").profile.values[0].second == "a#b");
}

TEST_CASE("profile: malformed input names the source and line", "[profile]") {
    for (const char* bad : {"just words\n", "dmodel =\n", "Bad_Key = 1\n", "dmodel = 1\ndmodel = 2\n", "name = a\nname = b\n"}) {
        const auto r = parse(bad, "f.profile");
        INFO(bad);
        CHECK_FALSE(r.ok);
        CHECK(r.error.rfind("f.profile:", 0) == 0);
    }
    CHECK(parse("ok = 1\nnope\n", "f").error == "f:2: expected 'key = value', got 'nope'");
}

TEST_CASE("profile: matches is case-insensitive over name and aliases", "[profile]") {
    const auto p = parse("name = Qwen-A\nalias = q4\n", "t").profile;
    CHECK(matches(p, "qwen-a"));
    CHECK(matches(p, "Q4"));
    CHECK_FALSE(matches(p, "q5"));
}

TEST_CASE("profile: path-versus-name detection", "[profile]") {
    CHECK(looks_like_path("dir/x"));
    CHECK(looks_like_path("x.profile"));
    CHECK_FALSE(looks_like_path("qwen3.8-27b"));
    CHECK_FALSE(looks_like_path(".profile"));   // a bare extension is a name, not a file
}

TEST_CASE("profile: discover finds *.profile, defaults the name to the stem, reports bad files", "[profile]") {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "sub0_profile_discover_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    auto put = [&](const char* n, const char* body) { std::ofstream(dir / n) << body; };
    put("zeta.profile", "dmodel = 1\n");
    put("alpha.profile", "name = Alpha\nalias = a\n");
    put("broken.profile", "nonsense\n");
    put("ignored.txt", "dmodel = 9\n");

    const Discovered d = discover(dir);
    REQUIRE(d.profiles.size() == 2);
    CHECK(d.profiles[0].name == "Alpha");   // sorted by name
    CHECK(d.profiles[1].name == "zeta");    // defaulted from the file stem
    REQUIRE(d.errors.size() == 1);
    CHECK(d.errors[0].find("broken.profile") != std::string::npos);
    CHECK(discover(dir / "missing").profiles.empty());   // a missing directory is empty, not an error
    fs::remove_all(dir);
}

TEST_CASE("profile: model-* keys are metadata, not options", "[profile]") {
    const auto r = parse("model-card = A/b\nmodel-source = u/b-GGUF\nmodel-dir = d\nmodel-include = *.gguf\ndmodel = 8\n", "t");
    REQUIRE(r.ok);
    CHECK(r.profile.model_card == "A/b");
    CHECK(r.profile.model_source == "u/b-GGUF");
    CHECK(r.profile.model_dir == "d");
    CHECK(r.profile.model_include == "*.gguf");
    CHECK(r.profile.values.size() == 1);            // only dmodel is a configurator option
    CHECK_FALSE(parse("model-dir = a\nmodel-dir = b\n", "t").ok);
}

TEST_CASE("profile: model_status distinguishes no metadata, no models dir, missing and present", "[profile]") {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "sub0_profile_model_status_test";
    fs::remove_all(root);
    Profile none;
    CHECK(model_status(none, root).state == ModelState::NoMetadata);
    Profile p;
    p.model_dir = "M";
    CHECK(model_status(p, {}).state == ModelState::NoModelsDir);
    CHECK(model_status(p, root).state == ModelState::Missing);
    fs::create_directories(root / "M" / "UD-IQ1_S");
    CHECK(model_status(p, root).state == ModelState::Missing);        // folder exists but holds no .gguf
    std::ofstream(root / "M" / "UD-IQ1_S" / "a.gguf") << std::string(100, 'x');
    std::ofstream(root / "M" / "readme.txt") << "ignored";
    const ModelStatus st = model_status(p, root);
    CHECK(st.state == ModelState::Present);                           // found recursively
    CHECK(st.gguf_files == 1);
    CHECK(st.bytes == 100);
    fs::remove_all(root);
}
