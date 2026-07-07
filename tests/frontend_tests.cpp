// frontend_tests.cpp — engine-free unit tests for the FRONTEND (pre-model) logic that the configurator
// and the stage tools share but the engine doesn't define: the device-memory planner (sub0/memplan.hpp)
// and the model registry's identity/compat rules (sub0/registry.hpp). Both are header-only and std-only,
// so they run in the fast engine-free sub0_tok_tests target -- no engine build, no GPU. This is the
// payoff of the frontend separation: pre-model logic gets covered without the engine. These two areas
// were directly behind this session's work (the gpu_batch VRAM clamp + `models --prune`).

#include <catch2/catch_test_macros.hpp>

#include "sub0/memplan.hpp"
#include "sub0/registry.hpp"
#include "sub0/log.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

// --- memplan: the VRAM math behind the configurator clamp + the GPU tuner's batch cap --------------
TEST_CASE("memplan: training footprint is monotonic in batch and scales with the model", "[frontend][memplan]") {
    using namespace sub0::memplan;
    const Dims d192{192,  6, 6,  768, 256,  4096};   // tinystories-scale
    const Dims d448{448, 11, 7, 1792, 256, 16384};   // fineweb-smoke-scale

    // Resident footprint never decreases as the batch grows (max_batch_for_vram relies on this to invert it).
    CHECK(train_resident_mb(d192, 1)  <= train_resident_mb(d192, 8));
    CHECK(train_resident_mb(d192, 8)  <= train_resident_mb(d192, 64));
    CHECK(train_resident_mb(d192, 64) <= train_resident_mb(d192, 256));

    // A bigger model needs more VRAM at the same batch; bf16 activations cost no more than f32.
    CHECK(train_resident_mb(d448, 64) > train_resident_mb(d192, 64));
    CHECK(train_resident_mb(d448, 64, /*bf16*/2) <= train_resident_mb(d448, 64, /*f32*/4));
}

TEST_CASE("memplan: max_batch_for_vram inverts the footprint (clamp + tuner cap)", "[frontend][memplan]") {
    using namespace sub0::memplan;
    const Dims d448{448, 11, 7, 1792, 256, 16384};
    constexpr int cap = 4096;                        // MAX_FWD_BATCH device-scratch ceiling

    // No device budget known -> defer to the hard cap.
    CHECK(max_batch_for_vram(d448, 0, cap) == cap);

    // On an 8 GiB budget the result FITS and is maximal: best+1 overflows (unless it hit the cap).
    const int b = max_batch_for_vram(d448, 8151, cap);
    REQUIRE(b >= 1);
    CHECK(train_resident_mb(d448, b) <= 8151);
    if (b < cap) CHECK(train_resident_mb(d448, b + 1) > 8151);

    // The configurator clamp invariant: a stale over-budget batch clamps DOWN to a batch that fits --
    // the same primitive the tuner bounds its sweep with, so the two agree. Reuse b+1 (already proven
    // just above to exceed the budget, given b < cap held there) instead of a hardcoded batch constant
    // -- a fixed "256 needs ~11 GiB" assumption went stale on its own once the B1 chunked-lm_head/CE
    // VRAM work (c05c6ff) cut real usage at that batch to ~7.2 GiB, silently flipping this REQUIRE.
    const int stale = b + 1;
    REQUIRE(train_resident_mb(d448, stale) > 8151);
    const int clamped = max_batch_for_vram(d448, 8151, stale);
    CHECK(clamped < stale);
    CHECK(clamped >= 1);
    CHECK(train_resident_mb(d448, clamped) <= 8151);

    // A budget too small for even batch 1 -> 0 (the configurator's hard-error path).
    CHECK(max_batch_for_vram(d448, 1, cap) == 0);
}

// --- log: leveled diagnostics + the file tee that backs <model_dir>/train.log --------------------
TEST_CASE("log: level filtering + file tee (prefix on leveled, none on raw lines)", "[frontend][log]") {
    namespace log = sub0::log;
    // Unique sink per run so a just-exited run's file handle can never collide/lock this one.
    const auto tmp = std::filesystem::temp_directory_path() /
                     std::format("sub0_log_test_{}.log", std::random_device{}());
    REQUIRE(log::set_file(tmp.string(), /*append=*/false));

    log::set_level(log::Level::Warn);          // threshold: drop info/debug, keep error/warn
    log::error("err {}", 1);
    log::warn("warn {}", 2);
    log::info("info {}", 3);                    // below threshold -> dropped
    log::line("raw {}", 4);                     // raw program output -> always tee'd, no level prefix
    log::close_file();

    std::ifstream f(tmp);
    std::stringstream ss; ss << f.rdbuf();
    const std::string out = ss.str();
    CHECK(out.find("[error] err 1") != std::string::npos);
    CHECK(out.find("[warn] warn 2")  != std::string::npos);
    CHECK(out.find("info 3")         == std::string::npos);   // filtered out by the threshold
    CHECK(out.find("raw 4")          != std::string::npos);   // raw line present...
    CHECK(out.find("[info]")         == std::string::npos);   // ...with no prefix, and no info leaked

    log::set_level(log::Level::Info);          // restore the default for any later test
    std::error_code ec; std::filesystem::remove(tmp, ec);   // best-effort cleanup
}

// --- registry: the model identity + compatibility behind auto-naming and `models --prune` ---------
TEST_CASE("registry: corpus_tag is a lowercased, filesystem-safe stem", "[frontend][registry]") {
    using sub0::registry::corpus_tag;
    CHECK(corpus_tag("data/TinyStories.txt") == "tinystories");   // stem, lowercased
    CHECK(corpus_tag("/abs/Fine-Web_99.txt") == "fine_web_99");   // non-alnum (- and _) -> '_'
    CHECK(corpus_tag("") == "corpus");                            // empty -> fallback
}

TEST_CASE("registry: model_dir encodes identity; compatible() gates loading (the prune rule)", "[frontend][registry]") {
    using namespace sub0::registry;

    // The directory name IS the identity: d/l/h/sq/v + ternary/rope/gated-ffn/tied-embed tags + git
    // sha (JOIN is the only tokenizer scheme now, so there is no 'j' tag).
    const auto dir = model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                               /*ternary*/0, /*pos=rope*/1, "abc123");
    CHECK(dir.filename().string() == "sub0llm_tinystories_d192l6h6sq256v4306r_abc123");
    CHECK(model_dir("models", "c.txt", 1,1,1,1,1, 0,0, "").filename().string().ends_with("_nogit"));
    CHECK(model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                    /*ternary*/0, /*pos=rope*/1, "abc123", /*gated_ffn*/1).filename().string()
          == "sub0llm_tinystories_d192l6h6sq256v4306rg_abc123");
    CHECK(model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                    /*ternary*/0, /*pos=rope*/1, "abc123", /*gated_ffn*/0, /*tied*/1).filename().string()
          == "sub0llm_tinystories_d192l6h6sq256v4306rw_abc123");
    CHECK(model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                    /*ternary*/0, /*pos=rope*/1, "abc123", /*gated_ffn*/0, /*tied*/0, /*qk_norm*/1).filename().string()
          == "sub0llm_tinystories_d192l6h6sq256v4306rq_abc123");

    // compatible(): an exact match loads; ANY differing dimension/flag makes it incompatible -- this is
    // what flagged the pruned d448 v2048 models 'x incompatible architecture' against a d192 v4306 build.
    ModelMeta m;
    m.d_model = 192; m.n_layers = 6; m.n_heads = 6; m.seq_len = 256; m.vocab = 4306;
    m.ternary = 0; m.pos_encoding = 1; m.gated_ffn = 0; m.tied_embeddings = 0; m.qk_norm = 0;
    CHECK(compatible(m, 192, 6, 6, 256, 4306, 0, 1));            // exact -> loadable
    CHECK_FALSE(compatible(m, 448, 6, 6, 256, 4306, 0, 1));      // d_model differs
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 2048, 0, 1));      // vocab differs (the pruned case)
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 4306, 0, 0));      // pos_encoding differs
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 4306, 0, 1, 1));   // gated_ffn differs
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 4306, 0, 1, 0, 1)); // tied_embeddings differs
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 4306, 0, 1, 0, 0, 1)); // qk_norm differs
}
