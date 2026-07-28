// frontend_tests.cpp — engine-free unit tests for the FRONTEND (pre-model) logic that the configurator
// and the stage tools share but the engine doesn't define: the device-memory planner (sub0/memplan.hpp)
// and the model registry's identity/compat rules (sub0/registry.hpp). Both are header-only and std-only,
// so they run in the fast engine-free sub0_tok_tests target -- no engine build, no GPU. This is the
// payoff of the frontend separation: pre-model logic gets covered without the engine. These two areas
// were directly behind this session's work (the gpu_batch VRAM clamp + `models --prune`).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/evalcache.hpp"
#include "sub0/memplan.hpp"
#include "sub0/registry.hpp"
#include "sub0/log.hpp"

#include <cctype>
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

    // The directory name IS the identity: d/l/h/sq/v + ternary/rope/gated-ffn/tied-embed tags + a
    // date+time tag for the training attempt (JOIN is the only tokenizer scheme now, so there is no
    // 'j' tag). The tag is NOT a compatibility check -- that's compatible() below, dims/flags only.
    const auto dir = model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                               /*ternary*/0, /*pos=rope*/1, "20260709-120000");
    CHECK(dir.filename().string() == "sub0llm_tinystories_d192l6h6sq256v4306r_20260709-120000");
    CHECK(model_dir("models", "c.txt", 1,1,1,1,1, 0,0, "").filename().string().ends_with("_unknown"));
    CHECK(model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                    /*ternary*/0, /*pos=rope*/1, "20260709-120000", /*gated_ffn*/1).filename().string()
          == "sub0llm_tinystories_d192l6h6sq256v4306rg_20260709-120000");
    CHECK(model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                    /*ternary*/0, /*pos=rope*/1, "20260709-120000", /*gated_ffn*/0, /*tied*/1).filename().string()
          == "sub0llm_tinystories_d192l6h6sq256v4306rw_20260709-120000");
    CHECK(model_dir("models", "data/tinystories.txt", 192, 6, 6, 256, 4306,
                    /*ternary*/0, /*pos=rope*/1, "20260709-120000", /*gated_ffn*/0, /*tied*/0, /*qk_norm*/1).filename().string()
          == "sub0llm_tinystories_d192l6h6sq256v4306rq_20260709-120000");

    // compatible() is decided by arch_id ALONE (sub0::MODEL_ARCH_ID), not by the individual fields.
    // The per-field comparison it replaced could not see n_kv_heads, LoopSplit's schedule or the rope
    // parameters, so it answered "compatible" for models load_model then REJECTS -- sending `train`
    // into a resume that fails. The dim parameters remain in the signature for callers that only have
    // loose dims to hand, but they no longer participate in the decision.
    ModelMeta m;
    m.d_model = 192; m.n_layers = 6; m.n_heads = 6; m.seq_len = 256; m.vocab = 4306;
    m.ternary = 0; m.pos_encoding = 1; m.gated_ffn = 0; m.tied_embeddings = 0; m.qk_norm = 0;
    m.arch_id = 0xfeedfacecafeb00dull;
    CHECK(compatible(m, 192, 6, 6, 256, 4306, 0, 1, 0, 0, 0, 0xfeedfacecafeb00dull));   // same arch
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 4306, 0, 1, 0, 0, 0, 0xfeedfacecafeb00eull)); // differs

    // Matching dims do NOT make it compatible when the arch id differs -- that is the entire point:
    // a GQA and an MHA build at identical dims used to compare equal here.
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 4306, 0, 1, 0, 0, 0, 1));

    // A meta with NO arch_id (0) is INCOMPATIBLE rather than falling back to a per-field match. The
    // fallback was removed with the rest of the legacy-format support: a per-field match is exactly
    // the wrong answer for the axes it cannot see.
    ModelMeta legacy = m; legacy.arch_id = 0;
    CHECK_FALSE(compatible(legacy, 192, 6, 6, 256, 4306, 0, 1, 0, 0, 0, 0xfeedfacecafeb00dull));
    // ...and a caller with no arch id of its own cannot claim compatibility either.
    CHECK_FALSE(compatible(m, 192, 6, 6, 256, 4306, 0, 1));
}

TEST_CASE("registry: now_datetag is fixed-width, filesystem-safe, and sortable", "[frontend][registry]") {
    using sub0::registry::now_datetag;
    const std::string tag = now_datetag();
    CHECK(tag.size() == 15);                          // YYYYMMDD-HHMMSS
    CHECK(tag[8] == '-');
    CHECK(tag.find(':') == std::string::npos);         // Windows disallows ':' in filenames
    CHECK(tag.find('T') == std::string::npos);
    for (std::size_t i = 0; i < tag.size(); ++i)
        if (i != 8) CHECK(std::isdigit(static_cast<unsigned char>(tag[i])));
}

TEST_CASE("registry: write_meta/read_meta round-trips every field, including optimizer", "[frontend][registry]") {
    using namespace sub0::registry;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_registry_meta_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);

    ModelMeta m;
    m.corpus = "tinystories"; m.git_sha = "a016d17"; m.created = now_iso(); m.updated = now_iso();
    m.status = "plateaued";
    m.d_model = 448; m.n_layers = 11; m.n_heads = 7; m.seq_len = 256; m.vocab = 16517; m.ternary = 0;
    m.pos_encoding = 1; m.gated_ffn = 1; m.tied_embeddings = 1; m.qk_norm = 1; m.optimizer = 1;
    m.steps = 12236; m.epochs = 1.90118; m.tokens_seen = 1205216838; m.batch = 385;
    m.lr = 0.00693722; m.seed = 42; m.best_val_nelbo = 1.19023;
    write_meta(dir, m);

    ModelMeta r;
    REQUIRE(read_meta(dir, r));
    CHECK(r.corpus == m.corpus);
    CHECK(r.git_sha == m.git_sha);
    CHECK(r.status == m.status);
    CHECK(r.d_model == m.d_model);
    CHECK(r.n_layers == m.n_layers);
    CHECK(r.n_heads == m.n_heads);
    CHECK(r.seq_len == m.seq_len);
    CHECK(r.vocab == m.vocab);
    CHECK(r.gated_ffn == m.gated_ffn);
    CHECK(r.tied_embeddings == m.tied_embeddings);
    CHECK(r.qk_norm == m.qk_norm);
    CHECK(r.optimizer == 1);
    CHECK(r.steps == m.steps);
    CHECK(r.best_val_nelbo == Catch::Approx(m.best_val_nelbo));

    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("registry: optimizer defaults to 0 (AdamW) when reading a pre-existing meta.txt without the field", "[frontend][registry]") {
    using namespace sub0::registry;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_registry_legacy_meta_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    {
        std::ofstream os(dir / "meta.txt");
        os << "model=sub0llm\ncorpus=tinystories\nd_model=448\nn_layers=11\nn_heads=7\n"
              "seq_len=256\nvocab=16517\nstatus=trained\n";   // no optimizer= line, matching an old file
    }
    ModelMeta m;
    REQUIRE(read_meta(dir, m));
    CHECK(m.optimizer == 0);

    std::filesystem::remove_all(dir, ec);
}

// --- evalcache: the per-model RAW-measurement cache (metrics.txt) behind `models --metrics` -------
TEST_CASE("evalcache: read_metrics returns false when no metrics.txt exists yet", "[frontend][evalcache]") {
    using namespace sub0::evalcache;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_evalcache_missing_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    EvalMetrics m;
    CHECK_FALSE(read_metrics(dir, m));

    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("evalcache: write_metrics/read_metrics round-trips scalars and the full grid", "[frontend][evalcache]") {
    using namespace sub0::evalcache;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_evalcache_roundtrip_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);

    EvalMetrics m;
    m.steps = 12236; m.measured_at = "2026-07-09T09:04:21Z";
    m.train_nelbo = 1.1027; m.val_nelbo = 1.1902; m.bytes_per_tok = 3.34;
    m.target_ppl = 3.04; m.target_rep = 0.038;
    m.grid.temp   = {0.3f, 0.7f, 1.0f, 1.4f};
    m.grid.gen_ppl = {1.513, 1.816, 2.560, 4.751};
    m.grid.rep4    = {0.055, 0.048, 0.038, 0.019};
    write_metrics(dir, m);

    EvalMetrics r;
    REQUIRE(read_metrics(dir, r));
    CHECK(r.steps == m.steps);
    CHECK(r.measured_at == m.measured_at);
    CHECK(r.train_nelbo == Catch::Approx(m.train_nelbo));
    CHECK(r.val_nelbo == Catch::Approx(m.val_nelbo));
    CHECK(r.bytes_per_tok == Catch::Approx(m.bytes_per_tok));
    CHECK(r.target_ppl == Catch::Approx(m.target_ppl));
    CHECK(r.target_rep == Catch::Approx(m.target_rep));
    REQUIRE(r.grid.temp.size() == m.grid.temp.size());
    for (std::size_t i = 0; i < m.grid.temp.size(); ++i) {
        CHECK(r.grid.temp[i] == Catch::Approx(m.grid.temp[i]));
        CHECK(r.grid.gen_ppl[i] == Catch::Approx(m.grid.gen_ppl[i]));
        CHECK(r.grid.rep4[i] == Catch::Approx(m.grid.rep4[i]));
    }

    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("evalcache: a read-modify-write preserves the OTHER tool's already-cached fields", "[frontend][evalcache]") {
    // Simulates report_stage writing first (train/val/bytes_per_tok only), then autotemp_stage
    // read-modify-writing its own fields on top -- the real usage pattern in train_stage.cpp, since
    // report and autotemp own disjoint subsets of EvalMetrics and neither should clobber the other.
    using namespace sub0::evalcache;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_evalcache_merge_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);

    EvalMetrics from_report;
    from_report.steps = 100; from_report.measured_at = "2026-07-09T00:00:00Z";
    from_report.train_nelbo = 1.10; from_report.val_nelbo = 1.19; from_report.bytes_per_tok = 3.34;
    write_metrics(dir, from_report);

    EvalMetrics merged;
    REQUIRE(read_metrics(dir, merged));
    merged.target_ppl = 3.04; merged.target_rep = 0.038;   // autotemp's own fields only
    merged.grid.temp = {1.0f}; merged.grid.gen_ppl = {2.56}; merged.grid.rep4 = {0.038};
    write_metrics(dir, merged);

    EvalMetrics final_read;
    REQUIRE(read_metrics(dir, final_read));
    CHECK(final_read.train_nelbo == Catch::Approx(1.10));   // report's fields survived the merge
    CHECK(final_read.val_nelbo == Catch::Approx(1.19));
    CHECK(final_read.bytes_per_tok == Catch::Approx(3.34));
    CHECK(final_read.target_ppl == Catch::Approx(3.04));    // autotemp's own fields landed
    REQUIRE(final_read.grid.temp.size() == 1);

    std::filesystem::remove_all(dir, ec);
}
