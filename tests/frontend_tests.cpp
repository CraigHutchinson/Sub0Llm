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

#include <algorithm>
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

// The LoopSplit/depth-attention arm shape, spelled out ONCE for the term-level tests below. This is the
// shape the memory audit was performed at, and the shape where the logits chunk cap truncates badly --
// d_ff 512 against vocab 16508 wants 33 chunks, four times the default cap. Batch 448 x seq 512 = the
// 229,376-row budget every MiB figure in docs/MEMORY_AUDIT.md is quoted at.
namespace {
constexpr sub0::memplan::Dims kArmShape{
    /*d_model=*/192, /*n_layers=*/10, /*n_heads=*/6, /*d_ff=*/512, /*seq_len=*/512, /*vocab=*/16508,
    /*tied=*/true, /*qk_norm=*/true, /*gated=*/true, /*pos_emb=*/false,
    /*n_kv_heads=*/3,        // GQA-2 -> D_KV = 3 * (192/6) = 96
    /*exec_layers=*/16,      // LoopSplit: 10 layers, 16 executions (arms B/D)
    /*depth_slots=*/4,       // depth attention at stride 4 (arm D)
};
constexpr int kArmBatch = 448;
constexpr sub0::memplan::u64 kBf16 = 2;
constexpr double kMiB = 1024.0 * 1024.0;
}  // namespace

// The itemization contract. train_scratch_bytes used to be a single opaque scalar, which is how a
// hand-recomputed census in docs/MEMORY_AUDIT.md could put the LARGEST term out by 4x while the total
// still looked plausible. Terms are now the reviewable unit, so pin two things a refactor could break:
// the total is exactly the sum (no term dropped from total(), no term double-counted), and no term that
// this shape genuinely uses silently collapses to zero.
TEST_CASE("memplan: scratch terms sum to the total and none silently vanishes", "[frontend][memplan]") {
    using namespace sub0::memplan;
    const ScratchTerms t = train_scratch_terms(kArmShape, kArmBatch, kBf16);

    // Sum integrity: total() must account for every field. Written as an explicit re-sum rather than
    // calling total() twice so that a field ADDED to the struct but forgotten in total() fails here.
    CHECK(t.per_exec + t.final_blk + t.logits + t.grad + t.qk_pre + t.depth == t.total());
    CHECK(train_scratch_bytes(kArmShape, kArmBatch, kBf16) == t.total());

    // Every term this shape uses is non-zero. The arm shape deliberately enables qk_norm AND depth
    // attention precisely so those two conditional terms are covered here rather than only in a build
    // that happens to have them on.
    CHECK(t.per_exec > 0);
    CHECK(t.final_blk > 0);
    CHECK(t.logits > 0);
    CHECK(t.grad > 0);
    CHECK(t.qk_pre > 0);
    CHECK(t.depth > 0);

    // The conditional terms really are conditional -- otherwise the two CHECKs above would pass for a
    // build that ignores the flags entirely.
    Dims off = kArmShape;
    off.qk_norm = false;
    off.depth_slots = 0;
    const ScratchTerms t_off = train_scratch_terms(off, kArmBatch, kBf16);
    CHECK(t_off.qk_pre == 0);
    CHECK(t_off.depth == 0);
    CHECK(t_off.total() < t.total());
}

// Regression test for a REAL divergence, not a hypothetical: the CUDA allocator branched on its own
// runtime chunk override while the predictor always re-derived the count, so forcing 32 chunks shrank the
// allocation by ~1.35 GiB while the prediction did not move -- a 4x-tolerance over-prediction that would
// make max_batch_for_vram under-size the very batch the lever exists to raise. Nothing caught it because
// the only caller that set the override was declared AFTER the footprint test in the same file.
TEST_CASE("memplan: a forced logits chunk count flows into the prediction", "[frontend][memplan]") {
    using namespace sub0::memplan;

    // The forced count is used verbatim -- NOT re-clamped to LOGITS_MAX_CHUNKS, which would silently
    // restore the disagreement for any override above the cap (32 > 8 is exactly the arm-D case).
    CHECK(logits_n_chunks(16508, 512, /*forced=*/32) == 32);
    CHECK(logits_n_chunks(16508, 512, /*forced=*/1) == 1);
    CHECK(logits_n_chunks(16508, 512) == std::min(33, LOGITS_MAX_CHUNKS));   // derivation still clamps

    Dims forced = kArmShape;
    forced.logits_chunks = 32;
    const ScratchTerms base = train_scratch_terms(kArmShape, kArmBatch, kBf16);
    const ScratchTerms with = train_scratch_terms(forced,    kArmBatch, kBf16);

    // Only the logits term moves, and it moves by the chunk-count ratio (32 vs the clamped derivation).
    CHECK(with.per_exec  == base.per_exec);
    CHECK(with.final_blk == base.final_blk);
    CHECK(with.grad      == base.grad);
    CHECK(with.depth     == base.depth);
    CHECK(with.logits < base.logits);
    const double ratio = static_cast<double>(base.logits) / static_cast<double>(with.logits);
    const double expect = 32.0 / static_cast<double>(logits_n_chunks(16508, 512));
    CHECK(ratio == Catch::Approx(expect).epsilon(0.01));

    // And the saving reaches the TOTAL -- the quantity max_batch_for_vram actually inverts. ~1.35 GiB at
    // this shape is the whole reason arm D fits an 8 GiB card at batch 448.
    const double saved_mib = static_cast<double>(base.total() - with.total()) / kMiB;
    CHECK(saved_mib > 1000.0);
    CHECK(max_batch_for_vram(forced, 7891, 4096, kBf16) > max_batch_for_vram(kArmShape, 7891, 4096, kBf16));
}

// The chunk lever's DESIGN INTENT, tested rather than asserted in a comment: the chunked logits buffer is
// supposed to land at roughly the same byte-scale as this model's other wide per-row activations, all of
// which scale with d_ff. The [1,LOGITS_MAX_CHUNKS] clamp is calibrated for a particular d_ff and truncates
// badly away from it -- at the arm shape it truncates 4x, which is how the logits term became 22% of the
// whole training footprint. This test makes that breach VISIBLE per shape instead of only discoverable as
// an out-of-VRAM run, and pins that some setting of the lever always brings the term back under budget.
TEST_CASE("memplan: the logits chunk cap's truncation is bounded per shape", "[frontend][memplan]") {
    using namespace sub0::memplan;
    struct Shape { const char* name; Dims d; int batch; };
    const Shape shapes[] = {
        {"production d448", Dims{448, 11, 7, 1792, 256, 16517, true, true, true, false}, 256},
        {"arm d192/ff512",  kArmShape, kArmBatch},
    };
    for (const auto& [name, d, batch] : shapes) {
        const int desired = (d.vocab + d.d_ff - 1) / d.d_ff;    // the UNclamped d_ff-matched ratio
        const int actual  = logits_n_chunks(d.vocab, d.d_ff);
        REQUIRE(actual >= 1);
        REQUIRE(actual <= desired);                             // the clamp only ever truncates
        const ScratchTerms t = train_scratch_terms(d, batch, kBf16);
        const double share = static_cast<double>(t.logits) / static_cast<double>(t.total());
        INFO(name << ": desired " << desired << " chunks, clamped to " << actual
             << " (truncation " << (double(desired) / actual) << "x) -> logits "
             << (static_cast<double>(t.logits) / kMiB) << " MiB = " << (100.0 * share) << "% of scratch");

        // With the lever set to the shape's OWN desired ratio, the term returns to the d_ff scale it was
        // designed to match: no more than ~12% of scratch. If this ever fails, the derivation itself is
        // wrong for that shape -- not merely capped -- and no override can rescue it.
        Dims tuned = d;
        tuned.logits_chunks = desired;
        const ScratchTerms tt = train_scratch_terms(tuned, batch, kBf16);
        const double tuned_share = static_cast<double>(tt.logits) / static_cast<double>(tt.total());
        INFO(name << ": at the desired " << desired << " chunks, logits share is " << (100.0 * tuned_share) << "%");
        CHECK(tuned_share < 0.12);

        // A shape whose clamp truncates by more than 2x needs -DSUB0_LOGITS_MAX_CHUNKS raised for it. That
        // is a build decision, so this is a WARN, not a failure -- but it must be LOUD, because the cost is
        // GPU-hours (arm D's first attempt spilled to shared memory and ran 2.7x slower for ~12 hours).
        if (double(desired) / actual > 2.0) {
            WARN(name << ": logits chunk cap truncates " << (double(desired) / actual)
                 << "x -- logits is " << (100.0 * share) << "% of scratch ("
                 << (static_cast<double>(t.logits) / kMiB) << " MiB at batch " << batch
                 << "). Build with -DSUB0_LOGITS_MAX_CHUNKS=" << desired << " for this shape.");
        }
    }
}

// Per-term scaling laws. The total's monotonicity in batch is already covered above (max_batch_for_vram
// inverts it), but a scaling regression inside ONE term is invisible on the total until it is large --
// these pin each term to the axis it is supposed to track, which is what makes a measured-vs-predicted
// gap attributable to a specific buffer group.
TEST_CASE("memplan: each scratch term scales on the axis it is supposed to", "[frontend][memplan]") {
    using namespace sub0::memplan;

    // per_exec tracks EXECUTIONS, not layers -- the whole point of the LoopSplit accounting. Same layer
    // count, double the executions, double the per-execution checkpoints and nothing else.
    Dims flat = kArmShape;
    flat.exec_layers = 8;
    Dims looped = kArmShape;
    looped.exec_layers = 16;
    const ScratchTerms tf = train_scratch_terms(flat, kArmBatch, kBf16);
    const ScratchTerms tl = train_scratch_terms(looped, kArmBatch, kBf16);
    CHECK(tl.per_exec == 2 * tf.per_exec);
    CHECK(tl.final_blk == tf.final_blk);
    CHECK(tl.grad == tf.grad);
    CHECK(tl.logits == tf.logits);

    // exec_layers == 0 means "unset -> fall back to n_layers", the meaning every pre-GQA call site relies on.
    Dims unset = kArmShape;
    unset.exec_layers = 0;
    Dims explicit_l = kArmShape;
    explicit_l.exec_layers = kArmShape.n_layers;
    CHECK(train_scratch_terms(unset, kArmBatch, kBf16).per_exec
          == train_scratch_terms(explicit_l, kArmBatch, kBf16).per_exec);

    // depth scales linearly in SLOTS, and each slot costs 3x a forward-only reading at bf16 -- two act_t
    // buffers (retained K + mixed V) PLUS two f32 gradient accumulators, i.e. (2*2 + 2*4) / (2*2). The
    // header used to claim "~1.5x" in prose; the real multiplier is dtype-dependent and 3x here, so pin
    // it as arithmetic instead of trusting the comment. This term is what decides which stride fits.
    Dims s1 = kArmShape, s8 = kArmShape;
    s1.depth_slots = 1;
    s8.depth_slots = 8;
    const auto d1 = train_scratch_terms(s1, kArmBatch, kBf16).depth;
    const auto d8 = train_scratch_terms(s8, kArmBatch, kBf16).depth;
    const auto Mm = static_cast<u64>(kArmBatch) * static_cast<u64>(kArmShape.seq_len);
    const auto per_slot = Mm * d_kv(kArmShape) * (2 * kBf16 + 2 * FLOAT);
    const auto shared_v = Mm * d_kv(kArmShape) * kBf16;             // the one shared pre-mix V
    CHECK(d1 == per_slot + shared_v);
    CHECK(d8 == 8 * per_slot + shared_v);
    CHECK(per_slot == 3 * (Mm * d_kv(kArmShape) * 2 * kBf16));      // 3x forward-only, NOT 1.5x

    // Every term is linear in the row budget (batch * seq): doubling the batch doubles each of them. This
    // is the property that makes a FIXED absolute footprint tolerance defensible -- if a term were
    // super-linear, a tolerance calibrated at batch 64 would not hold at 448.
    const ScratchTerms b1 = train_scratch_terms(kArmShape, 64, kBf16);
    const ScratchTerms b2 = train_scratch_terms(kArmShape, 128, kBf16);
    CHECK(b2.per_exec == 2 * b1.per_exec);
    CHECK(b2.final_blk == 2 * b1.final_blk);
    CHECK(b2.depth == 2 * b1.depth);
    CHECK(b2.qk_pre == 2 * b1.qk_pre);
    // logits and grad carry batch-INDEPENDENT pieces (a chunk-rounding remainder; dwqkv/lengths/active/
    // loss), so they only need to be bounded by linear growth, not exactly double.
    CHECK(b2.logits <= 2 * b1.logits);
    CHECK(b2.grad < 2 * b1.grad);
}

// --- log: leveled diagnostics + the file tee that backs <model_dir>/train.log --------------------
TEST_CASE("log: level filtering + file tee (prefix on leveled, none on raw lines)", "[frontend][log]") {
    namespace log = sub0::log;
    // Unique sink per run so a just-exited run's file handle can never collide/lock this one.
    const auto tmp = std::filesystem::temp_directory_path() /
                     std::format("sub0_log_test_{}.log", std::random_device{}());
    REQUIRE(log::set_file(tmp.string(), /*append=*/false));

    log::set_level(log::Level::Warn);          // threshold: drop info/debug, keep error/warn
    // Self-describing payloads: log::set_file tees to the console too, so a terse "err 1" here reads
    // like a real failure in test output -- spell out that it's expected.
    log::error("expected test log message (error) {}", 1);
    log::warn("expected test log message (warn) {}", 2);
    log::info("expected test log message (info) {}", 3);   // below threshold -> dropped
    log::line("expected test log message (raw) {}", 4);    // raw program output -> always tee'd, no level prefix
    log::close_file();

    std::ifstream f(tmp);
    std::stringstream ss; ss << f.rdbuf();
    const std::string out = ss.str();
    CHECK(out.find("[error] expected test log message (error) 1") != std::string::npos);
    CHECK(out.find("[warn] expected test log message (warn) 2")   != std::string::npos);
    CHECK(out.find("expected test log message (info) 3")          == std::string::npos);   // filtered out by the threshold
    CHECK(out.find("expected test log message (raw) 4")           != std::string::npos);   // raw line present...
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

TEST_CASE("registry: state.json carries provenance, config.json carries architecture, read_state merges both",
          "[frontend][registry]") {
    using namespace sub0::registry;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_registry_state_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);

    // The point of the split: NEITHER writer states a field the other one states. state.json's nine
    // fields are the run's provenance and progress; every architecture and recipe value comes from
    // config.json, which is generated from the X-macro and so cannot fall behind a new axis.
    ModelMeta m;
    m.git_sha = "a016d17"; m.created = now_iso(); m.updated = now_iso(); m.status = "plateaued";
    m.arch_id = 0xfeedfacecafeb00dULL;
    m.steps = 12236; m.epochs = 1.90118; m.tokens_seen = 1205216838; m.best_val_nelbo = 1.19023;
    write_state(dir, m);

    RunConfig c;
    c.corpus = "tinystories";
    c.d_model = 448; c.n_layers = 11; c.n_heads = 7; c.n_kv_heads = 4;
    c.seq_len = 256; c.vocab = 16517; c.ternary = 0; c.pos_encoding = 1;
    c.gated_ffn = 1; c.tied_embeddings = 1; c.qk_norm = 1; c.optimizer = 1;
    c.batch = 385; c.lr = 0.00693722; c.seed = 42u;
    write_config_json(c, dir);

    ModelMeta r;
    REQUIRE(read_state(dir, r));
    // ...from state.json. arch_id specifically: it round-trips through a hex STRING, and it is the
    // only thing compatible() consults, so a lossy round-trip here would silently mis-match models.
    CHECK(r.git_sha == m.git_sha);
    CHECK(r.created == m.created);
    CHECK(r.updated == m.updated);
    CHECK(r.status == m.status);
    CHECK(r.arch_id == m.arch_id);
    CHECK(r.steps == m.steps);
    CHECK(r.epochs == Catch::Approx(m.epochs));
    CHECK(r.tokens_seen == m.tokens_seen);
    CHECK(r.best_val_nelbo == Catch::Approx(m.best_val_nelbo));
    // ...and from config.json, merged in by read_state.
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
    CHECK(r.lr == Catch::Approx(c.lr));
    CHECK(r.seed == c.seed);
    CHECK(r.dir == dir);

    // compatible() consults arch_id alone, so the merge above must not disturb it.
    CHECK(compatible(r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m.arch_id));
    CHECK_FALSE(compatible(r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m.arch_id + 1));

    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("registry: a directory with no state.json is not a model directory", "[frontend][registry]") {
    using namespace sub0::registry;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_registry_nostate_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    // A config.json alone is not a model: `sub0llm configure` writes one before anything is trained.
    // scan() keys on state.json so a configured-but-never-trained directory does not list as a model.
    RunConfig c; c.corpus = "tinystories"; c.d_model = 448;
    write_config_json(c, dir);

    ModelMeta m;
    CHECK_FALSE(read_state(dir, m));

    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("registry: state.json without a config.json still lists, architecture left at defaults",
          "[frontend][registry]") {
    using namespace sub0::registry;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sub0_registry_noconfig_test";
    std::error_code ec; std::filesystem::remove_all(dir, ec);

    // An interrupted first run leaves real provenance, which a user needs to SEE in order to prune
    // it -- so a missing config.json is not a failure. The architecture fields stay at their
    // defaults rather than being invented from whatever the current build happens to be.
    ModelMeta m;
    m.git_sha = "deadbee"; m.status = "training"; m.steps = 7;
    write_state(dir, m);

    ModelMeta r;
    REQUIRE(read_state(dir, r));
    CHECK(r.status == "training");
    CHECK(r.steps == 7);
    CHECK(r.d_model == 0);
    CHECK(r.corpus.empty());

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
