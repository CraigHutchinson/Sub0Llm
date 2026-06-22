#pragma once

// run_config.hpp — the Ch29 training run configuration, as reflected sections.
//
// Each section is a plain value struct grouping related knobs; together they ARE the
// single architectural unit of configuration the rest of the trainer reads. The
// layered resolution is: compiled-in defaults  →  run_config.json in the checkpoint
// dir (if present)  →  command-line overrides. So `ch29_diffusion_training --ckpt-dir X`
// resumes with the EXACT corpus, architecture, and training settings of that run; any
// flag overrides just that field. Every flag, the JSON schema, and the config-SHA are
// derived from the field tables below (sub0llm/config/schema.hpp) — adding a knob is a
// member + one reflect() line, nothing else.
//
// SCOPE split: the model SHAPE (vocab/seq/embed/layers/heads/d_ff) is BuildTime — the
// dimensions a constexpr-specialized engine would bake in, and the only fields in the
// config-SHA. Everything operational (paths, optimizer, curriculum, diagnostics,
// verbosity) is Runtime: it steers a run without changing the engine binary.

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>

#include "sub0llm/config/schema.hpp"

namespace sub0diff::config {

using sub0llm::config::Scope;

// ── Model shape — BuildTime (the founded proportions are the defaults) ──────────────
// References bracketing this regime keep head_dim ≥ 32 and d_ff = 4·D (TRAINING_DESIGN
// §10): the founded D=256/L=6/H=8 (head_dim 32) won +11pt recall over the earlier
// mis-proportioned D=128/L=8/H=8. vocab_size is the post-BPE real vocab (written by the
// trainer after tokenizer build so a resume reconstructs the exact embedding/LM-head).
struct Model {
    std::int64_t  vocab_size = 512;   // real vocab (mask token added internally by the denoiser)
    std::int64_t  seq_len    = 128;   // canvas length T (frontier t = k/T) — ≫ mera_window so MERA is hierarchical
    std::int64_t  embed_dim  = 256;   // residual width D
    std::int64_t  n_layers   = 6;
    std::int64_t  d_ff       = 1024;  // 4·D
    std::int64_t  n_heads    = 8;     // head_dim = D/n_heads = 32
    std::int64_t  n_kv_heads = 4;     // 2:1 GQA
    // Architecture family (BuildTime). "flat" = the vanilla Denoiser (n_layers blocks). "mera" = the
    // recursive coarsen/refine MeraDenoiser (Ch32 P3): coarsen by `mera_coarsen` per level down to a top
    // level ≤ `mera_window`; n_layers is unused by MERA (its depth ≈ 2·levels+1 derives from seq_len).
    std::string   model_type      = "flat";   // flat | mera
    std::int64_t  mera_coarsen    = 4;        // MERA coarsen factor c per level
    std::int64_t  mera_window     = 16;       // MERA local window / top size w (seq 128 → levels 128·32·8)
    bool          mera_gated_pool = false;    // MERA: content-weighted (boundary-aware) coarsening vs mean-pool

    template <class Self, class V>
    static constexpr void reflect(Self& self, V&& v) {
        v("vocab_size",   Scope::BuildTime, self.vocab_size,   "real vocabulary size");
        v("seq_len",      Scope::BuildTime, self.seq_len,      "canvas length T");
        v("embed_dim",    Scope::BuildTime, self.embed_dim,    "residual width D");
        v("n_layers",     Scope::BuildTime, self.n_layers,     "transformer blocks (flat)");
        v("d_ff",         Scope::BuildTime, self.d_ff,         "FFN hidden width (4*D)");
        v("n_heads",      Scope::BuildTime, self.n_heads,      "attention heads");
        v("n_kv_heads",   Scope::BuildTime, self.n_kv_heads,   "KV heads (GQA)");
        v("model_type",   Scope::BuildTime, self.model_type,   "flat|mera architecture family");
        v("mera_coarsen",    Scope::BuildTime, self.mera_coarsen,    "MERA coarsen factor c per level");
        v("mera_window",     Scope::BuildTime, self.mera_window,     "MERA local window / top size w");
        v("mera_gated_pool", Scope::BuildTime, self.mera_gated_pool, "MERA content-weighted coarsening");
    }
};

// ── Data + checkpoint location — Runtime ────────────────────────────────────────────
struct Data {
    std::string  corpus     = "data/tinystories_clean.txt";
    std::string  ckpt_dir   = "/tmp/sub0diff_ch29";      // checkpoint directory (train_state.json + step_*.ckpt + step_*.opt)};
    std::string  name       = {};      // model name → in-repo dir models/<name>_g<gitSHA>_c<configSHA>
    std::int64_t paragraphs = 0;       // 0/-1 = all paragraphs
    bool         char_level = false;   // char tokenizer (whitespace preserved) + raw reader
    bool         word_level = true;   // word tokenizer (one token per whole word) + raw reader

    template <class Self, class V>
    static constexpr void reflect(Self& self, V&& v) {
        v("corpus",     Scope::Runtime, self.corpus,     "training corpus path");
        v("ckpt_dir",   Scope::Runtime, self.ckpt_dir,   "checkpoint directory");
        v("name",       Scope::Runtime, self.name,       "model name (-> models/<name>_g<sha>_c<sha>)");
        v("paragraphs", Scope::Runtime, self.paragraphs, "paragraph cap (0=all)");
        v("char_level", Scope::Runtime, self.char_level, "char-level tokenizer + raw reader");
        v("word_level", Scope::Runtime, self.word_level, "word-level tokenizer + raw reader");
    }
};

// ── Optimization + parallelism — Runtime ────────────────────────────────────────────
struct Optim {
    std::string   optimizer    = "adamw";   // adamw | adam | muon
    float         lr           = 1e-3f;
    bool          lr_explicit  = false;      // set when --lr was passed (persisted for honest resume)
    std::uint64_t warmup_steps = 0;          // 0 = derive
    std::uint64_t seed         = 42;
    float         t_max        = 1.0f;       // <1 trains under a Ch28-style noise ceiling
    std::int64_t  batch        = 8;          // effective batch B (quality, consistency ∝ √B); GPU handles 8 easily
    std::int64_t  threads      = 0;          // workers (speed only; 0 = auto = all P-cores)
    std::string   pin          = "auto";     // worker pin policy: auto|all|P|E|"lo-hi"|"a,b,c"
    bool          shared_t     = true;       // one t per step across workers (variance reduction)
    bool          exact_noise  = true;       // mask exactly round(t·T) positions
    bool          whole_word   = false;      // corrupt whole words, not independent subwords
    bool          contiguous   = false;      // mask a CONTIGUOUS span (§13.4 — trains long-range infill, not local interpolation)
    std::string   device       = "cpu";      // cpu (worker pool) | cuda (single GPU stream)

    template <class Self, class V>
    static constexpr void reflect(Self& self, V&& v) {
        v("optimizer",    Scope::Runtime, self.optimizer,    "adamw|adam|muon");
        v("lr",           Scope::Runtime, self.lr,           "peak LR after warmup");
        v("lr_explicit",  Scope::Runtime, self.lr_explicit,  "user-set LR (internal)");
        v("warmup_steps", Scope::Runtime, self.warmup_steps, "LR warmup steps (0=derive)");
        v("seed",         Scope::Runtime, self.seed,         "RNG seed");
        v("t_max",        Scope::Runtime, self.t_max,        "max noise level");
        v("batch",        Scope::Runtime, self.batch,        "effective batch B");
        v("threads",      Scope::Runtime, self.threads,      "worker threads (0=auto)");
        v("pin",          Scope::Runtime, self.pin,          "worker pin policy");
        v("shared_t",     Scope::Runtime, self.shared_t,     "shared t per step");
        v("exact_noise",  Scope::Runtime, self.exact_noise,  "exact masked count");
        v("contiguous",   Scope::Runtime, self.contiguous,   "mask a contiguous span (long-range infill)");
        v("whole_word",   Scope::Runtime, self.whole_word,   "whole-word masking");
        v("device",       Scope::Runtime, self.device,       "cpu|cuda (train on GPU)");
    }
};

// ── Schedule / early-stopping — Runtime ─────────────────────────────────────────────
struct Train {
    std::uint64_t steps       = 0;    // 0 = corpus-scaled safety bound
    std::uint64_t eval_every  = 0;    // 0 = derive from corpus
    double        eval_factor = 1.0;  // epochs of coverage per eval (floor 0.5)
    std::uint64_t patience    = 10;   // evals without improvement ⇒ stop
    std::uint64_t min_epochs  = 20;   // minimum passes before early-stop may fire

    template <class Self, class V>
    static constexpr void reflect(Self& self, V&& v) {
        v("steps",       Scope::Runtime, self.steps,       "max steps (0=corpus-scaled)");
        v("eval_every",  Scope::Runtime, self.eval_every,  "steps per eval (0=derive)");
        v("eval_factor", Scope::Runtime, self.eval_factor, "epochs coverage per eval");
        v("patience",    Scope::Runtime, self.patience,    "early-stop patience (evals)");
        v("min_epochs",  Scope::Runtime, self.min_epochs,  "min epochs before stopping");
    }
};

// ── Curriculum — Runtime ────────────────────────────────────────────────────────────
struct Curriculum {
    bool          curriculum            = false;   // token-gated EMA frontier-point curriculum
    float         curriculum_end        = 1.0f;
    std::uint64_t curriculum_min_tokens = 1500;
    bool          curriculum_converge   = false;   // convergence-gated integer-k curriculum
    std::uint64_t curriculum_patience   = 2;       // epochs of no level-NELBO gain ⇒ advance k
    std::int64_t  curriculum_k_step     = 1;       // masked tokens added per advancement
    std::int64_t  curriculum_k_start    = 1;       // starting level (manual resume fallback)

    template <class Self, class V>
    static constexpr void reflect(Self& self, V&& v) {
        v("curriculum",            Scope::Runtime, self.curriculum,            "token-gated curriculum");
        v("curriculum_end",        Scope::Runtime, self.curriculum_end,        "token-gated target ceiling");
        v("curriculum_min_tokens", Scope::Runtime, self.curriculum_min_tokens, "tokens per ceiling decision");
        v("curriculum_converge",   Scope::Runtime, self.curriculum_converge,   "convergence-gated integer-k");
        v("curriculum_patience",   Scope::Runtime, self.curriculum_patience,   "epochs to advance k");
        v("curriculum_k_step",     Scope::Runtime, self.curriculum_k_step,     "k added per level");
        v("curriculum_k_start",    Scope::Runtime, self.curriculum_k_start,    "starting k (resume fallback)");
    }
};

// ── Diagnostics / mode flags — Runtime ──────────────────────────────────────────────
struct Diag {
    bool          eval_only      = false;
    bool          track_recall   = false;
    bool          per_t          = false;
    bool          grad_probe     = false;
    bool          eval_train     = false;
    bool          profile        = false;
    std::int64_t  inspect        = 0;     // dump N best/worst recovered windows after eval
    std::int64_t  overfit        = 0;     // minimal-case: train only first N windows, stop on train recall
    std::uint64_t recall_windows = std::numeric_limits<std::uint64_t>::max();  // SIZE_MAX=corpus-scaled
    bool          founded        = false; // legacy --founded (a no-op now; defaults are founded)
    bool          oov_cliff      = false; // M1: NLL_rare/NLL_common cliff after the recall sweep
    double        oov_rare_frac  = 0.5;   // fraction of rarest types treated as the OOV proxy

    template <class Self, class V>
    static constexpr void reflect(Self& self, V&& v) {
        v("eval_only",      Scope::Runtime, self.eval_only,      "eval a checkpoint and exit");
        v("track_recall",   Scope::Runtime, self.track_recall,   "measure recall at each eval");
        v("per_t",          Scope::Runtime, self.per_t,          "per-noise-level CE/recall curve");
        v("grad_probe",     Scope::Runtime, self.grad_probe,     "gradient-conflict probe and exit");
        v("eval_train",     Scope::Runtime, self.eval_train,     "also sweep recall on train stream");
        v("profile",        Scope::Runtime, self.profile,        "report fwd/bwd/opt time split");
        v("inspect",        Scope::Runtime, self.inspect,        "dump N best/worst windows");
        v("overfit",        Scope::Runtime, self.overfit,        "minimal-case: first N windows");
        v("recall_windows", Scope::Runtime, self.recall_windows, "recall sweep budget");
        v("founded",        Scope::Runtime, self.founded,        "legacy founded preset (no-op)");
        v("oov_cliff",      Scope::Runtime, self.oov_cliff,      "M1: NLL rare vs common cliff");
        v("oov_rare_frac",  Scope::Runtime, self.oov_rare_frac,  "rarest type fraction (OOV proxy)");
    }
};

// ── The composed run configuration ──────────────────────────────────────────────────
struct RunConfig {
    Model      model;
    Data       data;
    Optim      optim;
    Train      train;
    Curriculum curriculum;
    Diag       diag;

    template <class Self, class V>
    static constexpr void reflect(Self& self, V&& v) {
        Model::reflect(self.model, v);
        Data::reflect(self.data, v);
        Optim::reflect(self.optim, v);
        Train::reflect(self.train, v);
        Curriculum::reflect(self.curriculum, v);
        Diag::reflect(self.diag, v);
    }
};

// Apply the founded proportions (re-asserts D=256/L6/H8/d_ff=4·D after the legacy flag).
void apply_founded(Model& m);

// Resolve the effective run configuration: defaults → run_config.json in the resolved
// ckpt-dir → command-line overrides. Throws std::runtime_error on an unknown flag, a
// missing flag value, or an invalid combination (e.g. both curriculum variants).
[[nodiscard]] RunConfig resolve(int argc, char** argv);

// Write run_config.json (full, the resume source) into `dir`, stamped with code_sha +
// config_sha. config.json (the model-only subset Ch30's sampler reads) stays separate.
void write_run_config(const std::filesystem::path& dir, const RunConfig& cfg,
                      std::string_view code_sha);

// config_sha — FNV-1a over the BuildTime (model shape) fields; tags a specialized build.
[[nodiscard]] std::uint64_t config_sha(const RunConfig& cfg);

}  // namespace sub0diff::config
