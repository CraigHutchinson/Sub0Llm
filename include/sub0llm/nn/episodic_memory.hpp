#pragma once

#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace sub0llm::nn {

// ── EpisodicProgress ──────────────────────────────────────────────────────────
//
// Emitted by episodic_encode via EpisodicConfig::on_progress so callers (the CLI)
// can show that a long write is making progress and whether the loss is actually
// falling. Phase distinguishes the one-shot comprehension report from the
// per-step rehearsal updates.
struct EpisodicProgress {
    enum class Phase { Comprehension, Rehearsal };
    Phase   phase;
    // Comprehension: mean_nll over the document, span_count = #spans to rehearse.
    // Rehearsal: which span / step, and the loss + pre-clip grad norm this step.
    float   mean_nll    = 0.0f;
    int     span_count  = 0;
    int     span_index  = 0;   // 0-based
    int64_t span_start  = 0;   // token offset of the span
    int64_t span_len    = 0;
    int     step        = 0;   // 0-based rehearsal step within the span
    int     total_steps = 0;
    float   loss        = 0.0f;
    float   grad_norm   = 0.0f;
};

// ── EpisodicConfig ────────────────────────────────────────────────────────────
//
// Controls how surprising spans are identified and how aggressively the
// episodic write updates the model delta.
struct EpisodicConfig {
    float learning_rate      = 5e-3f;  // gradient step size for episodic write
    float surprise_threshold = 1.5f;   // per-token NLL above which a token is "surprising"
    int   think_steps        = 2;      // elaborative rehearsal passes (max, when adaptive)
    // Scale the rehearsal steps by novelty: a fact whose mean comprehension NLL is
    // low (close to what the model already knows) needs fewer passes than a wholly
    // novel one. effective = clamp(round(think_steps · meanNLL / novelty_ref), 1,
    // think_steps).  false = always use think_steps.
    bool  adaptive_steps     = false;
    // Likewise scale the learning rate by novelty: a gentler step for content the
    // model already half-knows (floored so it still learns). effective_lr =
    // learning_rate · max(0.2, min(1, meanNLL / novelty_ref)).
    bool  adaptive_lr        = false;
    // Per-token novelty weighting: weight each token's cross-entropy by its base
    // surprisal, so the gradient concentrates on the genuinely novel tokens and
    // barely moves ones the model already predicts (which also helps specificity).
    bool  token_novelty_weight = false;
    float novelty_ref        = 6.0f;   // mean NLL treated as "fully novel" (=> full steps/lr)
    int   min_span_len       = 1;      // min consecutive surprising tokens to trigger write
    int   max_span_len       = 32;     // cap on span length to bound cost per write
    bool  accumulate         = true;   // add to existing deltas; false = reset first
    float max_grad_norm      = 1.0f;   // clip rehearsal gradients to this L2 norm (<=0 disables)
    // Restrict the write to the last N transformer blocks (+ final norm), freezing
    // the embedding and earlier blocks.  Cheaper (backprop skips frozen layers) and
    // more specific (delta stays local).  -1 = last half (default); 0 = full model.
    int   trainable_last_layers = -1;
    // >0 signals the model already has LoRA adapters attached (via
    // ModernGPT::enable_episodic_lora) and the base is frozen externally — so
    // episodic_encode leaves requires_grad alone and writes only into the adapters.
    int   lora_rank = 0;
    // Locality regularisation ("light on existing memory"): while learning the
    // fact, also penalise any change to the model's logits on `locality_anchor`
    // (generic, unrelated tokens) vs the base model — so the write can be strong
    // without disturbing existing knowledge.  0 = off.  Loss += weight · MSE(
    // logits_now(anchor), logits_base(anchor)).
    float                locality_weight = 0.0f;
    std::vector<int32_t> locality_anchor;
    // Optional progress sink (nullptr = silent). Called once for comprehension,
    // then once per rehearsal step.
    std::function<void(const EpisodicProgress&)> on_progress = nullptr;
};

// ── EpisodicState ─────────────────────────────────────────────────────────────
//
// Holds the accumulated gradient-delta tensors for each model parameter.
// These are merged into the base weights before inference and unmerged after,
// so the base model is never permanently changed within a session.
//
// Index order matches model.parameters() exactly.
struct EpisodicState {
    std::vector<Tensor> deltas;  // zero-initialised; same shapes as model.parameters()
    bool merged = false;

    // Zero all deltas. Throws if the state is currently merged; call unmerge()
    // first so the in-flight delta is properly removed from the model weights.
    void reset();

    // Add deltas to model weight data (idempotent: no-op if already merged).
    // Throws if the model's parameter count differs from deltas.size().
    void merge(ModernGPT& model);

    // Subtract deltas from model weight data.
    // Throws if the model's parameter count differs from deltas.size().
    void unmerge(ModernGPT& model);
};

// Allocate a zero-initialised EpisodicState matching the given model's parameter layout.
[[nodiscard]] EpisodicState make_episodic_state(ModernGPT& model);

// ── comprehension_pass ────────────────────────────────────────────────────────
//
// Forward pass collecting per-token NLL (surprisal signal).
// Returns T-1 loss values for a T-token sequence: losses[i] = NLL of predicting
// tokens[i+1] given tokens[0..i].
[[nodiscard]] std::vector<float> comprehension_pass(
    const ModernGPT& model, const std::vector<int32_t>& tokens);

// ── episodic_encode ───────────────────────────────────────────────────────────
//
// Full episodic write pipeline:
//   1. Comprehension pass  → per-token surprisal
//   2. Span detection      → greedy-extend runs where NLL > cfg.surprise_threshold
//   3. Elaborative rehearsal → cfg.think_steps gradient steps per span
//   4. Delta accumulation  → store negative gradient in state.deltas
//
// Does NOT merge into model weights — call state.merge(model) separately.
void episodic_encode(ModernGPT& model, EpisodicState& state,
                     const std::vector<int32_t>& tokens,
                     const EpisodicConfig& cfg = {});

// Multi-document variant: rehearses every document jointly in ONE session (one
// optimizer, one captured delta), so multiple framings of a fact co-train for
// retrieval linkage while the locality penalty constrains the combined write.
// Prefer this over multiple single-doc calls when encoding a fact + its angles.
void episodic_encode(ModernGPT& model, EpisodicState& state,
                     const std::vector<std::vector<int32_t>>& docs,
                     const EpisodicConfig& cfg = {});

// ── Session persistence ───────────────────────────────────────────────────────
//
// Binary format: 8-byte magic "SUB0EPIS" + 4-byte version (1) + 8-byte n_deltas
//   per delta: 4-byte n_dims + n_dims×8-byte shape + n_elements×4-byte data
// + 1-byte merged flag.
//
// Throws std::runtime_error on I/O error or format mismatch.
void save_episodic_state(const EpisodicState& state, const std::string& path);
[[nodiscard]] EpisodicState load_episodic_state(const std::string& path);

} // namespace sub0llm::nn
