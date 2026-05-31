#pragma once

#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// ── EpisodicConfig ────────────────────────────────────────────────────────────
//
// Controls how surprising spans are identified and how aggressively the
// episodic write updates the model delta.
struct EpisodicConfig {
    float learning_rate      = 5e-3f;  // gradient step size for episodic write
    float surprise_threshold = 1.5f;   // per-token NLL above which a token is "surprising"
    int   think_steps        = 2;      // elaborative rehearsal passes per span
    int   min_span_len       = 1;      // min consecutive surprising tokens to trigger write
    int   max_span_len       = 32;     // cap on span length to bound cost per write
    bool  accumulate         = true;   // add to existing deltas; false = reset first
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

    // Zero all deltas (call between unrelated sessions).
    void reset();

    // Add deltas to model weight data (idempotent: no-op if already merged).
    void merge(ModernGPT& model);

    // Subtract deltas from model weight data.
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

} // namespace sub0llm::nn
