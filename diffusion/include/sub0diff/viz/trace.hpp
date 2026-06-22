#pragma once

// trace.hpp (Ch32 Viz, Phase A) — the GENERATIONTRACE: a replayable record of the diffusion reverse
// process so a viewer can SCRUB the "thought process" iteration-by-iteration.
//
// The sampler (refine_canvas) fills one Frame per refine iteration — the canvas state, which positions
// committed this frame, and per-position confidence/entropy/prediction (it already computes these).
// Plain data only (no JSON/tokenizer deps here, so the header-template sampler stays light); a separate
// writer (viz/trace_json) serialises to JSON with a token decoder. See diffusion/VISUALIZATION_DESIGN.md.

#include <cstdint>
#include <string>
#include <vector>

namespace sub0diff::viz {

// One refine iteration. All per-position vectors are length T (the canvas length).
struct Frame {
    int                       iter = 0;
    double                    ms = 0.0;          // wall-time of this iteration (forward + commit), ms
    int                       committed_count = 0;  // positions newly committed this iter
    std::vector<std::int32_t> tokens;       // current canvas token id at each position (after this iter)
    std::vector<std::uint8_t> masked;       // still [MASK]? (1/0)
    std::vector<std::uint8_t> committed;    // newly committed THIS iter? (1/0)
    std::vector<std::uint8_t> remasked;     // re-opened THIS iter? (1/0)
    std::vector<float>        confidence;   // this iter's max prob at masked positions (0 elsewhere)
    std::vector<float>        entropy;      // this iter's entropy at masked positions (0 elsewhere)
    std::vector<std::int32_t> predicted;    // this iter's best-token prediction (token id; current id elsewhere)

    // Phase B — TRUE per-level internal activations from a forward pass over THIS frame's canvas.
    // One inner vector per MERA level (order matches GenerationTrace.levels: [N, N/c, …, top]); each is
    // the per-slot RMS magnitude of that level's D-dim representation. Empty for models without a level
    // hierarchy (flat) — the viewer then falls back to the derived settled-fraction view.
    std::vector<std::vector<float>> level_rms;
};

// A whole generation. meta fields are set by the runner; frames by refine_canvas.
struct GenerationTrace {
    std::int64_t       T = 0;               // canvas length
    std::string        prompt;
    std::string        model;               // "mera" | "flat" | "hier"
    std::string        commit_order;        // "confidence" | "spread"
    float              temperature = 0.0f;
    float              conf_threshold = 0.0f;
    float              min_commit_frac = 0.0f;
    float              remask_threshold = 0.0f;
    std::int64_t       N = 0, c = 0, w = 0; // model geometry (where applicable)
    std::vector<std::int64_t> levels;       // MERA level lengths [N, N/c, …, top] (coarse-to-fine viz)
    std::vector<Frame> frames;
};

}  // namespace sub0diff::viz
