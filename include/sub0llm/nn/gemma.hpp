#pragma once

// gemma.hpp (Ch27) — a faithful, self-contained Gemma 4 (12B) text forward.
//
// Gemma 4 is per-layer HETEROGENEOUS and cannot be expressed by the uniform
// ModernGPT: head_dim, kv-head count, RoPE base, and attention window all vary by
// layer (local vs global). This model is therefore its own type, built directly
// from the GGUF's per-layer tensor shapes (the authoritative source — the HF config
// and even some GGUF metadata disagree with the file).
//
// Architecture (resolved from tensor shapes + llama.cpp's gemma4.cpp, the reference
// interpreter of THIS file):
//   - 48 layers, D=3840, V=262144, d_ff=15360, eps 1e-6, tied embeddings.
//   - LOCAL layers (index%6 != 5): head_dim 256, 16 q / 8 kv heads, sliding window
//     1024, RoPE base 1e4, full rotary.
//   - GLOBAL layers (index%6 == 5): head_dim 512, 16 q / 1 kv head (MQA), full
//     attention, RoPE base 1e6, rope_freqs kills pairs 64..255 (only 64 pairs rotate).
//   - Block: cur = rmsnorm(inpL, attn_norm); Q/K get rmsnorm(q/k_norm)+RoPE per head;
//     V gets plain rmsnorm (NO weight); attn scale = 1.0 (NOT 1/sqrt(dh)); wo·;
//     cur = rmsnorm(cur, post_attn_norm); attn_out = cur + inpL;
//     ffn = down·(gelu_tanh(gate·)⊙(up·)); cur = rmsnorm(ffn, post_ffw_norm);
//     inpL = (cur + attn_out) * layer_output_scale[l].
//   - Final: rmsnorm(output_norm); logits = token_embd·(tied); 30·tanh(logits/30).
//   - Embedding: dequant(token_embd[id]) * sqrt(D).
// (1+w) RMSNorm is baked into the GGUF weights at conversion, so norms are a plain
// x_normed*w. Weights are Q8_0, loaded raw in (out,in) layout; norms are f32.

#include "sub0llm/backends/cpu/quant.hpp"
#include "sub0llm/nn/gguf_loader.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace sub0llm::backend::cuda { class GemmaGpuLayers; }   // device-resident GPU layers (Ch27)

namespace sub0llm::nn {

// Thread count for the Q8 GEMVs in the Gemma forward. 0 = auto (hardware_concurrency,
// capped at 32). Set to 1 for a single-core baseline. Applies to subsequent forwards.
void set_gemma_threads(int n);

// Toggle Q/K/V and gate/up GEMV fusion (one barrier per fused group). On by default;
// off dispatches each output separately — for drift-free A/B of the optimization.
void set_gemma_fuse(bool on);

// CPU topology: logical-CPU ids grouped by performance class, auto-detected so we never
// hardcode a core index. perf = the highest-performance class (Intel P-cores); efficiency
// = the rest (E-cores / LP-E). Both empty on a homogeneous CPU or if detection is
// unavailable (caller then uses all cores). On Windows, from
// GetLogicalProcessorInformationEx's EfficiencyClass (works on every Intel hybrid from
// 12th-gen Alder Lake through Core Ultra, and ARM big.LITTLE).
struct GemmaCoreTopology {
    std::vector<int> perf;        // performance-core logical CPU ids
    std::vector<int> efficiency;  // efficiency-core logical CPU ids
    int              n_logical = 0;
    [[nodiscard]] bool hybrid() const noexcept { return !perf.empty() && !efficiency.empty(); }
};
[[nodiscard]] GemmaCoreTopology gemma_detect_cores();

// Pin the GEMV pool to these logical CPUs (worker k → cpus[k]); the thread count becomes
// min(requested, cpus.size()). Empty = let the OS schedule (no pinning). Bandwidth-bound
// decode prefers the efficiency cores (measured equal/faster, frees the P-cores); set this
// to gemma_detect_cores().efficiency for that. Applied when the pool is next (re)built.
void set_gemma_cpus(const std::vector<int>& cpus);

// Per-layer weights + shape. Q8 weights are stored as the GGUF gives them: row-major
// (out_features, in_features), so backend::cpu::matvec_q8_0_q8_0 consumes them with
// M=out, K=in directly (q/k/v are sliced per head AFTER projection, in f32).
struct GemmaLayer {
    int64_t head_dim   = 0;   // 256 local / 512 global
    int64_t n_head     = 0;   // 16
    int64_t n_kv_head  = 0;   // 8 local / 1 global
    bool    is_global  = false;
    bool    has_wv     = true; // global layers omit attn_v → V is the raw K projection
    float   rope_base  = 1e4f;
    int64_t window     = 0;   // sliding window (local); 0 = full attention (global)
    float   out_scale  = 1.0f;

    // f32 norms.
    std::vector<float> attn_norm;        // (D)
    std::vector<float> post_attn_norm;   // (D)
    std::vector<float> ffn_norm;         // (D)
    std::vector<float> post_ffw_norm;    // (D)
    std::vector<float> q_norm;           // (head_dim)
    std::vector<float> k_norm;           // (head_dim)

    // Q8 projections (out, in).
    std::vector<backend::cpu::BlockQ8_0> wq;   // (n_head*head_dim, D)
    std::vector<backend::cpu::BlockQ8_0> wk;   // (n_kv*head_dim, D)
    std::vector<backend::cpu::BlockQ8_0> wv;   // (n_kv*head_dim, D)
    std::vector<backend::cpu::BlockQ8_0> wo;   // (D, n_head*head_dim)
    std::vector<backend::cpu::BlockQ8_0> gate; // (d_ff, D)
    std::vector<backend::cpu::BlockQ8_0> up;   // (d_ff, D)
    std::vector<backend::cpu::BlockQ8_0> down; // (D, d_ff)
};

// KV cache: per layer, the (already RoPE'd K / RMS-normed V) per kv-head per position,
// stored as f32 contiguous [kv_head][pos][head_dim].
struct GemmaKVCache {
    struct Layer {
        int64_t n_kv_head = 0;
        int64_t head_dim  = 0;
        std::vector<float> k;   // flat: kv_head * (max_pos*head_dim) + pos*head_dim + d
        std::vector<float> v;
    };
    std::vector<Layer> layers;
    int64_t max_pos = 0;
    int64_t len     = 0;   // positions filled so far
};

class GemmaModel {
public:
    // Holds a unique_ptr to an incomplete GemmaGpuLayers (Ch27 hybrid), so the special members
    // are out-of-line (defined where the type is complete). Move-only (owns device state).
    GemmaModel();
    ~GemmaModel();
    GemmaModel(GemmaModel&&) noexcept;
    GemmaModel& operator=(GemmaModel&&) noexcept;

    // Build from a Q8_0 Gemma 4 GGUF (quantize-on-load: no f32 weights materialized).
    [[nodiscard]] static GemmaModel load_q8(const GGUFReader& reader);

    [[nodiscard]] GemmaKVCache make_cache(int64_t max_pos) const;

    // ── Hybrid GPU/CPU decode (Ch27) ─────────────────────────────────────────────────────────
    // Upload the first `k_gpu` layers' weights to the GPU (resident) and run them on-device for
    // subsequent forward_one_hybrid calls; the remaining layers + head stay on the CPU. `max_pos`
    // sizes the device KV cache. Throws (from the CUDA backend) if CUDA is not compiled in. Call
    // once before the decode loop; k_gpu in [0, n_layers].
    void enable_gpu_layers(int k_gpu, int64_t max_pos, bool kv_q8 = false);
    [[nodiscard]] int n_gpu_layers() const noexcept { return n_gpu_layers_; }

    // Like forward_one, but the first n_gpu_layers() layers run on the GPU (device KV cache) and
    // the rest on the CPU (host `kv`), with ONE activation hand-off at the boundary. Numerically
    // ~equal to forward_one (per-layer ~5e-7); the gate is identical GREEDY tokens. Non-const: it
    // mutates the device KV held by the resident GPU layers. Requires enable_gpu_layers() first.
    [[nodiscard]] std::vector<float> forward_one_hybrid(int32_t token, int64_t pos, GemmaKVCache& kv,
                                                        bool apply_softcap = true,
                                                        bool compute_logits = true);

    // Hybrid PREFILL: the first n_gpu_layers() layers run as ONE batched GPU pass (IMMA tensor
    // cores — the compute-bound path), filling the device KV cache; the remaining layers run on the
    // CPU (filling host `kv`). Returns the LAST token's logits. Far faster than token-by-token
    // prefill for the GPU layers. Requires enable_gpu_layers() first; falls back to forward_prefill.
    [[nodiscard]] std::vector<float> forward_prefill_hybrid(const int32_t* tokens, int64_t T,
                                                            int64_t start_pos, GemmaKVCache& kv,
                                                            bool apply_softcap = true);

    // One decode step: append token at position `pos` to the cache, return logits (V).
    // `apply_softcap` controls the final logit soft-cap (30·tanh(z/30)). It is
    // order-preserving, so GREEDY decoding can pass false (identical argmax) and skip
    // 262 K tanh/token; pass true when the magnitudes matter (sampling, logit/nll parity).
    // `compute_logits=false` runs the layers (filling the KV cache) but SKIPS the final
    // norm + V-wide LM head and returns {} — for prompt prefill of every token except the
    // last, whose logits no one reads (saves the biggest single GEMV per prompt token).
    [[nodiscard]] std::vector<float> forward_one(int32_t token, int64_t pos,
                                                 GemmaKVCache& kv,
                                                 bool apply_softcap = true,
                                                 bool compute_logits = true) const;

    // Batched prompt prefill: process tokens[start_pos .. start_pos+T) in ONE pass per
    // layer using the row-reuse Q8 GEMM (weights streamed once, reused across all T
    // tokens — compute-bound, not the per-token bandwidth wall), filling the KV cache.
    // Returns the LAST token's logits (the only ones prefill needs). Equivalent to T
    // sequential forward_one calls but far faster for T>1; falls back to forward_one for
    // T==1. `apply_softcap` as above.
    [[nodiscard]] std::vector<float> forward_prefill(const int32_t* tokens, int64_t T,
                                                     int64_t start_pos, GemmaKVCache& kv,
                                                     bool apply_softcap = true) const;

    [[nodiscard]] int64_t vocab_size()  const noexcept { return V_; }
    [[nodiscard]] int64_t embed_dim()   const noexcept { return D_; }
    [[nodiscard]] int64_t n_layers()    const noexcept { return static_cast<int64_t>(layers_.size()); }
    [[nodiscard]] int64_t n_params()    const noexcept { return n_params_; }

private:
    // Shared building blocks of the forward (used by forward_one and forward_one_hybrid).
    [[nodiscard]] std::vector<float> embed_(int32_t token) const;     // dequant embedding · sqrt(D)
    void cpu_layer_(const GemmaLayer& L, GemmaKVCache::Layer& C,
                    std::vector<float>& x, int64_t pos, int64_t max_pos) const;  // one CPU layer
    void cpu_prefill_layer_(const GemmaLayer& L, GemmaKVCache::Layer& C, std::vector<float>& X,
                            int64_t start_pos, int64_t T, int64_t max_pos) const;  // one BATCHED CPU layer
    [[nodiscard]] std::vector<float> head_(std::vector<float> x, bool apply_softcap) const;

    int64_t V_ = 0, D_ = 0, d_ff_ = 0;
    float   eps_ = 1e-6f;
    float   embed_scale_ = 1.0f;
    float   final_softcap_ = 30.0f;
    int64_t n_params_ = 0;

    std::vector<GemmaLayer> layers_;
    std::vector<float>      rope_freqs_;   // global freq_factors (head_dim_global/2)
    std::vector<float>      output_norm_;  // (D)
    std::vector<backend::cpu::BlockQ8_0> token_embd_;  // (V, D) — embedding + tied LM head

    int n_gpu_layers_ = 0;
    std::unique_ptr<backend::cuda::GemmaGpuLayers> gpu_;   // resident GPU layers [0, n_gpu_layers_)
};

} // namespace sub0llm::nn
