// sub0/core.hpp — public API of the shared engine core (libsub0_core).
//
// The engine core holds the differentiable substrate that both backend stages
// (train, gen) share: the autograd node types, the model forward pass, the
// optimizer, serialization and the tokenizer. The heavy implementation lives in
// src/engine.cpp; this header exposes just enough for the stage libraries.
//
// All compile-time model dimensions come from the generated config header, which
// is produced by sub0-configure and placed on the include path by CMake.

#pragma once

#include "sub0_config.hpp"  // generated: D_MODEL, N_LAYERS, ..., VOCAB, VOCAB_CHARS, ...
#include "layout.hpp"       // DERIVED shape constants (sub0::D_KV, GQA_GROUP, LOOP_EXEC_COUNT, ...).
                            // Pulled in here deliberately: the generated header supplies the RAW config
                            // at global scope, but anything derived from it lives in namespace sub0, and
                            // consumers of this API routinely need both. Leaving it out cost three build
                            // breaks in one session -- each a consumer that reached for a derived
                            // constant (HAS_POS_EMB, D_KV) and got "no member named ... in namespace
                            // sub0" or an undeclared identifier. One include here, not N at the callers.

#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

// Export macro: with -fvisibility=hidden, only SUB0_API symbols leave the .so,
// isolating each shared library's internals.
#if defined(_WIN32)
  #define SUB0_API __declspec(dllexport)
#else
  #define SUB0_API __attribute__((visibility("default")))
#endif

static_assert(D_MODEL % N_HEADS == 0, "d_model must be divisible by n_heads");
static_assert(SEQ_LEN >= 1 && N_LAYERS >= 1 && VOCAB >= 2, "degenerate config");

namespace sub0 {

// --- Autograd node ----------------------------------------------------------
// A Node is a [rows x cols] view into the engine's activation arena, with a
// matching gradient view and the bookkeeping needed for reverse-mode autodiff.
// Stages treat Node pointers as opaque handles (forward -> loss -> backward),
// except for reading logits via the `data` span.
enum class Op : uint8_t { Leaf, Embed, Add, Linear, RMSNorm, GELU, Rope, Attn, CrossEnt, SwiGLU, TiedHead, QKNorm };

struct Node {
    Op op = Op::Leaf;
    int rows = 0, cols = 0;
    std::span<float> data, grad;
    Node *a = nullptr, *b = nullptr, *w = nullptr, *bias = nullptr;
    std::span<float> scratch;
    const int* ids = nullptr;
    int heads = 0;
    bool ternary = false;
};

// --- Model lifecycle --------------------------------------------------------
SUB0_API void build_model();                       // allocate param layout + random init
[[nodiscard]] SUB0_API bool load_model(const char* path);        // overwrite params from disk
[[nodiscard]] SUB0_API bool save_model(const char* path);        // write params to disk; false on an I/O failure
SUB0_API void print_config();                      // human-readable config + memory line

// Fast transcendental math (vectorized exp / tanh-form GELU) for the forward and
// backward passes. Selected at compile time: configure with SUB0_EXACT_MATH=ON (or
// define SUB0_EXACT_MATH) to compile the exact std::exp/std::erf path instead. The
// approximations are ~1e-6 and training-equivalent. fast_math() reports the
// compiled-in choice.
SUB0_API bool fast_math();
SUB0_API const char* default_corpus();             // baked-in corpus path
SUB0_API const char* default_corpus_tok();         // baked-in tokenized corpus (corpus.tok)
SUB0_API const char* default_tokenizer();          // baked-in runtime tokenizer (tokenizer.tok)

// --- Tokenizer (BPE, learned at build time, loaded at runtime) -------------
// The corpus is pre-tokenized by sub0-configure, so training reads token ids
// directly. Only generation needs the tokenizer: encode() turns a prompt into
// ids (corpus-aware truecasing + BPE), detokenize() turns ids back into text.
[[nodiscard]] SUB0_API bool load_tokenizer(const char* path);   // parse tokenizer.tok
SUB0_API std::vector<int> encode(const std::string& text);      // truecase + BPE -> ids
SUB0_API std::string detokenize(const std::vector<int>& ids);   // ids -> text
// 64-bit identity fingerprint of the loaded tokenizer (0 if none). save_model stamps this so a
// model and a mismatched decoder are caught loudly on load instead of detokenizing to garble.
SUB0_API std::uint64_t tokenizer_fingerprint();
// id of the loaded tokenizer's explicit end-of-document marker (<|endoftext|>), or -1 if none is
// loaded / the tokenizer predates it. gen stops sampling when it draws this id, instead of always
// running to a fixed token budget -- the model's learned stop signal (see casing.hpp's TOK_EOS).
SUB0_API int eos_token_id();

// Sample one token id from a logits row (length VOCAB) with temperature + top-k.
// Shared by generation and the training preview so both use the same correct sampler.
SUB0_API int sample_token(const float* logits, float temp, int topk, std::mt19937& rng);

// One row of the vocabulary, for inspection tooling. `text` is a printable
// rendering of the token's expansion (control/high bytes escaped, case markers
// shown as <|cap|>/<|up|>); `expansion_len` is how many base symbols it covers.
struct TokenEntry {
    enum class Kind : std::uint8_t { Byte, CapMarker, UpMarker, Merge };
    int         id   = 0;
    Kind        kind = Kind::Byte;
    std::string text;
    int         expansion_len = 0;
};
SUB0_API std::vector<TokenEntry> vocab_entries();  // requires load_tokenizer() first

// --- Forward / loss / backward ---------------------------------------------
SUB0_API void  graph_reset();                                   // reset arena + node pool
SUB0_API Node* forward(const int* ids, int T);                  // -> logits [T, VOCAB]
// Per-execution residual-stream diagnostic: for each of LOOP_EXEC_COUNT executions, ||h_in|| and
// ||h_out - h_in||. Under LoopSplit the same layer runs several times, so pass 1's contribution vs
// pass 2's for the SAME layers directly tests whether repeated passes are converging to a fixed point
// (the predicted reason looping under-delivers). Either output may be null; both are [LOOP_EXEC_COUNT].
SUB0_API void  loop_pass_stats(const int* ids, int T, float* out_delta, float* out_hnorm);
// Incremental single-token inference (KV-cache): kv_reset() before a generation, then forward_one()
// per token -> logits [VOCAB]. O(T) per token instead of forward()'s O(T^2). Dense (non-ternary),
// positions < SEQ_LEN. See src/backend_cpu.cpp.
SUB0_API void        kv_reset();
SUB0_API const float* forward_one(int id, int pos);
// Diagnostic-only: forward_one's residual-stream hidden state at its last call's position, right before
// ln_f/the head projection -- D_MODEL floats, thread-local, CPU backend only (see backend_cpu.cpp's
// Model::last_hidden). Lets a caller compare the fully-processed representation the model actually reads
// for its next-token prediction, not just the raw input embedding -- see docs/FACTSPIKE.md's "Pack-Aware
// Training" discussion on why raw-embedding-level fidelity (hrr_unbind) isn't the same question.
SUB0_API const float* last_hidden_ptr();

// --- KV-trace memoization primitives (spike, 2026-07-21) --------------------------------------------
// See docs/SCRATCH_TOKEN_FRAMING.md's "candidate 1: post-hoc per-layer KV pooling" -- these three
// functions are the minimal engine surface a caller needs to CAPTURE a word's real per-layer (K,V)
// trace (via an ordinary kv_reset()+forward_one() loop, already exposed above -- nothing new needed for
// that half), read it out, and SPLICE a (possibly pooled/compressed) trace into a different context's
// KV-cache later, bypassing forward_one's normal per-position computation for that position entirely
// (nothing downstream needs a packed word's own query/attention-output/FFN, only later positions'
// attention over its K/V -- see the framing doc's mechanism-B discussion). Deliberately NOT wired into
// any bindings struct or forward_one dispatch branch yet -- this is spike-scoped, orchestrated directly
// by the caller. NOTE its only consumer, tests/factspike_engine_tests.cpp, is a PARKED spike and is no
// longer in the default build (tests/CMakeLists.txt, -DSUB0_BUILD_SPIKE_TESTS=ON revives it). This
// surface is deliberately kept rather than deleted: it is the substrate the parked KV-trace work
// would resume from, and it is small, stable and inert when unused. Matches this project's own "don't
// build production abstractions before they're validated" discipline (memory only-add-arguments-we-need).
//
// Read accessors: raw pointers into the calling thread's KV-cache row at (layer, pos) -- D_KV floats
// (== D_MODEL unless grouped-query attention narrows K/V), thread-local, CPU backend only, same "valid
// until the next kv_reset()/forward_one() on this thread" contract as last_hidden_ptr() above. Two
// named functions (not one function + a bool selector) so a call site never has to decode what
// `true`/`false` means at the call.
//
// `layer` is an EXECUTION index in [0, LOOP_EXEC_COUNT), not a layer index. The two coincide unless
// LoopSplit is enabled, where a repeated middle layer runs several times per token and each execution
// keeps its own K/V history -- see layout.hpp's LAYER_EXEC_ORDER.
SUB0_API const float* kv_krow_ptr(int layer, int pos);
SUB0_API const float* kv_vrow_ptr(int layer, int pos);
// Rotate `row` (D_MODEL floats, K-shaped) in place by RoPE's angle at `pos` -- a thin export of the
// backend's existing rope_row (no new rotation math). `pos` may be negative: RoPE is a pure per-call
// rotation with no other position-dependent state, so calling with -pos exactly inverts a prior call
// with +pos (R(-pos)*R(pos) = I). Two uses this spike needs from the SAME primitive: de-rotating a
// captured K row's local capture-position angle away before pooling (the local capture position is an
// arbitrary artifact of the isolated precompute context, not signal -- must NOT be confused with the
// content-derived, causal-attention-refined part of the row, which carries the real multi-hop signal and
// must never be touched), and re-rotating a pooled trace to whatever position it's spliced into later.
SUB0_API void kv_rope_rotate(float* row, int pos);
// Write a word's (pooled) per-layer trace directly into this thread's KV-cache at (layer, pos), skipping
// forward_one's normal per-position computation for that position entirely. `k_canonical` is expected
// de-rotated (as produced by the capture+kv_rope_rotate(-pos) sequence above) -- this function rotates a
// local copy by `pos` before writing it into the K row; `v` (position-invariant, RoPE never touches V)
// is written unrotated. Call once per layer (N_LAYERS times) to splice one word into a target context.
// Downstream forward_one calls at later positions need no changes: their attention loop already
// iterates generically over whatever rows are populated in the KV-cache for j <= pos.
SUB0_API void kv_splice_row(int layer, int pos, const float* k_canonical, const float* v);

// --- Periodic packed-content re-injection (spike, 2026-07-21) ---------------------------------------
// Phase G/H found mechanism A's real weakness isn't a missing-information problem, it's that a packed
// slot's embedding is composed ONCE via encode_slot() at layer 0 and then has to survive N_LAYERS of
// ordinary computation alone -- exactly the shape Nanbeige4.5's NanbeigeNgramLayerFusion targets (project
// memory nanbeige-architecture-reference): re-inject the same side-channel signal as a residual addition
// at multiple depths instead of betting everything on one upfront injection. `set_scratch_reinject(stride,
// scale)` is the minimal opt-in probe of that idea: when `stride > 0`, forward_one adds a re-scaled copy
// of the packed vector back into a bound scratch slot's own hidden state every `stride` layers, reusing
// the SAME vector `encode_slot()` already computed at layer 0 (no new learned parameters). `scale` is a
// FRACTION OF THE RESIDUAL STREAM'S OWN CURRENT RMS NORM at that layer, not an absolute embedding-scale
// constant -- a first version used the packed vector's own fixed (small, ordinary-embedding-row-scale)
// magnitude directly and was measurably a near no-op even at full strength every layer, because the
// residual stream's norm grows across depth (standard pre-norm behavior) and dwarfed the fixed-scale
// addition by mid-stack; rescaling to match h's current norm before applying `scale` is what makes this a
// real perturbation regardless of depth. `set_scratch_reinject(0, 1.0f)` restores the default (unchanged)
// behavior.
//
// EVAL-ONLY: this is wired into forward_one (the incremental decode path) only, NOT into the training
// graph (Model::forward()/train_batch, a structurally separate autodiff implementation) -- so a model
// evaluated with this enabled was never trained to expect the extra signal. That makes any result here a
// directional, out-of-distribution probe on top of an ALREADY-trained model, not the full test; a real
// positive would justify the bigger investment of wiring this into training so the model can learn to use
// it, a null/negative result is still informative before making that investment. See docs/FACTSPIKE.md's
// Phase I entry.
SUB0_API void set_scratch_reinject(int stride, float scale);

// Content-derived scratch-slot embeddings: install the per-context bindings the next forward/forward_one
// on this thread uses (a BOUND scratch slot then embeds from its fragments instead of a fixed reserved-id
// row). null clears it -> plain tok_emb lookup, unchanged. See include/sub0/scratch_slots.hpp for the
// ScratchBindings view + the encoders; a forward declaration keeps this header free of that include.
struct ScratchBindings;
SUB0_API void set_scratch_bindings(const ScratchBindings* bindings);
// Persistent (unbounded) slot range -- SPIKE, see scratch_slots.hpp's own comment on PersistentBindings
// for the full design. Per-THREAD state like set_scratch_bindings (2026-07-17; originally a process
// global under a "set once, immutable" design -- real spike usage swaps per-document tables per training
// window, and train_batch's win_persist path installs them per worker thread). A caller wanting
// whole-process behavior sets it once on each thread that forwards (or once, on the only thread). null
// clears it -> no id is ever treated as persistent-slot-bound (id >= VOCAB then composes an all-zero row
// rather than falling through to a raw table lookup -- see is_persistent_slot's own comment for why that
// fallthrough must never happen).
struct PersistentBindings;
SUB0_API void set_persistent_bindings(const PersistentBindings* bindings);
// Sentinel-PAIR bindings -- SPIKE (Phase 2 of docs/SCRATCH_TOKENS.md's word-level plan): `<sigil> h`
// pairs where the token AFTER the sentinel embeds from the binding table keyed by that token. Same
// per-context/per-window thread_local lifetime as set_scratch_bindings (swapped per window on the
// installing thread), null clears. See scratch_slots.hpp's SentinelBindings for the full design.
struct SentinelBindings;
SUB0_API void set_sentinel_bindings(const SentinelBindings* bindings);
// A target of LOSS_IGNORE_INDEX means "do not train on this position": cross_entropy adds no loss and
// no gradient for it, and normalizes by the count of ACTIVE (non-ignored) positions (PyTorch's
// `ignore_index` + reduction='mean'). Valid token ids are [0,VOCAB), so a negative is unambiguous.
// The foundation for span loss-masking -- the spike's harness-injected spans, and future
// instruction-tuning's prompt/user turns (train only on the assistant span). When no target is
// negative, active == the row count, so the loss/gradient are IDENTICAL to the unmasked path.
constexpr int LOSS_IGNORE_INDEX = -1;
SUB0_API Node* cross_entropy(Node* logits, const int* targets); // -> mean loss [1,1]
SUB0_API void  backward(Node* loss, float seed);                // reverse walk
SUB0_API void  reduce_gradients();                              // publish single-window grad

// Data-parallel minibatch: forward+backward each window on its own thread, then sum
// the per-thread gradients into the shared gradient. Returns mean loss; then step().
// `starts[b]` is the token offset of window b (x = data+starts[b], y = +1). `lengths`,
// if non-null, gives window b's trained length (<= T) so short documents train at their
// own length instead of a fixed T; null means every window is exactly T.
// `loss_mask`, if non-null, is a per-token 0/1 array parallel to `data`: predicting token p
// contributes to the loss only if loss_mask[p] != 0 (masked target positions become
// LOSS_IGNORE_INDEX; see above). null = every in-length position trains, i.e. today's behavior.
// `win_binds`, if non-null, is a per-window array of `batch` pointers: win_binds[b] installs window b's
// scratch-slot bindings (content-derived slot embeddings) for its forward+backward, null for none. Each
// pointee must outlive this call. null (the default) => no content embeddings, today's behavior exactly.
// `win_sentinel`/`win_persist` extend the same per-window pattern to the OTHER two binding views
// (sentinel pairs / the persistent id>=VOCAB range) so their spikes can use this multi-threaded path
// instead of a single-core manual loop (a ~#cores speedup, 2026-07-17). Same contract as win_binds:
// per-window pointers, pointees outlive the call, null entries/arrays = that view uninstalled.
// LEARNED-encoder caveat carries over unchanged: a bindings entry carrying a non-null enc_w_grad must
// NOT be trained through this multi-threaded path (no per-thread reduction on encoder grads -- the
// documented CharEncoder/ConvPool single-thread limit); parameter-free encodings only.
SUB0_API float train_batch(const int* data, const std::size_t* starts, int batch, int T,
                           const int* lengths = nullptr, const std::uint8_t* loss_mask = nullptr,
                           const ScratchBindings* const* win_binds = nullptr,
                           const SentinelBindings* const* win_sentinel = nullptr,
                           const PersistentBindings* const* win_persist = nullptr);

// --- Optimizer (used by the train stage) -----------------------------------
// AdamW by default; optionally a HYBRID Muon+AdamW split when `use_muon` is set: the hidden 2D GEMM
// weight matrices (Wq/Wk/Wv/Wo/W1/W2/Wg) route through Muon's orthogonalized-momentum update
// (include/sub0/muon.hpp), everything else (embeddings, the output head, norm gains, biases) stays
// on plain AdamW -- this is Muon's own documented usage pattern (see muon.hpp), not a project
// invention. Muon needs its own, much larger learning rate (its updates are normalized very
// differently from AdamW's) -- set independently via set_muon_lr(), not derived from set_lr().
// This class (AdamW::step) is the CPU implementation; the CUDA backend has its own separate,
// numerically-equivalent device pipeline (backend_cuda.cu's device_adam_step / muon_step_matrix /
// muon_newton_schulz_device) reached the same way, via --optimizer muon on a GPU build.
//
// TODO(muon-constexpr): `use_muon` is a RUNTIME choice (a `--optimizer` CLI flag), which is a
// deliberate but TEMPORARY exception to this project's normal rule (every decision knowable upfront
// is baked as constexpr/compile-time, e.g. USE_TERNARY/USE_GATED_FFN/POS_ENCODING) -- chosen only so
// A/B'ing optimizers needs no rebuild (unlike a dims/architecture choice, the optimizer doesn't touch
// PARAM_LAYOUT/checkpoint shape). Once A/B evidence justifies committing to a default, convert to a
// configure-time `constexpr bool USE_MUON` (baked into sub0_system.hpp, a machine/training-run choice
// like the tuned runtime defaults, not a model-identity one) so AdamW::step()'s per-param dispatch
// becomes `if constexpr` and the unused path compiles away. See memory: compile-time-decisions-preferred.
class SUB0_API AdamW {
public:
    explicit AdamW(float lr, bool use_muon = false);
    void zero_grad();
    void step();
    void  set_lr(float lr) { lr_ = lr; }         // per-step LR schedule (warmup + decay), AdamW-routed params
    float lr() const { return lr_; }
    void  set_muon_lr(float lr) { muon_lr_ = lr; }   // per-step LR schedule for Muon-routed params
    float muon_lr() const { return muon_lr_; }
    bool  use_muon() const { return use_muon_; }
    long step_count() const { return t_; }      // bias-correction counter (for checkpoints)
    void set_step_count(long t) { t_ = t; }      // restore on resume
private:
    float lr_, b1_ = 0.9f, b2_ = 0.95f, eps_ = 1e-8f, wd_ = 0.01f, clip_ = 1.0f;
    long t_ = 0;
    bool  use_muon_ = false;
    float muon_lr_ = 0.02f;         // Muon's own reference default (unrelated scale to AdamW's lr_)
    float muon_beta_ = 0.95f;       // Muon's momentum EMA coefficient (reference default)
};

// --- Trainable-state access (for crash-safe checkpointing) ------------------
// The parameters and the Adam moments live in the engine's static arenas. These
// expose them so the train stage can serialize the *complete* optimizer state and
// resume a run exactly. All three buffers are trainable_floats() long.
SUB0_API std::size_t trainable_floats();   // == PARAM_FLOATS
SUB0_API float*      params_ptr();         // parameter values
SUB0_API float*      grad_ptr();           // parameter gradients (filled by backward)
SUB0_API float*      adam_m_ptr();         // Adam first-moment estimates
SUB0_API float*      adam_v_ptr();         // Adam second-moment estimates

// --- Backend host/device parameter sync -------------------------------------
// On a device (GPU) backend the live parameters and Adam moments reside in device
// memory; params_ptr()/adam_*_ptr() then return host staging buffers. These hooks
// move the trainable state across that boundary around serialization: call
// sync_params_to_host() before reading the buffers for a save, and
// sync_params_to_device() after writing them on a load. On the CPU backend the
// buffers ARE the live copy, so both are no-ops.
SUB0_API void sync_params_to_host();
SUB0_API void sync_params_to_device();

}  // namespace sub0
