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
SUB0_API bool load_model(const char* path);        // overwrite params from disk
SUB0_API bool save_model(const char* path);        // write params to disk; false on an I/O failure
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
SUB0_API bool        load_tokenizer(const char* path);          // parse tokenizer.tok
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
// Incremental single-token inference (KV-cache): kv_reset() before a generation, then forward_one()
// per token -> logits [VOCAB]. O(T) per token instead of forward()'s O(T^2). Dense (non-ternary),
// positions < SEQ_LEN. See src/backend_cpu.cpp.
SUB0_API void        kv_reset();
SUB0_API const float* forward_one(int id, int pos);
SUB0_API Node* cross_entropy(Node* logits, const int* targets); // -> mean loss [1,1]
SUB0_API void  backward(Node* loss, float seed);                // reverse walk
SUB0_API void  reduce_gradients();                              // publish single-window grad

// Data-parallel minibatch: forward+backward each window on its own thread, then sum
// the per-thread gradients into the shared gradient. Returns mean loss; then step().
// `starts[b]` is the token offset of window b (x = data+starts[b], y = +1). `lengths`,
// if non-null, gives window b's trained length (<= T) so short documents train at their
// own length instead of a fixed T; null means every window is exactly T.
SUB0_API float train_batch(const int* data, const std::size_t* starts, int batch, int T,
                           const int* lengths = nullptr);

// --- Optimizer (used by the train stage) -----------------------------------
// AdamW by default; optionally a HYBRID Muon+AdamW split when `use_muon` is set: the hidden 2D GEMM
// weight matrices (Wq/Wk/Wv/Wo/W1/W2/Wg) route through Muon's orthogonalized-momentum update
// (include/sub0/muon.hpp), everything else (embeddings, the output head, norm gains, biases) stays
// on plain AdamW -- this is Muon's own documented usage pattern (see muon.hpp), not a project
// invention. Muon needs its own, much larger learning rate (its updates are normalized very
// differently from AdamW's) -- set independently via set_muon_lr(), not derived from set_lr().
// CPU-only for now (the CUDA backend has its own separate optimizer kernel, untouched).
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
