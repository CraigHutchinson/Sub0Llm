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
enum class Op : uint8_t { Leaf, Embed, Add, Linear, RMSNorm, GELU, Attn, CrossEnt };

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
SUB0_API void save_model(const char* path);        // write params to disk
SUB0_API void print_config();                      // human-readable config + memory line

// Fast transcendental math (vectorized exp / tanh-form GELU) for the forward and
// backward passes. On by default; set SUB0_EXACT_MATH=1 in the environment, or call
// set_fast_math(false), to fall back to the exact std::exp/std::erf path. The
// approximations are ~1e-6 and training-equivalent, but the switch stays for A/B.
SUB0_API void set_fast_math(bool on);
SUB0_API bool fast_math();
SUB0_API const char* default_corpus();             // baked-in corpus path
SUB0_API const char* default_corpus_tok();         // baked-in tokenized corpus (corpus.tok)
SUB0_API const char* default_tokenizer();          // baked-in runtime tokenizer (tokenizer.bin)

// --- Tokenizer (BPE, learned at build time, loaded at runtime) -------------
// The corpus is pre-tokenized by sub0-configure, so training reads token ids
// directly. Only generation needs the tokenizer: encode() turns a prompt into
// ids (corpus-aware truecasing + BPE), detokenize() turns ids back into text.
SUB0_API bool        load_tokenizer(const char* path);          // parse tokenizer.bin
SUB0_API std::vector<int> encode(const std::string& text);      // truecase + BPE -> ids
SUB0_API std::string detokenize(const std::vector<int>& ids);   // ids -> text

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
SUB0_API Node* cross_entropy(Node* logits, const int* targets); // -> mean loss [1,1]
SUB0_API void  backward(Node* loss, float seed);                // reverse walk

// --- Optimizer (used by the train stage) -----------------------------------
class SUB0_API AdamW {
public:
    explicit AdamW(float lr);
    void zero_grad();
    void step();
    long step_count() const { return t_; }      // bias-correction counter (for checkpoints)
    void set_step_count(long t) { t_ = t; }      // restore on resume
private:
    float lr_, b1_ = 0.9f, b2_ = 0.95f, eps_ = 1e-8f, wd_ = 0.01f, clip_ = 1.0f;
    long t_ = 0;
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

}  // namespace sub0
