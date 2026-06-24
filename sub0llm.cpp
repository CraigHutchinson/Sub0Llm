// sub0llm.cpp — a tiny, CPU-only, single-file decoder-only Transformer (GPT-style)
// language model. Statically allocated: all model dimensions are constexpr, all
// memory (parameters + activations) lives in fixed-size std::array buffers that
// are allocated once (in BSS) and reused. There is no per-step heap allocation.
//
// Design notes (the "why"):
//   * A tensor's size is decided entirely by the config. Parameter tensors depend
//     only on the constexpr dimensions; activation tensors are [T x dim] where the
//     sequence length T is bounded by SEQ_LEN. So every buffer has a compile-time
//     maximum size and nothing needs dynamic allocation.
//   * Parameters live in one flat static array; each weight is a std::span view.
//   * Activations live in a reused "arena" (static array) that is reset at the
//     start of every forward pass; activation tensors are spans into it.
//   * The autograd graph is a fixed pool of enum-tagged Nodes (no std::function,
//     no closures), walked in reverse for backprop.
//   * Tokenization is byte-level, so VOCAB = 256 is a compile-time constant.
//
// It also implements an optional BitNet b1.58-style ternary linear layer (weights
// quantized to {-1,0,+1} with a straight-through estimator) — the "1-bit LLM"
// idea — toggled by the constexpr USE_TERNARY below.
//
// Build:  g++ -std=c++23 -O3 -march=native -fopenmp sub0llm.cpp -o sub0llm
// Train:  ./sub0llm train :builtin: model.bin --steps 3000
// Sample: ./sub0llm gen model.bin "the cat " --n 200 --temp 0.8 --topk 20

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

// ============================================================================
//  Section 1. Compile-time model configuration
//  Everything downstream — buffer sizes, node counts, the arena — is derived
//  from these constants at compile time.
// ============================================================================

constexpr int  D_MODEL     = 96;            // residual-stream / embedding width
constexpr int  N_LAYERS    = 3;             // transformer blocks
constexpr int  N_HEADS     = 4;             // attention heads
constexpr int  SEQ_LEN     = 64;            // max context window (bounds T)
constexpr int  D_FF        = 4 * D_MODEL;   // feed-forward inner width
constexpr int  VOCAB       = 256;           // byte-level tokenizer -> fixed
constexpr bool USE_TERNARY = false;         // BitNet-style ternary block weights

constexpr int  D_HEAD      = D_MODEL / N_HEADS;

static_assert(D_MODEL % N_HEADS == 0, "d_model must be divisible by n_heads");
static_assert(SEQ_LEN >= 1 && N_LAYERS >= 1, "degenerate config");

// ---- Derived sizes (all compile-time) --------------------------------------

// Number of distinct parameter tensors: tok_emb, pos_emb, then 10 per layer
// (ln1, ln2, Wq, Wk, Wv, Wo, W1, b1, W2, b2), then ln_f, lm_head, lm_bias.
constexpr int NUM_PARAMS = 2 + 10 * N_LAYERS + 3;

// Total parameter floats, computed exactly from the config.
consteval size_t calc_param_floats() {
    size_t n = 0;
    n += (size_t)VOCAB * D_MODEL;      // tok_emb
    n += (size_t)SEQ_LEN * D_MODEL;    // pos_emb
    for (int l = 0; l < N_LAYERS; ++l) {
        n += D_MODEL;                  // ln1 gain
        n += D_MODEL;                  // ln2 gain
        n += (size_t)4 * D_MODEL * D_MODEL;   // Wq,Wk,Wv,Wo
        n += (size_t)D_MODEL * D_FF;   // W1
        n += D_FF;                     // b1
        n += (size_t)D_FF * D_MODEL;   // W2
        n += D_MODEL;                  // b2
    }
    n += D_MODEL;                      // ln_f gain
    n += (size_t)D_MODEL * VOCAB;      // lm_head
    n += VOCAB;                        // lm_bias
    return n;
}
constexpr size_t PARAM_FLOATS = calc_param_floats();

// Worst-case activation floats for one forward pass (T = SEQ_LEN), plus margin.
consteval size_t calc_act_cap() {
    const size_t T = SEQ_LEN, C = D_MODEL, F = D_FF, H = N_HEADS, V = VOCAB;
    size_t base = 3 * T * C;  // tok emb, pos emb, their sum
    size_t per  = 10 * T * C      // rmsnorm,q,k,v,attn,o,add,rmsnorm2,w2,add2 (~T*C each)
                + 2 * T * F        // w1 out + gelu out
                + H * T * T        // attention probabilities (scratch)
                + 2 * T            // rmsnorm 1/rms scratch
                + (USE_TERNARY ? (size_t)4 * C * C + 2 * C * F : 0);  // ternary scratch
    size_t fin  = T * C + 2 * T * V + 64;  // final norm, logits, cross-entropy probs
    return base + (size_t)N_LAYERS * per + fin;
}
constexpr size_t ACT_CAP = calc_act_cap() * 3 / 2 + 8192;  // 1.5x safety margin

// Max graph nodes per forward: a handful of base nodes + ~16 per layer.
constexpr size_t MAX_NODES = 16 + 16 * (size_t)N_LAYERS;

// ============================================================================
//  Section 2. Static storage: parameters, activation arena, node pool
// ============================================================================

// --- Parameters: flat arrays; weights are spans into these. -----------------
static std::array<float, PARAM_FLOATS> g_param_data{};
static std::array<float, PARAM_FLOATS> g_param_grad{};
static std::array<float, PARAM_FLOATS> g_param_m{};
static std::array<float, PARAM_FLOATS> g_param_vel{};

// --- Activation arena: reset each forward pass. -----------------------------
static std::array<float, ACT_CAP> g_act_data{};
static std::array<float, ACT_CAP> g_act_grad{};
static size_t g_act_used = 0;

// Allocate n floats of activation (data + grad), zeroed. Aborts if undersized.
static std::pair<std::span<float>, std::span<float>> arena_alloc(size_t n) {
    if (g_act_used + n > ACT_CAP) {
        std::fprintf(stderr, "fatal: activation arena overflow (need %zu, cap %zu)\n",
                     g_act_used + n, ACT_CAP);
        std::abort();
    }
    size_t off = g_act_used;
    g_act_used += n;
    std::span<float> d(g_act_data.data() + off, n);
    std::span<float> gr(g_act_grad.data() + off, n);
    std::fill(d.begin(), d.end(), 0.f);
    std::fill(gr.begin(), gr.end(), 0.f);
    return {d, gr};
}

// ============================================================================
//  Section 3. Computation graph: enum-tagged Node pool (no closures, no heap)
// ============================================================================

enum class Op : uint8_t { Leaf, Embed, Add, Linear, RMSNorm, GELU, Attn, CrossEnt };

struct Node {
    Op op = Op::Leaf;
    int rows = 0, cols = 0;
    std::span<float> data, grad;
    // Inputs (meaning depends on op): generic a/b, weight w, bias.
    Node* a = nullptr;
    Node* b = nullptr;
    Node* w = nullptr;
    Node* bias = nullptr;
    std::span<float> scratch;     // saved forward state for backward
    const int* ids = nullptr;     // embed: row ids; cross-ent: targets
    int heads = 0;
    bool ternary = false;
};

// Parameter nodes persist for the program; pool nodes are per-forward.
static std::array<Node, NUM_PARAMS> g_param_nodes;
static std::array<Node, MAX_NODES>  g_pool;
static size_t g_pool_used = 0;

// Per-parameter view for the optimizer (offset/size into the flat arrays + decay).
struct ParamView { size_t off, n; bool decay; };
static std::array<ParamView, NUM_PARAMS> g_param_views;
static size_t g_param_used = 0;
static int g_pcount = 0;

// Register one parameter tensor; returns its (persistent) Node.
static Node* mk_param(int r, int c, bool decay) {
    size_t n = (size_t)r * c, off = g_param_used;
    g_param_used += n;
    Node& nd = g_param_nodes[g_pcount];
    nd.op = Op::Leaf;
    nd.rows = r; nd.cols = c;
    nd.data = std::span<float>(g_param_data.data() + off, n);
    nd.grad = std::span<float>(g_param_grad.data() + off, n);
    g_param_views[g_pcount] = {off, n, decay};
    ++g_pcount;
    return &nd;
}

// Allocate a fresh intermediate node bound to arena buffers of shape [r,c].
static Node* mk_node(Op op, int r, int c) {
    if (g_pool_used >= MAX_NODES) { std::fprintf(stderr, "fatal: node pool overflow\n"); std::abort(); }
    Node& nd = g_pool[g_pool_used++];
    nd = Node{};
    nd.op = op; nd.rows = r; nd.cols = c;
    auto [d, gr] = arena_alloc((size_t)r * c);
    nd.data = d; nd.grad = gr;
    return &nd;
}

// Reset the graph + arena for a new forward pass.
static void graph_reset() { g_pool_used = 0; g_act_used = 0; }

// Convenience accessors.
static inline float& el(std::span<float> s, int cols, int i, int j) { return s[(size_t)i * cols + j]; }

// ============================================================================
//  Section 4. Differentiable ops — forward builders
//  Each computes its forward result and stores whatever the backward needs.
// ============================================================================

// Gather rows of `table` by ids -> [T, C].
static Node* op_embed(Node* table, const int* ids, int T) {
    const int C = table->cols;
    Node* out = mk_node(Op::Embed, T, C);
    out->w = table; out->ids = ids;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < C; ++j) el(out->data, C, t, j) = el(table->data, C, ids[t], j);
    return out;
}

// Elementwise add (residual). a and b share shape.
static Node* op_add(Node* a, Node* b) {
    Node* out = mk_node(Op::Add, a->rows, a->cols);
    out->a = a; out->b = b;
    for (size_t i = 0; i < out->data.size(); ++i) out->data[i] = a->data[i] + b->data[i];
    return out;
}

// Absmean ternarization of a weight span -> {-s,0,+s}, s = mean(|w|).
static void ternarize_into(std::span<const float> w, std::span<float> q) {
    double s = 0.0;
    for (float v : w) s += std::fabs(v);
    float scale = (float)(s / std::max<size_t>(1, w.size())) + 1e-8f;
    for (size_t i = 0; i < w.size(); ++i) {
        float r = w[i] / scale;
        float t = r > 0.5f ? 1.f : (r < -0.5f ? -1.f : 0.f);
        q[i] = t * scale;
    }
}

// Linear: y = x @ W (+ bias). W:[in,out]. If ternary, forward uses ternarized
// weights (saved in scratch); backward updates full-precision W (straight-through).
static Node* op_linear(Node* x, Node* W, Node* bias, bool ternary) {
    const int T = x->rows, in = x->cols, out = W->cols;
    Node* y = mk_node(Op::Linear, T, out);
    y->a = x; y->w = W; y->bias = bias; y->ternary = ternary;
    const float* Wf = W->data.data();
    if (ternary) {
        auto [sd, sg] = arena_alloc(W->data.size());
        ternarize_into(W->data, sd);
        y->scratch = sd;
        Wf = sd.data();
    }
    const float* X = x->data.data();
    #pragma omp parallel for
    for (int t = 0; t < T; ++t)
        for (int p = 0; p < in; ++p) {
            float xtp = X[(size_t)t * in + p];
            if (xtp == 0.f) continue;
            const float* Wr = Wf + (size_t)p * out;
            float* Yr = y->data.data() + (size_t)t * out;
            for (int o = 0; o < out; ++o) Yr[o] += xtp * Wr[o];
        }
    if (bias)
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < out; ++o) el(y->data, out, t, o) += bias->data[o];
    return y;
}

// RMSNorm: per-row y = x / sqrt(mean(x^2)+eps) * gamma. gamma:[1,C].
static Node* op_rmsnorm(Node* x, Node* gamma) {
    const int T = x->rows, C = x->cols;
    const float eps = 1e-5f;
    Node* y = mk_node(Op::RMSNorm, T, C);
    y->a = x; y->w = gamma;
    auto [rinv, rinv_g] = arena_alloc(T);  // store 1/rms per row for backward
    y->scratch = rinv;
    for (int t = 0; t < T; ++t) {
        float ms = 0.f;
        for (int j = 0; j < C; ++j) ms += el(x->data, C, t, j) * el(x->data, C, t, j);
        ms /= C;
        float r = 1.f / std::sqrt(ms + eps);
        rinv[t] = r;
        for (int j = 0; j < C; ++j)
            el(y->data, C, t, j) = el(x->data, C, t, j) * r * gamma->data[j];
    }
    return y;
}

// GELU (exact, erf-based), elementwise.
static Node* op_gelu(Node* x) {
    Node* y = mk_node(Op::GELU, x->rows, x->cols);
    y->a = x;
    const float inv_sqrt2 = 0.70710678f;
    for (size_t i = 0; i < x->data.size(); ++i) {
        float v = x->data[i];
        y->data[i] = 0.5f * v * (1.f + std::erf(v * inv_sqrt2));
    }
    return y;
}

// Multi-head causal self-attention. q,k,v: [T,C]. Saves softmax probs (per head,
// flattened) in scratch for backward. Stores v in `bias` slot (spare pointer).
static Node* op_attn(Node* q, Node* k, Node* v, int H) {
    const int T = q->rows, C = q->cols, d = C / H;
    const float scale = 1.f / std::sqrt((float)d);
    Node* out = mk_node(Op::Attn, T, C);
    out->a = q; out->b = k; out->bias = v; out->heads = H;
    auto [P, Pg] = arena_alloc((size_t)H * T * T);
    out->scratch = P;
    auto Pidx = [T](int h, int i, int j) { return ((size_t)h * T + i) * T + j; };
    for (int h = 0; h < H; ++h) {
        int off = h * d;
        for (int i = 0; i < T; ++i) {
            float mx = -1e30f;
            std::array<float, SEQ_LEN> sc{};
            for (int j = 0; j <= i; ++j) {
                float s = 0.f;
                for (int a = 0; a < d; ++a) s += el(q->data, C, i, off + a) * el(k->data, C, j, off + a);
                s *= scale; sc[j] = s; mx = std::max(mx, s);
            }
            float Z = 0.f;
            for (int j = 0; j <= i; ++j) { sc[j] = std::exp(sc[j] - mx); Z += sc[j]; }
            for (int j = 0; j <= i; ++j) {
                float p = sc[j] / Z;
                P[Pidx(h, i, j)] = p;
                for (int a = 0; a < d; ++a)
                    el(out->data, C, i, off + a) += p * el(v->data, C, j, off + a);
            }
        }
    }
    return out;
}

// Softmax cross-entropy over rows. logits:[T,V]. Returns a [1,1] loss (mean over
// positions). Saves per-row probabilities in scratch and targets for backward.
static Node* op_cross_entropy(Node* logits, const int* targets) {
    const int T = logits->rows, V = logits->cols;
    Node* loss = mk_node(Op::CrossEnt, 1, 1);
    loss->a = logits; loss->ids = targets;
    auto [probs, pg] = arena_alloc((size_t)T * V);
    loss->scratch = probs;
    float total = 0.f;
    for (int t = 0; t < T; ++t) {
        float mx = -1e30f;
        for (int j = 0; j < V; ++j) mx = std::max(mx, el(logits->data, V, t, j));
        float Z = 0.f;
        for (int j = 0; j < V; ++j) { float e = std::exp(el(logits->data, V, t, j) - mx); probs[(size_t)t * V + j] = e; Z += e; }
        for (int j = 0; j < V; ++j) probs[(size_t)t * V + j] /= Z;
        total += -std::log(std::max(1e-9f, probs[(size_t)t * V + targets[t]]));
    }
    loss->data[0] = total / T;
    return loss;
}

// ============================================================================
//  Section 5. Backward dispatch (reverse walk over the node pool)
// ============================================================================

static void backward_node(Node& n) {
    switch (n.op) {
    case Op::Leaf: break;

    case Op::Embed: {
        const int T = n.rows, C = n.cols;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < C; ++j) el(n.w->grad, C, n.ids[t], j) += el(n.grad, C, t, j);
        break;
    }
    case Op::Add: {
        for (size_t i = 0; i < n.grad.size(); ++i) { n.a->grad[i] += n.grad[i]; n.b->grad[i] += n.grad[i]; }
        break;
    }
    case Op::Linear: {
        Node* x = n.a; Node* W = n.w;
        const int T = x->rows, in = x->cols, out = W->cols;
        const float* Wf = n.ternary ? n.scratch.data() : W->data.data();
        const float* dY = n.grad.data();
        const float* X = x->data.data();
        // dx = dY @ Wf^T
        #pragma omp parallel for
        for (int t = 0; t < T; ++t)
            for (int p = 0; p < in; ++p) {
                float s = 0.f;
                const float* dYr = dY + (size_t)t * out;
                const float* Wr = Wf + (size_t)p * out;
                for (int o = 0; o < out; ++o) s += dYr[o] * Wr[o];
                x->grad[(size_t)t * in + p] += s;
            }
        // dW = x^T @ dY (straight-through to full-precision weights)
        #pragma omp parallel for
        for (int p = 0; p < in; ++p)
            for (int o = 0; o < out; ++o) {
                float s = 0.f;
                for (int t = 0; t < T; ++t) s += X[(size_t)t * in + p] * dY[(size_t)t * out + o];
                W->grad[(size_t)p * out + o] += s;
            }
        if (n.bias)
            for (int t = 0; t < T; ++t)
                for (int o = 0; o < out; ++o) n.bias->grad[o] += dY[(size_t)t * out + o];
        break;
    }
    case Op::RMSNorm: {
        Node* x = n.a; Node* g = n.w;
        const int T = x->rows, C = x->cols;
        std::span<float> rinv = n.scratch;
        for (int t = 0; t < T; ++t) {
            float S = 0.f;
            for (int j = 0; j < C; ++j) S += el(n.grad, C, t, j) * g->data[j] * el(x->data, C, t, j);
            float r = rinv[t], r3 = r * r * r;
            for (int j = 0; j < C; ++j) {
                float xj = el(x->data, C, t, j), dy = el(n.grad, C, t, j), gj = g->data[j];
                el(x->grad, C, t, j) += r * dy * gj - (xj * r3 / C) * S;
                g->grad[j] += dy * xj * r;
            }
        }
        break;
    }
    case Op::GELU: {
        Node* x = n.a;
        const float inv_sqrt2 = 0.70710678f, inv_sqrt2pi = 0.39894228f;
        for (size_t i = 0; i < x->data.size(); ++i) {
            float v = x->data[i];
            float cdf = 0.5f * (1.f + std::erf(v * inv_sqrt2));
            float pdf = inv_sqrt2pi * std::exp(-0.5f * v * v);
            x->grad[i] += n.grad[i] * (cdf + v * pdf);
        }
        break;
    }
    case Op::Attn: {
        Node* q = n.a; Node* k = n.b; Node* v = n.bias;
        const int T = q->rows, C = q->cols, H = n.heads, d = C / H;
        const float scale = 1.f / std::sqrt((float)d);
        std::span<float> P = n.scratch;
        auto Pidx = [T](int h, int i, int j) { return ((size_t)h * T + i) * T + j; };
        for (int h = 0; h < H; ++h) {
            int off = h * d;
            for (int i = 0; i < T; ++i) {
                std::array<float, SEQ_LEN> dP{};
                for (int j = 0; j <= i; ++j) {
                    float p = P[Pidx(h, i, j)], dp = 0.f;
                    for (int a = 0; a < d; ++a) {
                        float dout = el(n.grad, C, i, off + a);
                        el(v->grad, C, j, off + a) += p * dout;
                        dp += dout * el(v->data, C, j, off + a);
                    }
                    dP[j] = dp;
                }
                float dot = 0.f;
                for (int j = 0; j <= i; ++j) dot += P[Pidx(h, i, j)] * dP[j];
                for (int j = 0; j <= i; ++j) {
                    float ds = P[Pidx(h, i, j)] * (dP[j] - dot) * scale;
                    for (int a = 0; a < d; ++a) {
                        el(q->grad, C, i, off + a) += ds * el(k->data, C, j, off + a);
                        el(k->grad, C, j, off + a) += ds * el(q->data, C, i, off + a);
                    }
                }
            }
        }
        break;
    }
    case Op::CrossEnt: {
        Node* logits = n.a;
        const int T = logits->rows, V = logits->cols;
        std::span<float> probs = n.scratch;
        float g = n.grad[0] / T;  // seed (set per-batch) times 1/T mean factor
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < V; ++j) {
                float p = probs[(size_t)t * V + j];
                el(logits->grad, V, t, j) += g * (p - (j == n.ids[t] ? 1.f : 0.f));
            }
        break;
    }
    }
}

// Seed the loss gradient and walk the pool in reverse.
static void backward(Node* loss, float seed) {
    loss->grad[0] = seed;
    for (size_t i = g_pool_used; i-- > 0; ) backward_node(g_pool[i]);
}

// ============================================================================
//  Section 6. Model: parameter layout + forward pass
// ============================================================================

struct Layer { Node *ln1, *ln2, *Wq, *Wk, *Wv, *Wo, *W1, *b1, *W2, *b2; };

struct Model {
    Node* tok_emb;
    Node* pos_emb;
    std::array<Layer, N_LAYERS> layers;
    Node* ln_f;
    Node* lm_head;
    Node* lm_bias;

    // Build parameter nodes (must match calc_param_floats order) and init values.
    void build() {
        std::mt19937 rng(1234);
        auto randn = [&](Node* t, float std) {
            std::normal_distribution<float> nd(0.f, std);
            for (auto& x : t->data) x = nd(rng);
        };
        auto ones = [](Node* t) { std::fill(t->data.begin(), t->data.end(), 1.f); };

        tok_emb = mk_param(VOCAB, D_MODEL, false);   randn(tok_emb, 0.02f);
        pos_emb = mk_param(SEQ_LEN, D_MODEL, false); randn(pos_emb, 0.02f);
        for (auto& L : layers) {
            L.ln1 = mk_param(1, D_MODEL, false); ones(L.ln1);
            L.ln2 = mk_param(1, D_MODEL, false); ones(L.ln2);
            L.Wq = mk_param(D_MODEL, D_MODEL, true); randn(L.Wq, 0.02f);
            L.Wk = mk_param(D_MODEL, D_MODEL, true); randn(L.Wk, 0.02f);
            L.Wv = mk_param(D_MODEL, D_MODEL, true); randn(L.Wv, 0.02f);
            L.Wo = mk_param(D_MODEL, D_MODEL, true); randn(L.Wo, 0.02f);
            L.W1 = mk_param(D_MODEL, D_FF, true); randn(L.W1, 0.02f);
            L.b1 = mk_param(1, D_FF, false);
            L.W2 = mk_param(D_FF, D_MODEL, true); randn(L.W2, 0.02f);
            L.b2 = mk_param(1, D_MODEL, false);
        }
        ln_f = mk_param(1, D_MODEL, false); ones(ln_f);
        lm_head = mk_param(D_MODEL, VOCAB, true); randn(lm_head, 0.02f);
        lm_bias = mk_param(1, VOCAB, false);
    }

    // Forward over `ids` (length T <= SEQ_LEN). Returns logits node [T, VOCAB].
    Node* forward(const int* ids, int T) {
        static int pos_ids[SEQ_LEN];
        for (int t = 0; t < T; ++t) pos_ids[t] = t;
        constexpr bool q = USE_TERNARY;
        Node* h = op_add(op_embed(tok_emb, ids, T), op_embed(pos_emb, pos_ids, T));
        for (auto& L : layers) {
            Node* a = op_rmsnorm(h, L.ln1);
            Node* att = op_attn(op_linear(a, L.Wq, nullptr, q),
                                op_linear(a, L.Wk, nullptr, q),
                                op_linear(a, L.Wv, nullptr, q), N_HEADS);
            h = op_add(h, op_linear(att, L.Wo, nullptr, q));
            Node* f = op_rmsnorm(h, L.ln2);
            f = op_linear(op_gelu(op_linear(f, L.W1, L.b1, q)), L.W2, L.b2, q);
            h = op_add(h, f);
        }
        h = op_rmsnorm(h, ln_f);
        return op_linear(h, lm_head, lm_bias, false);  // head stays full precision
    }
};

// ============================================================================
//  Section 7. AdamW optimizer (operates directly on the flat param arrays)
// ============================================================================

struct AdamW {
    float lr, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.01f, clip = 1.0f;
    long t = 0;
    explicit AdamW(float lr_) : lr(lr_) {}

    void zero_grad() { g_param_grad.fill(0.f); }

    void step() {
        double sq = 0.0;
        for (float g : g_param_grad) sq += (double)g * g;
        float norm = (float)std::sqrt(sq);
        float gs = (norm > clip) ? clip / (norm + 1e-6f) : 1.f;

        ++t;
        float bc1 = 1.f - std::pow(b1, (float)t);
        float bc2 = 1.f - std::pow(b2, (float)t);
        for (int pi = 0; pi < g_pcount; ++pi) {
            const ParamView& pv = g_param_views[pi];
            for (size_t i = pv.off; i < pv.off + pv.n; ++i) {
                float g = g_param_grad[i] * gs;
                g_param_m[i]   = b1 * g_param_m[i]   + (1 - b1) * g;
                g_param_vel[i] = b2 * g_param_vel[i] + (1 - b2) * g * g;
                float mhat = g_param_m[i] / bc1;
                float vhat = g_param_vel[i] / bc2;
                g_param_data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
                if (pv.decay) g_param_data[i] -= lr * wd * g_param_data[i];
            }
        }
    }
};

// ============================================================================
//  Section 8. Byte-level tokenizer (trivial: a token IS a byte)
// ============================================================================

static std::vector<int> encode(const std::string& s) {
    std::vector<int> out(s.size());
    for (size_t i = 0; i < s.size(); ++i) out[i] = (unsigned char)s[i];
    return out;
}
static char decode(int id) { return (char)(unsigned char)id; }

// ============================================================================
//  Section 9. Model serialization (header validates the constexpr config)
// ============================================================================

struct Header {
    char magic[4] = {'S', '0', 'L', '3'};
    int d_model = D_MODEL, n_layers = N_LAYERS, n_heads = N_HEADS;
    int d_ff = D_FF, seq_len = SEQ_LEN, vocab = VOCAB, ternary = USE_TERNARY;
    uint64_t param_floats = PARAM_FLOATS;
};

static void save_model(const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    Header h;
    os.write((const char*)&h, sizeof(h));
    os.write((const char*)g_param_data.data(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
}

static bool load_model(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    Header h, ref;
    is.read((char*)&h, sizeof(h));
    if (std::memcmp(h.magic, ref.magic, 4) != 0 ||
        h.d_model != ref.d_model || h.n_layers != ref.n_layers ||
        h.n_heads != ref.n_heads || h.d_ff != ref.d_ff ||
        h.seq_len != ref.seq_len || h.vocab != ref.vocab ||
        h.ternary != ref.ternary || h.param_floats != ref.param_floats) {
        std::fprintf(stderr, "error: model was built with a different (constexpr) config\n");
        return false;
    }
    is.read((char*)g_param_data.data(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
    return (bool)is;
}

// ============================================================================
//  Section 10. Generation and training
// ============================================================================

static std::string generate(Model& M, const std::string& prompt, int n,
                            float temp, int topk, std::mt19937& rng) {
    std::vector<int> ctx = encode(prompt);
    if (ctx.empty()) ctx.push_back((int)' ');
    std::string out = prompt;
    for (int s = 0; s < n; ++s) {
        int T = std::min((int)ctx.size(), SEQ_LEN);
        const int* win = ctx.data() + (ctx.size() - T);
        graph_reset();
        Node* logits = M.forward(win, T);
        const int last = logits->rows - 1;
        std::array<float, VOCAB> l;
        for (int j = 0; j < VOCAB; ++j) l[j] = el(logits->data, VOCAB, last, j) / std::max(1e-6f, temp);
        if (topk > 0 && topk < VOCAB) {
            std::array<int, VOCAB> idx;
            for (int j = 0; j < VOCAB; ++j) idx[j] = j;
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                              [&](int a, int b) { return l[a] > l[b]; });
            std::array<float, VOCAB> keep;
            keep.fill(-1e30f);
            for (int t = 0; t < topk; ++t) keep[idx[t]] = l[idx[t]];
            l = keep;
        }
        float mx = -1e30f;
        for (float x : l) mx = std::max(mx, x);
        float Z = 0.f;
        for (float& x : l) { x = std::exp(x - mx); Z += x; }
        for (float& x : l) x /= Z;
        std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng), acc = 0.f;
        int pick = VOCAB - 1;
        for (int j = 0; j < VOCAB; ++j) { acc += l[j]; if (r <= acc) { pick = j; break; } }
        ctx.push_back(pick);
        out.push_back(decode(pick));
    }
    graph_reset();
    return out;
}

static std::string load_text(const std::string& path);  // fwd decl (corpus below)

static void train(const std::string& corpus_path, const std::string& model_path,
                  int steps, int batch, float lr, unsigned seed) {
    std::string text = load_text(corpus_path);
    std::vector<int> data = encode(text);
    if ((int)data.size() <= SEQ_LEN + 1) {
        std::fprintf(stderr, "error: corpus too small (%zu) for seq_len %d\n", data.size(), SEQ_LEN);
        std::exit(1);
    }

    Model M;
    M.build();
    std::printf("corpus: %zu bytes | model: d=%d L=%d H=%d ff=%d seq=%d%s | params: %.2fM | "
                "static mem: params %.1fMB acts %.1fMB\n",
                text.size(), D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN,
                USE_TERNARY ? " (ternary)" : "", PARAM_FLOATS / 1e6,
                4 * PARAM_FLOATS * sizeof(float) / 1e6, 2 * ACT_CAP * sizeof(float) / 1e6);
    std::fflush(stdout);

    AdamW opt(lr);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> startd(0, data.size() - SEQ_LEN - 2);

    double run_loss = 0.0; int run_n = 0;
    for (int step = 1; step <= steps; ++step) {
        opt.zero_grad();
        float step_loss = 0.f;
        for (int b = 0; b < batch; ++b) {
            size_t s = startd(rng);
            const int* x = data.data() + s;
            const int* y = data.data() + s + 1;
            graph_reset();
            Node* logits = M.forward(x, SEQ_LEN);
            Node* loss = op_cross_entropy(logits, y);
            step_loss += loss->data[0] / batch;
            backward(loss, 1.f / batch);  // accumulate averaged grads over the batch
        }
        opt.step();
        run_loss += step_loss; ++run_n;

        if (step % 100 == 0 || step == 1) {
            std::printf("step %5d/%d  loss %.4f\n", step, steps, run_loss / run_n);
            std::fflush(stdout);
            run_loss = 0.0; run_n = 0;
        }
        if (step % 500 == 0 || step == steps) {
            std::string sample = generate(M, "the ", 120, 0.8f, 20, rng);
            std::printf("  --- sample ---\n  %s\n", sample.c_str());
            std::fflush(stdout);
        }
    }
    save_model(model_path);
    std::printf("saved model to %s\n", model_path.c_str());
}

// ============================================================================
//  Section 11. Built-in corpus (used when the corpus path is ":builtin:")
// ============================================================================

static const char* BUILTIN_CORPUS = R"CORPUS(
the sun is warm and the sky is blue. the cat sat on the mat in the sun.
a small dog ran across the green field and barked at the birds.
the birds flew up into the tall trees and sang a happy song.
a girl walked to the river and looked at the clear cold water.
the water moved slowly over the smooth grey stones near the bank.
a boy found a round red ball and threw it high into the air.
the ball came down and rolled into the soft green grass.
the old man sat on a wooden bench and read a long story.
the story was about a brave knight and a great golden dragon.
the knight rode a fast horse over the high hills to the castle.
the castle had tall stone walls and a deep dark gate.
inside the castle a kind queen gave bread to the poor people.
the people thanked the queen and went home to their small houses.
at night the moon was bright and the stars filled the dark sky.
a cool wind moved through the trees and the leaves began to fall.
in the morning the children ran out to play in the fresh snow.
they made a big white snowman with two black eyes and a long nose.
the dog jumped in the snow and chased its own little tail.
the mother called the children in for a warm bowl of soup.
the soup was hot and good and the bread was soft and sweet.
after lunch the family sat by the fire and told old stories.
the father told a tale of a ship that sailed across the wide sea.
the ship met a great storm with high waves and loud thunder.
the brave sailors held the ropes and steered the ship to land.
they found a green island with tall trees and sweet ripe fruit.
the men ate the fruit and drank cool water from a clear spring.
they built a small camp near the shore and rested in the shade.
when the storm passed they sailed home with many fine gifts.
the people in the town came down to the docks to greet them.
there was music and dancing in the streets all through the night.
a baker made fresh bread and a farmer brought sweet red apples.
the children laughed and ran between the bright market stalls.
an old woman sold warm honey cakes from a small wooden cart.
a young man played a soft tune on an old brown guitar.
the king watched from the high tower and smiled at his people.
he was glad to see them safe and happy and full of joy.
the next day the rain came down and washed the dusty roads.
the flowers in the garden opened wide and drank the cool rain.
bees moved from flower to flower and gathered the golden pollen.
a green frog sat on a wet leaf beside the quiet still pond.
a silver fish jumped up and caught a fly above the water.
the trees grew tall and strong and gave good shade in summer.
in autumn the leaves turned red and gold and brown and fell.
the wind carried the dry leaves across the empty open fields.
winter brought cold white snow that covered all the land.
the river froze and the children went out to skate on the ice.
spring came again with warm rain and bright new green leaves.
the birds returned from the south and built their small nests.
the world was full of color and light and the sound of life.
and the sun rose each morning warm and bright over the hills.
)CORPUS";

static std::string load_text(const std::string& path) {
    if (path == ":builtin:") return std::string(BUILTIN_CORPUS);
    std::ifstream is(path, std::ios::binary);
    if (!is) { std::fprintf(stderr, "error: cannot open corpus '%s'\n", path.c_str()); std::exit(1); }
    return std::string((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
}

// ============================================================================
//  Section 12. CLI
// ============================================================================

static int arg_int(int c, char** v, const char* k, int d) {
    for (int i = 1; i < c - 1; ++i) if (!std::strcmp(v[i], k)) return std::atoi(v[i + 1]);
    return d;
}
static float arg_float(int c, char** v, const char* k, float d) {
    for (int i = 1; i < c - 1; ++i) if (!std::strcmp(v[i], k)) return (float)std::atof(v[i + 1]);
    return d;
}

static void usage() {
    std::printf(
        "sub0llm — tiny CPU transformer LM (compile-time sized; edit constexpr config at top)\n\n"
        "  sub0llm train <corpus|:builtin:> <model.bin> [--steps N --batch N --lr F --seed N]\n"
        "  sub0llm gen   <model.bin> \"<prompt>\" [--n N --temp F --topk N --seed N]\n"
        "  sub0llm dump-corpus <out.txt>\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string mode = argv[1];

    if (mode == "train" && argc >= 4) {
        int steps = arg_int(argc, argv, "--steps", 3000);
        int batch = arg_int(argc, argv, "--batch", 8);
        float lr = arg_float(argc, argv, "--lr", 0.001f);
        unsigned seed = (unsigned)arg_int(argc, argv, "--seed", 42);
        train(argv[2], argv[3], steps, batch, lr, seed);
        return 0;
    }
    if (mode == "gen" && argc >= 4) {
        Model M; M.build();  // establish parameter-node layout + random init
        if (!load_model(argv[2])) { std::fprintf(stderr, "error: cannot load '%s'\n", argv[2]); return 1; }
        int n = arg_int(argc, argv, "--n", 200);
        float temp = arg_float(argc, argv, "--temp", 0.8f);
        int topk = arg_int(argc, argv, "--topk", 20);
        unsigned seed = (unsigned)arg_int(argc, argv, "--seed", (int)std::random_device{}());
        std::mt19937 rng(seed);
        std::printf("%s\n", generate(M, argv[3], n, temp, topk, rng).c_str());
        return 0;
    }
    if (mode == "dump-corpus" && argc >= 3) {
        std::ofstream os(argv[2], std::ios::binary); os << BUILTIN_CORPUS;
        std::printf("wrote built-in corpus to %s\n", argv[2]); return 0;
    }
    usage();
    return 1;
}
