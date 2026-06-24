// engine.cpp — shared differentiable substrate (libsub0_core).
//
// Statically allocated: every model dimension is a compile-time constant (from
// the generated config header), so all parameter and activation memory lives in
// fixed-size std::array buffers allocated once in BSS and reused. There is no
// per-step heap allocation. Reverse-mode autograd is a fixed enum-tagged node
// pool walked in reverse; activations come from a bump-allocated arena reset each
// forward pass. See include/sub0/core.hpp for the exposed API.

#include "sub0/core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>

namespace sub0 {

// ============================================================================
//  Derived compile-time sizes
// ============================================================================

constexpr int NUM_PARAMS = 2 + 10 * N_LAYERS + 3;

consteval size_t calc_param_floats() {
    size_t n = 0;
    n += (size_t)VOCAB * D_MODEL;       // tok_emb
    n += (size_t)SEQ_LEN * D_MODEL;     // pos_emb
    for (int l = 0; l < N_LAYERS; ++l) {
        n += D_MODEL + D_MODEL;                 // ln1, ln2
        n += (size_t)4 * D_MODEL * D_MODEL;     // Wq,Wk,Wv,Wo
        n += (size_t)D_MODEL * D_FF + D_FF;      // W1, b1
        n += (size_t)D_FF * D_MODEL + D_MODEL;   // W2, b2
    }
    n += D_MODEL;                       // ln_f
    n += (size_t)D_MODEL * VOCAB + VOCAB;        // lm_head, lm_bias
    return n;
}
constexpr size_t PARAM_FLOATS = calc_param_floats();

consteval size_t calc_act_cap() {
    const size_t T = SEQ_LEN, C = D_MODEL, F = D_FF, H = N_HEADS, V = VOCAB;
    size_t base = 3 * T * C;
    size_t per  = 10 * T * C + 2 * T * F + H * T * T + 2 * T
                + (USE_TERNARY ? (size_t)4 * C * C + 2 * C * F : 0);
    size_t fin  = T * C + 2 * T * V + 64;
    return base + (size_t)N_LAYERS * per + fin;
}
constexpr size_t ACT_CAP   = calc_act_cap() * 3 / 2 + 8192;
constexpr size_t MAX_NODES = 16 + 16 * (size_t)N_LAYERS;

// ============================================================================
//  Static storage
// ============================================================================

static std::array<float, PARAM_FLOATS> g_param_data{};
static std::array<float, PARAM_FLOATS> g_param_grad{};
static std::array<float, PARAM_FLOATS> g_param_m{};
static std::array<float, PARAM_FLOATS> g_param_vel{};

static std::array<float, ACT_CAP> g_act_data{};
static std::array<float, ACT_CAP> g_act_grad{};
static size_t g_act_used = 0;

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

static std::array<Node, NUM_PARAMS> g_param_nodes;
static std::array<Node, MAX_NODES>  g_pool;
static size_t g_pool_used = 0;

struct ParamView { size_t off, n; bool decay; };
static std::array<ParamView, NUM_PARAMS> g_param_views;
static size_t g_param_used = 0;
static int g_pcount = 0;

// When a model was loaded with weights already in their final (ternary) form,
// the linear op must not re-quantize them (absmean re-quantization is not
// idempotent). Training keeps this false so the straight-through estimator runs.
static bool g_packed_inference = false;

static Node* mk_param(int r, int c, bool decay) {
    size_t n = (size_t)r * c, off = g_param_used;
    g_param_used += n;
    Node& nd = g_param_nodes[g_pcount];
    nd = Node{};
    nd.op = Op::Leaf; nd.rows = r; nd.cols = c;
    nd.data = std::span<float>(g_param_data.data() + off, n);
    nd.grad = std::span<float>(g_param_grad.data() + off, n);
    g_param_views[g_pcount] = {off, n, decay};
    ++g_pcount;
    return &nd;
}

static Node* mk_node(Op op, int r, int c) {
    if (g_pool_used >= MAX_NODES) { std::fprintf(stderr, "fatal: node pool overflow\n"); std::abort(); }
    Node& nd = g_pool[g_pool_used++];
    nd = Node{};
    nd.op = op; nd.rows = r; nd.cols = c;
    auto [d, gr] = arena_alloc((size_t)r * c);
    nd.data = d; nd.grad = gr;
    return &nd;
}

static inline float& el(std::span<float> s, int cols, int i, int j) { return s[(size_t)i * cols + j]; }

// ============================================================================
//  Differentiable ops (forward builders)
// ============================================================================

static Node* op_embed(Node* table, const int* ids, int T) {
    const int C = table->cols;
    Node* out = mk_node(Op::Embed, T, C);
    out->w = table; out->ids = ids;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < C; ++j) el(out->data, C, t, j) = el(table->data, C, ids[t], j);
    return out;
}

static Node* op_add(Node* a, Node* b) {
    Node* out = mk_node(Op::Add, a->rows, a->cols);
    out->a = a; out->b = b;
    for (size_t i = 0; i < out->data.size(); ++i) out->data[i] = a->data[i] + b->data[i];
    return out;
}

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

static Node* op_linear(Node* x, Node* W, Node* bias, bool ternary) {
    const int T = x->rows, in = x->cols, out = W->cols;
    Node* y = mk_node(Op::Linear, T, out);
    y->a = x; y->w = W; y->bias = bias; y->ternary = ternary;
    const float* Wf = W->data.data();
    if (ternary && !g_packed_inference) {
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

static Node* op_rmsnorm(Node* x, Node* gamma) {
    const int T = x->rows, C = x->cols;
    const float eps = 1e-5f;
    Node* y = mk_node(Op::RMSNorm, T, C);
    y->a = x; y->w = gamma;
    auto [rinv, rinv_g] = arena_alloc(T);
    y->scratch = rinv;
    for (int t = 0; t < T; ++t) {
        float ms = 0.f;
        for (int j = 0; j < C; ++j) ms += el(x->data, C, t, j) * el(x->data, C, t, j);
        ms /= C;
        float r = 1.f / std::sqrt(ms + eps);
        rinv[t] = r;
        for (int j = 0; j < C; ++j) el(y->data, C, t, j) = el(x->data, C, t, j) * r * gamma->data[j];
    }
    return y;
}

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
                for (int a = 0; a < d; ++a) el(out->data, C, i, off + a) += p * el(v->data, C, j, off + a);
            }
        }
    }
    return out;
}

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
//  Backward dispatch
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
        const float* Wf = (n.ternary && !n.scratch.empty()) ? n.scratch.data() : W->data.data();
        const float* dY = n.grad.data();
        const float* X = x->data.data();
        #pragma omp parallel for
        for (int t = 0; t < T; ++t)
            for (int p = 0; p < in; ++p) {
                float s = 0.f;
                const float* dYr = dY + (size_t)t * out;
                const float* Wr = Wf + (size_t)p * out;
                for (int o = 0; o < out; ++o) s += dYr[o] * Wr[o];
                x->grad[(size_t)t * in + p] += s;
            }
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
        float g = n.grad[0] / T;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < V; ++j) {
                float p = probs[(size_t)t * V + j];
                el(logits->grad, V, t, j) += g * (p - (j == n.ids[t] ? 1.f : 0.f));
            }
        break;
    }
    }
}

// ============================================================================
//  Model (internal): parameter layout + forward pass
// ============================================================================

namespace {
struct Layer { Node *ln1, *ln2, *Wq, *Wk, *Wv, *Wo, *W1, *b1, *W2, *b2; };

struct Model {
    Node* tok_emb;
    Node* pos_emb;
    std::array<Layer, N_LAYERS> layers;
    Node* ln_f;
    Node* lm_head;
    Node* lm_bias;

    void build() {
        g_param_used = 0; g_pcount = 0;
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

static Model g_model;
}  // anonymous namespace

// ============================================================================
//  Serialization (header validates the generated config)
// ============================================================================

namespace {
struct Header {
    char magic[4] = {'S', '0', 'L', '4'};
    int d_model = D_MODEL, n_layers = N_LAYERS, n_heads = N_HEADS;
    int d_ff = D_FF, seq_len = SEQ_LEN, vocab = VOCAB, ternary = USE_TERNARY;
    uint64_t param_floats = PARAM_FLOATS;
};
}

// ============================================================================
//  Exposed API
// ============================================================================

void build_model() { g_model.build(); }

void save_model(const char* path) {
    std::ofstream os(path, std::ios::binary);
    Header h;
    os.write((const char*)&h, sizeof(h));
    os.write((const char*)g_param_data.data(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
}

bool load_model(const char* path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    Header h, ref;
    is.read((char*)&h, sizeof(h));
    if (std::memcmp(h.magic, ref.magic, 4) != 0 ||
        h.d_model != ref.d_model || h.n_layers != ref.n_layers || h.n_heads != ref.n_heads ||
        h.d_ff != ref.d_ff || h.seq_len != ref.seq_len || h.vocab != ref.vocab ||
        h.ternary != ref.ternary || h.param_floats != ref.param_floats) {
        std::fprintf(stderr, "error: model was built with a different (constexpr) config\n");
        return false;
    }
    is.read((char*)g_param_data.data(), (std::streamsize)(PARAM_FLOATS * sizeof(float)));
    return (bool)is;
}

void print_config() {
    std::printf("model: d=%d L=%d H=%d ff=%d seq=%d vocab=%d%s | params: %.2fM | "
                "static mem: params %.1fMB acts %.1fMB\n",
                D_MODEL, N_LAYERS, N_HEADS, D_FF, SEQ_LEN, VOCAB,
                USE_TERNARY ? " (ternary)" : "", PARAM_FLOATS / 1e6,
                4 * PARAM_FLOATS * sizeof(float) / 1e6, 2 * ACT_CAP * sizeof(float) / 1e6);
}

const char* default_corpus() { return DEFAULT_CORPUS; }

std::vector<int> encode(const std::string& text) {
    std::vector<int> out;
    out.reserve(text.size());
    for (unsigned char c : text) { int id = VOCAB_ID[c]; if (id >= 0) out.push_back(id); }
    return out;
}
char decode(int id) { return (id >= 0 && id < VOCAB) ? VOCAB_CHARS[id] : '?'; }

void graph_reset() { g_pool_used = 0; g_act_used = 0; }

Node* forward(const int* ids, int T) { return g_model.forward(ids, T); }
Node* cross_entropy(Node* logits, const int* targets) { return op_cross_entropy(logits, targets); }

void backward(Node* loss, float seed) {
    loss->grad[0] = seed;
    for (size_t i = g_pool_used; i-- > 0; ) backward_node(g_pool[i]);
}

// --- AdamW ------------------------------------------------------------------

AdamW::AdamW(float lr) : lr_(lr) {}

void AdamW::zero_grad() { g_param_grad.fill(0.f); }

void AdamW::step() {
    double sq = 0.0;
    for (float g : g_param_grad) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    float gs = (norm > clip_) ? clip_ / (norm + 1e-6f) : 1.f;

    ++t_;
    float bc1 = 1.f - std::pow(b1_, (float)t_);
    float bc2 = 1.f - std::pow(b2_, (float)t_);
    for (int pi = 0; pi < g_pcount; ++pi) {
        const ParamView& pv = g_param_views[pi];
        for (size_t i = pv.off; i < pv.off + pv.n; ++i) {
            float g = g_param_grad[i] * gs;
            g_param_m[i]   = b1_ * g_param_m[i]   + (1 - b1_) * g;
            g_param_vel[i] = b2_ * g_param_vel[i] + (1 - b2_) * g * g;
            float mhat = g_param_m[i] / bc1;
            float vhat = g_param_vel[i] / bc2;
            g_param_data[i] -= lr_ * mhat / (std::sqrt(vhat) + eps_);
            if (pv.decay) g_param_data[i] -= lr_ * wd_ * g_param_data[i];
        }
    }
}

}  // namespace sub0
