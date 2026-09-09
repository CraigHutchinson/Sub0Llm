// decode.cpp -- the CPU backend's incremental single-token (KV-cache) inference path.
//
// Split out of backend.cpp unchanged, so the decode path is a small translation unit of its own
// rather than a slice of a file that also carries the whole training/backward/optimizer
// machinery. What lives here is exactly what one generated token executes: the per-position row
// kernels, the three decode-persistent caches (KV, Gated DeltaNet recurrent state, QSA indexer
// keys), Model::forward_one, and the KV-cache entry points the engine API forwards to.
//
// The decode caches are deliberately a SEPARATE reset lifetime from anything training owns:
// kv_reset() below is the only thing that clears them, and Model::forward() (backend.cpp) never
// touches them at all. Everything shared with the batched path -- the arenas, the Worker pool,
// `Model`'s own node layout, the binding tables, the expert resolver -- comes from internal.hpp,
// so there is exactly one copy of each.
#include "internal.hpp"

#include "sub0/gated_residual_math.hpp"  // Gated Residual row math (gr::hc_norm/mix/gate/combine/tile)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

// OpenMP gate, the same one backend.cpp carries and for the same reason -- see its copy for the full
// argument. This code USED to live in that file and was covered by that guard, so repeating it here is
// keeping the guard, not adding one. It matters more here than there, in fact: tied_head_row's
// `#pragma omp simd reduction(+ : s)` is what PERMITS the compiler to reassociate that float sum, so a
// build that silently lost OpenMP for this translation unit would not merely decode more slowly, it
// would decode to different last bits than the batched forward() that still had it -- breaking the
// forward/forward_one parity check with no compile-time signal at all.
#if !defined(_OPENMP) && defined(SUB0_REQUIRE_OPENMP)
#error "OpenMP required but _OPENMP is undefined: this translation unit was compiled without OpenMP, so decode's `#pragma omp simd` reductions would change accumulation order relative to the batched forward path. Reconfigure with OpenMP available, or pass -DSUB0_REQUIRE_OPENMP=OFF to build single-threaded on purpose."
#endif

namespace sub0 {
namespace {
// --- Incremental single-token inference (KV-cache) --------------------------
// forward() recomputes the WHOLE context every token, so autoregressive gen is O(n*T^2). forward_one
// keeps a per-layer K/V cache and, given the new token + its position, runs the network over just
// that ONE row -- appending its K/V and attending the single new query against the cache. Per token
// this is O(T) attention + O(1) rows through the projections/FFN/head, instead of O(T) rows through
// everything. The row helpers below replicate the dense op_* math EXACTLY (same accumulation order),
// so forward_one's logits match forward()'s last row to fast-math tolerance (gated by a unit test).
// Dense weights only (the sparse-ternary linear path is not mirrored here); a ternary build keeps
// the full-forward gen path. Positions must stay < SEQ_LEN (gen falls back past the window).
struct KVCache {
    // Indexed by EXECUTION, not layer: under LoopSplit a repeated middle layer runs several times per
    // token and each execution attends over its own K/V history, so it needs its own slot. Equal to
    // N_LAYERS when looping is off.
    std::vector<float> k, v;                                    // [LOOP_EXEC_COUNT][SEQ_LEN][D_KV], flat
    void reset() {
        const size_t n = static_cast<size_t>(LOOP_EXEC_COUNT) * SEQ_LEN * D_KV;
        if (k.size() != n) { k.assign(n, 0.f); v.assign(n, 0.f); }
    }
    // Rows are D_KV wide, not D_MODEL: under GQA the cache holds N_KV_HEADS heads, which is the whole
    // point (the cache shrinks by N_HEADS/N_KV_HEADS). Identical to D_MODEL when not using GQA.
    float* krow(int l, int pos) { return k.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_KV; }
    float* vrow(int l, int pos) { return v.data() + ((static_cast<size_t>(l) * SEQ_LEN) + pos) * D_KV; }
};
thread_local KVCache g_kv;                                      // gen is single-threaded; lazily sized

// Gated DeltaNet's decode-persistent state (Stage 1; docs/GATED_DELTANET.md S2's "decode" bullet) --
// structurally NOT a bigger/smaller KVCache: it does not grow with position, it is READ AND WRITTEN IN
// PLACE by every forward_one call (an accumulator, not a per-position row store), and per-execution
// (LOOP_EXEC_COUNT slots) for the identical LoopSplit reason KVCache is. Because it is an accumulator,
// reset() must UNCONDITIONALLY re-zero it every time (unlike KVCache's reset(), which only assigns on a
// SIZE change -- safe there because every row it will ever READ was already WRITTEN earlier in the SAME
// generation; a GDN state left over from a PREVIOUS generation would otherwise silently leak into a new
// one, since nothing about starting a new generation naturally overwrites an accumulator's stale value).
struct GdnCache {
    std::vector<float> state, conv_hist;   // [LOOP_EXEC_COUNT][state/conv_hist floats-per-exec], flat
    void reset() {
        state.assign(static_cast<size_t>(LOOP_EXEC_COUNT) * gdn::state_floats(GDN_DIMS), 0.f);
        conv_hist.assign(static_cast<size_t>(LOOP_EXEC_COUNT) * gdn::conv_hist_floats(GDN_DIMS), 0.f);
    }
    float* state_of(int e) { return state.data() + static_cast<size_t>(e) * gdn::state_floats(GDN_DIMS); }
    float* conv_of(int e)  { return conv_hist.data() + static_cast<size_t>(e) * gdn::conv_hist_floats(GDN_DIMS); }
};
thread_local GdnCache g_gdn_cache;   // only ever populated/consulted when USE_GATED_DELTANET

// QSA's decode-persistent indexer key cache (Stage 1; docs/QSA.md S6's own note that this is QSA's ONLY
// cross-call state). Structurally a KVCache, not a GdnCache: it IS a per-position row store that grows
// with position (the indexer must be able to pool a block out of ANY earlier token's raw key), and it is
// per-EXECUTION for the identical LoopSplit reason KVCache is. Rows are the RAW, unnormed, unrotated
// indexer keys -- k_layernorm and RoPE happen later, on the POOLED block key, at the block's own start
// position (docs/QSA.md S1a). reset() follows KVCache's own assign-on-size-change rule, safe for the same
// reason: every row ever READ was WRITTEN earlier in the SAME generation.
//
// It ALSO carries the POOLED block-key cache (docs/QSA.md S11). A block's pooled+k_layernorm'd+RoPE'd
// key is a function of the block alone -- its fixed start position and the compress_ratio raw keys in
// it -- never of the querying position, so recomputing it on every decode step was O(T^2/ratio) work
// across a generation for an O(T/ratio) quantity. `n_cached[e]` is how many leading blocks of slot e
// are valid; qsa::indexer_select_row extends it by the (at most one) block that completed this step.
// UNLIKE raw_k, `n_cached` MUST be zeroed on every reset, not only when the size changes: a stale
// non-zero count would make a fresh generation trust the PREVIOUS generation's block keys. (raw_k keeps
// the assign-on-size-change rule, which is still safe for the same reason as before -- every row read
// was written earlier in the same generation -- and block_k needs no clearing at all, since only its
// first n_cached[e] entries are ever read.)
struct QsaCache {
    std::vector<float> raw_k;   // [LOOP_EXEC_COUNT][SEQ_LEN][QSA_IDX_HEAD_DIM_BUF], flat
    std::vector<float> block_k; // [LOOP_EXEC_COUNT][block_key_cache_floats(QSA_DIMS_BUF, SEQ_LEN)], flat
    std::vector<int>   n_cached;// [LOOP_EXEC_COUNT] -- valid leading block count per execution slot
    static constexpr size_t BLOCK_K_PER_EXEC = qsa::block_key_cache_floats(QSA_DIMS_BUF, SEQ_LEN);
    void reset() {
        const size_t n = static_cast<size_t>(LOOP_EXEC_COUNT) * SEQ_LEN * QSA_IDX_HEAD_DIM_BUF;
        if (raw_k.size() != n) raw_k.assign(n, 0.f);
        const size_t bn = static_cast<size_t>(LOOP_EXEC_COUNT) * BLOCK_K_PER_EXEC;
        if (block_k.size() != bn) block_k.assign(bn, 0.f);
        n_cached.assign(static_cast<size_t>(LOOP_EXEC_COUNT), 0);   // unconditional -- see above
    }
    float* base(int e) {
        return raw_k.data() + static_cast<size_t>(e) * SEQ_LEN * QSA_IDX_HEAD_DIM_BUF;
    }
    float* block_base(int e) { return block_k.data() + static_cast<size_t>(e) * BLOCK_K_PER_EXEC; }
    int*   n_cached_of(int e) { return n_cached.data() + e; }
};
thread_local QsaCache g_qsa_cache;   // only ever populated/consulted when USE_QSA

// y[out] = x[in] . W[in,out]  (+ bias); dense, same order as op_linear's non-ternary path.
static inline void linear_row(const float* __restrict x, const Node* W, const Node* bias,
                              float* __restrict y, int in, int out) {
    for (int o = 0; o < out; ++o) y[o] = 0.f;
    const float* __restrict Wf = W->data.data();
    for (int p = 0; p < in; ++p) {
        const float xp = x[p];
        if (xp == 0.f) continue;
        const float* __restrict Wr = Wf + static_cast<size_t>(p) * out;
        for (int o = 0; o < out; ++o) y[o] += xp * Wr[o];
    }
    if (bias) for (int o = 0; o < out; ++o) y[o] += bias->data[o];
}
// Tied-embedding head, single-row (generation) form: y[v] = dot(x[:], table[v,:]), no bias -- see
// op_tied_head's comment for why this is a dot-product pattern rather than linear_row's axpy one.
static inline void tied_head_row(const float* __restrict x, const Node* table,
                                 float* __restrict y, int C, int V) {
    const float* __restrict Tb = table->data.data();
    for (int v = 0; v < V; ++v) {
        const float* __restrict tv = Tb + static_cast<size_t>(v) * C;
        float s = 0.f;
        #pragma omp simd reduction(+ : s)
        for (int c = 0; c < C; ++c) s += x[c] * tv[c];
        y[v] = s;
    }
}
static inline void rmsnorm_row(const float* __restrict x, const Node* gamma, float* __restrict y, int C) {
    float ms = 0.f; for (int j = 0; j < C; ++j) ms += x[j] * x[j]; ms /= C;
    const float r = 1.f / std::sqrt(ms + 1e-5f);
    const float* __restrict G = gamma->data.data();
    for (int j = 0; j < C; ++j) y[j] = x[j] * r * G[j];
}
static inline void qknorm_row(float* __restrict x, const Node* gamma, int H, int C) {   // in place, mirrors rope_row
    const int d = C / H;
    const float* __restrict G = gamma->data.data();
    for (int h = 0; h < H; ++h) {
        const int off = h * d;
        float ms = 0.f; for (int j = 0; j < d; ++j) ms += x[off + j] * x[off + j]; ms /= d;
        const float r = 1.f / std::sqrt(ms + 1e-5f);
        for (int j = 0; j < d; ++j) x[off + j] *= r * G[j];
    }
}
static inline void rope_row(float* __restrict x, int pos, int H, int C) {   // rotate Q/K in place (t = pos)
    const int d = C / H;
    constexpr int rd = ROTARY_DIM, half = ROTARY_HALF;   // partial rotary -- mirrors op_rope; the
                                                          // un-rotated remainder needs no work in place
    for (int h = 0; h < H; ++h) {
        const int off = h * d;
        for (int m = 0; m < half; ++m) {
            const float ang = (static_cast<float>(pos) * ROPE_POS_SCALE) * std::pow(ROPE_THETA, -2.f * m / rd);
            const float cs = std::cos(ang), sn = std::sin(ang);
            const float x0 = x[off + 2 * m], x1 = x[off + 2 * m + 1];
            x[off + 2 * m]     = x0 * cs - x1 * sn;
            x[off + 2 * m + 1] = x0 * sn + x1 * cs;
        }
    }
}
// Depth attention for ONE position: forward_one's counterpart to op_depth_attn, which it must agree
// with numerically (the forward_one-vs-forward parity test is the gate). `dk`/`dv` are this token's
// depth cache, `n` its live entry count, and `v` is REWRITTEN IN PLACE with the mixture -- via a temp,
// since v is also entry `n` of the mix. Cheap because the softmax is over depth alone: a single
// position needs no sequence history, so decode needs no extra cache beyond the current token's rows.
static inline void depth_mix_row(const float* __restrict q, const float* __restrict k,
                                 float* __restrict v, const float* dk, const float* dv, int n) {
    constexpr int d = D_HEAD;
    const float scale = 1.f / std::sqrt((float)d);
    float mixed[D_KV];
    for (int hd = 0; hd < N_KV_HEADS; ++hd) {
        const int off = hd * d;
        float qb[D_HEAD] = {};
        for (int g = 0; g < GQA_GROUP; ++g) {
            const float* __restrict qh = q + (size_t)(hd * GQA_GROUP + g) * d;
            for (int a = 0; a < d; ++a) qb[a] += qh[a];
        }
        for (int a = 0; a < d; ++a) qb[a] *= 1.f / GQA_GROUP;
        float lg[DEPTH_CACHE_MAX + 1];
        float mx = -1e30f;
        for (int dd = 0; dd <= n; ++dd) {
            const float* __restrict kd = (dd < n) ? dk + (size_t)dd * D_KV + off : k + off;
            float s = 0.f;
            for (int a = 0; a < d; ++a) s += qb[a] * kd[a];
            s *= scale; lg[dd] = s; mx = std::max(mx, s);
        }
        float Z = 0.f;
        for (int dd = 0; dd <= n; ++dd) { lg[dd] = FAST_MATH ? fast_exp(lg[dd] - mx) : std::exp(lg[dd] - mx); Z += lg[dd]; }
        for (int a = 0; a < d; ++a) mixed[off + a] = 0.f;
        for (int dd = 0; dd <= n; ++dd) {
            const float p = lg[dd] / Z;
            const float* __restrict vd = (dd < n) ? dv + (size_t)dd * D_KV + off : v + off;
            for (int a = 0; a < d; ++a) mixed[off + a] += p * vd[a];
        }
    }
    for (int j = 0; j < D_KV; ++j) v[j] = mixed[j];
}
static inline float gelu_row(float v) {
    if constexpr (FAST_MATH) return gelu_fast(v);
    else return 0.5f * v * (1.f + std::erf(v * 0.70710678f));
}
static inline float silu_row(float v) {
    if constexpr (FAST_MATH) return silu_fast(v);
    else return v / (1.f + std::exp(-v));
}
}  // anonymous namespace

// Declared in internal.hpp's Model, where this function's contract comment lives.
const float* Model::forward_one(int id, int pos) {
    static thread_local std::array<float, VOCAB> logits;
    // Sentinel-pair detection needs the PREVIOUS fed token: decode feeds positions strictly
    // sequentially (prefill 0..n-1, then one per generated token), so "previous" is simply the last
    // (id,pos) this thread fed -- valid only when pos follows it directly. No explicit reset needed:
    // a NEW generation starts at pos 0, and 0 == prev_pos+1 is unsatisfiable for any real prev_pos
    // (>= 0), so a previous generation's tail can never leak in as this one's sigil.
    static thread_local int prev_id = -1, prev_pos = -2;
    const int prev = (pos == prev_pos + 1) ? prev_id : -1;
    prev_id = id; prev_pos = pos;
    // `d` is D_HEAD, NOT C / H: WP4b blocker A made the head width its own axis, so the two differ
    // the moment --head-dim is set (and the decode path's per-head strides are all in D_HEAD).
    // Spelling it C / H here was a real bug -- the forward-vs-forward_one parity check caught it at
    // 1.71 relative error on the first --head-dim build, exactly the AGENTS.md S10 class of defect
    // (a derived width re-spelled locally instead of read from the one source of truth).
    constexpr int C = D_MODEL, H = N_HEADS, d = D_HEAD;
    const float scale = 1.f / std::sqrt(static_cast<float>(d));
    // Gated Residual (Stage 1): `h` is HC_WIDE wide when GR is on (== D_MODEL, i.e. today's exact
    // size, when off -- layout.hpp's own collapse). Every OTHER per-layer temporary below (a, qn,
    // kn, vn, att, proj, f1, g1) stays exactly D_MODEL/D_FF-wide -- GR only ever widens the
    // persistent residual itself, never the sub-block's own internal working set (S2).
    // qn/att are D_Q = N_HEADS*D_HEAD wide and kn/vn are D_KV wide -- NOT D_MODEL, since WP4b
    // blocker A made D_HEAD its own axis. All four are sized to the widest of the three so one
    // constant covers them (D_KV <= D_Q by construction, see layout.hpp's static_assert); at the
    // derived head width this is exactly C, so these arrays are unchanged for every existing build.
    constexpr int ATT_W = sub0::D_Q > C ? sub0::D_Q : C;
    float h[HC_WIDE], a[C], qn[ATT_W], kn[ATT_W], vn[ATT_W], att[ATT_W], proj[C], f1[D_FF];
    [[maybe_unused]] float g1[D_FF];   // gate branch, gated FFN only
    [[maybe_unused]] float packed_copy[C];             // periodic-reinject spike, see below
    // Gated Residual scratch/output buffers -- sized 1 (never zero-length) when off, same idiom as
    // every other never-degenerate buffer in this function. gr_normed is shared by gr_read_row's
    // mix AND gate steps (both read-only consumers of the SAME hc_norm(wide) result) -- a cheaper,
    // equally correct alternative to op_gr_mix/op_gr_gate's own independent-recompute form
    // (docs/GATED_RESIDUAL.md S4c), available here because this is plain imperative code, not two
    // separate Node-graph ops that would each need their own self-contained scratch.
    [[maybe_unused]] float gr_normed[HC_WIDE];
    [[maybe_unused]] float gr_mixscr[GR_MIX_SCRATCH1 ? GR_MIX_SCRATCH1 : 1];
    [[maybe_unused]] float gr_mixed[C];
    [[maybe_unused]] float gr_inj[HC_COUNT_BUF];
    // Mixture of Experts (Stage 1, docs/MOE.md S4b) decode-path scratch -- MOE_SCRATCH1 is never
    // zero (see its own comment), unlike GR_MIX_SCRATCH1 above, so no ternary-guard is needed here.
    // (The NUM_EXPERTS-wide pointer arrays that used to sit here are gone: WP4e's moe_resolve()
    // hands moe::forward_row_via one expert at a time, so nothing needs a table of all 512.)
    [[maybe_unused]] float moe_scratch[MOE_SCRATCH1];
    // QSA (Stage 1, docs/QSA.md S4b) decode-path scratch. Every bound uses QSA_DIMS_BUF, whose widths
    // are never zero even when QSA is off, so these stay valid array bounds in a QSA-off build
    // (where nothing ever reads them) -- the same never-degenerate idiom GR_MIX_SCRATCH1 needed.
    // Sized for the WHOLE window (SEQ_LEN), not the current position: forward_one's kv window grows
    // with `pos`, and AGENTS.md S1 forbids sizing per call.
    [[maybe_unused]] float qsa_idx_q[QSA_DIMS_BUF.idx_q_width()];
    [[maybe_unused]] float qsa_gate_row[QSA_DIMS_BUF.q_width()];
    [[maybe_unused]] float qsa_mask[SEQ_LEN];
    [[maybe_unused]] float qsa_sel_scr[qsa::select_scratch_floats(QSA_DIMS_BUF, SEQ_LEN)];
    [[maybe_unused]] float qsa_att_scr[qsa::attn_scratch_floats(QSA_DIMS_BUF, SEQ_LEN)];
    // GR READ (mix+gate, into `out_a`) / WRITE (combine, in place on `wide`) row-helpers, T==1 --
    // the exact decode-path counterpart of forward()'s gr_read/gr_write lambdas (docs/GATED_RESIDUAL.md
    // S2). Safe to write `wide` in place in gr_write_row: combine()'s per-(stream,channel) output
    // depends only on that SAME index's own input (plus mixer_out, a disjoint buffer), never on any
    // other index, so there is no read-after-write hazard.
    auto gr_read_row = [&](const float* wide, Node* norm_w, Node* down_w, Node* up_w, Node* inject_w,
                            Node* ln, float* out_a) {
        if constexpr (USE_GATED_RESIDUAL) {
            gr::hc_norm(GR_DIMS, 1, wide, norm_w->data.data(), gr_normed);
            gr::mix(GR_DIMS, 1, gr_normed, down_w->data.data(), up_w->data.data(), gr_mixed, gr_mixscr);
            gr::gate(GR_DIMS, 1, gr_normed, inject_w->data.data(), gr_inj);
            // WP4b blocker D: the mixer reads mixed_input DIRECTLY -- no Ln1/Ln2 exists under GR
            // (`ln` is nullptr), and gr::hc_norm above already applied the real model's own
            // pre-block norm at its own 1e-6 eps. Mirrors forward()'s gr_read exactly; the
            // forward-vs-forward_one parity test is what gates that they stay mirrored.
            (void)ln;
            std::copy_n(gr_mixed, C, out_a);
        } else {
            rmsnorm_row(wide, ln, out_a, C);
        }
    };
    auto gr_write_row = [&](float* wide, const float* mixer_out) {
        if constexpr (USE_GATED_RESIDUAL) gr::combine(GR_DIMS, 1, wide, mixer_out, gr_inj, wide);
        else                              for (int j = 0; j < C; ++j) wide[j] += mixer_out[j];
    };
    // This token's depth cache: rows only, since the depth softmax never crosses positions (see
    // depth_mix_row). Scoped to one forward_one call -- unlike g_kv, nothing here spans tokens.
    [[maybe_unused]] float dep_k[(DEPTH_CACHE_MAX ? DEPTH_CACHE_MAX : 1) * D_KV];
    [[maybe_unused]] float dep_v[(DEPTH_CACHE_MAX ? DEPTH_CACHE_MAX : 1) * D_KV];
    [[maybe_unused]] int   dep_n = 0;
    bool do_reinject = false;

    if (g_sentinel_binds && prev == g_sentinel_binds->sigil && g_sentinel_binds->bound(id)) {
        encode_slot(tok_emb->data.data(), C, g_sentinel_binds->fragments(id), g_sentinel_binds->encoding,
                    h, g_sentinel_binds->enc_w);
    } else if (g_scratch_binds && is_scratch_slot(id) && g_scratch_binds->bound(id)) {
        encode_slot(tok_emb->data.data(), C, g_scratch_binds->fragments(id), g_scratch_binds->encoding, h,
                    g_scratch_binds->enc_w);
        if (g_scratch_reinject_stride > 0) {   // save the layer-0 packed vector for periodic re-injection
            for (int j = 0; j < C; ++j) packed_copy[j] = h[j];
            do_reinject = true;
        }
    } else if (is_persistent_slot(id, VOCAB)) {
        // Same unconditional guard as op_embed's forward branch (backend.cpp) -- see its
        // comment. Decode never runs backward, so this path only needs the forward compose (enc_w,
        // never enc_w_grad).
        const SlotEncoding enc = g_persistent_binds ? g_persistent_binds->encoding : SlotEncoding::MeanPool;
        encode_slot(tok_emb->data.data(), C, persistent_fragments(g_persistent_binds, id), enc, h,
                    g_persistent_binds ? g_persistent_binds->enc_w : nullptr);
    } else {
        const float* emb = tok_emb->data.data() + static_cast<size_t>(id) * C;
        for (int j = 0; j < C; ++j) h[j] = emb[j];
    }
    if constexpr (POS_ENCODING == PosEncoding::Absolute) {
        const float* pe = pos_emb->data.data() + static_cast<size_t>(pos) * C;
        for (int j = 0; j < C; ++j) h[j] += pe[j];
    }
    // N-gram embeddings, decode path: mirrors forward()'s block exactly (same hashing, same
    // block-matmul-via-row-slice composition), but the context comes from a small rolling history
    // of the last NGRAM_MAX_N-1 FED ids instead of indexing back into a batched window -- this
    // engine's analogue of the reference's `NgramCache.update_ngram_context`. `prev < 0` (computed
    // above, BEFORE it was overwritten) is the same "fresh generation or non-sequential jump"
    // signal forward_one's sentinel-pair detection already relies on, so history resets exactly
    // when that context would otherwise be wrong.
    if constexpr (NGRAM_EMBED) {
        static thread_local int hist[NGRAM_MAX_SHIFT_BUF];
        static thread_local int hist_len = 0;
        if (prev < 0) hist_len = 0;
        // Same guard as forward()'s ngram_tok: a persistent-slot id (>= VOCAB) is not a real
        // recurring vocabulary token and its raw integer value is unbounded, so it hashes as "no
        // signal" (id 0) -- see forward()'s comment for the full reasoning and the differential
        // test (persistent_slots_engine_tests.cpp) that caught this without the guard.
        const int id_tok = (id < VOCAB) ? id : 0;
        for (int e = 0; e < NGRAM_NUM_EMBEDDERS; ++e) {
            const int order     = NGRAM_ORDERS[static_cast<std::size_t>(e)];
            const int vocab_dim = NGRAM_VOCAB_DIMS[static_cast<std::size_t>(e)];
            std::int64_t acc = id_tok;
            for (int k = 2; k <= order; ++k) {
                const int shift = k - 1;
                const int ptok  = (shift - 1 < hist_len) ? hist[shift - 1] : 0;
                acc += static_cast<std::int64_t>(ptok) * NGRAM_VOCAB_MODS[static_cast<std::size_t>(e)][static_cast<std::size_t>(k - 2)];
            }
            const int tid = static_cast<int>(((acc % vocab_dim) + vocab_dim) % vocab_dim);
            const float* row = ngram_tab[static_cast<std::size_t>(e)]->data.data()
                             + static_cast<std::size_t>(tid) * NGRAM_EMB_DIM;
            const float* Wb = ngram_wblock[static_cast<std::size_t>(e)].data.data();
            for (int p = 0; p < NGRAM_EMB_DIM; ++p) {
                const float xp = row[p];
                const float* __restrict Wr = Wb + static_cast<std::size_t>(p) * C;
                for (int o = 0; o < C; ++o) h[o] += xp * Wr[o];
            }
        }
        for (int s = NGRAM_MAX_SHIFT_BUF - 1; s > 0; --s) hist[s] = hist[s - 1];
        hist[0] = id_tok;   // history stores the GUARDED value, so it stays consistent later too
        if (hist_len < NGRAM_MAX_SHIFT_BUF) ++hist_len;
    }
    // Gated Residual's model-level ENTRY tile (docs/GATED_RESIDUAL.md S1c): everything above (embed,
    // absolute pos, n-gram injection) wrote into h's first D_MODEL elements exactly as before --
    // this duplicates that across the other HC_COUNT-1 streams. Safe in place (out_wide aliases the
    // same buffer tile() reads from): stream 0's write is `h[j] = h[j]` (a no-op), and every later
    // stream's write target lies entirely outside the [0,D_MODEL) range tile() ever reads from.
    if constexpr (USE_GATED_RESIDUAL) gr::tile(GR_DIMS, 1, h, h);
    // `e` is the EXECUTION index (the KV-cache slot); `li` is which layer's weights run there.
    // They differ only under LoopSplit -- see LAYER_EXEC_ORDER and KVCache's own comments.
    for (int e = 0; e < LOOP_EXEC_COUNT; ++e) {
        const int l = LAYER_EXEC_ORDER[static_cast<std::size_t>(e)];
        Layer& L = layers[static_cast<std::size_t>(l)];
        // `h` is mutated IN PLACE by gr_write_row below (unlike forward()'s Node graph, where a new
        // Node replaces `h`), so gr_read_row's read and gr_write_row's later write on the SAME `h`
        // need no separate "before" snapshot -- nothing between them mutates it.
        gr_read_row(h, L.gr_attn_norm, L.gr_attn_down, L.gr_attn_up, L.gr_attn_inject, L.ln1, a);
        // The softmax-attention mixer sublayer, factored into a lambda (rather than duplicated
        // verbatim in both branches below) so the GDN_SCHEDULE dispatch reads as a single small
        // if/else rather than two copies of this block drifting apart over time.
        auto do_attention_mixer = [&] {
            // Wq is [C, D_Q] -- the OUT width is D_Q = N_HEADS*D_HEAD, not D_MODEL, since WP4b
            // blocker A. Identical at the derived head width.
            linear_row(a, L.Wq, nullptr, qn, C, sub0::D_Q);
            // K/V project to D_KV (N_KV_HEADS heads), Q to D_Q (N_HEADS heads). qknorm_row/rope_row
            // both derive their per-head width as (width / heads), so passing the KV pair yields
            // the same D_HEAD they always did -- the rotation and norm are per-head, so nothing
            // else changes.
            linear_row(a, L.Wk, nullptr, kn, C, D_KV);
            linear_row(a, L.Wv, nullptr, vn, C, D_KV);
            if constexpr (USE_QK_NORM) { qknorm_row(qn, L.q_norm, H, sub0::D_Q); qknorm_row(kn, L.k_norm, N_KV_HEADS, D_KV); }
            if constexpr (POS_ENCODING == PosEncoding::Rope) { rope_row(qn, pos, H, sub0::D_Q); rope_row(kn, pos, N_KV_HEADS, D_KV); }
            // Depth attention, before the KV-cache append: the sequence cache must hold the MIXED
            // V, because that is what op_attn receives in the batched forward. Mirrors forward()'s
            // block.
            if constexpr (USE_DEPTH_ATTN) {
                depth_mix_row(qn, kn, vn, dep_k, dep_v, dep_n);
                if (DEPTH_SCHEDULE.own[static_cast<std::size_t>(e)] >= 0) {
                    for (int j = 0; j < D_KV; ++j) {
                        dep_k[(size_t)dep_n * D_KV + j] = kn[j];
                        dep_v[(size_t)dep_n * D_KV + j] = vn[j];
                    }
                    ++dep_n;
                }
            }
            float* kc = g_kv.krow(e, pos); float* vc = g_kv.vrow(e, pos);    // append this token's K/V
            for (int j = 0; j < D_KV; ++j) { kc[j] = kn[j]; vc[j] = vn[j]; }
            for (int hd = 0; hd < H; ++hd) {                                 // attend query pos over j<=pos
                const int off    = hd * d;
                const int off_kv = (hd / GQA_GROUP) * d;    // shared KV head for this query group
                std::array<float, SEQ_LEN> sc{};
                float mx = -1e30f;
                for (int j = 0; j <= pos; ++j) {
                    const float* kj = g_kv.krow(e, j) + off_kv;
                    float s = 0.f; for (int aa = 0; aa < d; ++aa) s += qn[off + aa] * kj[aa];
                    s *= scale; sc[j] = s; mx = std::max(mx, s);
                }
                float Z = 0.f;
                for (int j = 0; j <= pos; ++j) { sc[j] = FAST_MATH ? fast_exp(sc[j] - mx) : std::exp(sc[j] - mx); Z += sc[j]; }
                for (int aa = 0; aa < d; ++aa) att[off + aa] = 0.f;
                for (int j = 0; j <= pos; ++j) {
                    const float p = sc[j] / Z; const float* vj = g_kv.vrow(e, j) + off_kv;
                    for (int aa = 0; aa < d; ++aa) att[off + aa] += p * vj[aa];
                }
            }
            linear_row(att, L.Wo, nullptr, proj, sub0::D_Q, C);   // Wo is [D_Q, D_MODEL] (blocker A)
            gr_write_row(h, proj);                                           // residual (write step)
        };
        // The QSA decode counterpart of the same sublayer -- the T==1 form of op_qsa's own batched
        // call, built from the SAME qsa_math.hpp row helpers op_qsa's qsa::forward() loops over, so
        // the two provably compute identical arithmetic (the forward-vs-forward_one parity test is
        // what gates that -- docs/QSA.md S9). Reuses g_kv for K/V exactly as the softmax path does
        // (kv_width() == D_KV), and g_qsa_cache for the indexer's own raw keys, which g_kv cannot
        // hold (a different width and a different -- unnormed, unrotated -- content).
        [[maybe_unused]] auto do_qsa_mixer = [&] {
            float* raw_k_base = g_qsa_cache.base(e);
            // Row stride is ROTARY_DIM (the cos/sin tables are [SEQ_LEN][ROTARY_DIM]) -- see
            // QsaRopeTables (internal.hpp); it was D_HEAD before --rotary-dim became an axis.
            const float* cos_pos = g_qsa_rope.cos.data() + static_cast<size_t>(pos) * ROTARY_DIM;
            const float* sin_pos = g_qsa_rope.sin.data() + static_cast<size_t>(pos) * ROTARY_DIM;
            qsa::indexer_project_row(QSA_DIMS, a, L.qsa_idx_qk->data.data(),
                                      L.qsa_idx_qnorm->data.data(), cos_pos, sin_pos, qsa::RMS_EPS,
                                      qsa_idx_q,
                                      raw_k_base + static_cast<size_t>(pos) * QSA_INDEXER_HEAD_DIM);
            qsa::attn_project_row(QSA_DIMS, a, L.qsa_q->data.data(), L.qsa_gate->data.data(),
                                   L.qsa_k->data.data(), L.qsa_v->data.data(),
                                   L.qsa_qnorm->data.data(), L.qsa_knorm->data.data(),
                                   cos_pos, sin_pos, qsa::RMS_EPS,
                                   qn, qsa_gate_row, g_kv.krow(e, pos), g_kv.vrow(e, pos));
            // The pooled block keys persist across decode steps in this execution's own QsaCache
            // slot, exactly as the raw keys above already do -- the decode counterpart of the
            // batched path's in-scratch cache, filled by the SAME primitive (docs/QSA.md S11).
            qsa::indexer_select_row(QSA_DIMS, qsa_idx_q, raw_k_base, pos + 1,
                                     L.qsa_idx_knorm->data.data(), g_qsa_rope.cos.data(),
                                     g_qsa_rope.sin.data(), qsa::RMS_EPS,
                                     g_qsa_cache.block_base(e), g_qsa_cache.n_cached_of(e),
                                     qsa_mask, qsa_sel_scr);
            qsa::attn_row(QSA_DIMS, qn, qsa_gate_row, g_kv.krow(e, 0), g_kv.vrow(e, 0), pos + 1,
                           qsa_mask, L.qsa_o->data.data(), proj, qsa_att_scr);
            gr_write_row(h, proj);                                           // residual (write step)
        };
        // Which of the two full-attention forms this build uses, decided ONCE at compile time. This
        // is the seam the MoE stage's own SIGSEGV came through (docs/MOE.md S9: a replacement wired
        // into only ONE of two call sites), so it is expressed as a single named lambda that BOTH
        // the GDN-on and GDN-off dispatch paths below call, rather than duplicated in each.
        auto do_full_attn_mixer = [&] {
            if constexpr (USE_QSA) do_qsa_mixer();
            else                   do_attention_mixer();
        };
        // GDN_SCHEDULE.full_attn[l] mirrors Model::forward()'s own dispatch exactly (per-LAYER, not
        // per-execution) -- see that function's comment. `if constexpr` keeps a GDN-off build
        // exactly the original single call, no branch at all.
        if constexpr (USE_GATED_DELTANET) {
            if (GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)]) {
                do_full_attn_mixer();
            } else {
                // Gated DeltaNet decode: T=1, persistent per-execution state/conv history
                // (GdnCache, reset by kv_reset() at the start of each generation). op_gdn's whole
                // output (in_proj*, conv, recurrence, gated-norm, out_proj) IS the mixer sublayer,
                // so it goes straight into the residual the same way `proj` does above -- no
                // separate Wo, no RoPE, no QK-norm, no depth-attention (see Model::forward()'s
                // matching comment for why those are out of scope for a GDN layer).
                // Same dt_bias/a_log ARGUMENT-ORDER fix as op_gdn's batched forward (backend.cpp) (this
                // decode path had the identical swap, independently) -- see that call site's comment.
                float gdn_scratch[GDN_SCRATCH1];
                gdn::forward(GDN_DIMS, 1, a,
                             L.gdn_in_qkv->data.data(), L.gdn_in_z->data.data(),
                             L.gdn_in_b->data.data(), L.gdn_in_a->data.data(),
                             L.gdn_conv->data.data(), L.gdn_dt_bias->data.data(),
                             L.gdn_a_log->data.data(), L.gdn_norm->data.data(),
                             L.gdn_out_proj->data.data(),
                             g_gdn_cache.state_of(e), g_gdn_cache.conv_of(e), proj, gdn_scratch);
                gr_write_row(h, proj);                                       // residual (write step)
            }
        } else {
            do_full_attn_mixer();
        }
        gr_read_row(h, L.gr_mlp_norm, L.gr_mlp_down, L.gr_mlp_up, L.gr_mlp_inject, L.ln2, a);
        if constexpr (USE_MOE) {
            // WP4e: the same moe_resolve() op_moe uses, so the decode path and the batched path
            // resolve experts identically -- which is what keeps forward/forward_one parity (the
            // check docs/WP4_SCOPE.md S6 records as having caught a real bug in every one of WP1-3)
            // a check of the MODEL rather than of two different residency strategies.
            if constexpr (USE_MOE_QUANT) W->moe_cache.allocate();
            moe::forward_row_via(MOE_DIMS, a, L.moe_router->data.data(),
                                  [&](int e) { return moe_resolve(L, l, e); },
                                  L.moe_shared_gate->data.data(), L.moe_shared_up->data.data(),
                                  L.moe_shared_down->data.data(), L.moe_shared_gate_proj->data.data(),
                                  proj, moe_scratch);
        } else if constexpr (USE_GATED_FFN) {
            linear_row(a, L.Wg, nullptr, g1, C, D_FF);
            linear_row(a, L.W1, nullptr, f1, C, D_FF);
            for (int j = 0; j < D_FF; ++j) f1[j] = silu_row(g1[j]) * f1[j];
            linear_row(f1, L.W2, nullptr, proj, D_FF, C);
        } else {
            linear_row(a, L.W1, L.b1, f1, C, D_FF);
            for (int j = 0; j < D_FF; ++j) f1[j] = gelu_row(f1[j]);
            linear_row(f1, L.W2, L.b2, proj, D_FF, C);
        }
        gr_write_row(h, proj);                                              // residual (write step)
        // Periodic packed-content re-injection spike (Nanbeige-inspired, see core.hpp's
        // set_scratch_reinject doc comment): every `stride` layers, add the SAME layer-0 packed
        // vector back into this position's hidden state -- tests whether reinforcing the signal
        // partway through the stack counters the dilution a single upfront injection leaves (axis 9).
        // SCALE-ADAPTIVE: the residual stream's own RMS norm grows across depth (standard pre-norm
        // behavior -- it's why ln_f exists before the head at all), while `packed_copy` sits at fixed
        // ordinary-embedding-row scale (scratch_slots.hpp's own amplitude convention) -- a first,
        // fixed-scale version of this spike was measurably a near no-op even at full strength every
        // layer, because the addition was dwarfed by h's already-larger accumulated norm by mid-stack.
        // Rescaling packed_copy to match h's CURRENT norm before applying `scale` (now a fraction of
        // h's own magnitude, not an absolute embedding-scale constant) is what makes this a real test.
        //
        // NOT Gated-Residual-aware: this spike only ever touches h's first D_MODEL elements (stream 0
        // when GR is wide), same deliberate-scope-gap shape as GDN+depth-attention's own "neither doc
        // discusses the combination" (docs/GATED_DELTANET.md) -- this spike and GR have not been
        // designed to interact, and this pass does not attempt it.
        if (do_reinject && ((l + 1) % g_scratch_reinject_stride == 0)) {
            float h_ms = 0.f; for (int j = 0; j < C; ++j) h_ms += h[j] * h[j]; h_ms /= C;
            float pk_ms = 0.f; for (int j = 0; j < C; ++j) pk_ms += packed_copy[j] * packed_copy[j]; pk_ms /= C;
            const float norm_scale = g_scratch_reinject_scale * std::sqrt((h_ms + 1e-8f) / (pk_ms + 1e-8f));
            for (int j = 0; j < C; ++j) h[j] += norm_scale * packed_copy[j];
        }
    }
    // Gated Residual's model-level EXIT collapse (docs/GATED_RESIDUAL.md S1c), mirroring forward()'s
    // own exit-collapse placement: use_combine=False, so just the mix half (no gate/combine) --
    // last_hidden captures the FULLY-COLLAPSED, D_MODEL-wide representation, same as the GR-off path.
    if constexpr (USE_GATED_RESIDUAL) {
        gr::hc_norm(GR_DIMS, 1, h, gr_top_norm->data.data(), gr_normed);
        gr::mix(GR_DIMS, 1, gr_normed, gr_top_down->data.data(), gr_top_up->data.data(), gr_mixed, gr_mixscr);
        for (int j = 0; j < C; ++j) last_hidden[static_cast<std::size_t>(j)] = gr_mixed[j];
        // No final norm under GR -- the exit collapse's own hc_norm is it, so mixed_input feeds the
        // head directly. Mirrors forward()'s own branch exactly (the forward-vs-forward_one parity
        // check is what gates that the two stay mirrored).
        for (int j = 0; j < C; ++j) a[j] = gr_mixed[j];
    } else {
        for (int j = 0; j < C; ++j) last_hidden[static_cast<std::size_t>(j)] = h[j];   // diagnostic capture
        rmsnorm_row(h, ln_f, a, C);
    }
    if constexpr (USE_TIED_EMBEDDINGS) tied_head_row(a, tok_emb, logits.data(), C, VOCAB);
    else                               linear_row(a, lm_head, lm_bias, logits.data(), C, VOCAB);
    return logits.data();
}

namespace cpu_detail {

// Incremental single-token inference (KV-cache). kv_reset() clears/sizes the cache at the start of a
// generation; forward_one(id, pos) returns the logits [VOCAB] for the next token. See KVCache above.
// Also resets GdnCache's decode-persistent recurrent state -- `if constexpr` so this costs nothing
// (not even the vector-emptiness check) on a build with GDN off, per AGENTS.md S4.
void kv_reset() {
    g_kv.reset();
    if constexpr (USE_GATED_DELTANET) g_gdn_cache.reset();
    if constexpr (USE_QSA) g_qsa_cache.reset();   // the indexer's own raw-key store -- docs/QSA.md S6
}
const float* forward_one(int id, int pos) { ensure_thread_built(); return g_model.forward_one(id, pos); }
const float* last_hidden_ptr() { return g_model.last_hidden.data(); }   // see Model::last_hidden's comment

// KV-trace memoization primitives (spike, see core.hpp's declarations for the full design comment).
// Caller-responsibility contract matches forward_one's own: valid only after kv_reset() has sized g_kv
// on this thread (no redundant guard here, same discipline as g_kv's other two entry points above).
const float* kv_krow_ptr(int layer, int pos) { return g_kv.krow(layer, pos); }
const float* kv_vrow_ptr(int layer, int pos) { return g_kv.vrow(layer, pos); }
// KV-cache rows are D_KV wide with N_KV_HEADS heads (== D_MODEL/N_HEADS when not using GQA); the
// per-head rotation width is D_HEAD either way.
void kv_rope_rotate(float* row, int pos) { rope_row(row, pos, N_KV_HEADS, D_KV); }
void kv_splice_row(int layer, int pos, const float* k_canonical, const float* v) {
    float k[D_KV];
    for (int j = 0; j < D_KV; ++j) k[j] = k_canonical[j];
    rope_row(k, pos, N_KV_HEADS, D_KV);          // rotate the de-rotated canonical row to its splice position
    float* kc = g_kv.krow(layer, pos);
    float* vc = g_kv.vrow(layer, pos);
    for (int j = 0; j < D_KV; ++j) { kc[j] = k[j]; vc[j] = v[j]; }      // V is position-invariant, no rotation
}

}  // namespace cpu_detail
}  // namespace sub0
