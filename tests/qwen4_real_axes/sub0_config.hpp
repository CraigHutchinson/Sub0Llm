// tests/qwen4_real_axes/sub0_config.hpp -- Qwen3.8-Flash-Next's REAL RunConfig axes, hand-written.
//
// NOT generated, and deliberately NOT on any engine target's include path: this header exists so ONE
// translation unit (tests/qwen4_real_shape_tests.cpp) can compile include/sub0/layout.hpp against the
// real model's own axes and check, at COMPILE TIME, that make_param_layout() produces the real model's
// tensor count and float total. That is WP4b's headline gate (docs/WP4_SCOPE.md S6): it fails before
// the four shape blockers land and passes after, which is what makes "the S2h shape gap is closed" a
// proven statement rather than a claim.
//
// Every value below is transcribed from docs/WP4_SCOPE.md S1's table, which is itself cross-checked
// field-by-field against docs/QWEN4_PREVIEW_REFERENCE.md's verified facts table and the real
// config.json values re-fetched independently by docs/QSA.md S0 and docs/MOE.md S0. The real
// config.json field name is named in a comment beside each one, so a future reader can re-verify a
// single value without re-deriving the whole table.
//
// Two axes are DELIBERATELY not the real model's, and both are recorded rather than silently chosen:
//   * SEQ_LEN -- the real max_position_embeddings is 262,144, which is not affordable (docs/WP4_SCOPE.md
//     S4) and, crucially, CANNOT change the answer: under RoPE no parameter tensor's shape depends on
//     SEQ_LEN at all (layout.hpp's HAS_POS_EMB comment -- that omission is what decouples the window
//     from the checkpoint). A small value keeps the compile cheap and is provably shape-neutral here.
//   * NGRAM_* -- the n-gram/PLE table is explicitly OUT of the first real run (docs/WP4_SCOPE.md S5),
//     so it is off here, and the expected totals below are stated for a build without it.

#pragma once

constexpr int  D_MODEL     = 2560;    // hidden_size
constexpr int  N_LAYERS    = 48;      // num_hidden_layers
constexpr int  N_HEADS     = 24;      // num_attention_heads
constexpr int  N_KV_HEADS  = 2;       // num_key_value_heads
constexpr int  LOOP_MIDDLE_LAYERS = 0;   // LoopSplit is this project's own axis; the real model has none
constexpr int  LOOP_REPEATS       = 1;
constexpr int  DEPTH_ATTN_STRIDE  = 0;   // likewise -- this project's own axis, off
constexpr int  GDN_FULL_ATTN_STRIDE = 4; // full_attention_interval (docs/QSA.md S0 verified the real
                                          // 48-entry layer_types array IS gdn_schedule_for(4))
// WP4b blocker B -- GDN's own four head axes, previously aliased onto N_KV_HEADS/N_HEADS/D_HEAD/D_HEAD.
constexpr int  GDN_KEY_HEADS       = 16;   // linear_num_key_heads
constexpr int  GDN_VALUE_HEADS     = 48;   // linear_num_value_heads
constexpr int  GDN_KEY_HEAD_DIM    = 128;  // linear_key_head_dim
constexpr int  GDN_VALUE_HEAD_DIM  = 128;  // linear_value_head_dim
constexpr int  NGRAM_MAX_N         = 0;    // PLE table deliberately OUT of scope (see header comment)
constexpr int  NGRAM_TABLES_PER_ORDER = 1;
constexpr int  NGRAM_TABLE_SIZE    = 0;
constexpr int  HC_COUNT   = 4;        // hc_count
constexpr int  HC_LOWRANK = 320;      // hc_lowrank
constexpr int  NUM_EXPERTS     = 512; // num_experts
constexpr int  EXPERTS_PER_TOK = 10;  // num_experts_per_tok
constexpr int  QSA_INDEXER_N_HEADS       = 4;     // indexer_n_heads
constexpr int  QSA_INDEXER_KV_HEADS      = 1;     // indexer_kv_heads
constexpr int  QSA_INDEXER_HEAD_DIM      = 128;   // indexer_head_dim
constexpr int  QSA_INDEXER_BUDGET        = 2048;  // indexer_budget
constexpr int  QSA_INDEXER_COMPRESS_RATIO = 4;    // indexer_compress_ratio
// WP4b blocker C -- partial_rotary_factor 0.25 x head_dim 256 = 64.
constexpr int  ROTARY_DIM  = 64;
constexpr int   ROPE_SCALING      = 0;
constexpr float ROPE_SCALE_FACTOR = 1.0000f;
constexpr int  SEQ_LEN     = 128;     // NOT the real 262144 -- provably shape-neutral, see header comment
constexpr int  D_FF        = 640;     // moe_intermediate_size (== shared_expert_intermediate_size)
// WP4b blocker A -- head_dim is its own axis now. Note 24 * 256 = 6144 != hidden_size 2560: the whole
// point of the blocker, and the reason D_HEAD could not previously be spelled at all.
constexpr int  D_HEAD      = 256;     // head_dim
constexpr bool USE_TERNARY = false;
constexpr bool USE_GATED_FFN = true;
constexpr bool USE_TIED_EMBEDDINGS = false;   // tie_word_embeddings: false
constexpr bool USE_QK_NORM = true;
constexpr int  VOCAB       = 248320;  // vocab_size
// --- Positional encoding (compile-time) --------------------------------
enum class PosEncoding { Absolute, Rope };
constexpr PosEncoding POS_ENCODING = PosEncoding::Rope;
constexpr float       ROPE_THETA   = 10000000.0f;   // rope_theta
