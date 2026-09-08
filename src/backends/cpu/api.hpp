#pragma once

#include "sub0/core.hpp"

namespace sub0::cpu_detail {

void build_model();
void set_scratch_bindings(const ScratchBindings* bindings);
void set_scratch_reinject(int stride, float scale);
void set_persistent_bindings(const PersistentBindings* bindings);
void set_sentinel_bindings(const SentinelBindings* bindings);
void print_host_memplan();
void print_config();
bool load_moe_quant_sidecar(const char* model_path);
std::size_t trainable_floats();
float* params_ptr();
float* grad_ptr();
float* adam_m_ptr();
float* adam_v_ptr();
void sync_params_to_host();
void sync_params_to_device();
void graph_reset();
Node* forward(const int* ids, int T);
void loop_pass_stats(const int* ids, int T, float* out_delta, float* out_hnorm);
Node* forward_capture(const int* ids, int T, HiddenSink sink, void* context);
Node* cross_entropy(Node* logits, const int* targets);
void kv_reset();
const float* forward_one(int id, int pos);
const float* last_hidden_ptr();
const float* kv_krow_ptr(int layer, int pos);
const float* kv_vrow_ptr(int layer, int pos);
void kv_rope_rotate(float* row, int pos);
void kv_splice_row(int layer, int pos, const float* k_canonical, const float* v);
void backward(Node* loss, float seed);
void reduce_gradients();
float train_batch(const int* data, const std::size_t* starts, int batch, int T,
                  const int* lengths, const std::uint8_t* loss_mask,
                  const ScratchBindings* const* window_bindings,
                  const SentinelBindings* const* window_sentinels,
                  const PersistentBindings* const* window_persistent);

}  // namespace sub0::cpu_detail
