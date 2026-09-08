#include "api.hpp"

namespace sub0 {

void build_model() { cpu_detail::build_model(); }
void set_scratch_bindings(const ScratchBindings* bindings) { cpu_detail::set_scratch_bindings(bindings); }
void set_scratch_reinject(int stride, float scale) { cpu_detail::set_scratch_reinject(stride, scale); }
void set_persistent_bindings(const PersistentBindings* bindings) { cpu_detail::set_persistent_bindings(bindings); }
void set_sentinel_bindings(const SentinelBindings* bindings) { cpu_detail::set_sentinel_bindings(bindings); }
void print_host_memplan() { cpu_detail::print_host_memplan(); }
void print_config() { cpu_detail::print_config(); }
bool load_moe_quant_sidecar(const char* model_path) { return cpu_detail::load_moe_quant_sidecar(model_path); }
float* params_ptr() { return cpu_detail::params_ptr(); }
float* grad_ptr() { return cpu_detail::grad_ptr(); }
float* adam_m_ptr() { return cpu_detail::adam_m_ptr(); }
float* adam_v_ptr() { return cpu_detail::adam_v_ptr(); }
void sync_params_to_host() { cpu_detail::sync_params_to_host(); }
void sync_params_to_device() { cpu_detail::sync_params_to_device(); }
void graph_reset() { cpu_detail::graph_reset(); }
Node* forward(const int* ids, int T) { return cpu_detail::forward(ids, T); }
void loop_pass_stats(const int* ids, int T, float* out_delta, float* out_hnorm) {
    cpu_detail::loop_pass_stats(ids, T, out_delta, out_hnorm);
}
Node* forward_capture(const int* ids, int T, HiddenSink sink, void* context) {
    return cpu_detail::forward_capture(ids, T, sink, context);
}
Node* cross_entropy(Node* logits, const int* targets) { return cpu_detail::cross_entropy(logits, targets); }
void kv_reset() { cpu_detail::kv_reset(); }
const float* forward_one(int id, int pos) { return cpu_detail::forward_one(id, pos); }
const float* last_hidden_ptr() { return cpu_detail::last_hidden_ptr(); }
const float* kv_krow_ptr(int layer, int pos) { return cpu_detail::kv_krow_ptr(layer, pos); }
const float* kv_vrow_ptr(int layer, int pos) { return cpu_detail::kv_vrow_ptr(layer, pos); }
void kv_rope_rotate(float* row, int pos) { cpu_detail::kv_rope_rotate(row, pos); }
void kv_splice_row(int layer, int pos, const float* k_canonical, const float* v) {
    cpu_detail::kv_splice_row(layer, pos, k_canonical, v);
}
void backward(Node* loss, float seed) { cpu_detail::backward(loss, seed); }
void reduce_gradients() { cpu_detail::reduce_gradients(); }
float train_batch(const int* data, const std::size_t* starts, int batch, int T,
                  const int* lengths, const std::uint8_t* loss_mask,
                  const ScratchBindings* const* window_bindings,
                  const SentinelBindings* const* window_sentinels,
                  const PersistentBindings* const* window_persistent) {
    return cpu_detail::train_batch(data, starts, batch, T, lengths, loss_mask,
                                   window_bindings, window_sentinels, window_persistent);
}

}  // namespace sub0
