#include "sub0llm/autograd/embedding_ops.hpp"
#include "sub0llm/core/block_pool.hpp"
#include "../backends/cpu/kernels.hpp"
#include "../backends/cuda/backend.hpp"   // device dispatch for embed-scatter bwd (Stage 4 Phase 5)

#include <cstring>
#include <format>
#include <stdexcept>

namespace sub0llm::autograd {

using Shape = Tensor::Shape;

namespace {

std::shared_ptr<Node> make_node(Tensor data, bool rg) {
    auto n           = std::allocate_shared<Node>(PoolAllocator<Node>{});
    n->data          = std::move(data);
    n->requires_grad = rg;
    n->is_leaf       = false;
    return n;
}

Edge make_edge(std::shared_ptr<Node> node,
               SmallFunction<Tensor(const Tensor&)> fn) {
    Edge e;
    e.node        = std::move(node);
    e.backward_fn = std::move(fn);
    return e;
}

} // anonymous namespace

// ── embedding_lookup ──────────────────────────────────────────────────────────

Variable embedding_lookup(const Variable& weight, const Tensor& indices) {
    const auto& wd = weight.data();
    if (wd.ndim() != 2)
        throw std::runtime_error(
            "autograd::embedding_lookup: weight must be 2D (V, D)");
    if (indices.dtype() != DType::Int32)
        throw std::runtime_error(
            "autograd::embedding_lookup: indices must be int32");
    if (!indices.is_contiguous())
        throw std::runtime_error(
            "autograd::embedding_lookup: indices must be contiguous");

    const auto V = static_cast<std::size_t>(wd.shape()[0]);
    const auto D = static_cast<std::size_t>(wd.shape()[1]);
    const auto N = static_cast<std::size_t>(indices.numel());

    // Forward: gather rows.
    Tensor out_flat = zeros({static_cast<int64_t>(N), static_cast<int64_t>(D)},
                            DType::Float32, wd.device());
    const auto ws  = wd.data_as<float>();
    auto       os  = out_flat.data_as<float>();
    const auto idx = indices.data_as<int32_t>();

    for (std::size_t i = 0; i < N; ++i) {
        const auto tok = static_cast<std::size_t>(idx[i]);
        if (tok >= V)
            throw std::runtime_error(std::format(
                "autograd::embedding_lookup: index {} out of vocab [0,{})", tok, V));
        std::memcpy(os.data() + i * D, ws.data() + tok * D, D * sizeof(float));
    }

    // Reshape to match indices shape + D.
    Shape out_shape;
    for (std::size_t d = 0; d < static_cast<std::size_t>(indices.ndim()); ++d)
        out_shape.push_back(indices.shape()[d]);
    out_shape.push_back(static_cast<int64_t>(D));
    Tensor out_data = out_flat.reshape(out_shape);

    auto out = make_node(std::move(out_data), weight.requires_grad());
    if (out->requires_grad) {
        const Device w_device = wd.device();
        // Snapshot indices on the weight's device so the scatter kernel can read them.
        Tensor idx_copy = w_device.is_cuda() ? indices.to(w_device) : copy(indices);
        out->edges.push_back(make_edge(weight.impl(),
            [idx_copy, N, V, D, w_device](const Tensor& g) {
                // Flatten upstream gradient to (N, D) for uniform indexing.
                const Tensor g_flat = g.reshape(
                    {static_cast<int64_t>(N), static_cast<int64_t>(D)}).contiguous();

                Tensor grad_w = zeros(   // pre-zeroed: both kernels accumulate into it
                    {static_cast<int64_t>(V), static_cast<int64_t>(D)},
                    DType::Float32, w_device);

                if (w_device.is_cuda())
                    backend::cuda::embed_bwd(
                        reinterpret_cast<const float*>(g_flat.raw_ptr()),
                        reinterpret_cast<const int*>(idx_copy.raw_ptr()),
                        reinterpret_cast<float*>(grad_w.raw_ptr()),
                        static_cast<int>(N), static_cast<int>(D));
                else
                    backend::cpu::embed_bwd_f32(
                        g_flat.data_as<float>().data(), idx_copy.data_as<int32_t>().data(),
                        grad_w.data_as<float>().data(), N, D);
                return grad_w;
            }));
    }
    return Variable::wrap(std::move(out));
}

} // namespace sub0llm::autograd
