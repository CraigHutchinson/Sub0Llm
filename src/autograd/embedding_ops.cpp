#include "sub0llm/autograd/embedding_ops.hpp"
#include "../backends/cpu/kernels.hpp"

#include <cstring>
#include <format>
#include <stdexcept>

namespace sub0llm::autograd {

using Shape = Tensor::Shape;

namespace {

std::shared_ptr<Node> make_node(Tensor data, bool rg) {
    auto n           = std::make_shared<Node>();
    n->data          = std::move(data);
    n->requires_grad = rg;
    n->is_leaf       = false;
    return n;
}

Edge make_edge(std::shared_ptr<Node> node,
               std::function<Tensor(const Tensor&)> fn) {
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
        Tensor idx_copy = copy(indices);
        const Device w_device = wd.device();
        out->edges.push_back(make_edge(weight.impl(),
            [idx_copy, N, V, D, w_device](const Tensor& g) {
                // Flatten upstream gradient to (N, D) for uniform indexing.
                const Tensor g_flat = g.reshape(
                    {static_cast<int64_t>(N), static_cast<int64_t>(D)}).contiguous();
                const auto gs  = g_flat.data_as<float>();
                const auto tc  = idx_copy.data_as<int32_t>();

                Tensor grad_w = zeros(
                    {static_cast<int64_t>(V), static_cast<int64_t>(D)},
                    DType::Float32, w_device);
                auto gwd = grad_w.data_as<float>();

                backend::cpu::embed_bwd_f32(
                    gs.data(),
                    reinterpret_cast<const int32_t*>(idx_copy.raw_ptr()),
                    gwd.data(), N, D);
                return grad_w;
            }));
    }
    return Variable::wrap(std::move(out));
}

} // namespace sub0llm::autograd
