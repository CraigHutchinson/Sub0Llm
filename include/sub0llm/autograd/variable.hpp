#pragma once

#include "sub0llm/core/tensor.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sub0llm::autograd {

struct Node;

// One backward edge: a reference to an input node plus the VJP that maps the
// output gradient to the gradient contribution for that input.
struct Edge {
    std::shared_ptr<Node>               node;
    std::function<Tensor(const Tensor&)> backward_fn;
};

// A node in the dynamic computation graph.
// Leaf nodes (parameters / inputs created by the user) have is_leaf=true.
// Non-leaf nodes are produced by differentiable ops.
struct Node {
    Tensor      data;
    Tensor      grad;           // accumulated gradient (same shape as data)
    bool        requires_grad = false;
    bool        is_leaf       = true;
    std::string name;           // optional debug label

    std::vector<Edge> edges;    // backward edges (empty for leaves)

    // Accumulate upstream into grad (initialises grad on first call).
    void accumulate_grad(const Tensor& upstream);

    void zero_grad();
};

// Variable is a reference-counted handle to a Node.
// Cheap to copy; semantics match std::shared_ptr.
class Variable {
public:
    Variable() = default;
    explicit Variable(Tensor data, bool requires_grad = false, std::string name = {});

    [[nodiscard]] const Tensor& data()          const;
    [[nodiscard]] Tensor&       data();
    [[nodiscard]] Tensor&       grad();
    [[nodiscard]] const Tensor& grad()          const;
    [[nodiscard]] bool          requires_grad() const noexcept;
    // Toggle gradient tracking on a leaf (e.g. to freeze parameters). Ops read
    // this when building the backward graph, so a frozen leaf gets no edges and
    // its subgraph is pruned from backward.
    void                        set_requires_grad(bool rg) noexcept;
    [[nodiscard]] bool          is_leaf()       const noexcept;
    [[nodiscard]] bool          defined()       const noexcept { return impl_ != nullptr; }
    [[nodiscard]] const std::string& name()     const noexcept;

    // Run reverse-mode AD from this node.  Call only on scalar (numel=1) outputs;
    // for tensor outputs supply an explicit upstream_grad of matching shape.
    void backward(Tensor upstream_grad = {});

    // Zero this variable's gradient (sets grad to zeros, same shape as data).
    void zero_grad();

    // Internal: expose the underlying node to ops that build the backward graph.
    [[nodiscard]] std::shared_ptr<Node> impl() const noexcept { return impl_; }
    [[nodiscard]] static Variable       wrap(std::shared_ptr<Node> n) noexcept;

private:
    std::shared_ptr<Node> impl_;
};

// ── Free utilities ─────────────────────────────────────────────────────────────

// Detach a tensor from the computation graph (leaf, requires_grad=false).
// Returns a deep copy of the data; mutating the result does not affect v.
[[nodiscard]] Variable detach(const Variable& v);

} // namespace sub0llm::autograd
