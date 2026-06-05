#include "sub0llm/autograd/variable.hpp"

#include "sub0llm/core/ops.hpp"

#include <algorithm>
#include <format>
#include <functional>
#include <stdexcept>
#include <unordered_set>

namespace sub0llm::autograd {

// ── Node ──────────────────────────────────────────────────────────────────────

void Node::accumulate_grad(const Tensor& upstream) {
    if (grad.numel() == 0) {
        grad = copy(upstream);
    } else {
        grad = ops::add(grad, upstream);
    }
}

void Node::zero_grad() {
    grad = zeros(data.shape(), data.dtype(), data.device());
}

// ── Variable ──────────────────────────────────────────────────────────────────

Variable::Variable(Tensor data, bool requires_grad, std::string name) {
    impl_                 = std::make_shared<Node>();
    impl_->data           = std::move(data);
    impl_->requires_grad  = requires_grad;
    impl_->is_leaf        = true;
    impl_->name           = std::move(name);
}

Variable Variable::wrap(std::shared_ptr<Node> n) noexcept {
    Variable v;
    v.impl_ = std::move(n);
    return v;
}

const Tensor& Variable::data() const {
    return impl_->data;
}
Tensor& Variable::data() {
    return impl_->data;
}
Tensor& Variable::grad() {
    return impl_->grad;
}
const Tensor& Variable::grad() const {
    return impl_->grad;
}
bool Variable::requires_grad() const noexcept {
    return impl_ && impl_->requires_grad;
}
void Variable::set_requires_grad(bool rg) noexcept {
    if (impl_) impl_->requires_grad = rg;
}
bool Variable::is_leaf() const noexcept {
    return impl_ && impl_->is_leaf;
}
const std::string& Variable::name() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->name : empty;
}

void Variable::zero_grad() {
    if (impl_) impl_->zero_grad();
}

// ── backward ──────────────────────────────────────────────────────────────────

void Variable::backward(Tensor upstream_grad) {
    if (!impl_)
        throw std::runtime_error("Variable::backward: called on undefined variable");

    if (!impl_->requires_grad)
        throw std::runtime_error("Variable::backward: variable does not require grad");

    // Default upstream gradient for scalar outputs.
    if (upstream_grad.numel() == 0) {
        if (impl_->data.numel() != 1)
            throw std::runtime_error(std::format(
                "Variable::backward: upstream_grad required for non-scalar output "
                "(numel={})", impl_->data.numel()));
        upstream_grad = ones({1}, DType::Float32, impl_->data.device());
    }

    // Build reverse topological order via iterative DFS.
    std::vector<std::shared_ptr<Node>> topo;
    std::unordered_set<Node*>          visited;

    std::function<void(const std::shared_ptr<Node>&)> dfs =
        [&](const std::shared_ptr<Node>& n) {
            if (!n || !n->requires_grad || visited.count(n.get())) return;
            visited.insert(n.get());
            for (const auto& e : n->edges) dfs(e.node);
            topo.push_back(n);
        };

    dfs(impl_);
    std::reverse(topo.begin(), topo.end()); // output first

    // Seed the output gradient.
    impl_->accumulate_grad(upstream_grad);

    // Propagate.
    for (const auto& node : topo) {
        if (node->grad.numel() == 0) continue;
        for (const auto& edge : node->edges) {
            if (!edge.node || !edge.node->requires_grad) continue;
            const Tensor input_grad = edge.backward_fn(node->grad);
            edge.node->accumulate_grad(input_grad);
        }
    }
}

// ── detach ────────────────────────────────────────────────────────────────────

Variable detach(const Variable& v) {
    return Variable(copy(v.data()), /*requires_grad=*/false);
}

} // namespace sub0llm::autograd
