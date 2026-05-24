// Chapter 01 — Core Foundations
//
// This chapter introduces the Tensor abstraction that underpins every later
// chapter.  In Python projects like LLMs-from-scratch, NumPy/PyTorch handle
// this transparently; in C++23 we build it ourselves so we understand every
// byte of the model.
//
// Topics covered:
//   • DType system and compile-time / runtime type traits
//   • Device abstraction (CPU now, CUDA/OpenVINO in Ch02)
//   • Tensor construction, shape, strides, and contiguous layout
//   • Factory functions: zeros, ones, randn, arange
//   • Typed data access via std::span
//   • reshape and transpose (view semantics)
//   • Basic element-wise ops and matmul

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;
using namespace sub0llm::ops;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: print section headers for chapter narrative
static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 01: Core Foundations", version_string);

    // ── 1. DType system ───────────────────────────────────────────────────────
    section("1. DType system");

    std::println("DType sizes:");
    for (auto dt : {DType::Float32, DType::Float16, DType::BFloat16,
                    DType::Float64, DType::Int8, DType::Int32, DType::Int64}) {
        std::println("  {:10} — {} bytes", dtype_name(dt), dtype_size(dt));
    }
    std::println("dtype_of<float>  = {}", dtype_name(dtype_of<float>));
    std::println("dtype_of<int32>  = {}", dtype_name(dtype_of<std::int32_t>));

    // ── 2. Device abstraction ─────────────────────────────────────────────────
    section("2. Device abstraction");

    auto cpu = Device::cpu();
    std::println("Default device: {}", cpu.str());
    std::println("CUDA:0 device:  {}", Device::cuda(0).str());

    // ── 3. Creating tensors ───────────────────────────────────────────────────
    section("3. Tensor creation");

    Tensor z = zeros({3, 4});
    std::println("zeros(3,4):  {}", z.to_string());

    Tensor o = ones({2, 3});
    std::println("ones(2,3):   {}", o.to_string());

    Tensor r = randn({4, 4});
    std::println("randn(4,4):  {}", r.to_string());

    Tensor a = arange(6);
    std::println("arange(6):   {}", a.to_string());

    // ── 4. Shape & strides ────────────────────────────────────────────────────
    section("4. Shape & strides");

    Tensor t = arange(24);
    std::println("t.shape() = {}", t.shape_str());
    std::println("t.numel() = {}", t.numel());

    // Reshape into a 2×3×4 tensor (view — no data copy when contiguous).
    Tensor t3d = t.reshape({2, 3, 4});
    std::println("After reshape(2,3,4): shape = {}, contiguous = {}",
        t3d.shape_str(), t3d.is_contiguous());

    // ── 5. Transpose (view) ───────────────────────────────────────────────────
    section("5. Transpose");

    Tensor mat = arange(6).reshape({2, 3});
    Tensor mat_T = mat.transpose(0, 1);
    std::println("mat (2,3) shape:          {}", mat.shape_str());
    std::println("mat.T (3,2) shape:        {}", mat_T.shape_str());
    std::println("mat.T is_contiguous:      {}", mat_T.is_contiguous());
    std::println("mat.T.contiguous() shape: {}", mat_T.contiguous().shape_str());

    // ── 6. Typed data access via std::span ────────────────────────────────────
    section("6. std::span data access");

    Tensor x = zeros({5}, DType::Float32);
    auto span = x.data_as<float>();
    for (std::size_t i = 0; i < span.size(); ++i) span[i] = static_cast<float>(i) * 0.5f;
    std::println("x after manual fill: {}", x.to_string());

    // ── 7. Element-wise ops ───────────────────────────────────────────────────
    section("7. Element-wise ops");

    Tensor a2 = arange(4);                     // [0, 1, 2, 3]
    Tensor b2 = ones({4});                     // [1, 1, 1, 1]
    std::println("a     = {}", a2.to_string());
    std::println("b     = {}", b2.to_string());
    std::println("a + b = {}", add(a2, b2).to_string());
    std::println("a * b = {}", mul(a2, b2).to_string());
    std::println("a * 2 = {}", mul(a2, 2.0f).to_string());

    // ── 8. Reductions ─────────────────────────────────────────────────────────
    section("8. Reductions");

    std::println("sum(a)  = {}", sum(a2));
    std::println("mean(a) = {}", mean(a2));
    std::println("max(a)  = {}", max(a2));

    // ── 9. Activations ────────────────────────────────────────────────────────
    section("9. Activations");

    Tensor logits = zeros({1, 4});
    {
        auto sp = logits.data_as<float>();
        sp[0] = 1.0f; sp[1] = 2.0f; sp[2] = 3.0f; sp[3] = 4.0f;
    }
    std::println("logits:          {}", logits.to_string());
    std::println("relu(logits-2):  {}", relu(add(logits, -2.0f)).to_string());
    std::println("softmax(logits): {}", softmax(logits).to_string());
    std::println("gelu([0..3]):    {}", gelu(arange(4)).to_string());

    // ── 10. Matrix multiply ───────────────────────────────────────────────────
    section("10. Matrix multiply (naive O(M·N·K))");

    // (2,3) × (3,2) → (2,2)
    Tensor A = arange(6).reshape({2, 3});
    Tensor B = arange(6).reshape({3, 2});
    Tensor C = matmul(A, B);
    std::println("A (2,3): {}", A.to_string());
    std::println("B (3,2): {}", B.to_string());
    std::println("A @ B  : {}", C.to_string());

    // ── 11. What's next ───────────────────────────────────────────────────────
    section("What's next — Ch02: Compute Backends");
    std::println(
        "Ch02 replaces the naive loops above with:\n"
        "  • AVX2/AVX-512 SIMD kernels (via intrinsics)\n"
        "  • CUDA kernels for GPU acceleration\n"
        "  • OpenVINO dispatch for Intel hardware\n"
        "The Tensor API above stays identical — only the dispatch layer changes."
    );

    return 0;
}
