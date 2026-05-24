#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/core/tensor.hpp"

using namespace sub0llm;
using Catch::Matchers::WithinAbs;

TEST_CASE("Tensor construction and metadata", "[tensor]") {
    Tensor t({2, 3, 4}, DType::Float32, Device::cpu());

    REQUIRE(t.ndim()  == 3);
    REQUIRE(t.numel() == 24);
    REQUIRE(t.shape() == Tensor::Shape{2, 3, 4});
    REQUIRE(t.dtype() == DType::Float32);
    REQUIRE(t.device().is_cpu());
    REQUIRE(t.defined());
    REQUIRE(t.is_contiguous());
}

TEST_CASE("zeros factory", "[tensor]") {
    Tensor t = zeros({3, 3});
    auto sp = t.data_as<float>();
    for (float v : sp) REQUIRE(v == 0.0f);
}

TEST_CASE("ones factory", "[tensor]") {
    Tensor t = ones({4});
    auto sp = t.data_as<float>();
    for (float v : sp) REQUIRE(v == 1.0f);
}

TEST_CASE("arange factory", "[tensor]") {
    Tensor t = arange(5);
    auto sp = t.data_as<float>();
    for (std::size_t i = 0; i < 5; ++i)
        REQUIRE(sp[i] == static_cast<float>(i));
}

TEST_CASE("arange int64", "[tensor]") {
    Tensor t = arange(6, DType::Int64);
    auto sp = t.data_as<std::int64_t>();
    for (std::size_t i = 0; i < 6; ++i)
        REQUIRE(sp[i] == static_cast<std::int64_t>(i));
}

TEST_CASE("randn shape and dtype", "[tensor]") {
    Tensor t = randn({10, 10});
    REQUIRE(t.shape() == Tensor::Shape{10, 10});
    REQUIRE(t.numel() == 100);
    REQUIRE(t.dtype() == DType::Float32);
}

TEST_CASE("dtype mismatch throws", "[tensor]") {
    Tensor t = zeros({4}, DType::Int32);
    REQUIRE_THROWS_AS(t.data_as<float>(), std::runtime_error);
}

TEST_CASE("reshape — same numel", "[tensor]") {
    Tensor t = arange(12);
    Tensor r = t.reshape({3, 4});

    REQUIRE(r.ndim()  == 2);
    REQUIRE(r.shape(0) == 3);
    REQUIRE(r.shape(1) == 4);
    REQUIRE(r.numel()  == 12);
    REQUIRE(r.is_contiguous());

    // Data shared with original.
    auto orig = t.data_as<float>();
    auto view = r.data_as<float>();
    for (std::size_t i = 0; i < 12; ++i)
        REQUIRE(view[i] == orig[i]);
}

TEST_CASE("reshape — wrong numel throws", "[tensor]") {
    Tensor t = arange(12);
    REQUIRE_THROWS_AS(t.reshape({3, 5}), std::runtime_error);
}

TEST_CASE("transpose — shape swap", "[tensor]") {
    Tensor t = arange(6).reshape({2, 3});
    Tensor tT = t.transpose(0, 1);

    REQUIRE(tT.shape(0) == 3);
    REQUIRE(tT.shape(1) == 2);
    REQUIRE_FALSE(tT.is_contiguous());
}

TEST_CASE("contiguous copy", "[tensor]") {
    Tensor t  = arange(6).reshape({2, 3});
    Tensor tT = t.transpose(0, 1);
    Tensor c  = tT.contiguous();

    REQUIRE(c.is_contiguous());
    REQUIRE(c.shape() == tT.shape());
    REQUIRE(c.numel() == tT.numel());
}

TEST_CASE("item() on scalar tensor", "[tensor]") {
    Tensor t = zeros({1});
    t.data_as<float>()[0] = 3.14f;
    REQUIRE_THAT(t.item<float>(), WithinAbs(3.14f, 1e-6f));
}

TEST_CASE("item() on non-scalar throws", "[tensor]") {
    Tensor t = zeros({2});
    REQUIRE_THROWS_AS(t.item<float>(), std::runtime_error);
}

TEST_CASE("shape_str formatting", "[tensor]") {
    Tensor t({2, 3, 4});
    REQUIRE(t.shape_str() == "(2, 3, 4)");
}

TEST_CASE("copy function", "[tensor]") {
    Tensor a = ones({3, 3});
    Tensor b = copy(a);

    // Modifying a should not affect b.
    a.data_as<float>()[0] = 99.0f;
    REQUIRE(b.data_as<float>()[0] == 1.0f);
}

TEST_CASE("device assignment", "[tensor]") {
    Tensor t = zeros({2, 2}, DType::Float32, Device::cpu());
    REQUIRE(t.device().is_cpu());
}

TEST_CASE("to() same device is a no-op copy", "[tensor]") {
    Tensor a = ones({4});
    Tensor b = a.to(Device::cpu());
    REQUIRE(b.numel() == 4);
    REQUIRE(b.data_as<float>()[0] == 1.0f);
}

TEST_CASE("to() GPU device throws placeholder error", "[tensor]") {
    Tensor a = ones({4});
    REQUIRE_THROWS_AS(a.to(Device::cuda()), std::runtime_error);
}

// ── Edge cases identified in code review ─────────────────────────────────────

TEST_CASE("default-constructed tensor is undefined", "[tensor][edge]") {
    Tensor t;
    REQUIRE_FALSE(t.defined());
    REQUIRE(t.numel() == 0);
    REQUIRE_THROWS_AS(t.raw_ptr(), std::runtime_error);
}

TEST_CASE("zero-element tensor (shape containing 0)", "[tensor][edge]") {
    Tensor t({2, 0, 4});
    REQUIRE(t.numel() == 0);
    REQUIRE(t.shape(1) == 0);
    REQUIRE(t.is_contiguous());
    // data_as on zero-element tensor returns an empty span — no allocation
    auto sp = t.data_as<float>();
    REQUIRE(sp.empty());
}

TEST_CASE("non-contiguous copy — stride-aware float32", "[tensor][edge]") {
    // Build (2,3) and transpose to get a non-contiguous (3,2)
    Tensor t = arange(6).reshape({2, 3});
    Tensor tT = t.transpose(0, 1);
    REQUIRE_FALSE(tT.is_contiguous());

    Tensor c = tT.contiguous();
    REQUIRE(c.is_contiguous());

    // Verify: tT[0,0]=0, tT[0,1]=3, tT[1,0]=1, tT[1,1]=4, tT[2,0]=2, tT[2,1]=5
    auto sp = c.data_as<float>();
    REQUIRE(sp[0] == 0.0f);
    REQUIRE(sp[1] == 3.0f);
    REQUIRE(sp[2] == 1.0f);
    REQUIRE(sp[3] == 4.0f);
    REQUIRE(sp[4] == 2.0f);
    REQUIRE(sp[5] == 5.0f);
}

TEST_CASE("non-contiguous copy — non-float32 throws", "[tensor][edge]") {
    Tensor t = arange(6, DType::Int64).reshape({2, 3});
    Tensor tT = t.transpose(0, 1);
    REQUIRE_FALSE(tT.is_contiguous());
    REQUIRE_THROWS_AS(tT.contiguous(), std::runtime_error);
}

TEST_CASE("ones — unsupported dtype throws", "[tensor][edge]") {
    REQUIRE_THROWS_AS(ones({4}, DType::Float16), std::runtime_error);
    REQUIRE_THROWS_AS(ones({4}, DType::BFloat16), std::runtime_error);
    REQUIRE_THROWS_AS(ones({4}, DType::Bool), std::runtime_error);
}

TEST_CASE("ones — Int8 and Int16 supported", "[tensor][edge]") {
    Tensor a = ones({4}, DType::Int8);
    for (auto v : a.data_as<std::int8_t>()) REQUIRE(v == 1);

    Tensor b = ones({4}, DType::Int16);
    for (auto v : b.data_as<std::int16_t>()) REQUIRE(v == 1);
}

TEST_CASE("dtype_name covers all enum values", "[dtype]") {
    // Exhaustive check — catches missing cases in the switch.
    REQUIRE(dtype_name(DType::Float32)  == "float32");
    REQUIRE(dtype_name(DType::Float16)  == "float16");
    REQUIRE(dtype_name(DType::BFloat16) == "bfloat16");
    REQUIRE(dtype_name(DType::Float64)  == "float64");
    REQUIRE(dtype_name(DType::Int8)     == "int8");
    REQUIRE(dtype_name(DType::Int16)    == "int16");
    REQUIRE(dtype_name(DType::Int32)    == "int32");
    REQUIRE(dtype_name(DType::Int64)    == "int64");
    REQUIRE(dtype_name(DType::Bool)     == "bool");
}
