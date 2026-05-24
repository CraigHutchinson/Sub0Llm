#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace sub0llm {

// ── Supported scalar types ────────────────────────────────────────────────────
enum class DType : std::uint8_t {
    Float32,
    Float16,   // storage only initially; compute promotes to float32
    BFloat16,  // storage only initially
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
};

// ── Compile-time traits ───────────────────────────────────────────────────────
template<DType D> struct dtype_traits;

template<> struct dtype_traits<DType::Float32>  { using type = float;         static constexpr std::string_view name = "float32";  static constexpr bool is_float = true;  };
template<> struct dtype_traits<DType::Float64>  { using type = double;        static constexpr std::string_view name = "float64";  static constexpr bool is_float = true;  };
template<> struct dtype_traits<DType::Int8>     { using type = std::int8_t;   static constexpr std::string_view name = "int8";     static constexpr bool is_float = false; };
template<> struct dtype_traits<DType::Int16>    { using type = std::int16_t;  static constexpr std::string_view name = "int16";    static constexpr bool is_float = false; };
template<> struct dtype_traits<DType::Int32>    { using type = std::int32_t;  static constexpr std::string_view name = "int32";    static constexpr bool is_float = false; };
template<> struct dtype_traits<DType::Int64>    { using type = std::int64_t;  static constexpr std::string_view name = "int64";    static constexpr bool is_float = false; };
template<> struct dtype_traits<DType::Bool>     { using type = bool;          static constexpr std::string_view name = "bool";     static constexpr bool is_float = false; };

// ── Runtime helpers ───────────────────────────────────────────────────────────

// Returns the byte size of a dtype at runtime.
[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32:  return 4;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Float64:  return 8;
        case DType::Int8:     return 1;
        case DType::Int16:    return 2;
        case DType::Int32:    return 4;
        case DType::Int64:    return 8;
        case DType::Bool:     return 1;
    }
    return 0;
}

[[nodiscard]] constexpr std::string_view dtype_name(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float32:  return "float32";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Float64:  return "float64";
        case DType::Int8:     return "int8";
        case DType::Int16:    return "int16";
        case DType::Int32:    return "int32";
        case DType::Int64:    return "int64";
        case DType::Bool:     return "bool";
    }
    return "unknown";
}

[[nodiscard]] constexpr bool dtype_is_float(DType dtype) noexcept {
    return dtype == DType::Float32 || dtype == DType::Float64
        || dtype == DType::Float16 || dtype == DType::BFloat16;
}

// ── Concept: valid compute scalar (promotes to float32/float64) ───────────────
template<typename T>
concept ComputeScalar = std::is_same_v<T, float>
                     || std::is_same_v<T, double>
                     || std::is_same_v<T, std::int8_t>
                     || std::is_same_v<T, std::int16_t>
                     || std::is_same_v<T, std::int32_t>
                     || std::is_same_v<T, std::int64_t>;

// Maps a C++ type to the corresponding DType enum at compile time.
template<typename T> struct type_to_dtype;
template<> struct type_to_dtype<float>          { static constexpr DType value = DType::Float32; };
template<> struct type_to_dtype<double>         { static constexpr DType value = DType::Float64; };
template<> struct type_to_dtype<std::int8_t>    { static constexpr DType value = DType::Int8;    };
template<> struct type_to_dtype<std::int16_t>   { static constexpr DType value = DType::Int16;   };
template<> struct type_to_dtype<std::int32_t>   { static constexpr DType value = DType::Int32;   };
template<> struct type_to_dtype<std::int64_t>   { static constexpr DType value = DType::Int64;   };
template<> struct type_to_dtype<bool>           { static constexpr DType value = DType::Bool;    };

template<ComputeScalar T>
inline constexpr DType dtype_of = type_to_dtype<T>::value;

} // namespace sub0llm
