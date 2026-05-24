#include "backend.hpp"
#include <stdexcept>

#ifdef SUB0LLM_OPENVINO
#  include <openvino/openvino.hpp>
#endif

namespace sub0llm::backend::openvino {

Tensor matmul(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_OPENVINO
    // Full implementation in Ch14 when we cover whole-model inference.
    // Single-op dispatch here for benchmarking parity.
    (void)a; (void)b;
    throw std::runtime_error("OpenVINO single-op matmul: not yet implemented (Ch14)");
#else
    (void)a; (void)b;
    throw std::runtime_error("OpenVINO backend not compiled in (rebuild with -DSUB0LLM_ENABLE_OPENVINO=ON)");
#endif
}

Tensor add(const Tensor& a, const Tensor& b) {
#ifdef SUB0LLM_OPENVINO
    (void)a; (void)b;
    throw std::runtime_error("OpenVINO single-op add: not yet implemented (Ch14)");
#else
    (void)a; (void)b;
    throw std::runtime_error("OpenVINO backend not compiled in");
#endif
}

} // namespace sub0llm::backend::openvino
