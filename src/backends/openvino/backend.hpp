#pragma once
#include "sub0llm/core/tensor.hpp"

// OpenVINO backend — added in Ch02 as a stub; full integration in a later pass.
//
// OpenVINO dispatches operations through its runtime via ov::Tensor and
// ov::Core.  For LLM use, typical flow is:
//   1. Convert model to IR (OpenVINO Model Optimizer or torch.export)
//   2. Compile to device: ov::CompiledModel model = core.compile_model(ir, "CPU");
//   3. Run inference:     ov::InferRequest req = model.create_infer_request();
//
// The ops below exist to allow single-op dispatch (for benchmarking), not full
// model execution.  Full model inference via OpenVINO is added in Ch14.

namespace sub0llm::backend::openvino {

[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor add   (const Tensor& a, const Tensor& b);

} // namespace sub0llm::backend::openvino
