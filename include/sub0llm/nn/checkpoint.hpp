#pragma once
#include "sub0llm/autograd/variable.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace sub0llm {

// Checkpoint format version. Bumped whenever a change makes previously-saved
// weights numerically incompatible with the current code.
//   1 (implicit/legacy): pre-versioned checkpoints; interleaved RoPE convention.
//   2: half-split (NeoX/HF) RoPE convention — matches GGUF and forward_one().
// load_checkpoint rejects any checkpoint whose version != kCheckpointFormatVersion.
inline constexpr int kCheckpointFormatVersion = 2;

// Save all param tensors to dir/step_XXXXXXXXX.ckpt
// Format: 4-byte header-length + JSON header (format_version + step + shapes)
//         + packed float32 data
void save_checkpoint(const std::vector<autograd::Variable>& params,
                     const std::string& dir, int64_t step);

// Load params from path. Validates param count and element counts.
// Returns the step number stored in the checkpoint.
// Throws std::runtime_error if file not found or shapes mismatch.
[[nodiscard]] int64_t load_checkpoint(const std::vector<autograd::Variable>& params,
                                       const std::string& path);

// Returns path of the checkpoint with the highest step in dir, or "" if none.
[[nodiscard]] std::string latest_checkpoint_path(const std::string& dir);

// Returns the step of the latest checkpoint, or -1 if none.
[[nodiscard]] int64_t latest_checkpoint_step(const std::string& dir);

} // namespace sub0llm
