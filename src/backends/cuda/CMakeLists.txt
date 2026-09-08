# CUDA backend source manifest. Keep backend-owned source selection beside the backend.
#
# NOT `PARENT_SCOPE`, for exactly the reason the CPU manifest beside this one spells out: this file is
# `include()`d, not `add_subdirectory()`d, so PARENT_SCOPE writes above the scope that reads it and
# `add_library(sub0_backend_cuda SHARED ${SUB0_CUDA_BACKEND_SOURCES})` gets nothing. Unlike the CPU
# case this one fails only in a GPU build, which is why it was not caught alongside it.
set(SUB0_CUDA_BACKEND_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/backend.cu")
