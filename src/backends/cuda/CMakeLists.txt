# CUDA backend source manifest. Keep backend-owned source selection beside the backend.
# NOTE: included via include(), not add_subdirectory() -- see src/backends/cpu/CMakeLists.txt's
# matching note; PARENT_SCOPE here is a no-op and leaves this variable empty in the caller.
set(SUB0_CUDA_BACKEND_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/backend.cu")
