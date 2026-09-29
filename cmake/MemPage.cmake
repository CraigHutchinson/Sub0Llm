include_guard(GLOBAL)
include(FetchContent)
# Published lower revision shared by the optional engine transport and standalone storage tests.
set(SUB0_STORAGE_SUB0MEMPAGE_REVISION "213acdd2121cec369c6b9606c514db2231e94f27")
set(SUB0MEMPAGE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_INTEL_USM_PROBE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Sub0MemPage
  GIT_REPOSITORY https://github.com/CraigHutchinson/Sub0MemPage.git
  GIT_TAG ${SUB0_STORAGE_SUB0MEMPAGE_REVISION})
FetchContent_MakeAvailable(Sub0MemPage)
