include_guard(GLOBAL)
include(FetchContent)
# Published lower revision shared by the optional engine transport and standalone storage tests.
# f74b65e: FileAccess::uncached (the expert cache's --moe-cache-io uncached), on top of TransferSet::submit
# scanning only live claims (needed by chunked cache fills). Local, not yet
# published: build with FETCHCONTENT_SOURCE_DIR_SUB0MEMPAGE at exactly this revision until it is pushed.
set(SUB0_STORAGE_SUB0MEMPAGE_REVISION "f74b65e")
set(SUB0MEMPAGE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_INTEL_USM_PROBE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Sub0MemPage
  GIT_REPOSITORY https://github.com/CraigHutchinson/Sub0MemPage.git
  GIT_TAG ${SUB0_STORAGE_SUB0MEMPAGE_REVISION})
FetchContent_MakeAvailable(Sub0MemPage)
