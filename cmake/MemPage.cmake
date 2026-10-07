include_guard(GLOBAL)
include(FetchContent)
# Published lower revision shared by the optional engine transport and standalone storage tests.
# f74b65e: FileAccess::uncached (the expert cache's --moe-cache-io uncached), on top of TransferSet::submit
# scanning only live claims (needed by chunked cache fills). Published. For offline work, set
# FETCHCONTENT_SOURCE_DIR_SUB0MEMPAGE to a checkout at exactly this revision.
set(SUB0_STORAGE_SUB0MEMPAGE_REVISION "f74b65ea494af6e9b7d70069a3b0732e8c6f9da0")
set(SUB0MEMPAGE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(SUB0MEMPAGE_BUILD_INTEL_USM_PROBE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Sub0MemPage
  GIT_REPOSITORY https://github.com/CraigHutchinson/Sub0MemPage.git
  GIT_TAG ${SUB0_STORAGE_SUB0MEMPAGE_REVISION})
FetchContent_MakeAvailable(Sub0MemPage)
