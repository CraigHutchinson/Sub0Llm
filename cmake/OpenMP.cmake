# cmake/OpenMP.cmake — OpenMP detection for the data-parallel training path.
#
# Exposes SUB0_OPENMP_TARGET (a link target carrying the OpenMP flags) when OpenMP is
# available. A missing OpenMP is a hard, explained error by default: without it the
# engine silently falls back to single-thread stubs and `tune`/`bench` show no scaling
# at all -- an easily missed perf regression. Included from the top-level build only on
# the real (non-super-build) path, so an OpenMP-less host never blocks the orchestrator.

find_package(OpenMP)

option(SUB0_REQUIRE_OPENMP "Fail configuration if OpenMP is unavailable (the data-parallel path needs it)" ON)

if(OpenMP_CXX_FOUND)
  set(SUB0_OPENMP_TARGET OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP: enabled via find_package (${OpenMP_CXX_FLAGS})")
else()
  # find_package's library probe is fragile with clang on Windows: it can fail to
  # locate libomp even though the compiler fully supports -fopenmp=libomp (which
  # auto-links the runtime). Probe the flag directly and drive it ourselves if it
  # works, so a flaky find_package no longer silently disables all parallelism.
  include(CheckCXXSourceCompiles)
  block()
    set(CMAKE_REQUIRED_FLAGS "-fopenmp=libomp")   # scoped to this block, never leaks
    check_cxx_source_compiles(
      "#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }"
      SUB0_OPENMP_FLAG_WORKS)
  endblock()
  if(SUB0_OPENMP_FLAG_WORKS)
    add_library(sub0_openmp INTERFACE)
    target_compile_options(sub0_openmp INTERFACE -fopenmp=libomp)
    target_link_options(sub0_openmp INTERFACE -fopenmp=libomp)
    set(SUB0_OPENMP_TARGET sub0_openmp)
    message(STATUS "OpenMP: find_package failed; using -fopenmp=libomp directly")
  elseif(SUB0_REQUIRE_OPENMP)
    message(FATAL_ERROR
      "OpenMP not found and '-fopenmp=libomp' does not compile.\n"
      "  The data-parallel training path REQUIRES OpenMP; without it the engine runs\n"
      "  single-threaded and `tune`/`bench` show no thread scaling (a silent perf\n"
      "  regression). Install an OpenMP runtime (libomp) for this toolchain and\n"
      "  reconfigure. To build single-threaded ON PURPOSE, pass -DSUB0_REQUIRE_OPENMP=OFF.")
  else()
    message(WARNING "OpenMP unavailable -- building SINGLE-THREADED (SUB0_REQUIRE_OPENMP=OFF).")
  endif()
endif()
