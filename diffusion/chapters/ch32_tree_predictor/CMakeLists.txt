# Ch32 tree-predictor / hierarchy track. P1 1c OOV A/B experiment.
add_executable(ch32_oov_ab oov_ab.cpp)
target_link_libraries(ch32_oov_ab PRIVATE sub0diff)

# Phase-0 M2 topic-drift baseline (corpus vs flat-model generations).
add_executable(ch32_m2_drift m2_drift.cpp)
target_link_libraries(ch32_m2_drift PRIVATE sub0diff)

# P2 2b gist-conditioning A/B (flat Denoiser vs GistDenoiser).
add_executable(ch32_gist_ab gist_ab.cpp)
target_link_libraries(ch32_gist_ab PRIVATE sub0diff)

# P2 2e coarse-to-fine vs flat (efficiency/context benchmark).
add_executable(ch32_hier_ab hier_ab.cpp)
target_link_libraries(ch32_hier_ab PRIVATE sub0diff)

# P2 HierDenoiser generation + M2-on-generations (flat vs coarse-to-fine).
add_executable(ch32_hier_gen hier_gen.cpp)
target_link_libraries(ch32_hier_gen PRIVATE sub0diff)
