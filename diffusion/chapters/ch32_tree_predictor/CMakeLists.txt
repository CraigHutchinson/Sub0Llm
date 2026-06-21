# Ch32 tree-predictor / hierarchy track. P1 1c OOV A/B experiment.
add_executable(ch32_oov_ab oov_ab.cpp)
target_link_libraries(ch32_oov_ab PRIVATE sub0diff)

# Phase-0 M2 topic-drift baseline (corpus vs flat-model generations).
add_executable(ch32_m2_drift m2_drift.cpp)
target_link_libraries(ch32_m2_drift PRIVATE sub0diff)
