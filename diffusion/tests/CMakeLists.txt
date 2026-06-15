add_executable(tests_diffusion
    test_diffusion.cpp
    test_schedule.cpp
)

target_link_libraries(tests_diffusion
    PRIVATE
        sub0diff
        Catch2::Catch2WithMain
)

sub0llm_apply_compile_options(tests_diffusion)

include(Catch)
catch_discover_tests(tests_diffusion)
