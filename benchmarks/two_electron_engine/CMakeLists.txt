add_executable(twoe_timing src/twoe_timing.cpp)

target_link_libraries(twoe_timing PRIVATE tlno::twoe_engine tlno::basis tlno_benchmark_io)

option(TLNO_BUILD_D1 "Build the D1 SIMD lane experiment" OFF)
option(TLNO_BUILD_EXPERIMENTS "Build the available two-electron experiments" OFF)

if(TLNO_BUILD_D1 OR TLNO_BUILD_EXPERIMENTS)
    if(NOT TLNO_SIMD MATCHES "^(avx2|avx512)$")
        message(FATAL_ERROR "D1 requires TLNO_SIMD=avx2 or avx512")
    endif()

    add_executable(rys_native_lane_ablation
        src/rys_native_lane_ablation.cpp)

    target_link_libraries(rys_native_lane_ablation PRIVATE
        tlno::shell
        tlno::math)
endif()
