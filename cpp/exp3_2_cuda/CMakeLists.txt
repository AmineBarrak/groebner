cmake_minimum_required(VERSION 3.18)
project(exp3_2_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── CUDA architecture ──────────────────────────────────────────────────────
# RTX 40xx = Ada Lovelace = sm_89
# RTX 30xx = Ampere      = sm_86
# Detect at build time so the same CMakeLists works for both:
set(CMAKE_CUDA_ARCHITECTURES 89)

# ── Dependencies ───────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP      REQUIRED)

# ── Target ─────────────────────────────────────────────────────────────────
add_executable(f4_cuda f4_cuda.cu)

target_include_directories(f4_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/..
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(f4_cuda PRIVATE
    CUDA::cusparse
    OpenMP::OpenMP_CXX
)

set_target_properties(f4_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(f4_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

message(STATUS "CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit       : ${CUDAToolkit_VERSION}")
message(STATUS "")
message(STATUS "Build: cmake -B build && cmake --build build -j4")
message(STATUS "Run  : ./build/f4_cuda")
