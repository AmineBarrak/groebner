cmake_minimum_required(VERSION 3.18)
project(exp6_l2_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── CUDA architecture ──────────────────────────────────────────────────────
# Detect architecture or set manually:
#   Tesla V100 (Matilda HPC) = sm_70
#   RTX 30xx = sm_86,  RTX 40xx = sm_89
# Default to V100; override with -DCMAKE_CUDA_ARCHITECTURES=89 for local
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# ── Dependencies ───────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP      REQUIRED)
find_package(MPI         REQUIRED)

# GMP / GMPXX
find_library(GMP_LIB    gmp    REQUIRED)
find_library(GMPXX_LIB  gmpxx  REQUIRED)
find_path(GMP_INCLUDE_DIR gmpxx.h)

# ── Target ─────────────────────────────────────────────────────────────────
add_executable(l2_pipeline l2_pipeline.cu)

target_include_directories(l2_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/..
    ${CUDAToolkit_INCLUDE_DIRS}
    ${MPI_CXX_INCLUDE_DIRS}
    ${GMP_INCLUDE_DIR}
)

target_link_libraries(l2_pipeline PRIVATE
    CUDA::cusparse
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    ${GMP_LIB}
    ${GMPXX_LIB}
)

set_target_properties(l2_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(l2_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

message(STATUS "CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit       : ${CUDAToolkit_VERSION}")
message(STATUS "MPI implementation : ${MPI_CXX_COMPILER}")
message(STATUS "GMP library        : ${GMP_LIB}")
message(STATUS "")
message(STATUS "Build:  cmake -B build && cmake --build build -j4")
message(STATUS "Run  :  mpirun -np 4 ./build/l2_pipeline")
message(STATUS "   or:  ./build/l2_pipeline  (single process)")
