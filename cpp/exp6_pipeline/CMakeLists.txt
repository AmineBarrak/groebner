cmake_minimum_required(VERSION 3.17)
project(exp6_l2_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -- CUDA architecture --------------------------------------------------
# Tesla V100 (Matilda HPC) = sm_70
# RTX 30xx = sm_86,  RTX 40xx = sm_89
# Override: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

# -- Dependencies -------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP      REQUIRED)
find_package(MPI         REQUIRED)

# GMP — Matilda installs to /cm/shared/apps/gmp/6.1.2
# Note: Matilda's GMP lacks libgmpxx (no --enable-cxx at build time).
# We use the C API (gmp.h / mpz_t) directly and provide our own C++ wrapper.
set(GMP_ROOT "/cm/shared/apps/gmp/6.1.2" CACHE PATH "GMP install prefix")
find_library(GMP_LIB gmp HINTS ${GMP_ROOT}/lib REQUIRED)
find_library(GMPXX_LIB gmpxx HINTS ${GMP_ROOT}/lib)
find_path(GMP_INCLUDE_DIR gmp.h
    HINTS ${GMP_ROOT}/include $ENV{HOME}/local/include /usr/include)

if(NOT GMPXX_LIB)
  message(STATUS "libgmpxx not found — using C API wrapper (gmp_wrapper.h)")
  add_definitions(-DUSE_GMP_C_API)
  set(GMPXX_LIB "")  # don't link it
endif()

# -- Target -------------------------------------------------------------
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
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

message(STATUS "CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit       : ${CUDAToolkit_VERSION}")
message(STATUS "MPI                : ${MPI_CXX_COMPILER}")
message(STATUS "GMP                : ${GMP_LIB}")
message(STATUS "")
message(STATUS "Build:  cmake -B build && cmake --build build -j8")
message(STATUS "Run  :  mpirun ./build/l2_pipeline")
