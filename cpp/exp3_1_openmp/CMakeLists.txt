cmake_minimum_required(VERSION 3.18)
project(exp3_1_openmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with full optimisation
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native)

find_package(OpenMP REQUIRED)

add_executable(hom_speedup hom_speedup.cpp)
target_include_directories(hom_speedup PRIVATE ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(hom_speedup PRIVATE OpenMP::OpenMP_CXX)

message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Build:  cmake -B build && cmake --build build -j4")
message(STATUS "Run  :  ./build/hom_speedup")
