cmake_minimum_required(VERSION 3.24)

project(
  Helix
  VERSION 0.1.0
  DESCRIPTION "A C++/CUDA LLM inference runtime built from first principles"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(GNUInstallDirs)
include(HelixCompilerOptions)

option(HELIX_ENABLE_CUDA "Build the optional CUDA backend" OFF)
option(HELIX_BUILD_TESTS "Build Helix unit tests" ${BUILD_TESTING})
option(HELIX_BUILD_BENCHMARKS "Build Helix microbenchmarks" ON)
option(HELIX_BUILD_EXAMPLES "Build Helix examples" ON)
option(HELIX_ENABLE_NCCL "Enable the future NCCL distributed backend" OFF)
option(HELIX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(HELIX_ENABLE_CLANG_TIDY "Run clang-tidy while compiling Helix targets" OFF)
option(HELIX_WARNINGS_AS_ERRORS "Treat Helix compiler warnings as errors" OFF)
option(HELIX_FETCH_DEPENDENCIES "Fetch missing test/benchmark dependencies" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(HELIX_ENABLE_NCCL)
  message(FATAL_ERROR
    "HELIX_ENABLE_NCCL is reserved for Milestone 25 and is not implemented. "
    "Leave it OFF; Helix does not currently claim NCCL support.")
endif()

if(HELIX_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "HELIX_ENABLE_CUDA=ON, but CMake could not find a CUDA compiler. "
      "Install the NVIDIA CUDA toolkit or configure with HELIX_ENABLE_CUDA=OFF.")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

set(HELIX_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${HELIX_GENERATED_INCLUDE_DIR}/helix/core")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hpp.in"
  "${HELIX_GENERATED_INCLUDE_DIR}/helix/core/config.hpp"
  @ONLY)

add_library(helix_core
  src/core/device.cpp
  src/core/dtype.cpp
  src/core/error.cpp
  src/core/logging.cpp)
add_library(helix::core ALIAS helix_core)

target_compile_features(helix_core PUBLIC cxx_std_20)
target_include_directories(
  helix_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${HELIX_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

helix_configure_target(helix_core)

if(HELIX_ENABLE_CUDA)
  add_library(helix_cuda_runtime
    src/core/cuda_build_probe.cu
    src/runtime/cuda/device.cpp
    src/runtime/cuda/error.cpp
    src/runtime/cuda/event.cpp
    src/runtime/cuda/memory.cpp
    src/runtime/cuda/stream.cpp)
  add_library(helix::cuda_runtime ALIAS helix_cuda_runtime)
  target_compile_features(helix_cuda_runtime PUBLIC cxx_std_20)
  target_link_libraries(helix_cuda_runtime PUBLIC helix::core CUDA::cudart)
  target_include_directories(
    helix_cuda_runtime
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${HELIX_GENERATED_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  helix_configure_target(helix_cuda_runtime)
endif()

add_library(helix_tensor
  src/tensor/shape.cpp
  src/tensor/storage.cpp
  src/tensor/tensor.cpp)
add_library(helix::tensor ALIAS helix_tensor)
target_compile_features(helix_tensor PUBLIC cxx_std_20)
target_link_libraries(helix_tensor PUBLIC helix::core)
if(HELIX_ENABLE_CUDA)
  target_link_libraries(helix_tensor PRIVATE helix::cuda_runtime)
endif()
target_include_directories(
  helix_tensor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${HELIX_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
helix_configure_target(helix_tensor)

if(HELIX_ENABLE_CUDA)
  add_library(helix_cuda_kernels
    src/kernels/cuda/elementwise.cu
    src/kernels/cuda/matmul.cu
    src/kernels/cuda/reduction.cu
    src/kernels/cuda/softmax.cu)
  add_library(helix::cuda_kernels ALIAS helix_cuda_kernels)
  target_compile_features(helix_cuda_kernels PUBLIC cxx_std_20)
  target_link_libraries(helix_cuda_kernels PUBLIC helix::tensor helix::cuda_runtime)
  target_include_directories(
    helix_cuda_kernels
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${HELIX_GENERATED_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  helix_configure_target(helix_cuda_kernels)
endif()

add_library(helix_cpu_kernels src/kernels/cpu/reference.cpp)
add_library(helix::cpu_kernels ALIAS helix_cpu_kernels)
target_compile_features(helix_cpu_kernels PUBLIC cxx_std_20)
target_link_libraries(helix_cpu_kernels PUBLIC helix::tensor)
target_include_directories(
  helix_cpu_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${HELIX_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
helix_configure_target(helix_cpu_kernels)

helix_add_format_targets()

if(HELIX_BUILD_TESTS)
  enable_testing()
  include(HelixDependencies)
  helix_require_googletest()
  add_subdirectory(tests)
endif()

if(HELIX_BUILD_BENCHMARKS)
  include(HelixDependencies)
  helix_require_google_benchmark()
  add_subdirectory(benchmarks)
endif()

if(HELIX_BUILD_EXAMPLES)
  add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()

set(HELIX_INSTALL_TARGETS helix_core helix_tensor helix_cpu_kernels)
if(HELIX_ENABLE_CUDA)
  list(APPEND HELIX_INSTALL_TARGETS helix_cuda_runtime helix_cuda_kernels)
endif()
install(
  TARGETS ${HELIX_INSTALL_TARGETS}
  EXPORT HelixTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
  FILES "${HELIX_GENERATED_INCLUDE_DIR}/helix/core/config.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/helix/core")

message(STATUS "Helix ${PROJECT_VERSION}")
message(STATUS "  CUDA backend:       ${HELIX_ENABLE_CUDA}")
message(STATUS "  Tests:              ${HELIX_BUILD_TESTS}")
message(STATUS "  Benchmarks:         ${HELIX_BUILD_BENCHMARKS}")
message(STATUS "  Sanitizers:         ${HELIX_ENABLE_SANITIZERS}")
message(STATUS "  clang-tidy:         ${HELIX_ENABLE_CLANG_TIDY}")
