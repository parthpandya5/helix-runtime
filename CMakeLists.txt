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

if(HELIX_ENABLE_CUDA)
  target_sources(helix_core PRIVATE src/core/cuda_build_probe.cu)
  set_target_properties(helix_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

helix_configure_target(helix_core)
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

install(
  TARGETS helix_core
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
