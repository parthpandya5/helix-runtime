cmake_minimum_required(VERSION 3.24)

project(HelixInstallConsumer LANGUAGES CXX)

if(NOT DEFINED HELIX_EXPECT_CUDA OR NOT DEFINED HELIX_EXPECT_VERSION)
  message(FATAL_ERROR
          "HELIX_EXPECT_CUDA and HELIX_EXPECT_VERSION must be provided by the install smoke")
endif()

find_package(Helix 0.1 CONFIG REQUIRED)

if(NOT Helix_VERSION VERSION_EQUAL HELIX_EXPECT_VERSION)
  message(FATAL_ERROR
          "Expected Helix ${HELIX_EXPECT_VERSION}, found ${Helix_VERSION}")
endif()

foreach(expected_target
        core
        tensor
        memory
        cpu_kernels
        attention
        kv_cache
        scheduler
        model
        runtime)
  if(NOT TARGET helix::${expected_target})
    message(FATAL_ERROR "Helix package is missing helix::${expected_target}")
  endif()
endforeach()

if(HELIX_EXPECT_CUDA)
  if(NOT Helix_CUDA_ENABLED OR NOT TARGET helix::cuda_runtime OR
     NOT TARGET helix::cuda_kernels)
    message(FATAL_ERROR "CUDA-enabled Helix package did not export its CUDA targets")
  endif()
else()
  if(Helix_CUDA_ENABLED OR TARGET helix::cuda_runtime OR TARGET helix::cuda_kernels)
    message(FATAL_ERROR "CPU-only Helix package unexpectedly exported CUDA support")
  endif()
endif()

add_executable(helix_install_consumer main.cpp)
target_compile_features(helix_install_consumer PRIVATE cxx_std_20)
target_compile_definitions(
  helix_install_consumer PRIVATE HELIX_CONSUMER_EXPECT_CUDA=$<BOOL:${HELIX_EXPECT_CUDA}>)
target_link_libraries(helix_install_consumer PRIVATE helix::runtime)

enable_testing()
add_test(NAME helix_install_consumer.run COMMAND helix_install_consumer)
