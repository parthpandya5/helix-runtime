cmake_minimum_required(VERSION 3.24)

foreach(required_variable
        HELIX_BINARY_DIR
        HELIX_CONSUMER_SOURCE_DIR
        HELIX_CONSUMER_ROOT
        HELIX_CTEST_COMMAND
        HELIX_CXX_COMPILER
        HELIX_EXPECT_CUDA
        HELIX_EXPECT_VERSION
        HELIX_GENERATOR)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "install-consumer smoke requires ${required_variable}")
  endif()
endforeach()

cmake_path(
  IS_PREFIX HELIX_BINARY_DIR "${HELIX_CONSUMER_ROOT}" NORMALIZE consumer_root_is_safe)
cmake_path(
  COMPARE "${HELIX_BINARY_DIR}" EQUAL "${HELIX_CONSUMER_ROOT}" consumer_root_is_build)
if(NOT consumer_root_is_safe OR consumer_root_is_build)
  message(FATAL_ERROR
          "install-consumer root must be a child of the Helix build directory")
endif()

set(install_prefix "${HELIX_CONSUMER_ROOT}/prefix")
set(consumer_build "${HELIX_CONSUMER_ROOT}/build")
file(REMOVE_RECURSE "${HELIX_CONSUMER_ROOT}")

set(install_command
    "${CMAKE_COMMAND}" --install "${HELIX_BINARY_DIR}" --prefix "${install_prefix}")
if(DEFINED HELIX_INSTALL_CONFIG AND NOT HELIX_INSTALL_CONFIG STREQUAL "")
  list(APPEND install_command --config "${HELIX_INSTALL_CONFIG}")
endif()
execute_process(
  COMMAND ${install_command}
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
  message(FATAL_ERROR
          "Helix installation failed (${install_result})\n${install_output}\n${install_error}")
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${HELIX_CONSUMER_SOURCE_DIR}"
    -B "${consumer_build}"
    -G "${HELIX_GENERATOR}"
    "-DCMAKE_CXX_COMPILER=${HELIX_CXX_COMPILER}"
    "-DCMAKE_PREFIX_PATH=${install_prefix}"
    "-DHELIX_EXPECT_CUDA=${HELIX_EXPECT_CUDA}"
    "-DHELIX_EXPECT_VERSION=${HELIX_EXPECT_VERSION}")
if(DEFINED HELIX_INSTALL_CONFIG AND NOT HELIX_INSTALL_CONFIG STREQUAL "")
  list(APPEND configure_command "-DCMAKE_BUILD_TYPE=${HELIX_INSTALL_CONFIG}")
endif()
if(DEFINED HELIX_GENERATOR_PLATFORM AND NOT HELIX_GENERATOR_PLATFORM STREQUAL "")
  list(APPEND configure_command -A "${HELIX_GENERATOR_PLATFORM}")
endif()
if(DEFINED HELIX_GENERATOR_TOOLSET AND NOT HELIX_GENERATOR_TOOLSET STREQUAL "")
  list(APPEND configure_command -T "${HELIX_GENERATOR_TOOLSET}")
endif()
execute_process(
  COMMAND ${configure_command}
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
  message(
    FATAL_ERROR
      "Helix consumer configure failed (${configure_result})\n${configure_output}\n${configure_error}")
endif()

set(build_command "${CMAKE_COMMAND}" --build "${consumer_build}")
if(DEFINED HELIX_INSTALL_CONFIG AND NOT HELIX_INSTALL_CONFIG STREQUAL "")
  list(APPEND build_command --config "${HELIX_INSTALL_CONFIG}")
endif()
execute_process(
  COMMAND ${build_command}
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error)
if(NOT build_result EQUAL 0)
  message(FATAL_ERROR
          "Helix consumer build failed (${build_result})\n${build_output}\n${build_error}")
endif()

set(test_command "${HELIX_CTEST_COMMAND}" --test-dir "${consumer_build}" --output-on-failure)
if(DEFINED HELIX_INSTALL_CONFIG AND NOT HELIX_INSTALL_CONFIG STREQUAL "")
  list(APPEND test_command -C "${HELIX_INSTALL_CONFIG}")
endif()
execute_process(
  COMMAND ${test_command}
  RESULT_VARIABLE test_result
  OUTPUT_VARIABLE test_output
  ERROR_VARIABLE test_error)
if(NOT test_result EQUAL 0)
  message(FATAL_ERROR
          "Helix installed consumer failed (${test_result})\n${test_output}\n${test_error}")
endif()
