cmake_minimum_required(VERSION 3.24)

foreach(required_variable IN ITEMS BASH_EXECUTABLE PROFILE_SCRIPT TEST_BINARY_DIR)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

set(output_dir "${TEST_BINARY_DIR}/profile-script-contract-output")
file(REMOVE_RECURSE "${output_dir}")

execute_process(
  COMMAND
    "${BASH_EXECUTABLE}" "${PROFILE_SCRIPT}" all
    --dry-run
    --build-dir "${TEST_BINARY_DIR}"
    --output-dir "${output_dir}"
    --output-name contract
    --benchmark-filter "^BM_CudaMatmulTiled/256/256/256/manual_time$"
    --iterations 2
    --launch-skip 10
    --launch-count 1
    --ncu-set full
  RESULT_VARIABLE dry_run_result
  OUTPUT_VARIABLE dry_run_output
  ERROR_VARIABLE dry_run_error)

if(NOT dry_run_result EQUAL 0)
  message(FATAL_ERROR
    "profiling dry run failed (${dry_run_result})\nstdout:\n${dry_run_output}\nstderr:\n${dry_run_error}")
endif()

foreach(expected_text IN ITEMS
    "DRY-RUN Nsight Systems command:"
    "nsys profile"
    "--sample=none"
    "--cpuctxsw=none"
    "--force-overwrite=false"
    "DRY-RUN Nsight Compute command:"
    "ncu --set=full"
    "--launch-skip=10"
    "--launch-count=1"
    "--benchmark_min_time=2x"
    "DRY-RUN CUDA preflight command:"
    "--benchmark_min_time=1x"
    "DRY-RUN environment manifest:")
  string(FIND "${dry_run_output}" "${expected_text}" found_at)
  if(found_at EQUAL -1)
    message(FATAL_ERROR
      "profiling dry run omitted '${expected_text}'\noutput:\n${dry_run_output}")
  endif()
endforeach()

if(EXISTS "${output_dir}")
  message(FATAL_ERROR "--dry-run unexpectedly created ${output_dir}")
endif()

execute_process(
  COMMAND
    "${BASH_EXECUTABLE}" "${PROFILE_SCRIPT}" ncu
    --dry-run
    --force
    --output-dir "${output_dir}"
    --output-name forced
  RESULT_VARIABLE force_result
  OUTPUT_VARIABLE force_output
  ERROR_VARIABLE force_error)
if(NOT force_result EQUAL 0)
  message(FATAL_ERROR
    "forced profiling dry run failed (${force_result})\nstdout:\n${force_output}\nstderr:\n${force_error}")
endif()
string(FIND "${force_output}" "--force-overwrite" force_found_at)
if(force_found_at EQUAL -1)
  message(FATAL_ERROR "--force did not reach the Nsight Compute command")
endif()

function(expect_profile_failure description)
  execute_process(
    COMMAND "${BASH_EXECUTABLE}" "${PROFILE_SCRIPT}" ${ARGN}
    RESULT_VARIABLE failure_result
    OUTPUT_VARIABLE failure_output
    ERROR_VARIABLE failure_error)
  if(failure_result EQUAL 0)
    message(FATAL_ERROR
      "${description} unexpectedly succeeded\nstdout:\n${failure_output}\nstderr:\n${failure_error}")
  endif()
endfunction()

expect_profile_failure(
  "zero measured iterations"
  nsys --dry-run --iterations 0)
expect_profile_failure(
  "path traversal in the output stem"
  nsys --dry-run --output-name ../escape)
expect_profile_failure(
  "unknown profiler mode"
  invalid --dry-run)
