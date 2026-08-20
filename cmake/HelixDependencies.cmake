include_guard(GLOBAL)

include(FetchContent)

function(helix_require_googletest)
  if(TARGET GTest::gtest_main)
    return()
  endif()

  find_package(GTest CONFIG QUIET)
  if(TARGET GTest::gtest_main)
    message(STATUS "Using installed GoogleTest")
    return()
  endif()

  if(NOT HELIX_FETCH_DEPENDENCIES)
    message(FATAL_ERROR
      "GoogleTest was not found and HELIX_FETCH_DEPENDENCIES=OFF. "
      "Install GoogleTest, set GTest_ROOT/CMAKE_PREFIX_PATH, or enable fetching.")
  endif()

  message(STATUS "GoogleTest not found; fetching release 1.15.2")
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
endfunction()

function(helix_require_google_benchmark)
  if(TARGET benchmark::benchmark)
    return()
  endif()

  find_package(benchmark CONFIG QUIET)
  if(TARGET benchmark::benchmark)
    message(STATUS "Using installed Google Benchmark")
    return()
  endif()

  if(NOT HELIX_FETCH_DEPENDENCIES)
    message(FATAL_ERROR
      "Google Benchmark was not found and HELIX_FETCH_DEPENDENCIES=OFF. "
      "Install it, set benchmark_ROOT/CMAKE_PREFIX_PATH, or enable fetching.")
  endif()

  message(STATUS "Google Benchmark not found; fetching release 1.9.1")
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    URL_HASH SHA256=32131c08ee31eeff2c8968d7e874f3cb648034377dfc32a4c377fa8796d84981
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googlebenchmark)
endfunction()
