add_executable(helix_benchmarks
  core_benchmarks.cpp
  cpu_kernel_benchmarks.cpp
  tensor_benchmarks.cpp)
target_link_libraries(helix_benchmarks PRIVATE helix::cpu_kernels benchmark::benchmark_main)
helix_configure_target(helix_benchmarks)

add_test(
  NAME benchmark.smoke
  COMMAND helix_benchmarks
          --benchmark_filter=BM_DTypeMetadata
          --benchmark_min_time=0.001
          --benchmark_repetitions=1)
set_tests_properties(benchmark.smoke PROPERTIES LABELS "benchmark;smoke" TIMEOUT 30)
