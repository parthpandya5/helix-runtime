add_executable(helix_benchmarks
  allocator_benchmarks.cpp
  attention_benchmarks.cpp
  core_benchmarks.cpp
  cpu_kernel_benchmarks.cpp
  kv_cache_benchmarks.cpp
  paged_attention_benchmarks.cpp
  tensor_benchmarks.cpp)
target_link_libraries(
  helix_benchmarks
  PRIVATE
    helix::attention
    helix::cpu_kernels
    helix::kv_cache
    helix::memory
    benchmark::benchmark_main)
if(HELIX_ENABLE_CUDA)
  target_sources(
    helix_benchmarks
    PRIVATE
      cuda_attention_benchmarks.cpp
      cuda_elementwise_benchmarks.cpp
      cuda_allocator_benchmarks.cpp
      cuda_matmul_benchmarks.cpp
      cuda_reduction_benchmarks.cpp
      cuda_transformer_benchmarks.cpp)
  target_link_libraries(helix_benchmarks PRIVATE helix::cuda_kernels)
endif()
helix_configure_target(helix_benchmarks)

add_test(
  NAME benchmark.smoke
  COMMAND helix_benchmarks
          --benchmark_filter=BM_DTypeMetadata
          --benchmark_min_time=0.001
          --benchmark_repetitions=1)
set_tests_properties(benchmark.smoke PROPERTIES LABELS "benchmark;smoke" TIMEOUT 30)
