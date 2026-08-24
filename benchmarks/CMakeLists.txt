add_executable(helix_benchmarks
  allocator_benchmarks.cpp
  attention_benchmarks.cpp
  core_benchmarks.cpp
  cpu_kernel_benchmarks.cpp
  kv_cache_benchmarks.cpp
  model_benchmarks.cpp
  paged_attention_benchmarks.cpp
  runtime_benchmarks.cpp
  scheduler_benchmarks.cpp
  tensor_benchmarks.cpp)
target_link_libraries(
  helix_benchmarks
  PRIVATE
    helix::attention
    helix::cpu_kernels
    helix::kv_cache
    helix::memory
    helix::model
    helix::runtime
    helix::scheduler
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
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(benchmark.smoke PROPERTIES LABELS "benchmark;smoke")

add_test(
  NAME benchmark.scheduler_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_(ContinuousBatchingPolicy|StaticCohortPolicy)"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.scheduler_smoke
  PROPERTIES LABELS "benchmark;scheduler;smoke")

add_test(
  NAME benchmark.model_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_CpuLlamaFullPrefill/hidden:32/intermediate:64/layers:2/heads:4/tokens:7/vocabulary:128$"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.model_smoke
  PROPERTIES LABELS "benchmark;model;smoke")

add_test(
  NAME benchmark.model_load_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_LlamaModelLoadFromFile/(Copied|ReadOnlyMapped)/hidden:32/intermediate:64/layers:2/heads:4/tokens:7/vocabulary:128$"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.model_load_smoke
  PROPERTIES LABELS "benchmark;model;smoke")

add_test(
  NAME benchmark.runtime_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_CpuRuntimeContinuousTrace/FullPromptPrefill/hidden:16/intermediate:32/layers:1/heads:4/requests:8/prompt:4/output:4/active:4$"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.runtime_smoke
  PROPERTIES LABELS "benchmark;runtime;smoke")

add_test(
  NAME benchmark.runtime_paged_decode_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_CpuRuntimePagedDecodeTrace/DirectBlockTable/hidden:16/intermediate:32/layers:1/heads:4/requests:8/prompt:4/output:4/active:4$"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.runtime_paged_decode_smoke
  PROPERTIES LABELS "benchmark;runtime;smoke")

add_test(
  NAME benchmark.runtime_prefix_reuse_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_CpuRuntimeHotPrefixReuseTrace/Enabled/hidden:16/intermediate:32/layers:1/heads:4/requests:16/prompt:8$"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.runtime_prefix_reuse_smoke
  PROPERTIES LABELS "benchmark;runtime;smoke")

add_test(
  NAME benchmark.kv_prefix_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_PagedKvPrefixForkLifecycle/SharedBlocks/layers:2/heads:2/head_dimension:32/block_tokens:8/prefix_tokens:32$"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.kv_prefix_smoke
  PROPERTIES LABELS "benchmark;kv_cache;smoke")

add_test(
  NAME benchmark.kv_prefix_index_smoke
  COMMAND helix_benchmarks
          "--benchmark_filter=BM_PagedPrefixCacheHotHitLifecycle/entries:16$"
          --benchmark_min_time=1x
          --benchmark_repetitions=1)
set_tests_properties(
  benchmark.kv_prefix_index_smoke
  PROPERTIES LABELS "benchmark;kv_cache;smoke")
