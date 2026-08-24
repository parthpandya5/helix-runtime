include(GoogleTest)

if(UNIX)
  find_program(HELIX_BASH_EXECUTABLE bash)
  if(HELIX_BASH_EXECUTABLE)
    add_test(
      NAME profiling.script_contract
      COMMAND
        "${CMAKE_COMMAND}"
        "-DBASH_EXECUTABLE=${HELIX_BASH_EXECUTABLE}"
        "-DPROFILE_SCRIPT=${PROJECT_SOURCE_DIR}/scripts/profile_cuda.sh"
        "-DTEST_BINARY_DIR=${PROJECT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/profile_script_contract.cmake")
    set_tests_properties(
      profiling.script_contract
      PROPERTIES LABELS "profiling;smoke" TIMEOUT 10)
  endif()
endif()

add_executable(helix_core_tests core_test.cpp)
target_link_libraries(helix_core_tests PRIVATE helix::core GTest::gtest_main)
helix_configure_target(helix_core_tests)

gtest_discover_tests(
  helix_core_tests
  TEST_PREFIX "core."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_tensor_tests tensor_test.cpp)
target_link_libraries(helix_tensor_tests PRIVATE helix::tensor GTest::gtest_main)
helix_configure_target(helix_tensor_tests)

gtest_discover_tests(
  helix_tensor_tests
  TEST_PREFIX "tensor."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_cpu_kernel_tests cpu_kernels_test.cpp)
target_link_libraries(helix_cpu_kernel_tests PRIVATE helix::cpu_kernels GTest::gtest_main)
helix_configure_target(helix_cpu_kernel_tests)

gtest_discover_tests(
  helix_cpu_kernel_tests
  TEST_PREFIX "cpu_kernels."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_attention_tests attention_test.cpp)
target_link_libraries(helix_attention_tests PRIVATE helix::attention GTest::gtest_main)
helix_configure_target(helix_attention_tests)

gtest_discover_tests(
  helix_attention_tests
  TEST_PREFIX "attention."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_contiguous_kv_cache_tests contiguous_kv_cache_test.cpp)
target_link_libraries(
  helix_contiguous_kv_cache_tests
  PRIVATE helix::attention helix::kv_cache GTest::gtest_main)
helix_configure_target(helix_contiguous_kv_cache_tests)

gtest_discover_tests(
  helix_contiguous_kv_cache_tests
  TEST_PREFIX "contiguous_kv."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_paged_kv_cache_tests paged_kv_cache_test.cpp)
target_link_libraries(
  helix_paged_kv_cache_tests
  PRIVATE helix::attention helix::kv_cache GTest::gtest_main)
helix_configure_target(helix_paged_kv_cache_tests)

gtest_discover_tests(
  helix_paged_kv_cache_tests
  TEST_PREFIX "paged_kv."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_prefix_cache_tests prefix_cache_test.cpp)
target_link_libraries(
  helix_prefix_cache_tests
  PRIVATE helix::kv_cache GTest::gtest_main)
helix_configure_target(helix_prefix_cache_tests)

gtest_discover_tests(
  helix_prefix_cache_tests
  TEST_PREFIX "prefix_cache."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_paged_attention_tests paged_attention_test.cpp)
target_link_libraries(
  helix_paged_attention_tests
  PRIVATE helix::attention GTest::gtest_main)
helix_configure_target(helix_paged_attention_tests)

gtest_discover_tests(
  helix_paged_attention_tests
  TEST_PREFIX "paged_attention."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_request_tests request_test.cpp)
target_link_libraries(helix_request_tests PRIVATE helix::scheduler GTest::gtest_main)
helix_configure_target(helix_request_tests)

gtest_discover_tests(
  helix_request_tests
  TEST_PREFIX "request."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_fifo_scheduler_tests fifo_scheduler_test.cpp)
target_link_libraries(helix_fifo_scheduler_tests PRIVATE helix::scheduler GTest::gtest_main)
helix_configure_target(helix_fifo_scheduler_tests)

gtest_discover_tests(
  helix_fifo_scheduler_tests
  TEST_PREFIX "fifo_scheduler."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_scheduler_metrics_tests scheduler_metrics_test.cpp)
target_link_libraries(helix_scheduler_metrics_tests PRIVATE helix::scheduler GTest::gtest_main)
helix_configure_target(helix_scheduler_metrics_tests)

gtest_discover_tests(
  helix_scheduler_metrics_tests
  TEST_PREFIX "scheduler_metrics."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_llama_model_tests llama_model_test.cpp)
target_link_libraries(helix_llama_model_tests PRIVATE helix::model GTest::gtest_main)
helix_configure_target(helix_llama_model_tests)

gtest_discover_tests(
  helix_llama_model_tests
  TEST_PREFIX "llama_model."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_model_format_tests model_format_test.cpp)
target_link_libraries(helix_model_format_tests PRIVATE helix::model GTest::gtest_main)
helix_configure_target(helix_model_format_tests)

gtest_discover_tests(
  helix_model_format_tests
  TEST_PREFIX "model_format."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_generation_tests generation_test.cpp)
target_link_libraries(helix_generation_tests PRIVATE helix::model GTest::gtest_main)
helix_configure_target(helix_generation_tests)

gtest_discover_tests(
  helix_generation_tests
  TEST_PREFIX "generation."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_cpu_runtime_engine_tests cpu_runtime_engine_test.cpp)
target_link_libraries(helix_cpu_runtime_engine_tests PRIVATE helix::runtime GTest::gtest_main)
helix_configure_target(helix_cpu_runtime_engine_tests)

gtest_discover_tests(
  helix_cpu_runtime_engine_tests
  TEST_PREFIX "cpu_runtime."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_memory_tests memory_allocator_test.cpp)
target_link_libraries(helix_memory_tests PRIVATE helix::memory GTest::gtest_main)
helix_configure_target(helix_memory_tests)

gtest_discover_tests(
  helix_memory_tests
  TEST_PREFIX "memory."
  DISCOVERY_MODE PRE_TEST)

add_executable(helix_deferred_allocator_tests deferred_allocator_test.cpp)
target_link_libraries(
  helix_deferred_allocator_tests
  PRIVATE helix::memory GTest::gtest_main)
helix_configure_target(helix_deferred_allocator_tests)

gtest_discover_tests(
  helix_deferred_allocator_tests
  TEST_PREFIX "deferred_memory."
  DISCOVERY_MODE PRE_TEST)

if(HELIX_ENABLE_CUDA)
  add_executable(helix_cuda_embedding_tests cuda_embedding_test.cpp)
  target_link_libraries(
    helix_cuda_embedding_tests
    PRIVATE helix::cuda_kernels helix::cpu_kernels GTest::gtest_main)
  helix_configure_target(helix_cuda_embedding_tests)

  gtest_discover_tests(
    helix_cuda_embedding_tests
    TEST_PREFIX "cuda_embedding."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 120)

  add_executable(helix_cuda_runtime_tests cuda_runtime_test.cpp)
  target_link_libraries(
    helix_cuda_runtime_tests
    PRIVATE helix::cuda_runtime helix::tensor GTest::gtest_main)
  helix_configure_target(helix_cuda_runtime_tests)

  gtest_discover_tests(
    helix_cuda_runtime_tests
    TEST_PREFIX "cuda_runtime."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 60)

  add_executable(helix_cuda_elementwise_tests cuda_elementwise_test.cpp)
  target_link_libraries(
    helix_cuda_elementwise_tests
    PRIVATE helix::cuda_kernels helix::cpu_kernels GTest::gtest_main)
  helix_configure_target(helix_cuda_elementwise_tests)

  gtest_discover_tests(
    helix_cuda_elementwise_tests
    TEST_PREFIX "cuda_elementwise."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 120)

  add_executable(helix_cuda_reduction_tests cuda_reduction_test.cpp)
  target_link_libraries(
    helix_cuda_reduction_tests
    PRIVATE helix::cuda_kernels helix::cpu_kernels GTest::gtest_main)
  helix_configure_target(helix_cuda_reduction_tests)

  gtest_discover_tests(
    helix_cuda_reduction_tests
    TEST_PREFIX "cuda_reduction."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 120)

  add_executable(helix_cuda_softmax_tests cuda_softmax_test.cpp)
  target_link_libraries(
    helix_cuda_softmax_tests
    PRIVATE helix::cuda_kernels helix::cpu_kernels GTest::gtest_main)
  helix_configure_target(helix_cuda_softmax_tests)

  gtest_discover_tests(
    helix_cuda_softmax_tests
    TEST_PREFIX "cuda_softmax."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 120)

  add_executable(helix_cuda_matmul_tests cuda_matmul_test.cpp)
  target_link_libraries(
    helix_cuda_matmul_tests
    PRIVATE helix::cuda_kernels helix::cpu_kernels GTest::gtest_main)
  helix_configure_target(helix_cuda_matmul_tests)

  gtest_discover_tests(
    helix_cuda_matmul_tests
    TEST_PREFIX "cuda_matmul."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 180)

  add_executable(helix_cuda_transformer_tests cuda_transformer_test.cpp)
  target_link_libraries(
    helix_cuda_transformer_tests
    PRIVATE helix::cuda_kernels helix::cpu_kernels GTest::gtest_main)
  helix_configure_target(helix_cuda_transformer_tests)

  gtest_discover_tests(
    helix_cuda_transformer_tests
    TEST_PREFIX "cuda_transformer."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 180)

  add_executable(helix_cuda_attention_tests cuda_attention_test.cpp)
  target_link_libraries(
    helix_cuda_attention_tests
    PRIVATE helix::attention helix::cuda_runtime GTest::gtest_main)
  helix_configure_target(helix_cuda_attention_tests)

  gtest_discover_tests(
    helix_cuda_attention_tests
    TEST_PREFIX "cuda_attention."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 180)

  add_executable(helix_cuda_llama_model_tests cuda_llama_model_test.cpp)
  target_link_libraries(
    helix_cuda_llama_model_tests
    PRIVATE helix::model helix::cuda_runtime GTest::gtest_main)
  helix_configure_target(helix_cuda_llama_model_tests)

  gtest_discover_tests(
    helix_cuda_llama_model_tests
    TEST_PREFIX "cuda_llama_model."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 300)

  add_executable(helix_cuda_allocator_tests cuda_allocator_test.cpp)
  target_link_libraries(
    helix_cuda_allocator_tests
    PRIVATE helix::memory helix::cuda_runtime GTest::gtest_main)
  helix_configure_target(helix_cuda_allocator_tests)

  gtest_discover_tests(
    helix_cuda_allocator_tests
    TEST_PREFIX "cuda_allocator."
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS gpu TIMEOUT 120)
endif()
