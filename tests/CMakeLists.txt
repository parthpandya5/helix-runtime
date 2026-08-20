include(GoogleTest)

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
