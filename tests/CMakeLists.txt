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
