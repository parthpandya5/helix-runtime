include(GoogleTest)

add_executable(helix_core_tests core_test.cpp)
target_link_libraries(helix_core_tests PRIVATE helix::core GTest::gtest_main)
helix_configure_target(helix_core_tests)

gtest_discover_tests(
  helix_core_tests
  TEST_PREFIX "core."
  DISCOVERY_MODE PRE_TEST)
