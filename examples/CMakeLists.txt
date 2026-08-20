add_executable(helix_info core_info.cpp)
target_link_libraries(helix_info PRIVATE helix::core)
helix_configure_target(helix_info)
