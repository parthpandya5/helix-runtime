add_executable(helix_info core_info.cpp)
target_link_libraries(helix_info PRIVATE helix::core)
helix_configure_target(helix_info)

add_executable(helix_tensor_demo tensor_demo.cpp)
target_link_libraries(helix_tensor_demo PRIVATE helix::tensor)
helix_configure_target(helix_tensor_demo)

add_executable(helix_cpu_kernels_demo cpu_kernels_demo.cpp)
target_link_libraries(helix_cpu_kernels_demo PRIVATE helix::cpu_kernels)
helix_configure_target(helix_cpu_kernels_demo)
