#include "helix/core/build_info.hpp"
#include "helix/core/device.hpp"
#include "helix/core/dtype.hpp"

#include <iostream>

int main() {
  const auto info = helix::build_info();
  std::cout << "Helix " << info.version << '\n';
  std::cout << "CUDA backend compiled: " << (info.cuda_enabled ? "yes" : "no") << '\n';
  std::cout << "Default device: " << helix::Device::cpu() << '\n';
  std::cout << "Initial executable dtype: " << helix::dtype_name(helix::DType::kFloat32) << '\n';
  return 0;
}
