#include "helix/kernels/cpu/reference.hpp"

#include <array>
#include <cstddef>
#include <iostream>

namespace {

void print_tensor(const helix::Tensor& tensor) {
  std::cout << helix::shape_string(tensor.shape()) << ':';
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    std::cout << ' ' << tensor.data_as<float>()[index];
  }
  std::cout << '\n';
}

} // namespace

int main() {
  const helix::Tensor lhs =
      helix::Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const helix::Tensor rhs =
      helix::Tensor::from_values({3, 2}, std::array{7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

  const helix::Tensor product = helix::cpu::matmul(lhs, rhs);
  const helix::Tensor probabilities = helix::cpu::softmax(product);

  std::cout << "matmul ";
  print_tensor(product);
  std::cout << "row-wise softmax ";
  print_tensor(probabilities);
  return 0;
}
