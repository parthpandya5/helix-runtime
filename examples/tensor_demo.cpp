#include "helix/tensor/tensor.hpp"

#include <iostream>

int main() {
  helix::Tensor tensor = helix::Tensor::random_uniform({2, 3}, 42);
  const helix::Tensor matrix_view = tensor.reshape({3, 2});

  std::cout << "Tensor shape: " << helix::shape_string(tensor.shape()) << '\n';
  std::cout << "View shape:   " << helix::shape_string(matrix_view.shape()) << '\n';
  std::cout << "Storage is shared: " << (tensor.data() == matrix_view.data() ? "yes" : "no")
            << '\n';
  std::cout << "Values:";
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    std::cout << ' ' << tensor.data_as<float>()[index];
  }
  std::cout << '\n';
  return 0;
}
