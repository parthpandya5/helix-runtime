#pragma once

#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

namespace helix::cuda::kernels {

enum class SoftmaxAlgorithm {
  kNaive,
  kSharedMemory,
};

void softmax_out(const Tensor& input, Tensor& output, const Stream& stream,
                 SoftmaxAlgorithm algorithm = SoftmaxAlgorithm::kSharedMemory);
[[nodiscard]] Tensor softmax(const Tensor& input, const Stream& stream,
                             SoftmaxAlgorithm algorithm = SoftmaxAlgorithm::kSharedMemory);

} // namespace helix::cuda::kernels
