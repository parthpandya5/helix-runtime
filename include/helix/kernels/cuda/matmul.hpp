#pragma once

#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

namespace helix::cuda::kernels {

enum class MatmulAlgorithm {
  kNaive,
  kTiled,
};

void matmul_out(const Tensor& lhs, const Tensor& rhs, Tensor& output, const Stream& stream,
                MatmulAlgorithm algorithm = MatmulAlgorithm::kTiled);
[[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs, const Stream& stream,
                            MatmulAlgorithm algorithm = MatmulAlgorithm::kTiled);

} // namespace helix::cuda::kernels
