#pragma once

#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

namespace helix::cuda::kernels {

enum class ReductionAlgorithm {
  kNaive,
  kSharedMemory,
  kWarpShuffle,
};

void sum_last_dim_out(const Tensor& input, Tensor& output, const Stream& stream,
                      bool keep_dim = false,
                      ReductionAlgorithm algorithm = ReductionAlgorithm::kWarpShuffle);
[[nodiscard]] Tensor sum_last_dim(const Tensor& input, const Stream& stream, bool keep_dim = false,
                                  ReductionAlgorithm algorithm = ReductionAlgorithm::kWarpShuffle);

} // namespace helix::cuda::kernels
