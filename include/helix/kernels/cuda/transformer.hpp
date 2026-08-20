#pragma once

#include "helix/kernels/cuda/matmul.hpp"
#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

namespace helix::cuda::kernels {

void rms_norm_out(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output,
                  const Stream& stream);
[[nodiscard]] Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon,
                              const Stream& stream);

void swiglu_out(const Tensor& gate, const Tensor& up, Tensor& output, const Stream& stream);
[[nodiscard]] Tensor swiglu(const Tensor& gate, const Tensor& up, const Stream& stream);

// Applies adjacent-pair rotary embedding to the complete last dimension. Input is
// [tokens, ..., head_dimension]; cosine/sine are [tokens, head_dimension / 2].
void rope_out(const Tensor& input, const Tensor& cosine, const Tensor& sine, Tensor& output,
              const Stream& stream);
[[nodiscard]] Tensor rope(const Tensor& input, const Tensor& cosine, const Tensor& sine,
                          const Stream& stream);

// Weight layout is [input_features, output_features]; bias is intentionally absent.
void linear_out(const Tensor& input, const Tensor& weight, Tensor& output, const Stream& stream,
                MatmulAlgorithm algorithm = MatmulAlgorithm::kTiled);
[[nodiscard]] Tensor linear(const Tensor& input, const Tensor& weight, const Stream& stream,
                            MatmulAlgorithm algorithm = MatmulAlgorithm::kTiled);

} // namespace helix::cuda::kernels
