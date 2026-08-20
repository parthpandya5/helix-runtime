#pragma once

#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

namespace helix::cuda::kernels {

void add_out(const Tensor& lhs, const Tensor& rhs, Tensor& output, const Stream& stream);
[[nodiscard]] Tensor add(const Tensor& lhs, const Tensor& rhs, const Stream& stream);

void multiply_out(const Tensor& lhs, const Tensor& rhs, Tensor& output, const Stream& stream);
[[nodiscard]] Tensor multiply(const Tensor& lhs, const Tensor& rhs, const Stream& stream);

void relu_out(const Tensor& input, Tensor& output, const Stream& stream);
[[nodiscard]] Tensor relu(const Tensor& input, const Stream& stream);

void silu_out(const Tensor& input, Tensor& output, const Stream& stream);
[[nodiscard]] Tensor silu(const Tensor& input, const Stream& stream);

} // namespace helix::cuda::kernels
