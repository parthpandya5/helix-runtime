#pragma once

#include "helix/tensor/tensor.hpp"

namespace helix::cpu {

void add_out(const Tensor& lhs, const Tensor& rhs, Tensor& output);
[[nodiscard]] Tensor add(const Tensor& lhs, const Tensor& rhs);

void multiply_out(const Tensor& lhs, const Tensor& rhs, Tensor& output);
[[nodiscard]] Tensor multiply(const Tensor& lhs, const Tensor& rhs);

void relu_out(const Tensor& input, Tensor& output);
[[nodiscard]] Tensor relu(const Tensor& input);

void silu_out(const Tensor& input, Tensor& output);
[[nodiscard]] Tensor silu(const Tensor& input);

void sum_out(const Tensor& input, Tensor& output);
[[nodiscard]] Tensor sum(const Tensor& input);

void sum_last_dim_out(const Tensor& input, Tensor& output, bool keep_dim = false);
[[nodiscard]] Tensor sum_last_dim(const Tensor& input, bool keep_dim = false);

void softmax_out(const Tensor& input, Tensor& output);
[[nodiscard]] Tensor softmax(const Tensor& input);

void matmul_out(const Tensor& lhs, const Tensor& rhs, Tensor& output);
[[nodiscard]] Tensor matmul(const Tensor& lhs, const Tensor& rhs);

void rms_norm_out(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output);
[[nodiscard]] Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon);

void swiglu_out(const Tensor& gate, const Tensor& up, Tensor& output);
[[nodiscard]] Tensor swiglu(const Tensor& gate, const Tensor& up);

} // namespace helix::cpu
