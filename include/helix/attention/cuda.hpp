#pragma once

#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>

namespace helix::attention::cuda {

[[nodiscard]] Shape score_workspace_shape(const Tensor& query, const Tensor& key);
[[nodiscard]] std::size_t score_workspace_bytes(const Tensor& query, const Tensor& key);

void scaled_dot_product_attention_out(const Tensor& query, const Tensor& key, const Tensor& value,
                                      std::size_t query_position_offset, Tensor& output,
                                      Tensor& score_workspace, const helix::cuda::Stream& stream);
[[nodiscard]] Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key,
                                                  const Tensor& value,
                                                  std::size_t query_position_offset,
                                                  const helix::cuda::Stream& stream);

} // namespace helix::attention::cuda
