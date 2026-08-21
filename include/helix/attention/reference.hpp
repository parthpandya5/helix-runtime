#pragma once

#include "helix/tensor/tensor.hpp"

#include <cstddef>

namespace helix::attention::cpu {

[[nodiscard]] Shape score_workspace_shape(const Tensor& query, const Tensor& key);
[[nodiscard]] std::size_t score_workspace_bytes(const Tensor& query, const Tensor& key);

void scaled_dot_product_attention_out(const Tensor& query, const Tensor& key, const Tensor& value,
                                      std::size_t query_position_offset, Tensor& output,
                                      Tensor& score_workspace);

// Retained as the original measurement baseline.  It computes the same result
// but traverses all KV rows once per output dimension.
void scaled_dot_product_attention_dimension_major_out(const Tensor& query, const Tensor& key,
                                                      const Tensor& value,
                                                      std::size_t query_position_offset,
                                                      Tensor& output, Tensor& score_workspace);

[[nodiscard]] Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key,
                                                  const Tensor& value,
                                                  std::size_t query_position_offset = 0);

} // namespace helix::attention::cpu
