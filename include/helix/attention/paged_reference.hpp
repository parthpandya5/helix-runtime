#pragma once

#include "helix/kv_cache/paged.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>

namespace helix::attention::cpu {

[[nodiscard]] Shape
paged_score_workspace_shape(const Tensor& query,
                            const kv_cache::PagedKvCache::LayerReadView& kv_layer);
[[nodiscard]] std::size_t
paged_score_workspace_bytes(const Tensor& query,
                            const kv_cache::PagedKvCache::LayerReadView& kv_layer);

void paged_scaled_dot_product_attention_out(const Tensor& query,
                                            const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                            std::size_t query_position_offset, Tensor& output,
                                            Tensor& score_workspace);
[[nodiscard]] Tensor
paged_scaled_dot_product_attention(const Tensor& query,
                                   const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                   std::size_t query_position_offset = 0);

} // namespace helix::attention::cpu
