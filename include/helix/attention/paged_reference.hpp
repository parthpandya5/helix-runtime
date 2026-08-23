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

// Extends an immutable paged history with an uncached contiguous tail. Query,
// tail_key, and tail_value use [heads, tail_tokens, head_dimension]; each query
// position attends to all history plus its causal prefix of the tail. This is the
// decode/continuation path used before the caller commits the tail to the cache.
[[nodiscard]] Shape
paged_tail_score_workspace_shape(const Tensor& query,
                                 const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                 const Tensor& tail_key, const Tensor& tail_value);
[[nodiscard]] std::size_t
paged_tail_score_workspace_bytes(const Tensor& query,
                                 const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                 const Tensor& tail_key, const Tensor& tail_value);

void paged_scaled_dot_product_attention_out(const Tensor& query,
                                            const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                            std::size_t query_position_offset, Tensor& output,
                                            Tensor& score_workspace);

void paged_scaled_dot_product_attention_with_tail_out(
    const Tensor& query, const kv_cache::PagedKvCache::LayerReadView& kv_layer,
    const Tensor& tail_key, const Tensor& tail_value, Tensor& output, Tensor& score_workspace);

// Retained as a measurement baseline for the optimized value traversal.  It
// computes the same result but walks every physical page once per output
// dimension and is not intended for normal runtime use.
void paged_scaled_dot_product_attention_dimension_major_out(
    const Tensor& query, const kv_cache::PagedKvCache::LayerReadView& kv_layer,
    std::size_t query_position_offset, Tensor& output, Tensor& score_workspace);

[[nodiscard]] Tensor
paged_scaled_dot_product_attention(const Tensor& query,
                                   const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                   std::size_t query_position_offset = 0);

[[nodiscard]] Tensor
paged_scaled_dot_product_attention_with_tail(const Tensor& query,
                                             const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                             const Tensor& tail_key, const Tensor& tail_value);

} // namespace helix::attention::cpu
