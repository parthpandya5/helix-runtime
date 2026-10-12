#pragma once

#include "helix/kv_cache/cuda_block_table.hpp"
#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>

namespace helix::attention::cuda {

// Backward-compatible attention spelling; ownership belongs to the lower KV
// layer so cache implementations never depend upward on attention.
using PagedBlockTable = helix::kv_cache::cuda::PagedBlockTable;

// query:        [heads, query_tokens, key_head_dimension]
// key_blocks:   [physical_blocks, heads, block_tokens, key_head_dimension]
// value_blocks: [physical_blocks, heads, block_tokens, value_head_dimension]
// workspace:    [heads, query_tokens, kv_tokens]
// output:       [heads, query_tokens, value_head_dimension]
[[nodiscard]] Shape paged_score_workspace_shape(const Tensor& query, const Tensor& key_blocks,
                                                const PagedBlockTable& block_table,
                                                std::size_t kv_tokens);
[[nodiscard]] std::size_t paged_score_workspace_bytes(const Tensor& query, const Tensor& key_blocks,
                                                      const PagedBlockTable& block_table,
                                                      std::size_t kv_tokens);

// Extends immutable packed history with an uncommitted contiguous tail.
// query and tail_key are [heads, tail_tokens, key_head_dimension]; tail_value
// is [heads, tail_tokens, value_head_dimension]. Query row q attends to all
// history tokens and tail rows through q. This lets a caller commit K/V only
// after the complete model step succeeds.
[[nodiscard]] Shape paged_tail_score_workspace_shape(const Tensor& query, const Tensor& key_blocks,
                                                     const Tensor& value_blocks,
                                                     const PagedBlockTable& block_table,
                                                     std::size_t history_tokens,
                                                     const Tensor& tail_key,
                                                     const Tensor& tail_value);
[[nodiscard]] std::size_t
paged_tail_score_workspace_bytes(const Tensor& query, const Tensor& key_blocks,
                                 const Tensor& value_blocks, const PagedBlockTable& block_table,
                                 std::size_t history_tokens, const Tensor& tail_key,
                                 const Tensor& tail_value);

void paged_scaled_dot_product_attention_out(
    const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
    const PagedBlockTable& block_table, std::size_t kv_tokens, std::size_t query_position_offset,
    Tensor& output, Tensor& score_workspace, const helix::cuda::Stream& stream);

void paged_scaled_dot_product_attention_with_tail_out(
    const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
    const PagedBlockTable& block_table, std::size_t history_tokens, const Tensor& tail_key,
    const Tensor& tail_value, Tensor& output, Tensor& score_workspace,
    const helix::cuda::Stream& stream);

// Allocates private score scratch and therefore synchronizes stream before
// returning so that scratch remains alive. Use the _out form for asynchronous
// execution with caller-managed output and workspace lifetime.
[[nodiscard]] Tensor
paged_scaled_dot_product_attention(const Tensor& query, const Tensor& key_blocks,
                                   const Tensor& value_blocks, const PagedBlockTable& block_table,
                                   std::size_t kv_tokens, std::size_t query_position_offset,
                                   const helix::cuda::Stream& stream);

// Allocates private score scratch and synchronizes before returning. The
// returned output remains device-resident.
[[nodiscard]] Tensor paged_scaled_dot_product_attention_with_tail(
    const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
    const PagedBlockTable& block_table, std::size_t history_tokens, const Tensor& tail_key,
    const Tensor& tail_value, const helix::cuda::Stream& stream);

} // namespace helix::attention::cuda
