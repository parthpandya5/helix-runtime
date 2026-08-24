#pragma once

#include "helix/runtime/cuda/memory.hpp"
#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace helix::attention::cuda {

// Immutable device copy of one sequence's logical-to-physical page mapping.
// Construction validates every physical ID before synchronously uploading it.
// Duplicate IDs are valid because attention only reads the referenced pages.
// Callers must retain the table, tensors, and stream until enqueued attention
// work completes.
class PagedBlockTable final {
public:
  PagedBlockTable(std::span<const std::uint32_t> physical_blocks, std::size_t physical_block_count,
                  int device_index);
  ~PagedBlockTable() = default;

  PagedBlockTable(const PagedBlockTable&) = delete;
  PagedBlockTable& operator=(const PagedBlockTable&) = delete;
  PagedBlockTable(PagedBlockTable&& other) noexcept;
  PagedBlockTable& operator=(PagedBlockTable&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const std::uint32_t* data() const noexcept;
  [[nodiscard]] std::size_t logical_block_count() const noexcept {
    return logical_block_count_;
  }
  [[nodiscard]] std::size_t physical_block_count() const noexcept {
    return physical_block_count_;
  }
  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return logical_block_count_ * sizeof(std::uint32_t);
  }
  [[nodiscard]] int device_index() const noexcept {
    return device_index_;
  }

private:
  std::optional<helix::cuda::Buffer> device_blocks_;
  std::size_t logical_block_count_ = 0;
  std::size_t physical_block_count_ = 0;
  int device_index_ = -1;
};

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

void paged_scaled_dot_product_attention_out(
    const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
    const PagedBlockTable& block_table, std::size_t kv_tokens, std::size_t query_position_offset,
    Tensor& output, Tensor& score_workspace, const helix::cuda::Stream& stream);

// Allocates private score scratch and therefore synchronizes stream before
// returning so that scratch remains alive. Use the _out form for asynchronous
// execution with caller-managed output and workspace lifetime.
[[nodiscard]] Tensor
paged_scaled_dot_product_attention(const Tensor& query, const Tensor& key_blocks,
                                   const Tensor& value_blocks, const PagedBlockTable& block_table,
                                   std::size_t kv_tokens, std::size_t query_position_offset,
                                   const helix::cuda::Stream& stream);

} // namespace helix::attention::cuda
