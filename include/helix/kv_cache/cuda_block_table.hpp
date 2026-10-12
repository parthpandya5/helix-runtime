#pragma once

#include "helix/runtime/cuda/memory.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace helix::kv_cache::cuda {

// Immutable device copy of one sequence's logical-to-physical page mapping.
// Construction validates every physical ID before synchronously uploading it.
// An empty mapping is valid and owns no device allocation, allowing position-
// zero decode. Duplicate IDs are valid because consumers only read pages.
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

} // namespace helix::kv_cache::cuda
