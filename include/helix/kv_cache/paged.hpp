#pragma once

#include "helix/kv_cache/layer.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace helix::kv_cache {

class PhysicalBlockId final {
public:
  PhysicalBlockId() = default;
  explicit PhysicalBlockId(std::uint32_t value) noexcept : value_(value) {}

  [[nodiscard]] bool valid() const noexcept {
    return value_ != std::numeric_limits<std::uint32_t>::max();
  }
  [[nodiscard]] std::uint32_t value() const noexcept {
    return value_;
  }

  friend bool operator==(const PhysicalBlockId&, const PhysicalBlockId&) = default;

private:
  std::uint32_t value_ = std::numeric_limits<std::uint32_t>::max();
};

class PagedSequenceHandle final {
public:
  PagedSequenceHandle() = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return sequence_id_ != 0;
  }
  [[nodiscard]] std::uint64_t id() const noexcept {
    return sequence_id_;
  }
  [[nodiscard]] std::uint64_t cache_id() const noexcept {
    return cache_id_;
  }

  friend bool operator==(const PagedSequenceHandle&, const PagedSequenceHandle&) = default;

private:
  friend class PagedKvCache;

  PagedSequenceHandle(std::uint64_t sequence_id, std::uint64_t cache_id) noexcept
      : sequence_id_(sequence_id), cache_id_(cache_id) {}

  std::uint64_t sequence_id_ = 0;
  std::uint64_t cache_id_ = 0;
};

struct PagedKvCacheConfig {
  std::size_t layers = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dimension = 0;
  std::size_t block_tokens = 0;
  std::size_t physical_blocks = 0;
};

struct PagedKvCacheStats {
  std::size_t bytes_reserved = 0;
  std::size_t bytes_used = 0;
  std::size_t bytes_free = 0;
  std::size_t peak_bytes_used = 0;
  std::size_t total_blocks = 0;
  std::size_t used_blocks = 0;
  std::size_t free_blocks = 0;
  std::size_t active_sequence_count = 0;
  std::size_t active_tokens = 0;
  std::size_t allocated_token_slots = 0;
  std::uint64_t sequence_create_count = 0;
  std::uint64_t append_count = 0;
  std::uint64_t release_count = 0;
  std::uint64_t block_allocation_count = 0;
  std::uint64_t block_reuse_count = 0;

  [[nodiscard]] std::size_t unused_token_slots() const noexcept {
    return allocated_token_slots - active_tokens;
  }

  [[nodiscard]] double token_slot_utilization() const noexcept {
    if (allocated_token_slots == 0) {
      return 0.0;
    }
    return static_cast<double>(active_tokens) / static_cast<double>(allocated_token_slots);
  }
};

class PagedKvCache final {
public:
  explicit PagedKvCache(PagedKvCacheConfig config);
  ~PagedKvCache();

  PagedKvCache(const PagedKvCache&) = delete;
  PagedKvCache& operator=(const PagedKvCache&) = delete;
  PagedKvCache(PagedKvCache&&) = delete;
  PagedKvCache& operator=(PagedKvCache&&) = delete;

  [[nodiscard]] PagedSequenceHandle create_sequence();
  void append(const PagedSequenceHandle& sequence, const Tensor& token_keys,
              const Tensor& token_values);
  [[nodiscard]] KvLayer retrieve_layer(const PagedSequenceHandle& sequence,
                                       std::size_t layer) const;
  void copy_layer_out(const PagedSequenceHandle& sequence, std::size_t layer, Tensor& keys,
                      Tensor& values) const;
  void release_sequence(const PagedSequenceHandle& sequence);

  [[nodiscard]] bool contains(const PagedSequenceHandle& sequence) const;
  [[nodiscard]] std::size_t length(const PagedSequenceHandle& sequence) const;
  [[nodiscard]] std::vector<PhysicalBlockId> block_table(const PagedSequenceHandle& sequence) const;
  [[nodiscard]] std::size_t bytes_per_token() const noexcept;
  [[nodiscard]] std::size_t block_bytes() const noexcept;
  [[nodiscard]] std::size_t token_capacity() const noexcept;
  [[nodiscard]] PagedKvCacheStats stats() const;
  [[nodiscard]] const PagedKvCacheConfig& config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace helix::kv_cache
