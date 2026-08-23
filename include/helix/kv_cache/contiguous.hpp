#pragma once

#include "helix/kv_cache/layer.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace helix::kv_cache {

struct ContiguousKvCacheConfig {
  std::size_t layers = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dimension = 0;
  std::size_t maximum_bytes = std::numeric_limits<std::size_t>::max();
};

class SequenceHandle final {
public:
  SequenceHandle() = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return sequence_id_ != 0;
  }
  [[nodiscard]] std::uint64_t id() const noexcept {
    return sequence_id_;
  }
  [[nodiscard]] std::uint64_t cache_id() const noexcept {
    return cache_id_;
  }

  friend bool operator==(const SequenceHandle&, const SequenceHandle&) = default;

private:
  friend class ContiguousKvCache;

  SequenceHandle(std::uint64_t sequence_id, std::uint64_t cache_id) noexcept
      : sequence_id_(sequence_id), cache_id_(cache_id) {}

  std::uint64_t sequence_id_ = 0;
  std::uint64_t cache_id_ = 0;
};

struct ContiguousKvCacheStats {
  std::size_t bytes_allocated = 0;
  std::size_t peak_bytes_allocated = 0;
  std::size_t active_sequence_count = 0;
  std::size_t active_capacity_tokens = 0;
  std::size_t active_tokens = 0;
  std::uint64_t sequence_create_count = 0;
  std::uint64_t append_count = 0;
  std::uint64_t release_count = 0;
};

class ContiguousKvCache final {
public:
  explicit ContiguousKvCache(ContiguousKvCacheConfig config);
  ~ContiguousKvCache();

  ContiguousKvCache(const ContiguousKvCache&) = delete;
  ContiguousKvCache& operator=(const ContiguousKvCache&) = delete;
  ContiguousKvCache(ContiguousKvCache&&) = delete;
  ContiguousKvCache& operator=(ContiguousKvCache&&) = delete;

  [[nodiscard]] SequenceHandle create_sequence(std::size_t capacity_tokens);
  void append(const SequenceHandle& sequence, const Tensor& token_keys, const Tensor& token_values);
  [[nodiscard]] KvLayer retrieve_layer(const SequenceHandle& sequence, std::size_t layer) const;
  void copy_layer_out(const SequenceHandle& sequence, std::size_t layer, Tensor& keys,
                      Tensor& values) const;
  // Copies the current cached history into the prefix of caller-owned
  // [kv_heads, destination_tokens, head_dimension] tensors. The destination
  // may reserve trailing token rows, which are left untouched. Requiring the
  // expected length prevents a caller from silently consuming a stale snapshot
  // if the sequence changes before the copy acquires the cache lock.
  void copy_layer_prefix_out(const SequenceHandle& sequence, std::size_t layer,
                             std::size_t expected_length, Tensor& keys, Tensor& values) const;
  void release_sequence(const SequenceHandle& sequence);

  [[nodiscard]] bool contains(const SequenceHandle& sequence) const;
  [[nodiscard]] std::size_t length(const SequenceHandle& sequence) const;
  [[nodiscard]] std::size_t capacity(const SequenceHandle& sequence) const;
  [[nodiscard]] std::size_t bytes_per_token() const noexcept;
  [[nodiscard]] std::size_t sequence_bytes(std::size_t capacity_tokens) const;
  [[nodiscard]] ContiguousKvCacheStats stats() const;
  [[nodiscard]] const ContiguousKvCacheConfig& config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace helix::kv_cache
