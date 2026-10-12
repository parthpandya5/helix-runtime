#pragma once

#include "helix/kv_cache/cuda_block_table.hpp"
#include "helix/kv_cache/paged.hpp"
#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>

namespace helix::kv_cache {

class CudaPagedSequenceHandle final {
public:
  CudaPagedSequenceHandle() = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return sequence_id_ != 0;
  }
  [[nodiscard]] std::uint64_t id() const noexcept {
    return sequence_id_;
  }
  [[nodiscard]] std::uint64_t cache_id() const noexcept {
    return cache_id_;
  }

  friend bool operator==(const CudaPagedSequenceHandle&, const CudaPagedSequenceHandle&) = default;

private:
  friend class CudaPagedKvCache;

  CudaPagedSequenceHandle(std::uint64_t sequence_id, std::uint64_t cache_id) noexcept
      : sequence_id_(sequence_id), cache_id_(cache_id) {}

  std::uint64_t sequence_id_ = 0;
  std::uint64_t cache_id_ = 0;
};

// GPU-resident paged K/V ownership for synchronous model execution. Each layer
// owns key and value tensors with physical layout
// [physical_blocks, kv_heads, block_tokens, head_dimension].
class CudaPagedKvCache final {
  struct SequenceState;
  struct Impl;

public:
  // Pins the pool and sequence mapping under the cache mutex. The exposed
  // device tensors and table are borrowed: every asynchronous consumer must be
  // synchronized before this view is destroyed. Successful CudaLlamaModel paged
  // decode satisfies that contract; a caller with uncertain exceptional
  // completion must poison the view before destruction.
  class DecodeView final {
  public:
    ~DecodeView();

    DecodeView(const DecodeView&) = delete;
    DecodeView& operator=(const DecodeView&) = delete;
    DecodeView(DecodeView&&) noexcept;
    DecodeView& operator=(DecodeView&&) noexcept;

    [[nodiscard]] std::size_t length() const;
    [[nodiscard]] std::span<const std::uint32_t> physical_blocks() const;
    [[nodiscard]] std::span<const Tensor> layer_key_blocks() const;
    [[nodiscard]] std::span<const Tensor> layer_value_blocks() const;
    [[nodiscard]] const cuda::PagedBlockTable& block_table() const;

    // Conservatively poisons the cache and transfers the device table into its
    // quarantine when a consumer throws without a trustworthy completion
    // boundary. The cache remains inspectable/releasable but cannot admit,
    // append, or lend pages again. Calling this on an empty view is a no-op.
    void poison_after_consumer_failure() noexcept;

  private:
    friend class CudaPagedKvCache;

    DecodeView(std::shared_ptr<Impl> cache, std::unique_lock<std::mutex> lock,
               const SequenceState* sequence, cuda::PagedBlockTable block_table) noexcept;
    void require_valid(std::string_view operation) const;

    std::shared_ptr<Impl> cache_;
    std::unique_lock<std::mutex> lock_;
    const SequenceState* sequence_ = nullptr;
    std::optional<cuda::PagedBlockTable> block_table_;
  };

  explicit CudaPagedKvCache(PagedKvCacheConfig config, int device_index = 0);
  ~CudaPagedKvCache();

  CudaPagedKvCache(const CudaPagedKvCache&) = delete;
  CudaPagedKvCache& operator=(const CudaPagedKvCache&) = delete;
  CudaPagedKvCache(CudaPagedKvCache&&) = delete;
  CudaPagedKvCache& operator=(CudaPagedKvCache&&) = delete;

  [[nodiscard]] CudaPagedSequenceHandle create_sequence();
  // Copies one aggregate [layers, kv_heads, head_dimension] key/value token on
  // stream and synchronizes it before committing sequence metadata. If a copy
  // or synchronization fails, the logical length and block table do not move.
  void append(const CudaPagedSequenceHandle& sequence, const Tensor& token_keys,
              const Tensor& token_values, const helix::cuda::Stream& stream);
  void release_sequence(const CudaPagedSequenceHandle& sequence);

  [[nodiscard]] DecodeView read_sequence(const CudaPagedSequenceHandle& sequence) const;
  [[nodiscard]] bool is_healthy() const;
  [[nodiscard]] bool contains(const CudaPagedSequenceHandle& sequence) const;
  [[nodiscard]] std::size_t length(const CudaPagedSequenceHandle& sequence) const;
  [[nodiscard]] PagedKvCacheStats stats() const;
  [[nodiscard]] std::size_t bytes_per_token() const noexcept;
  [[nodiscard]] std::size_t block_bytes() const noexcept;
  [[nodiscard]] std::size_t token_capacity() const noexcept;
  [[nodiscard]] int device_index() const noexcept;
  [[nodiscard]] const PagedKvCacheConfig& config() const noexcept;

private:
  std::shared_ptr<Impl> impl_;
};

} // namespace helix::kv_cache
