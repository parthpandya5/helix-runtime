#pragma once

#include "helix/kv_cache/paged.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>

namespace helix::kv_cache {

class PrefixNamespaceId final {
public:
  PrefixNamespaceId() = default;
  explicit PrefixNamespaceId(std::uint64_t value) noexcept : value_(value) {}

  [[nodiscard]] bool valid() const noexcept {
    return value_ != 0;
  }
  [[nodiscard]] std::uint64_t value() const noexcept {
    return value_;
  }

  auto operator<=>(const PrefixNamespaceId&) const = default;

private:
  std::uint64_t value_ = 0;
};

struct PagedPrefixCacheConfig {
  std::size_t maximum_entries = 0;
  std::size_t maximum_block_references = 0;
};

enum class PrefixInsertStatus : std::uint8_t {
  kInserted,
  kAlreadyPresent,
  kRejectedTooLarge,
};

struct PagedPrefixMatch {
  // The caller owns this logical cache sequence and must eventually pass it to
  // PagedKvCache::release_sequence.
  PagedSequenceHandle sequence;
  std::size_t matched_tokens = 0;
};

struct PagedPrefixCacheStats {
  std::size_t maximum_entries = 0;
  std::size_t maximum_block_references = 0;
  std::size_t entry_count = 0;
  std::size_t retained_tokens = 0;
  std::size_t retained_block_references = 0;
  std::size_t peak_entries = 0;
  std::size_t peak_block_references = 0;
  std::uint64_t lookup_count = 0;
  std::uint64_t hit_count = 0;
  std::uint64_t miss_count = 0;
  std::uint64_t insertion_count = 0;
  std::uint64_t duplicate_insertion_count = 0;
  std::uint64_t rejected_insertion_count = 0;
  std::uint64_t eviction_count = 0;
  std::uint64_t erase_count = 0;
};

// Thread-safe bounded metadata index over persistent PagedKvCache anchor
// sequences. Keys contain exact token IDs plus a namespace, so equality never
// depends on a digest alone. Longest lookup hashes each block-aligned prefix of
// the query, probes an indexed candidate set, and verifies the complete token
// vector. Entries remain in most-recently-used order for bounded eviction.
//
// Public methods acquire the prefix-cache mutex before the paged-cache mutex.
// Callers must not invoke them while holding a LayerReadView from the same pool.
class PagedPrefixCache final {
public:
  PagedPrefixCache(std::shared_ptr<PagedKvCache> cache, PagedPrefixCacheConfig config);
  ~PagedPrefixCache();

  PagedPrefixCache(const PagedPrefixCache&) = delete;
  PagedPrefixCache& operator=(const PagedPrefixCache&) = delete;
  PagedPrefixCache(PagedPrefixCache&&) = delete;
  PagedPrefixCache& operator=(PagedPrefixCache&&) = delete;

  // The caller guarantees that source contains the K/V state for the exact
  // namespace/tokens pair. Only positive, block-aligned prefixes are accepted.
  [[nodiscard]] PrefixInsertStatus insert(PrefixNamespaceId namespace_id,
                                          std::span<const std::uint32_t> tokens,
                                          const PagedSequenceHandle& source);

  // Returns the longest stored block-aligned prefix no longer than
  // maximum_prefix_tokens. A hit forks the retained anchor; the caller must
  // release the returned sequence.
  [[nodiscard]] std::optional<PagedPrefixMatch>
  lookup_longest(PrefixNamespaceId namespace_id, std::span<const std::uint32_t> tokens,
                 std::size_t maximum_prefix_tokens = std::numeric_limits<std::size_t>::max());

  [[nodiscard]] bool contains(PrefixNamespaceId namespace_id,
                              std::span<const std::uint32_t> tokens) const;
  [[nodiscard]] bool erase(PrefixNamespaceId namespace_id, std::span<const std::uint32_t> tokens);
  [[nodiscard]] bool evict_lru();
  void clear();

  [[nodiscard]] PagedPrefixCacheStats stats() const;
  [[nodiscard]] const PagedPrefixCacheConfig& config() const noexcept;
  [[nodiscard]] std::size_t block_tokens() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace helix::kv_cache
