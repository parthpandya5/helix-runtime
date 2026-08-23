#include "helix/kv_cache/prefix.hpp"

#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace helix::kv_cache {
namespace {

constexpr std::string_view kTypeName = "PagedPrefixCache";

void require_namespace(PrefixNamespaceId namespace_id, std::string_view operation) {
  HELIX_CHECK(namespace_id.valid(), ErrorCode::kInvalidArgument,
              std::string(kTypeName) + "::" + std::string(operation) +
                  ": prefix namespace zero is reserved as invalid");
}

void increment_saturating(std::uint64_t& counter) noexcept {
  if (counter != std::numeric_limits<std::uint64_t>::max()) {
    ++counter;
  }
}

bool token_vectors_equal(const std::vector<std::uint32_t>& stored,
                         std::span<const std::uint32_t> candidate) noexcept {
  return stored.size() == candidate.size() &&
         std::equal(stored.begin(), stored.end(), candidate.begin());
}

bool token_vector_is_prefix(const std::vector<std::uint32_t>& stored,
                            std::span<const std::uint32_t> candidate) noexcept {
  return stored.size() <= candidate.size() &&
         std::equal(stored.begin(), stored.end(), candidate.begin());
}

} // namespace

struct PagedPrefixCache::Impl {
  struct Entry {
    PrefixNamespaceId namespace_id;
    std::vector<std::uint32_t> tokens;
    PagedSequenceHandle anchor;
    std::size_t block_references = 0;
  };

  using EntryList = std::list<Entry>;
  using Iterator = EntryList::iterator;
  using ConstIterator = EntryList::const_iterator;

  Impl(std::shared_ptr<PagedKvCache> owned_cache, PagedPrefixCacheConfig prefix_config)
      : cache(std::move(owned_cache)), config(prefix_config) {
    HELIX_CHECK(cache != nullptr, ErrorCode::kInvalidArgument,
                "PagedPrefixCache: paged cache must not be null");
    HELIX_CHECK(config.maximum_entries != 0, ErrorCode::kInvalidArgument,
                "PagedPrefixCache: maximum entry count must be positive");
    HELIX_CHECK(config.maximum_block_references != 0, ErrorCode::kInvalidArgument,
                "PagedPrefixCache: maximum block-reference count must be positive");
    const std::size_t page_tokens = cache->config().block_tokens;
    HELIX_CHECK(
        config.maximum_block_references <= std::numeric_limits<std::size_t>::max() / page_tokens,
        ErrorCode::kOutOfRange, "PagedPrefixCache: retained token accounting would overflow");
    cache_stats.maximum_entries = config.maximum_entries;
    cache_stats.maximum_block_references = config.maximum_block_references;
    validate_invariants();
  }

  [[nodiscard]] Iterator find_exact(PrefixNamespaceId namespace_id,
                                    std::span<const std::uint32_t> tokens) noexcept {
    return std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) {
      return entry.namespace_id == namespace_id && token_vectors_equal(entry.tokens, tokens);
    });
  }

  [[nodiscard]] ConstIterator find_exact(PrefixNamespaceId namespace_id,
                                         std::span<const std::uint32_t> tokens) const noexcept {
    return std::find_if(entries.cbegin(), entries.cend(), [&](const Entry& entry) {
      return entry.namespace_id == namespace_id && token_vectors_equal(entry.tokens, tokens);
    });
  }

  void validate_key(PrefixNamespaceId namespace_id, std::span<const std::uint32_t> tokens,
                    std::string_view operation) const {
    require_namespace(namespace_id, operation);
    HELIX_CHECK(!tokens.empty(), ErrorCode::kInvalidArgument,
                std::string(kTypeName) + "::" + std::string(operation) +
                    ": prefix must contain at least one token");
    HELIX_CHECK(tokens.size() % cache->config().block_tokens == 0, ErrorCode::kInvalidArgument,
                std::string(kTypeName) + "::" + std::string(operation) +
                    ": prefix must end on a KV block boundary");
  }

  void erase_locked(Iterator position, bool eviction) {
    HELIX_ASSERT(position != entries.end(), "prefix cache attempted to erase its end iterator");
    cache->release_sequence(position->anchor);
    cache_stats.retained_tokens -= position->tokens.size();
    cache_stats.retained_block_references -= position->block_references;
    --cache_stats.entry_count;
    if (eviction) {
      increment_saturating(cache_stats.eviction_count);
    } else {
      increment_saturating(cache_stats.erase_count);
    }
    entries.erase(position);
  }

  void evict_for_insert_locked(std::size_t inserted_blocks) {
    HELIX_ASSERT(inserted_blocks <= config.maximum_block_references,
                 "prefix cache insertion exceeds its block-reference limit");
    while (entries.size() > config.maximum_entries ||
           cache_stats.retained_block_references >
               config.maximum_block_references - inserted_blocks) {
      HELIX_ASSERT(entries.size() > 1,
                   "prefix cache insertion has no older capacity-eviction candidate");
      erase_locked(std::prev(entries.end()), true);
    }
  }

  void validate_invariants() const {
#ifndef NDEBUG
    std::size_t retained_tokens = 0;
    std::size_t retained_blocks = 0;
    for (auto outer = entries.cbegin(); outer != entries.cend(); ++outer) {
      HELIX_ASSERT(outer->namespace_id.valid(), "prefix cache entry has an invalid namespace");
      HELIX_ASSERT(!outer->tokens.empty() &&
                       outer->tokens.size() % cache->config().block_tokens == 0,
                   "prefix cache entry is empty or not block aligned");
      HELIX_ASSERT(outer->block_references == outer->tokens.size() / cache->config().block_tokens,
                   "prefix cache entry block count disagrees with its token count");
      HELIX_ASSERT(cache->contains(outer->anchor), "prefix cache anchor is not live");
      HELIX_ASSERT(cache->length(outer->anchor) == outer->tokens.size(),
                   "prefix cache anchor length disagrees with its content key");
      HELIX_ASSERT(outer->tokens.size() <=
                       std::numeric_limits<std::size_t>::max() - retained_tokens,
                   "prefix cache retained-token invariant sum would overflow");
      retained_tokens += outer->tokens.size();
      HELIX_ASSERT(outer->block_references <=
                       std::numeric_limits<std::size_t>::max() - retained_blocks,
                   "prefix cache retained-block invariant sum would overflow");
      retained_blocks += outer->block_references;
      for (auto inner = std::next(outer); inner != entries.cend(); ++inner) {
        HELIX_ASSERT(outer->namespace_id != inner->namespace_id || outer->tokens != inner->tokens,
                     "prefix cache contains a duplicate exact key");
      }
    }
    HELIX_ASSERT(entries.size() <= config.maximum_entries, "prefix cache entry limit is exceeded");
    HELIX_ASSERT(retained_blocks <= config.maximum_block_references,
                 "prefix cache block-reference limit is exceeded");
    HELIX_ASSERT(cache_stats.entry_count == entries.size(),
                 "prefix cache entry statistic is inconsistent");
    HELIX_ASSERT(cache_stats.retained_tokens == retained_tokens,
                 "prefix cache retained-token statistic is inconsistent");
    HELIX_ASSERT(cache_stats.retained_block_references == retained_blocks,
                 "prefix cache retained-block statistic is inconsistent");
    HELIX_ASSERT(cache_stats.peak_entries >= cache_stats.entry_count,
                 "prefix cache peak-entry statistic is inconsistent");
    HELIX_ASSERT(cache_stats.peak_block_references >= cache_stats.retained_block_references,
                 "prefix cache peak-block statistic is inconsistent");
#endif
  }

  std::shared_ptr<PagedKvCache> cache;
  PagedPrefixCacheConfig config;
  mutable std::mutex mutex;
  EntryList entries;
  PagedPrefixCacheStats cache_stats;
};

PagedPrefixCache::PagedPrefixCache(std::shared_ptr<PagedKvCache> cache,
                                   PagedPrefixCacheConfig config) try
    : impl_(std::make_unique<Impl>(std::move(cache), config)) {
} catch (const std::bad_alloc&) {
  throw_error(ErrorCode::kResourceExhausted,
              "PagedPrefixCache construction could not allocate metadata");
}

PagedPrefixCache::~PagedPrefixCache() {
  try {
    clear();
  } catch (...) {
    try {
      HELIX_LOG_ERROR("PagedPrefixCache destructor could not release every retained anchor");
    } catch (...) {
      // Cleanup diagnostics are best-effort; destructors must not throw.
    }
  }
}

PrefixInsertStatus PagedPrefixCache::insert(PrefixNamespaceId namespace_id,
                                            std::span<const std::uint32_t> tokens,
                                            const PagedSequenceHandle& source) {
  constexpr std::string_view kOperation = "insert";
  std::scoped_lock lock(impl_->mutex);
  impl_->validate_key(namespace_id, tokens, kOperation);
  const std::size_t source_length = impl_->cache->length(source);
  HELIX_CHECK(tokens.size() <= source_length, ErrorCode::kOutOfRange,
              "PagedPrefixCache::insert prefix exceeds the source sequence length");

  const auto existing = impl_->find_exact(namespace_id, tokens);
  if (existing != impl_->entries.end()) {
    impl_->entries.splice(impl_->entries.begin(), impl_->entries, existing);
    increment_saturating(impl_->cache_stats.duplicate_insertion_count);
    impl_->validate_invariants();
    return PrefixInsertStatus::kAlreadyPresent;
  }

  const std::size_t block_references = tokens.size() / impl_->cache->config().block_tokens;
  if (block_references > impl_->config.maximum_block_references) {
    increment_saturating(impl_->cache_stats.rejected_insertion_count);
    impl_->validate_invariants();
    return PrefixInsertStatus::kRejectedTooLarge;
  }

  Impl::Entry staged;
  staged.namespace_id = namespace_id;
  staged.block_references = block_references;
  try {
    staged.tokens.assign(tokens.begin(), tokens.end());
  } catch (const std::bad_alloc&) {
    throw_error(ErrorCode::kResourceExhausted,
                "PagedPrefixCache::insert could not allocate the content key");
  }
  const PagedSequenceHandle anchor = impl_->cache->fork_sequence(source, tokens.size());
  staged.anchor = anchor;
  try {
    impl_->entries.push_front(std::move(staged));
  } catch (const std::bad_alloc&) {
    impl_->cache->release_sequence(anchor);
    throw_error(ErrorCode::kResourceExhausted,
                "PagedPrefixCache::insert could not allocate the LRU entry");
  }

  impl_->evict_for_insert_locked(block_references);
  ++impl_->cache_stats.entry_count;
  impl_->cache_stats.retained_tokens += tokens.size();
  impl_->cache_stats.retained_block_references += block_references;
  impl_->cache_stats.peak_entries =
      std::max(impl_->cache_stats.peak_entries, impl_->cache_stats.entry_count);
  impl_->cache_stats.peak_block_references = std::max(impl_->cache_stats.peak_block_references,
                                                      impl_->cache_stats.retained_block_references);
  increment_saturating(impl_->cache_stats.insertion_count);
  impl_->validate_invariants();
  return PrefixInsertStatus::kInserted;
}

std::optional<PagedPrefixMatch>
PagedPrefixCache::lookup_longest(PrefixNamespaceId namespace_id,
                                 std::span<const std::uint32_t> tokens,
                                 std::size_t maximum_prefix_tokens) {
  constexpr std::string_view kOperation = "lookup_longest";
  std::scoped_lock lock(impl_->mutex);
  require_namespace(namespace_id, kOperation);
  const std::size_t page_tokens = impl_->cache->config().block_tokens;
  const std::size_t bounded_tokens = std::min(tokens.size(), maximum_prefix_tokens);
  const std::size_t aligned_limit = bounded_tokens - bounded_tokens % page_tokens;

  Impl::Iterator best = impl_->entries.end();
  std::size_t best_tokens = 0;
  for (auto position = impl_->entries.begin(); position != impl_->entries.end(); ++position) {
    if (position->namespace_id == namespace_id && position->tokens.size() <= aligned_limit &&
        position->tokens.size() > best_tokens && token_vector_is_prefix(position->tokens, tokens)) {
      best = position;
      best_tokens = position->tokens.size();
    }
  }

  if (best == impl_->entries.end()) {
    increment_saturating(impl_->cache_stats.lookup_count);
    increment_saturating(impl_->cache_stats.miss_count);
    impl_->validate_invariants();
    return std::nullopt;
  }

  const PagedSequenceHandle sequence = impl_->cache->fork_sequence(best->anchor, best_tokens);
  impl_->entries.splice(impl_->entries.begin(), impl_->entries, best);
  increment_saturating(impl_->cache_stats.lookup_count);
  increment_saturating(impl_->cache_stats.hit_count);
  impl_->validate_invariants();
  return PagedPrefixMatch{sequence, best_tokens};
}

bool PagedPrefixCache::contains(PrefixNamespaceId namespace_id,
                                std::span<const std::uint32_t> tokens) const {
  constexpr std::string_view kOperation = "contains";
  std::scoped_lock lock(impl_->mutex);
  impl_->validate_key(namespace_id, tokens, kOperation);
  return impl_->find_exact(namespace_id, tokens) != impl_->entries.cend();
}

bool PagedPrefixCache::erase(PrefixNamespaceId namespace_id,
                             std::span<const std::uint32_t> tokens) {
  constexpr std::string_view kOperation = "erase";
  std::scoped_lock lock(impl_->mutex);
  impl_->validate_key(namespace_id, tokens, kOperation);
  const auto position = impl_->find_exact(namespace_id, tokens);
  if (position == impl_->entries.end()) {
    return false;
  }
  impl_->erase_locked(position, false);
  impl_->validate_invariants();
  return true;
}

bool PagedPrefixCache::evict_lru() {
  std::scoped_lock lock(impl_->mutex);
  if (impl_->entries.empty()) {
    return false;
  }
  impl_->erase_locked(std::prev(impl_->entries.end()), true);
  impl_->validate_invariants();
  return true;
}

void PagedPrefixCache::clear() {
  std::scoped_lock lock(impl_->mutex);
  while (!impl_->entries.empty()) {
    impl_->erase_locked(std::prev(impl_->entries.end()), false);
  }
  impl_->validate_invariants();
}

PagedPrefixCacheStats PagedPrefixCache::stats() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->cache_stats;
}

const PagedPrefixCacheConfig& PagedPrefixCache::config() const noexcept {
  return impl_->config;
}

std::size_t PagedPrefixCache::block_tokens() const noexcept {
  return impl_->cache->config().block_tokens;
}

} // namespace helix::kv_cache
