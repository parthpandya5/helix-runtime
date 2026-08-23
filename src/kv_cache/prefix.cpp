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
#include <unordered_map>
#include <utility>
#include <vector>

namespace helix::kv_cache {
namespace {

constexpr std::string_view kTypeName = "PagedPrefixCache";
constexpr std::uint64_t kFnvOffsetBasis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;

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

std::uint64_t extend_token_digest(std::uint64_t digest, std::uint32_t token) noexcept {
  for (std::uint32_t shift = 0; shift < 32; shift += 8) {
    digest ^= static_cast<std::uint8_t>(token >> shift);
    digest *= kFnvPrime;
  }
  return digest;
}

std::uint64_t token_digest(std::span<const std::uint32_t> tokens) noexcept {
  std::uint64_t digest = kFnvOffsetBasis;
  for (const std::uint32_t token : tokens) {
    digest = extend_token_digest(digest, token);
  }
  return digest;
}

std::size_t fold_digest(std::uint64_t value) noexcept {
  value ^= value >> 30;
  value *= 0xbf58'476d'1ce4'e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d0'49bb'1331'11ebULL;
  value ^= value >> 31;
  if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
    value ^= value >> 32;
  }
  return static_cast<std::size_t>(value);
}

} // namespace

struct PagedPrefixCache::Impl {
  struct Entry {
    PrefixNamespaceId namespace_id;
    std::vector<std::uint32_t> tokens;
    PagedSequenceHandle anchor;
    std::size_t block_references = 0;
    std::uint64_t digest = kFnvOffsetBasis;
  };

  using EntryList = std::list<Entry>;
  using Iterator = EntryList::iterator;
  using ConstIterator = EntryList::const_iterator;

  struct IndexKey {
    PrefixNamespaceId namespace_id;
    std::size_t token_count = 0;
    std::uint64_t digest = kFnvOffsetBasis;

    friend bool operator==(const IndexKey&, const IndexKey&) = default;
  };

  struct IndexKeyHash {
    [[nodiscard]] std::size_t operator()(const IndexKey& key) const noexcept {
      std::uint64_t mixed = key.digest;
      mixed ^= key.namespace_id.value() + 0x9e37'79b9'7f4a'7c15ULL + (mixed << 6) + (mixed >> 2);
      mixed ^= static_cast<std::uint64_t>(key.token_count) + 0x9e37'79b9'7f4a'7c15ULL +
               (mixed << 6) + (mixed >> 2);
      return fold_digest(mixed);
    }
  };

  // List iterators survive MRU splices, so the content index never needs repair
  // when an entry is touched. Exact token verification makes digest collisions
  // a performance concern rather than an identity or correctness concern.
  using ContentIndex = std::unordered_multimap<IndexKey, Iterator, IndexKeyHash>;

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

  [[nodiscard]] static IndexKey index_key(PrefixNamespaceId namespace_id, std::size_t token_count,
                                          std::uint64_t digest) noexcept {
    return IndexKey{namespace_id, token_count, digest};
  }

  [[nodiscard]] Iterator find_exact(PrefixNamespaceId namespace_id,
                                    std::span<const std::uint32_t> tokens) {
    const auto candidates =
        index.equal_range(index_key(namespace_id, tokens.size(), token_digest(tokens)));
    const auto match =
        std::find_if(candidates.first, candidates.second, [&](const auto& candidate) {
          return token_vectors_equal(candidate.second->tokens, tokens);
        });
    return match == candidates.second ? entries.end() : match->second;
  }

  [[nodiscard]] ConstIterator find_exact(PrefixNamespaceId namespace_id,
                                         std::span<const std::uint32_t> tokens) const {
    const auto candidates =
        index.equal_range(index_key(namespace_id, tokens.size(), token_digest(tokens)));
    const auto match =
        std::find_if(candidates.first, candidates.second, [&](const auto& candidate) {
          return token_vectors_equal(candidate.second->tokens, tokens);
        });
    return match == candidates.second ? entries.cend() : ConstIterator(match->second);
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
    const auto candidates = index.equal_range(
        index_key(position->namespace_id, position->tokens.size(), position->digest));
    const auto indexed =
        std::find_if(candidates.first, candidates.second,
                     [&](const auto& candidate) { return candidate.second == position; });
    HELIX_ASSERT(indexed != candidates.second,
                 "prefix cache LRU entry is absent from the content index");
    cache->release_sequence(position->anchor);
    index.erase(indexed);
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
      HELIX_ASSERT(outer->digest == token_digest(outer->tokens),
                   "prefix cache entry digest disagrees with its content key");
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
      const auto candidates =
          index.equal_range(index_key(outer->namespace_id, outer->tokens.size(), outer->digest));
      std::size_t exact_key_count = 0;
      std::size_t entry_mapping_count = 0;
      for (auto candidate = candidates.first; candidate != candidates.second; ++candidate) {
        exact_key_count +=
            static_cast<std::size_t>(token_vectors_equal(candidate->second->tokens, outer->tokens));
        entry_mapping_count += static_cast<std::size_t>(candidate->second == outer);
      }
      HELIX_ASSERT(exact_key_count == 1, "prefix cache contains a duplicate exact key");
      HELIX_ASSERT(entry_mapping_count == 1,
                   "prefix cache entry does not have exactly one index mapping");
    }
    HELIX_ASSERT(index.size() == entries.size(),
                 "prefix cache content-index size disagrees with its LRU size");
    for (const auto& [key, position] : index) {
      HELIX_ASSERT(position != entries.end(), "prefix cache index stores its end iterator");
      HELIX_ASSERT(key.namespace_id == position->namespace_id &&
                       key.token_count == position->tokens.size() && key.digest == position->digest,
                   "prefix cache index key disagrees with its LRU entry");
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
  ContentIndex index;
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
  const std::uint64_t digest = token_digest(tokens);
  staged.digest = digest;
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
  try {
    impl_->index.emplace(Impl::index_key(namespace_id, tokens.size(), digest),
                         impl_->entries.begin());
  } catch (const std::bad_alloc&) {
    impl_->cache->release_sequence(anchor);
    impl_->entries.pop_front();
    throw_error(ErrorCode::kResourceExhausted,
                "PagedPrefixCache::insert could not allocate the content index entry");
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
  std::uint64_t digest = kFnvOffsetBasis;
  for (std::size_t token = 0; token < aligned_limit; ++token) {
    digest = extend_token_digest(digest, tokens[token]);
    const std::size_t prefix_tokens = token + 1;
    if (prefix_tokens % page_tokens != 0) {
      continue;
    }
    const auto candidates =
        impl_->index.equal_range(Impl::index_key(namespace_id, prefix_tokens, digest));
    for (auto candidate = candidates.first; candidate != candidates.second; ++candidate) {
      if (token_vector_is_prefix(candidate->second->tokens, tokens)) {
        best = candidate->second;
        best_tokens = prefix_tokens;
        break;
      }
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
