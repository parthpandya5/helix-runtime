#include "helix/core/error.hpp"
#include "helix/kv_cache/prefix.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace helix::kv_cache {
namespace {

void append_tokens(PagedKvCache& cache, const PagedSequenceHandle& sequence,
                   std::size_t token_count, float base) {
  for (std::size_t token = 0; token < token_count; ++token) {
    const Tensor key = Tensor::from_values({1, 1, 1}, std::array{base + static_cast<float>(token)});
    const Tensor value =
        Tensor::from_values({1, 1, 1}, std::array{base + 1'000.0F + static_cast<float>(token)});
    cache.append(sequence, key, value);
  }
}

void expect_payload(const PagedKvCache& cache, const PagedSequenceHandle& sequence,
                    std::size_t token_count, float base) {
  const KvLayer layer = cache.retrieve_layer(sequence, 0);
  ASSERT_EQ(layer.keys.shape(), (Shape{1, static_cast<Dimension>(token_count), 1}));
  ASSERT_EQ(layer.values.shape(), layer.keys.shape());
  for (std::size_t token = 0; token < token_count; ++token) {
    EXPECT_FLOAT_EQ(layer.keys.data_as<float>()[token], base + static_cast<float>(token));
    EXPECT_FLOAT_EQ(layer.values.data_as<float>()[token],
                    base + 1'000.0F + static_cast<float>(token));
  }
}

TEST(PagedPrefixCacheTest, FindsLongestAlignedExactPrefixAndAccountsReferences) {
  auto cache = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 8});
  const PagedSequenceHandle source = cache->create_sequence();
  append_tokens(*cache, source, 6, 100.0F);
  const std::array<std::uint32_t, 6> tokens{10, 11, 12, 13, 14, 15};
  const std::span<const std::uint32_t> token_span(tokens);
  const PrefixNamespaceId namespace_id(7);
  PagedPrefixCache prefixes(cache, PagedPrefixCacheConfig{4, 8});

  EXPECT_EQ(prefixes.insert(namespace_id, token_span.first(2), source),
            PrefixInsertStatus::kInserted);
  EXPECT_EQ(prefixes.insert(namespace_id, token_span.first(4), source),
            PrefixInsertStatus::kInserted);
  EXPECT_TRUE(prefixes.contains(namespace_id, token_span.first(2)));
  EXPECT_TRUE(prefixes.contains(namespace_id, token_span.first(4)));
  EXPECT_EQ(prefixes.block_tokens(), 2U);
  EXPECT_EQ(prefixes.config().maximum_entries, 4U);

  PagedPrefixCacheStats prefix_stats = prefixes.stats();
  EXPECT_EQ(prefix_stats.entry_count, 2U);
  EXPECT_EQ(prefix_stats.retained_tokens, 6U);
  EXPECT_EQ(prefix_stats.retained_block_references, 3U);
  EXPECT_EQ(prefix_stats.peak_entries, 2U);
  EXPECT_EQ(prefix_stats.peak_block_references, 3U);
  EXPECT_EQ(prefix_stats.insertion_count, 2U);

  PagedKvCacheStats cache_stats = cache->stats();
  EXPECT_EQ(cache_stats.active_sequence_count, 3U);
  EXPECT_EQ(cache_stats.active_tokens, 12U);
  EXPECT_EQ(cache_stats.resident_tokens, 6U);
  EXPECT_EQ(cache_stats.block_reference_count, 6U);
  EXPECT_EQ(cache_stats.shared_blocks, 2U);

  const std::array<std::uint32_t, 5> extended{10, 11, 12, 13, 99};
  const auto longest = prefixes.lookup_longest(namespace_id, extended);
  ASSERT_TRUE(longest.has_value());
  EXPECT_EQ(longest->matched_tokens, 4U);
  const std::vector<PhysicalBlockId> source_blocks = cache->block_table(source);
  const std::vector<PhysicalBlockId> expected_longest_blocks(source_blocks.begin(),
                                                             source_blocks.begin() + 2);
  EXPECT_EQ(cache->block_table(longest->sequence), expected_longest_blocks);

  const auto bounded = prefixes.lookup_longest(namespace_id, extended, 3);
  ASSERT_TRUE(bounded.has_value());
  EXPECT_EQ(bounded->matched_tokens, 2U);
  EXPECT_EQ(cache->length(bounded->sequence), 2U);

  const std::array<std::uint32_t, 5> different{10, 90, 12, 13, 99};
  EXPECT_FALSE(prefixes.lookup_longest(namespace_id, different).has_value());
  EXPECT_FALSE(prefixes.lookup_longest(PrefixNamespaceId(8), extended).has_value());
  cache->release_sequence(longest->sequence);
  cache->release_sequence(bounded->sequence);

  EXPECT_EQ(prefixes.insert(namespace_id, token_span.first(2), source),
            PrefixInsertStatus::kAlreadyPresent);
  prefix_stats = prefixes.stats();
  EXPECT_EQ(prefix_stats.entry_count, 2U);
  EXPECT_EQ(prefix_stats.lookup_count, 4U);
  EXPECT_EQ(prefix_stats.hit_count, 2U);
  EXPECT_EQ(prefix_stats.miss_count, 2U);
  EXPECT_EQ(prefix_stats.duplicate_insertion_count, 1U);

  prefixes.clear();
  cache->release_sequence(source);
  cache_stats = cache->stats();
  EXPECT_EQ(cache_stats.active_sequence_count, 0U);
  EXPECT_EQ(cache_stats.used_blocks, 0U);
  EXPECT_EQ(cache_stats.block_reference_count, 0U);
}

TEST(PagedPrefixCacheTest, ExactKeysSeparateTokenContentAndNamespacesWithoutReplacement) {
  auto cache = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 6});
  const PagedSequenceHandle first_source = cache->create_sequence();
  const PagedSequenceHandle second_source = cache->create_sequence();
  const PagedSequenceHandle namespace_source = cache->create_sequence();
  append_tokens(*cache, first_source, 2, 100.0F);
  append_tokens(*cache, second_source, 2, 200.0F);
  append_tokens(*cache, namespace_source, 2, 300.0F);
  const std::array<std::uint32_t, 2> first_key{7, 8};
  const std::array<std::uint32_t, 2> second_key{7, 9};
  const PrefixNamespaceId first_namespace(1);
  const PrefixNamespaceId second_namespace(2);
  PagedPrefixCache prefixes(cache, PagedPrefixCacheConfig{4, 4});

  EXPECT_EQ(prefixes.insert(first_namespace, first_key, first_source),
            PrefixInsertStatus::kInserted);
  EXPECT_EQ(prefixes.insert(first_namespace, second_key, second_source),
            PrefixInsertStatus::kInserted);
  EXPECT_EQ(prefixes.insert(second_namespace, first_key, namespace_source),
            PrefixInsertStatus::kInserted);
  EXPECT_EQ(prefixes.insert(first_namespace, first_key, second_source),
            PrefixInsertStatus::kAlreadyPresent);

  const auto first = prefixes.lookup_longest(first_namespace, first_key);
  const auto second = prefixes.lookup_longest(first_namespace, second_key);
  const auto namespaced = prefixes.lookup_longest(second_namespace, first_key);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_TRUE(namespaced.has_value());
  EXPECT_EQ(cache->block_table(first->sequence).front(), cache->block_table(first_source).front());
  EXPECT_EQ(cache->block_table(second->sequence).front(),
            cache->block_table(second_source).front());
  EXPECT_EQ(cache->block_table(namespaced->sequence).front(),
            cache->block_table(namespace_source).front());
  expect_payload(*cache, first->sequence, 2, 100.0F);
  expect_payload(*cache, second->sequence, 2, 200.0F);
  expect_payload(*cache, namespaced->sequence, 2, 300.0F);
  EXPECT_EQ(prefixes.stats().duplicate_insertion_count, 1U);

  cache->release_sequence(first->sequence);
  cache->release_sequence(second->sequence);
  cache->release_sequence(namespaced->sequence);
  prefixes.clear();
  cache->release_sequence(first_source);
  cache->release_sequence(second_source);
  cache->release_sequence(namespace_source);
}

TEST(PagedPrefixCacheTest, EnforcesBlockReferenceBudgetWithDeterministicLruEviction) {
  auto cache = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 8});
  const PagedSequenceHandle first_source = cache->create_sequence();
  const PagedSequenceHandle second_source = cache->create_sequence();
  const PagedSequenceHandle third_source = cache->create_sequence();
  append_tokens(*cache, first_source, 2, 10.0F);
  append_tokens(*cache, second_source, 4, 20.0F);
  append_tokens(*cache, third_source, 2, 30.0F);
  const std::array<std::uint32_t, 2> first_key{1, 2};
  const std::array<std::uint32_t, 4> second_key{3, 4, 5, 6};
  const std::array<std::uint32_t, 2> third_key{7, 8};
  const PrefixNamespaceId namespace_id(9);
  PagedPrefixCache prefixes(cache, PagedPrefixCacheConfig{4, 3});

  EXPECT_EQ(prefixes.insert(namespace_id, first_key, first_source), PrefixInsertStatus::kInserted);
  EXPECT_EQ(prefixes.insert(namespace_id, second_key, second_source),
            PrefixInsertStatus::kInserted);
  const auto touched = prefixes.lookup_longest(namespace_id, first_key);
  ASSERT_TRUE(touched.has_value());
  cache->release_sequence(touched->sequence);

  EXPECT_EQ(prefixes.insert(namespace_id, third_key, third_source), PrefixInsertStatus::kInserted);
  EXPECT_TRUE(prefixes.contains(namespace_id, first_key));
  EXPECT_FALSE(prefixes.contains(namespace_id, second_key));
  EXPECT_TRUE(prefixes.contains(namespace_id, third_key));
  PagedPrefixCacheStats stats = prefixes.stats();
  EXPECT_EQ(stats.entry_count, 2U);
  EXPECT_EQ(stats.retained_block_references, 2U);
  EXPECT_EQ(stats.peak_block_references, 3U);
  EXPECT_EQ(stats.eviction_count, 1U);

  EXPECT_TRUE(prefixes.evict_lru());
  EXPECT_FALSE(prefixes.contains(namespace_id, first_key));
  EXPECT_TRUE(prefixes.contains(namespace_id, third_key));
  EXPECT_TRUE(prefixes.evict_lru());
  EXPECT_FALSE(prefixes.evict_lru());
  stats = prefixes.stats();
  EXPECT_EQ(stats.entry_count, 0U);
  EXPECT_EQ(stats.retained_tokens, 0U);
  EXPECT_EQ(stats.retained_block_references, 0U);
  EXPECT_EQ(stats.eviction_count, 3U);

  PagedPrefixCache entry_bounded(cache, PagedPrefixCacheConfig{2, 8});
  EXPECT_EQ(entry_bounded.insert(namespace_id, first_key, first_source),
            PrefixInsertStatus::kInserted);
  EXPECT_EQ(entry_bounded.insert(namespace_id, second_key, second_source),
            PrefixInsertStatus::kInserted);
  EXPECT_EQ(entry_bounded.insert(namespace_id, first_key, third_source),
            PrefixInsertStatus::kAlreadyPresent);
  EXPECT_EQ(entry_bounded.insert(namespace_id, third_key, third_source),
            PrefixInsertStatus::kInserted);
  EXPECT_TRUE(entry_bounded.contains(namespace_id, first_key));
  EXPECT_FALSE(entry_bounded.contains(namespace_id, second_key));
  EXPECT_TRUE(entry_bounded.contains(namespace_id, third_key));
  stats = entry_bounded.stats();
  EXPECT_EQ(stats.entry_count, 2U);
  EXPECT_EQ(stats.peak_entries, 2U);
  EXPECT_EQ(stats.eviction_count, 1U);
  EXPECT_EQ(stats.duplicate_insertion_count, 1U);
  EXPECT_FALSE(entry_bounded.erase(namespace_id, second_key));
  EXPECT_TRUE(entry_bounded.erase(namespace_id, first_key));
  EXPECT_EQ(entry_bounded.stats().erase_count, 1U);
  entry_bounded.clear();

  cache->release_sequence(first_source);
  cache->release_sequence(second_source);
  cache->release_sequence(third_source);
  EXPECT_EQ(cache->stats().used_blocks, 0U);
}

TEST(PagedPrefixCacheTest, RejectsInvalidKeysSourcesAndOversizedEntriesTransactionally) {
  EXPECT_THROW(static_cast<void>(PagedPrefixCache({}, PagedPrefixCacheConfig{1, 1})), Error);
  auto cache = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 4});
  EXPECT_THROW(static_cast<void>(PagedPrefixCache(cache, PagedPrefixCacheConfig{0, 1})), Error);
  EXPECT_THROW(static_cast<void>(PagedPrefixCache(cache, PagedPrefixCacheConfig{1, 0})), Error);

  const PagedSequenceHandle source = cache->create_sequence();
  append_tokens(*cache, source, 4, 10.0F);
  const std::array<std::uint32_t, 6> tokens{1, 2, 3, 4, 5, 6};
  PagedPrefixCache prefixes(cache, PagedPrefixCacheConfig{4, 1});
  EXPECT_EQ(prefixes.insert(PrefixNamespaceId(1), std::span(tokens).first(4), source),
            PrefixInsertStatus::kRejectedTooLarge);
  EXPECT_EQ(prefixes.stats().entry_count, 0U);
  EXPECT_EQ(prefixes.stats().rejected_insertion_count, 1U);
  EXPECT_EQ(cache->stats().sequence_fork_count, 0U);
  EXPECT_EQ(cache->stats().active_sequence_count, 1U);

  EXPECT_THROW(
      static_cast<void>(prefixes.insert(PrefixNamespaceId{}, std::span(tokens).first(2), source)),
      Error);
  EXPECT_THROW(static_cast<void>(
                   prefixes.insert(PrefixNamespaceId(1), std::span<const std::uint32_t>{}, source)),
               Error);
  EXPECT_THROW(
      static_cast<void>(prefixes.insert(PrefixNamespaceId(1), std::span(tokens).first(3), source)),
      Error);
  EXPECT_THROW(static_cast<void>(prefixes.insert(PrefixNamespaceId(1), tokens, source)), Error);
  EXPECT_THROW(
      static_cast<void>(prefixes.contains(PrefixNamespaceId{}, std::span(tokens).first(2))), Error);
  EXPECT_THROW(
      static_cast<void>(prefixes.contains(PrefixNamespaceId(1), std::span(tokens).first(3))),
      Error);
  EXPECT_FALSE(
      prefixes.lookup_longest(PrefixNamespaceId(1), std::span<const std::uint32_t>{}).has_value());
  EXPECT_FALSE(prefixes.lookup_longest(PrefixNamespaceId(1), tokens, 0).has_value());

  auto foreign = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 2});
  const PagedSequenceHandle foreign_source = foreign->create_sequence();
  append_tokens(*foreign, foreign_source, 2, 20.0F);
  EXPECT_THROW(static_cast<void>(prefixes.insert(PrefixNamespaceId(1), std::span(tokens).first(2),
                                                 foreign_source)),
               Error);
  cache->release_sequence(source);
  EXPECT_THROW(
      static_cast<void>(prefixes.insert(PrefixNamespaceId(1), std::span(tokens).first(2), source)),
      Error);
  EXPECT_EQ(prefixes.stats().entry_count, 0U);
  EXPECT_EQ(prefixes.stats().insertion_count, 0U);
  EXPECT_EQ(cache->stats().active_sequence_count, 0U);
  foreign->release_sequence(foreign_source);
}

TEST(PagedPrefixCacheTest, AnchorsKeepPayloadAliveUntilTheLastLookupOwnerReleases) {
  auto cache = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 2});
  const PagedSequenceHandle source = cache->create_sequence();
  append_tokens(*cache, source, 4, 50.0F);
  const std::array<std::uint32_t, 4> tokens{40, 41, 42, 43};
  PagedPrefixCache prefixes(cache, PagedPrefixCacheConfig{1, 2});
  EXPECT_EQ(prefixes.insert(PrefixNamespaceId(5), tokens, source), PrefixInsertStatus::kInserted);
  cache->release_sequence(source);
  EXPECT_EQ(cache->stats().active_sequence_count, 1U);
  EXPECT_EQ(cache->stats().free_blocks, 0U);

  const auto match = prefixes.lookup_longest(PrefixNamespaceId(5), tokens);
  ASSERT_TRUE(match.has_value());
  prefixes.clear();
  EXPECT_EQ(prefixes.stats().entry_count, 0U);
  EXPECT_EQ(prefixes.stats().erase_count, 1U);
  EXPECT_EQ(cache->stats().active_sequence_count, 1U);
  EXPECT_EQ(cache->stats().free_blocks, 0U);
  expect_payload(*cache, match->sequence, 4, 50.0F);

  cache->release_sequence(match->sequence);
  EXPECT_EQ(cache->stats().active_sequence_count, 0U);
  EXPECT_EQ(cache->stats().used_blocks, 0U);
  EXPECT_EQ(cache->stats().free_blocks, 2U);
}

TEST(PagedPrefixCacheTest, DestructorReleasesPersistentAnchors) {
  auto cache = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 2});
  const PagedSequenceHandle source = cache->create_sequence();
  append_tokens(*cache, source, 4, 75.0F);
  const std::array<std::uint32_t, 4> tokens{1, 2, 3, 4};
  {
    PagedPrefixCache prefixes(cache, PagedPrefixCacheConfig{1, 2});
    EXPECT_EQ(prefixes.insert(PrefixNamespaceId(1), tokens, source), PrefixInsertStatus::kInserted);
    cache->release_sequence(source);
    EXPECT_EQ(cache->stats().active_sequence_count, 1U);
  }
  EXPECT_EQ(cache->stats().active_sequence_count, 0U);
  EXPECT_EQ(cache->stats().used_blocks, 0U);
  EXPECT_EQ(cache->stats().block_reference_count, 0U);
}

TEST(PagedPrefixCacheTest, SerializesConcurrentLookupForkAndReleaseLifecycles) {
  constexpr std::size_t kThreads = 8;
  constexpr std::size_t kLookupsPerThread = 32;
  auto cache = std::make_shared<PagedKvCache>(PagedKvCacheConfig{1, 1, 1, 2, 4});
  const PagedSequenceHandle source = cache->create_sequence();
  append_tokens(*cache, source, 8, 100.0F);
  const std::array<std::uint32_t, 8> tokens{10, 11, 12, 13, 14, 15, 16, 17};
  PagedPrefixCache prefixes(cache, PagedPrefixCacheConfig{1, 4});
  ASSERT_EQ(prefixes.insert(PrefixNamespaceId(1), tokens, source), PrefixInsertStatus::kInserted);

  std::array<std::exception_ptr, kThreads> failures{};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&prefixes, &failures, cache, tokens, thread] {
      try {
        for (std::size_t lookup = 0; lookup < kLookupsPerThread; ++lookup) {
          const auto match = prefixes.lookup_longest(PrefixNamespaceId(1), tokens);
          if (!match.has_value() || match->matched_tokens != tokens.size()) {
            throw std::runtime_error("concurrent prefix lookup unexpectedly missed");
          }
          cache->release_sequence(match->sequence);
        }
      } catch (...) {
        failures[thread] = std::current_exception();
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  for (const std::exception_ptr& failure : failures) {
    ASSERT_EQ(failure, nullptr);
  }

  const PagedPrefixCacheStats stats = prefixes.stats();
  EXPECT_EQ(stats.lookup_count, kThreads * kLookupsPerThread);
  EXPECT_EQ(stats.hit_count, kThreads * kLookupsPerThread);
  EXPECT_EQ(stats.miss_count, 0U);
  EXPECT_EQ(stats.entry_count, 1U);
  EXPECT_EQ(cache->stats().active_sequence_count, 2U);
  EXPECT_EQ(cache->stats().block_reference_count, 8U);

  prefixes.clear();
  cache->release_sequence(source);
  EXPECT_EQ(cache->stats().active_sequence_count, 0U);
  EXPECT_EQ(cache->stats().used_blocks, 0U);
}

} // namespace
} // namespace helix::kv_cache
