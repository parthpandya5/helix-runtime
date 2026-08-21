#include "helix/attention/reference.hpp"
#include "helix/kv_cache/contiguous.hpp"
#include "helix/kv_cache/paged.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace helix::kv_cache {
namespace {

Tensor make_token(std::size_t layers, std::size_t heads, std::size_t dimension, float base) {
  const std::size_t elements = layers * heads * dimension;
  std::vector<float> values(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    values[index] = base + static_cast<float>(index);
  }
  return Tensor::from_values({static_cast<Dimension>(layers), static_cast<Dimension>(heads),
                              static_cast<Dimension>(dimension)},
                             values);
}

void expect_values(const Tensor& tensor, std::span<const float> expected) {
  ASSERT_EQ(tensor.numel(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(tensor.data_as<float>()[index], expected[index]) << "at index " << index;
  }
}

TEST(PagedKvCacheTest, MapsLogicalHistoryAcrossPhysicalBlockBoundaries) {
  PagedKvCache cache(PagedKvCacheConfig{2, 2, 2, 2, 4});
  EXPECT_EQ(cache.bytes_per_token(), 64U);
  EXPECT_EQ(cache.block_bytes(), 128U);
  EXPECT_EQ(cache.token_capacity(), 8U);
  const PagedSequenceHandle sequence = cache.create_sequence();

  for (std::size_t token = 0; token < 5; ++token) {
    cache.append(sequence, make_token(2, 2, 2, static_cast<float>(token * 100)),
                 make_token(2, 2, 2, static_cast<float>(1'000 + token * 100)));
  }

  const std::vector<PhysicalBlockId> table = cache.block_table(sequence);
  ASSERT_EQ(table.size(), 3U);
  EXPECT_EQ(table[0].value(), 0U);
  EXPECT_EQ(table[1].value(), 1U);
  EXPECT_EQ(table[2].value(), 2U);
  const KvLayer layer = cache.retrieve_layer(sequence, 1);
  EXPECT_EQ(layer.keys.shape(), (Shape{2, 5, 2}));
  expect_values(layer.keys, std::array{4.0F,   5.0F,   104.0F, 105.0F, 204.0F, 205.0F, 304.0F,
                                       305.0F, 404.0F, 405.0F, 6.0F,   7.0F,   106.0F, 107.0F,
                                       206.0F, 207.0F, 306.0F, 307.0F, 406.0F, 407.0F});
  expect_values(layer.values,
                std::array{1'004.0F, 1'005.0F, 1'104.0F, 1'105.0F, 1'204.0F, 1'205.0F, 1'304.0F,
                           1'305.0F, 1'404.0F, 1'405.0F, 1'006.0F, 1'007.0F, 1'106.0F, 1'107.0F,
                           1'206.0F, 1'207.0F, 1'306.0F, 1'307.0F, 1'406.0F, 1'407.0F});

  const PagedKvCacheStats stats = cache.stats();
  EXPECT_EQ(stats.bytes_reserved, 512U);
  EXPECT_EQ(stats.bytes_used, 384U);
  EXPECT_EQ(stats.bytes_free, 128U);
  EXPECT_EQ(stats.peak_bytes_used, 384U);
  EXPECT_EQ(stats.total_blocks, 4U);
  EXPECT_EQ(stats.used_blocks, 3U);
  EXPECT_EQ(stats.free_blocks, 1U);
  EXPECT_EQ(stats.active_sequence_count, 1U);
  EXPECT_EQ(stats.active_tokens, 5U);
  EXPECT_EQ(stats.allocated_token_slots, 6U);
  EXPECT_EQ(stats.unused_token_slots(), 1U);
  EXPECT_DOUBLE_EQ(stats.token_slot_utilization(), 5.0 / 6.0);
  EXPECT_EQ(stats.block_allocation_count, 3U);
  EXPECT_EQ(stats.block_reuse_count, 0U);
}

TEST(PagedKvCacheTest, RetrievedBlocksFeedContiguousDecodeAttention) {
  PagedKvCache cache(PagedKvCacheConfig{1, 1, 2, 2, 2});
  const PagedSequenceHandle sequence = cache.create_sequence();
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{1.0F, 0.0F}),
               Tensor::from_values({1, 1, 2}, std::array{10.0F, 1.0F}));
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{0.0F, 1.0F}),
               Tensor::from_values({1, 1, 2}, std::array{20.0F, 2.0F}));
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{1.0F, 1.0F}),
               Tensor::from_values({1, 1, 2}, std::array{30.0F, 3.0F}));

  const KvLayer history = cache.retrieve_layer(sequence, 0);
  const Tensor query = Tensor::from_values({1, 1, 2}, std::array{1.0F, 1.0F});
  const Tensor paged_result =
      attention::cpu::scaled_dot_product_attention(query, history.keys, history.values, 2);
  const Tensor explicit_result = attention::cpu::scaled_dot_product_attention(
      query, Tensor::from_values({1, 3, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F}),
      Tensor::from_values({1, 3, 2}, std::array{10.0F, 1.0F, 20.0F, 2.0F, 30.0F, 3.0F}), 2);
  expect_values(paged_result,
                std::span<const float>(explicit_result.data_as<float>(), explicit_result.numel()));
}

TEST(PagedKvCacheTest, BlockExhaustionIsTransactionalAndPartialBlocksRemainWritable) {
  PagedKvCache cache(PagedKvCacheConfig{1, 1, 2, 2, 1});
  const PagedSequenceHandle first = cache.create_sequence();
  const PagedSequenceHandle second = cache.create_sequence();
  const Tensor token = Tensor::from_values({1, 1, 2}, std::array{1.0F, 2.0F});
  cache.append(first, token, token);

  try {
    cache.append(second, token, token);
    FAIL() << "append unexpectedly succeeded without a free physical block";
  } catch (const Error& error) {
    EXPECT_EQ(error.code(), ErrorCode::kResourceExhausted);
  }
  EXPECT_EQ(cache.length(second), 0U);
  EXPECT_TRUE(cache.block_table(second).empty());
  EXPECT_EQ(cache.stats().append_count, 1U);

  cache.append(first, token, token);
  EXPECT_EQ(cache.length(first), 2U);
  EXPECT_THROW(cache.append(first, token, token), Error);
  EXPECT_EQ(cache.length(first), 2U);
  EXPECT_EQ(cache.block_table(first).size(), 1U);
  EXPECT_EQ(cache.stats().active_tokens, 2U);

  cache.release_sequence(first);
  const Tensor replacement = Tensor::from_values({1, 1, 2}, std::array{9.0F, 8.0F});
  cache.append(second, replacement, replacement);
  ASSERT_EQ(cache.block_table(second).size(), 1U);
  EXPECT_EQ(cache.block_table(second)[0].value(), 0U);
  EXPECT_EQ(cache.stats().block_reuse_count, 1U);
  expect_values(cache.retrieve_layer(second, 0).keys, std::array{9.0F, 8.0F});
}

TEST(PagedKvCacheTest, FreePoolIsDeterministicFifoAndReleaseReclaimsEveryBlock) {
  PagedKvCache cache(PagedKvCacheConfig{1, 1, 1, 1, 3});
  const Tensor token = Tensor::from_values({1, 1, 1}, std::array{1.0F});
  const PagedSequenceHandle first = cache.create_sequence();
  const PagedSequenceHandle second = cache.create_sequence();
  const PagedSequenceHandle third = cache.create_sequence();
  cache.append(first, token, token);
  cache.append(second, token, token);
  cache.append(third, token, token);
  EXPECT_EQ(cache.block_table(first)[0].value(), 0U);
  EXPECT_EQ(cache.block_table(second)[0].value(), 1U);
  EXPECT_EQ(cache.block_table(third)[0].value(), 2U);

  cache.release_sequence(second);
  cache.release_sequence(first);
  const PagedSequenceHandle fourth = cache.create_sequence();
  const PagedSequenceHandle fifth = cache.create_sequence();
  cache.append(fourth, token, token);
  cache.append(fifth, token, token);
  EXPECT_EQ(cache.block_table(fourth)[0].value(), 1U);
  EXPECT_EQ(cache.block_table(fifth)[0].value(), 0U);
  EXPECT_EQ(cache.stats().block_reuse_count, 2U);

  cache.release_sequence(third);
  cache.release_sequence(fourth);
  cache.release_sequence(fifth);
  const PagedKvCacheStats stats = cache.stats();
  EXPECT_EQ(stats.bytes_used, 0U);
  EXPECT_EQ(stats.bytes_free, stats.bytes_reserved);
  EXPECT_EQ(stats.used_blocks, 0U);
  EXPECT_EQ(stats.free_blocks, 3U);
  EXPECT_EQ(stats.active_tokens, 0U);
  EXPECT_EQ(stats.allocated_token_slots, 0U);
  EXPECT_THROW(cache.release_sequence(first), Error);
}

TEST(PagedKvCacheTest, RejectsInvalidConfigurationHandlesShapesAndOutputs) {
  EXPECT_THROW(PagedKvCache(PagedKvCacheConfig{0, 1, 2, 2, 2}), Error);
  EXPECT_THROW(PagedKvCache(PagedKvCacheConfig{1, 0, 2, 2, 2}), Error);
  EXPECT_THROW(PagedKvCache(PagedKvCacheConfig{1, 1, 0, 2, 2}), Error);
  EXPECT_THROW(PagedKvCache(PagedKvCacheConfig{1, 1, 2, 0, 2}), Error);
  EXPECT_THROW(PagedKvCache(PagedKvCacheConfig{1, 1, 2, 2, 0}), Error);
  if constexpr (std::numeric_limits<std::size_t>::max() >
                std::numeric_limits<std::uint32_t>::max()) {
    const std::size_t too_many_blocks =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
    EXPECT_THROW(PagedKvCache(PagedKvCacheConfig{1, 1, 1, 1, too_many_blocks}), Error);
  }
  EXPECT_THROW(PagedKvCache(PagedKvCacheConfig{
                   1, 1, static_cast<std::size_t>(std::numeric_limits<Dimension>::max()), 2, 1}),
               Error);

  PagedKvCache first(PagedKvCacheConfig{1, 1, 2, 2, 2});
  PagedKvCache second(PagedKvCacheConfig{1, 1, 2, 2, 2});
  const PagedSequenceHandle first_sequence = first.create_sequence();
  const PagedSequenceHandle second_sequence = second.create_sequence();
  ASSERT_EQ(first_sequence.id(), second_sequence.id());
  EXPECT_FALSE(first.contains(second_sequence));
  EXPECT_THROW(first.append(second_sequence, Tensor::zeros({1, 1, 2}), Tensor::zeros({1, 1, 2})),
               Error);
  EXPECT_THROW(first.release_sequence(PagedSequenceHandle{}), Error);
  EXPECT_FALSE(PhysicalBlockId{}.valid());

  EXPECT_THROW(first.append(first_sequence, Tensor::zeros({1, 2}), Tensor::zeros({1, 1, 2})),
               Error);
  EXPECT_THROW(first.append(first_sequence, Tensor::zeros({1, 1, 2}, DType::kFloat16),
                            Tensor::zeros({1, 1, 2})),
               Error);
  EXPECT_THROW(static_cast<void>(first.retrieve_layer(first_sequence, 1)), Error);

  const KvLayer empty = first.retrieve_layer(first_sequence, 0);
  EXPECT_EQ(empty.keys.shape(), (Shape{1, 0, 2}));
  Tensor wrong_keys = Tensor::zeros({1, 1, 2});
  Tensor empty_values = Tensor::zeros({1, 0, 2});
  EXPECT_THROW(first.copy_layer_out(first_sequence, 0, wrong_keys, empty_values), Error);

  const Tensor token = Tensor::zeros({1, 1, 2});
  first.append(first_sequence, token, token);
  Tensor aliased = Tensor::zeros({1, 1, 2});
  EXPECT_THROW(first.copy_layer_out(first_sequence, 0, aliased, aliased), Error);
}

TEST(PagedKvCacheTest, BoundsTailSlackForAnOverReservedContiguousTrace) {
  ContiguousKvCache contiguous(ContiguousKvCacheConfig{1, 1, 1, 1'024});
  PagedKvCache paged(PagedKvCacheConfig{1, 1, 1, 4, 3});
  const SequenceHandle contiguous_short = contiguous.create_sequence(10);
  const SequenceHandle contiguous_long = contiguous.create_sequence(10);
  const PagedSequenceHandle paged_short = paged.create_sequence();
  const PagedSequenceHandle paged_long = paged.create_sequence();
  const Tensor token = Tensor::from_values({1, 1, 1}, std::array{2.0F});

  contiguous.append(contiguous_short, token, token);
  paged.append(paged_short, token, token);
  for (std::size_t position = 0; position < 5; ++position) {
    contiguous.append(contiguous_long, token, token);
    paged.append(paged_long, token, token);
  }

  const ContiguousKvCacheStats contiguous_stats = contiguous.stats();
  const PagedKvCacheStats paged_stats = paged.stats();
  ASSERT_EQ(contiguous.bytes_per_token(), paged.bytes_per_token());
  EXPECT_EQ(contiguous_stats.active_capacity_tokens, 20U);
  EXPECT_EQ(contiguous_stats.active_capacity_tokens - contiguous_stats.active_tokens, 14U);
  EXPECT_EQ(paged_stats.allocated_token_slots, 12U);
  EXPECT_EQ(paged_stats.unused_token_slots(), 6U);
  EXPECT_EQ(contiguous_stats.bytes_allocated, 160U);
  EXPECT_EQ(paged_stats.bytes_used, 96U);
}

TEST(PagedKvCacheTest, SerializesConcurrentAppendsAndReclaimsAllTables) {
  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kTokens = 25;
  PagedKvCache cache(PagedKvCacheConfig{1, 1, 2, 7, 16});
  std::array<PagedSequenceHandle, kThreads> sequences{};
  for (PagedSequenceHandle& sequence : sequences) {
    sequence = cache.create_sequence();
  }
  const Tensor token = Tensor::from_values({1, 1, 2}, std::array{3.0F, 4.0F});
  std::vector<std::thread> threads;
  for (std::size_t thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&cache, &sequences, &token, thread] {
      for (std::size_t position = 0; position < kTokens; ++position) {
        cache.append(sequences[thread], token, token);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(cache.stats().active_tokens, kThreads * kTokens);
  EXPECT_EQ(cache.stats().used_blocks, 16U);
  EXPECT_EQ(cache.stats().unused_token_slots(), 12U);
  for (const PagedSequenceHandle sequence : sequences) {
    EXPECT_EQ(cache.length(sequence), kTokens);
    EXPECT_EQ(cache.block_table(sequence).size(), 4U);
    cache.release_sequence(sequence);
  }
  EXPECT_EQ(cache.stats().free_blocks, 16U);
  EXPECT_EQ(cache.stats().active_sequence_count, 0U);
  EXPECT_EQ(cache.stats().active_tokens, 0U);
}

} // namespace
} // namespace helix::kv_cache
