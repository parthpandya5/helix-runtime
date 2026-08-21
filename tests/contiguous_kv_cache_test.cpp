#include "helix/attention/reference.hpp"
#include "helix/kv_cache/contiguous.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <thread>
#include <vector>

namespace helix::kv_cache {
namespace {

void expect_values(const Tensor& tensor, std::span<const float> expected) {
  ASSERT_EQ(tensor.numel(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_FLOAT_EQ(tensor.data_as<float>()[index], expected[index]) << "at index " << index;
  }
}

TEST(ContiguousKvCacheTest, AccountsMemoryAndRetrievesLayerHeadMajorHistory) {
  ContiguousKvCache cache(ContiguousKvCacheConfig{2, 2, 3, 4'096});
  EXPECT_EQ(cache.bytes_per_token(), 96U);
  EXPECT_EQ(cache.sequence_bytes(5), 480U);
  const SequenceHandle sequence = cache.create_sequence(5);
  EXPECT_TRUE(sequence);
  EXPECT_TRUE(cache.contains(sequence));
  EXPECT_EQ(cache.length(sequence), 0U);
  EXPECT_EQ(cache.capacity(sequence), 5U);

  const Tensor first_keys =
      Tensor::from_values({2, 2, 3}, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
                                                9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor first_values =
      Tensor::from_values({2, 2, 3}, std::array{-1.0F, -2.0F, -3.0F, -4.0F, -5.0F, -6.0F, -7.0F,
                                                -8.0F, -9.0F, -10.0F, -11.0F, -12.0F});
  const Tensor second_keys =
      Tensor::from_values({2, 2, 3}, std::array{101.0F, 102.0F, 103.0F, 104.0F, 105.0F, 106.0F,
                                                107.0F, 108.0F, 109.0F, 110.0F, 111.0F, 112.0F});
  const Tensor second_values = Tensor::from_values(
      {2, 2, 3}, std::array{-101.0F, -102.0F, -103.0F, -104.0F, -105.0F, -106.0F, -107.0F, -108.0F,
                            -109.0F, -110.0F, -111.0F, -112.0F});
  cache.append(sequence, first_keys, first_values);
  cache.append(sequence, second_keys, second_values);

  const KvLayer layer = cache.retrieve_layer(sequence, 1);
  EXPECT_EQ(layer.keys.shape(), (Shape{2, 2, 3}));
  expect_values(layer.keys, std::array{7.0F, 8.0F, 9.0F, 107.0F, 108.0F, 109.0F, 10.0F, 11.0F,
                                       12.0F, 110.0F, 111.0F, 112.0F});
  expect_values(layer.values, std::array{-7.0F, -8.0F, -9.0F, -107.0F, -108.0F, -109.0F, -10.0F,
                                         -11.0F, -12.0F, -110.0F, -111.0F, -112.0F});

  const ContiguousKvCacheStats stats = cache.stats();
  EXPECT_EQ(stats.bytes_allocated, 480U);
  EXPECT_EQ(stats.peak_bytes_allocated, 480U);
  EXPECT_EQ(stats.active_sequence_count, 1U);
  EXPECT_EQ(stats.active_capacity_tokens, 5U);
  EXPECT_EQ(stats.active_tokens, 2U);
  EXPECT_EQ(stats.sequence_create_count, 1U);
  EXPECT_EQ(stats.append_count, 2U);
}

TEST(ContiguousKvCacheTest, RetrievedHistoryFeedsDecodeAttentionWithoutRecomputingPriorKv) {
  ContiguousKvCache cache(ContiguousKvCacheConfig{1, 1, 2, 1'024});
  const SequenceHandle sequence = cache.create_sequence(3);
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{1.0F, 0.0F}),
               Tensor::from_values({1, 1, 2}, std::array{10.0F, 1.0F}));
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{0.0F, 1.0F}),
               Tensor::from_values({1, 1, 2}, std::array{20.0F, 2.0F}));
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{1.0F, 1.0F}),
               Tensor::from_values({1, 1, 2}, std::array{30.0F, 3.0F}));

  const KvLayer history = cache.retrieve_layer(sequence, 0);
  const Tensor query = Tensor::from_values({1, 1, 2}, std::array{1.0F, 1.0F});
  const Tensor cached_result =
      attention::cpu::scaled_dot_product_attention(query, history.keys, history.values, 2);
  const Tensor explicit_result = attention::cpu::scaled_dot_product_attention(
      query, Tensor::from_values({1, 3, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F}),
      Tensor::from_values({1, 3, 2}, std::array{10.0F, 1.0F, 20.0F, 2.0F, 30.0F, 3.0F}), 2);
  expect_values(cached_result,
                std::span<const float>(explicit_result.data_as<float>(), explicit_result.numel()));
}

TEST(ContiguousKvCacheTest, CapacityAndReservationFailuresPreserveExistingState) {
  ContiguousKvCache cache(ContiguousKvCacheConfig{1, 1, 2, 32});
  const SequenceHandle sequence = cache.create_sequence(2);
  const Tensor token = Tensor::from_values({1, 1, 2}, std::array{1.0F, 2.0F});
  cache.append(sequence, token, token);
  cache.append(sequence, token, token);
  EXPECT_THROW(cache.append(sequence, token, token), Error);
  EXPECT_EQ(cache.length(sequence), 2U);
  EXPECT_EQ(cache.stats().append_count, 2U);

  EXPECT_THROW(static_cast<void>(cache.create_sequence(1)), Error);
  EXPECT_EQ(cache.stats().active_sequence_count, 1U);
  EXPECT_EQ(cache.stats().bytes_allocated, 32U);

  cache.release_sequence(sequence);
  EXPECT_FALSE(cache.contains(sequence));
  EXPECT_EQ(cache.stats().bytes_allocated, 0U);
  EXPECT_EQ(cache.stats().active_tokens, 0U);
  EXPECT_EQ(cache.stats().release_count, 1U);
  EXPECT_THROW(cache.release_sequence(sequence), Error);
}

TEST(ContiguousKvCacheTest, RejectsInvalidConfigurationHandlesShapesAndOutputs) {
  EXPECT_THROW(ContiguousKvCache(ContiguousKvCacheConfig{0, 1, 2, 128}), Error);
  EXPECT_THROW(ContiguousKvCache(ContiguousKvCacheConfig{1, 0, 2, 128}), Error);
  EXPECT_THROW(ContiguousKvCache(ContiguousKvCacheConfig{1, 1, 0, 128}), Error);
  EXPECT_THROW(ContiguousKvCache(ContiguousKvCacheConfig{1, 1, 2, 0}), Error);

  ContiguousKvCache first(ContiguousKvCacheConfig{1, 1, 2, 1'024});
  ContiguousKvCache second(ContiguousKvCacheConfig{1, 1, 2, 1'024});
  EXPECT_THROW(static_cast<void>(first.create_sequence(0)), Error);
  const SequenceHandle first_sequence = first.create_sequence(2);
  const SequenceHandle second_sequence = second.create_sequence(2);
  ASSERT_EQ(first_sequence.id(), second_sequence.id());
  EXPECT_FALSE(first.contains(second_sequence));
  EXPECT_THROW(first.append(second_sequence, Tensor::zeros({1, 1, 2}), Tensor::zeros({1, 1, 2})),
               Error);
  EXPECT_THROW(first.release_sequence(SequenceHandle{}), Error);

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

TEST(ContiguousKvCacheTest, SerializesConcurrentAppendsToIndependentSequences) {
  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kTokens = 50;
  ContiguousKvCache cache(ContiguousKvCacheConfig{1, 1, 2, 16'384});
  std::array<SequenceHandle, kThreads> sequences{};
  for (SequenceHandle& sequence : sequences) {
    sequence = cache.create_sequence(kTokens);
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
  EXPECT_EQ(cache.stats().append_count, kThreads * kTokens);
  for (const SequenceHandle sequence : sequences) {
    EXPECT_EQ(cache.length(sequence), kTokens);
    const KvLayer layer = cache.retrieve_layer(sequence, 0);
    EXPECT_EQ(layer.keys.shape(), (Shape{1, static_cast<Dimension>(kTokens), 2}));
  }
}

} // namespace
} // namespace helix::kv_cache
