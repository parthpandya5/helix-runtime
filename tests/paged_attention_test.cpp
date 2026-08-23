#include "helix/attention/paged_reference.hpp"
#include "helix/attention/reference.hpp"
#include "helix/kv_cache/paged.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>

namespace helix::attention::cpu {
namespace {

void expect_near(const Tensor& actual, const Tensor& expected, float tolerance = 1.0e-5F) {
  ASSERT_EQ(actual.shape(), expected.shape());
  for (std::size_t index = 0; index < actual.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected.data_as<float>()[index], tolerance)
        << "at index " << index;
  }
}

Tensor concatenate_token_axis(const Tensor& history, const Tensor& tail) {
  const std::size_t heads = static_cast<std::size_t>(history.shape()[0]);
  const std::size_t history_tokens = static_cast<std::size_t>(history.shape()[1]);
  const std::size_t tail_tokens = static_cast<std::size_t>(tail.shape()[1]);
  const std::size_t dimensions = static_cast<std::size_t>(history.shape()[2]);
  Tensor result =
      Tensor::empty({history.shape()[0], history.shape()[1] + tail.shape()[1], history.shape()[2]});
  for (std::size_t head = 0; head < heads; ++head) {
    float* destination =
        result.data_as<float>() + head * (history_tokens + tail_tokens) * dimensions;
    std::copy_n(history.data_as<float>() + head * history_tokens * dimensions,
                history_tokens * dimensions, destination);
    std::copy_n(tail.data_as<float>() + head * tail_tokens * dimensions, tail_tokens * dimensions,
                destination + history_tokens * dimensions);
  }
  return result;
}

TEST(PagedAttentionTest, MatchesContiguousCausalPrefillAcrossBlockBoundaries) {
  kv_cache::PagedKvCache cache(kv_cache::PagedKvCacheConfig{1, 1, 2, 2, 2});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{1.0F, 0.0F}),
               Tensor::from_values({1, 1, 2}, std::array{10.0F, 1.0F}));
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{0.0F, 1.0F}),
               Tensor::from_values({1, 1, 2}, std::array{20.0F, 2.0F}));
  cache.append(sequence, Tensor::from_values({1, 1, 2}, std::array{1.0F, 1.0F}),
               Tensor::from_values({1, 1, 2}, std::array{30.0F, 3.0F}));
  const kv_cache::KvLayer compact = cache.retrieve_layer(sequence, 0);
  const Tensor query =
      Tensor::from_values({1, 3, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F});
  const Tensor expected = scaled_dot_product_attention(query, compact.keys, compact.values, 0);

  auto view = cache.read_layer(sequence, 0);
  EXPECT_EQ(paged_score_workspace_shape(query, view), (Shape{1, 3, 3}));
  EXPECT_EQ(paged_score_workspace_bytes(query, view), 36U);
  Tensor output = Tensor::empty({1, 3, 2});
  Tensor workspace = Tensor::empty({1, 3, 3});
  paged_scaled_dot_product_attention_out(query, view, 0, output, workspace);
  expect_near(output, expected);

  Tensor expected_output = Tensor::empty({1, 3, 2});
  Tensor expected_workspace = Tensor::empty({1, 3, 3});
  scaled_dot_product_attention_out(query, compact.keys, compact.values, 0, expected_output,
                                   expected_workspace);
  expect_near(workspace, expected_workspace);

  Tensor baseline_output = Tensor::empty({1, 3, 2});
  Tensor baseline_workspace = Tensor::empty({1, 3, 3});
  paged_scaled_dot_product_attention_dimension_major_out(query, view, 0, baseline_output,
                                                         baseline_workspace);
  expect_near(baseline_output, expected);
  expect_near(baseline_workspace, expected_workspace);
}

TEST(PagedAttentionTest, MatchesOddMultiHeadChunkedDecodeWithoutCompaction) {
  constexpr std::size_t kHeads = 3;
  constexpr std::size_t kDimension = 5;
  constexpr std::size_t kTokens = 9;
  kv_cache::PagedKvCache cache(kv_cache::PagedKvCacheConfig{2, kHeads, kDimension, 4, 3});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  for (std::size_t token = 0; token < kTokens; ++token) {
    Tensor keys = Tensor::random_uniform(
        {2, static_cast<Dimension>(kHeads), static_cast<Dimension>(kDimension)}, 91'000 + token,
        -1.0F, 1.0F);
    Tensor values = Tensor::random_uniform(
        {2, static_cast<Dimension>(kHeads), static_cast<Dimension>(kDimension)}, 92'000 + token,
        -1.0F, 1.0F);
    cache.append(sequence, keys, values);
  }
  const kv_cache::KvLayer compact = cache.retrieve_layer(sequence, 1);
  const Tensor query = Tensor::random_uniform(
      {static_cast<Dimension>(kHeads), 3, static_cast<Dimension>(kDimension)}, 93'000, -1.0F, 1.0F);
  const Tensor expected = scaled_dot_product_attention(query, compact.keys, compact.values, 6);

  auto view = cache.read_layer(sequence, 1);
  const Tensor actual = paged_scaled_dot_product_attention(query, view, 6);
  expect_near(actual, expected, 2.0e-5F);
}

TEST(PagedAttentionTest, TailContinuationMatchesCompactedHistoryAcrossOddPages) {
  constexpr std::size_t kHeads = 2;
  constexpr std::size_t kDimension = 3;
  constexpr std::size_t kHistoryTokens = 3;
  constexpr std::size_t kTailTokens = 2;
  kv_cache::PagedKvCache cache(kv_cache::PagedKvCacheConfig{1, kHeads, kDimension, 2, 2});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  for (std::size_t token = 0; token < kHistoryTokens; ++token) {
    cache.append(sequence,
                 Tensor::random_uniform({1, kHeads, kDimension}, 94'000 + token, -1.0F, 1.0F),
                 Tensor::random_uniform({1, kHeads, kDimension}, 95'000 + token, -1.0F, 1.0F));
  }

  const Tensor query =
      Tensor::random_uniform({kHeads, kTailTokens, kDimension}, 96'000, -1.0F, 1.0F);
  const Tensor tail_key =
      Tensor::random_uniform({kHeads, kTailTokens, kDimension}, 96'001, -1.0F, 1.0F);
  const Tensor tail_value =
      Tensor::random_uniform({kHeads, kTailTokens, kDimension}, 96'002, -1.0F, 1.0F);
  const kv_cache::KvLayer compact = cache.retrieve_layer(sequence, 0);
  const Tensor combined_key = concatenate_token_axis(compact.keys, tail_key);
  const Tensor combined_value = concatenate_token_axis(compact.values, tail_value);
  Tensor expected_output = Tensor::empty({kHeads, kTailTokens, kDimension});
  Tensor expected_workspace = Tensor::empty({kHeads, kTailTokens, kHistoryTokens + kTailTokens});
  scaled_dot_product_attention_out(query, combined_key, combined_value, kHistoryTokens,
                                   expected_output, expected_workspace);

  auto view = cache.read_layer(sequence, 0);
  EXPECT_EQ(paged_tail_score_workspace_shape(query, view, tail_key, tail_value),
            (Shape{kHeads, kTailTokens, kHistoryTokens + kTailTokens}));
  EXPECT_EQ(paged_tail_score_workspace_bytes(query, view, tail_key, tail_value), 80U);
  Tensor output = Tensor::empty({kHeads, kTailTokens, kDimension});
  Tensor workspace = Tensor::empty({kHeads, kTailTokens, kHistoryTokens + kTailTokens});
  paged_scaled_dot_product_attention_with_tail_out(query, view, tail_key, tail_value, output,
                                                   workspace);
  expect_near(output, expected_output, 2.0e-5F);
  expect_near(workspace, expected_workspace, 2.0e-5F);
  expect_near(paged_scaled_dot_product_attention_with_tail(query, view, tail_key, tail_value),
              expected_output, 2.0e-5F);
}

TEST(PagedAttentionTest, TailContinuationSupportsEmptyHistoryAndRejectsInvalidContracts) {
  kv_cache::PagedKvCache cache(kv_cache::PagedKvCacheConfig{1, 2, 3, 2, 1});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  const Tensor query = Tensor::random_uniform({2, 1, 3}, 97'000, -1.0F, 1.0F);
  const Tensor tail_key = Tensor::random_uniform({2, 1, 3}, 97'001, -1.0F, 1.0F);
  const Tensor tail_value = Tensor::random_uniform({2, 1, 3}, 97'002, -1.0F, 1.0F);
  const Tensor expected = scaled_dot_product_attention(query, tail_key, tail_value, 0);
  auto view = cache.read_layer(sequence, 0);
  expect_near(paged_scaled_dot_product_attention_with_tail(query, view, tail_key, tail_value),
              expected);

  EXPECT_THROW(static_cast<void>(paged_tail_score_workspace_shape(Tensor::zeros({2, 0, 3}), view,
                                                                  Tensor::zeros({2, 0, 3}),
                                                                  Tensor::zeros({2, 0, 3}))),
               Error);
  EXPECT_THROW(static_cast<void>(paged_tail_score_workspace_shape(
                   query, view, Tensor::zeros({1, 1, 3}), tail_value)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_tail_score_workspace_shape(query, view, tail_key,
                                                                  Tensor::zeros({2, 1, 2}))),
               Error);
  EXPECT_THROW(static_cast<void>(paged_tail_score_workspace_shape(Tensor::zeros({2, 2, 3}), view,
                                                                  tail_key, tail_value)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_tail_score_workspace_shape(
                   query, view, Tensor::zeros({2, 1, 3}, DType::kFloat16), tail_value)),
               Error);

  Tensor output_alias = tail_key;
  Tensor workspace = Tensor::empty({2, 1, 1});
  EXPECT_THROW(paged_scaled_dot_product_attention_with_tail_out(query, view, tail_key, tail_value,
                                                                output_alias, workspace),
               Error);
  Tensor output = Tensor::empty({2, 1, 3});
  Tensor wrong_workspace = Tensor::empty({2, 1, 2});
  EXPECT_THROW(paged_scaled_dot_product_attention_with_tail_out(query, view, tail_key, tail_value,
                                                                output, wrong_workspace),
               Error);
}

TEST(PagedAttentionTest, SupportsZeroQueriesWithPositiveKvHistory) {
  kv_cache::PagedKvCache cache(kv_cache::PagedKvCacheConfig{1, 2, 3, 4, 1});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  cache.append(sequence, Tensor::zeros({1, 2, 3}), Tensor::zeros({1, 2, 3}));
  auto view = cache.read_layer(sequence, 0);
  const Tensor query = Tensor::empty({2, 0, 3});
  EXPECT_EQ(paged_score_workspace_shape(query, view), (Shape{2, 0, 1}));
  EXPECT_EQ(paged_score_workspace_bytes(query, view), 0U);
  const Tensor output = paged_scaled_dot_product_attention(query, view, 1);
  EXPECT_EQ(output.shape(), (Shape{2, 0, 3}));
  EXPECT_EQ(output.numel(), 0U);
}

TEST(PagedAttentionTest, RejectsEmptyHistoryShapeOffsetDTypeAndAliasingErrors) {
  kv_cache::PagedKvCache empty_cache(kv_cache::PagedKvCacheConfig{1, 1, 2, 2, 1});
  const kv_cache::PagedSequenceHandle empty_sequence = empty_cache.create_sequence();
  const Tensor query = Tensor::zeros({1, 1, 2});
  {
    auto empty_view = empty_cache.read_layer(empty_sequence, 0);
    EXPECT_THROW(static_cast<void>(paged_score_workspace_shape(query, empty_view)), Error);
  }

  kv_cache::PagedKvCache cache(kv_cache::PagedKvCacheConfig{1, 1, 2, 2, 1});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  const Tensor token = Tensor::zeros({1, 1, 2});
  cache.append(sequence, token, token);
  cache.append(sequence, token, token);
  auto view = cache.read_layer(sequence, 0);

  EXPECT_THROW(static_cast<void>(paged_score_workspace_shape(Tensor::zeros({1, 2}), view)), Error);
  EXPECT_THROW(static_cast<void>(paged_score_workspace_shape(Tensor::zeros({2, 1, 2}), view)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_score_workspace_shape(Tensor::zeros({1, 1, 3}), view)),
               Error);
  EXPECT_THROW(static_cast<void>(
                   paged_score_workspace_shape(Tensor::zeros({1, 1, 2}, DType::kFloat16), view)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(query, view, 2)), Error);
  EXPECT_THROW(
      static_cast<void>(paged_scaled_dot_product_attention(Tensor::zeros({1, 2, 2}), view, 1)),
      Error);

  Tensor output = Tensor::empty({1, 1, 2});
  Tensor workspace = Tensor::empty({1, 1, 2});
  Tensor wrong_output = Tensor::empty({1, 1, 3});
  Tensor wrong_workspace = Tensor::empty({1, 1, 1});
  EXPECT_THROW(paged_scaled_dot_product_attention_out(query, view, 1, wrong_output, workspace),
               Error);
  EXPECT_THROW(paged_scaled_dot_product_attention_out(query, view, 1, output, wrong_workspace),
               Error);

  Tensor output_alias = query;
  EXPECT_THROW(paged_scaled_dot_product_attention_out(query, view, 1, output_alias, workspace),
               Error);
  Tensor shared_output_workspace = Tensor::empty({1, 1, 2});
  EXPECT_THROW(paged_scaled_dot_product_attention_out(query, view, 1, shared_output_workspace,
                                                      shared_output_workspace),
               Error);
}

} // namespace
} // namespace helix::attention::cpu
