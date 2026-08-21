#include "helix/attention/paged_reference.hpp"
#include "helix/attention/reference.hpp"
#include "helix/kv_cache/paged.hpp"

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
