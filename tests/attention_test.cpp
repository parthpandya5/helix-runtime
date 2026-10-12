#include "helix/attention/reference.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>

namespace helix::attention::cpu {
namespace {

void expect_values_near(const Tensor& tensor, std::span<const float> expected,
                        float tolerance = 1.0e-5F) {
  ASSERT_EQ(tensor.numel(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(tensor.data_as<float>()[index], expected[index], tolerance) << "at index " << index;
  }
}

TEST(AttentionTest, ComputesKnownCausalAttentionAndMaterializesProbabilities) {
  const Tensor query = Tensor::from_values({1, 2, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F});
  const Tensor key = query.clone();
  const Tensor value = Tensor::from_values({1, 2, 2}, std::array{10.0F, 0.0F, 0.0F, 20.0F});
  Tensor output = Tensor::empty({1, 2, 2});
  Tensor workspace = Tensor::empty({1, 2, 2});

  scaled_dot_product_attention_out(query, key, value, 0, output, workspace);

  const double exponential = std::exp(1.0 / std::sqrt(2.0));
  const float first_probability = static_cast<float>(1.0 / (1.0 + exponential));
  const float second_probability = static_cast<float>(exponential / (1.0 + exponential));
  expect_values_near(workspace, std::array{1.0F, 0.0F, first_probability, second_probability});
  expect_values_near(
      output, std::array{10.0F, 0.0F, 10.0F * first_probability, 20.0F * second_probability});
  EXPECT_EQ(score_workspace_shape(query, key), (Shape{1, 2, 2}));
  EXPECT_EQ(score_workspace_bytes(query, key), 4U * sizeof(float));

  Tensor baseline_output = Tensor::empty({1, 2, 2});
  Tensor baseline_workspace = Tensor::empty({1, 2, 2});
  scaled_dot_product_attention_dimension_major_out(query, key, value, 0, baseline_output,
                                                   baseline_workspace);
  expect_values_near(baseline_output, std::array{10.0F, 0.0F, 10.0F * first_probability,
                                                 20.0F * second_probability});
  expect_values_near(baseline_workspace,
                     std::array{1.0F, 0.0F, first_probability, second_probability});
}

TEST(AttentionTest, QueryOffsetModelsSingleTokenDecodeAgainstFullKvHistory) {
  const Tensor query = Tensor::from_values({1, 1, 1}, std::array{1.0F});
  const Tensor key = Tensor::from_values({1, 3, 1}, std::array{0.0F, 1.0F, 2.0F});
  const Tensor value =
      Tensor::from_values({1, 3, 2}, std::array{1.0F, 2.0F, 10.0F, 20.0F, 100.0F, 200.0F});
  const Tensor output = scaled_dot_product_attention(query, key, value, 2);

  const double denominator = 1.0 + std::exp(1.0) + std::exp(2.0);
  const float expected_first =
      static_cast<float>((1.0 + 10.0 * std::exp(1.0) + 100.0 * std::exp(2.0)) / denominator);
  const float expected_second =
      static_cast<float>((2.0 + 20.0 * std::exp(1.0) + 200.0 * std::exp(2.0)) / denominator);
  expect_values_near(output, std::array{expected_first, expected_second}, 2.0e-5F);
}

TEST(AttentionTest, CausalMaskDoesNotReadFutureNonFiniteValues) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor query = Tensor::from_values({1, 1, 1}, std::array{1.0F});
  const Tensor key = Tensor::from_values({1, 2, 1}, std::array{0.0F, nan});
  const Tensor value = Tensor::from_values({1, 2, 2}, std::array{3.0F, 4.0F, nan, nan});

  Tensor output = Tensor::empty({1, 1, 2});
  Tensor workspace = Tensor::empty({1, 1, 2});
  scaled_dot_product_attention_out(query, key, value, 0, output, workspace);
  expect_values_near(output, std::array{3.0F, 4.0F});
  expect_values_near(workspace, std::array{1.0F, 0.0F});

  scaled_dot_product_attention_dimension_major_out(query, key, value, 0, output, workspace);
  expect_values_near(output, std::array{3.0F, 4.0F});
}

TEST(AttentionTest, HandlesMultipleHeadsOddDimensionsAndEmptyOutput) {
  const Tensor query = Tensor::random_uniform({2, 3, 5}, 6'001, -1.0F, 1.0F);
  const Tensor key = Tensor::random_uniform({2, 5, 5}, 6'002, -1.0F, 1.0F);
  const Tensor value = Tensor::random_uniform({2, 5, 7}, 6'003, -1.0F, 1.0F);
  Tensor output = Tensor::empty({2, 3, 7});
  Tensor workspace = Tensor::empty({2, 3, 5});
  scaled_dot_product_attention_out(query, key, value, 2, output, workspace);

  for (std::size_t row = 0; row < 6; ++row) {
    float probability_sum = 0.0F;
    for (std::size_t column = 0; column < 5; ++column) {
      const float probability = workspace.data_as<float>()[row * 5 + column];
      EXPECT_GE(probability, 0.0F);
      probability_sum += probability;
    }
    EXPECT_NEAR(probability_sum, 1.0F, 1.0e-6F);
  }
  for (std::size_t index = 0; index < output.numel(); ++index) {
    EXPECT_TRUE(std::isfinite(output.data_as<float>()[index]));
  }

  const Tensor empty_query = Tensor::zeros({2, 0, 5});
  const Tensor empty_output = scaled_dot_product_attention(empty_query, key, value, 5);
  EXPECT_EQ(empty_output.shape(), (Shape{2, 0, 7}));
  EXPECT_EQ(empty_output.numel(), 0U);
  EXPECT_EQ(score_workspace_bytes(empty_query, key), 0U);

  const Tensor no_heads = scaled_dot_product_attention(
      Tensor::zeros({0, 3, 5}), Tensor::zeros({0, 5, 5}), Tensor::zeros({0, 5, 7}), 2);
  EXPECT_EQ(no_heads.shape(), (Shape{0, 3, 7}));
}

TEST(AttentionTest, RejectsShapeOffsetDTypeAndAliasingViolations) {
  const Tensor query = Tensor::zeros({1, 2, 2});
  const Tensor key = Tensor::zeros({1, 2, 2});
  const Tensor value = Tensor::zeros({1, 2, 2});
  Tensor output = Tensor::zeros({1, 2, 2});
  Tensor workspace = Tensor::zeros({1, 2, 2});

  EXPECT_THROW(
      static_cast<void>(scaled_dot_product_attention(Tensor::zeros({2, 2}), key, value, 0)), Error);
  EXPECT_THROW(
      static_cast<void>(scaled_dot_product_attention(query, Tensor::zeros({2, 2, 2}), value, 0)),
      Error);
  EXPECT_THROW(
      static_cast<void>(scaled_dot_product_attention(query, Tensor::zeros({1, 2, 3}), value, 0)),
      Error);
  EXPECT_THROW(
      static_cast<void>(scaled_dot_product_attention(query, key, Tensor::zeros({1, 3, 2}), 0)),
      Error);
  EXPECT_THROW(static_cast<void>(scaled_dot_product_attention(query, key, value, 1)), Error);
  EXPECT_THROW(
      static_cast<void>(scaled_dot_product_attention(Tensor::zeros({1, 0, 2}), key, value, 3)),
      Error);
  EXPECT_THROW(static_cast<void>(scaled_dot_product_attention(query, Tensor::zeros({1, 0, 2}),
                                                              Tensor::zeros({1, 0, 2}), 0)),
               Error);
  EXPECT_THROW(static_cast<void>(scaled_dot_product_attention(
                   Tensor::zeros({1, 2, 2}, DType::kFloat16), key, value, 0)),
               Error);

  Tensor output_alias = query;
  EXPECT_THROW(scaled_dot_product_attention_out(query, key, value, 0, output_alias, workspace),
               Error);
  Tensor workspace_alias = query;
  EXPECT_THROW(scaled_dot_product_attention_out(query, key, value, 0, output, workspace_alias),
               Error);
  EXPECT_THROW(scaled_dot_product_attention_out(query, key, value, 0, workspace, workspace), Error);

  Tensor wrong_output = Tensor::zeros({1, 2, 3});
  EXPECT_THROW(scaled_dot_product_attention_out(query, key, value, 0, wrong_output, workspace),
               Error);

  struct alignas(256) Buffer final {
    std::array<std::byte, 768> bytes{};
  };
  const auto owner = std::make_shared<Buffer>();
  const Tensor partial_query = Tensor::from_mutable_cpu_buffer(
      {1, 1, 128}, DType::kFloat32, owner->bytes.data(), 512, owner, 256);
  Tensor partial_output = Tensor::from_mutable_cpu_buffer(
      {1, 1, 128}, DType::kFloat32, owner->bytes.data() + 256, 512, owner, 256);
  Tensor partial_workspace = Tensor::zeros({1, 1, 1});
  EXPECT_THROW(scaled_dot_product_attention_out(partial_query, Tensor::zeros({1, 1, 128}),
                                                Tensor::zeros({1, 1, 128}), 0, partial_output,
                                                partial_workspace),
               Error);
}

} // namespace
} // namespace helix::attention::cpu
