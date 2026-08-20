#include "helix/kernels/cpu/reference.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace helix::cpu {
namespace {

void expect_values_near(const Tensor& tensor, std::span<const float> expected,
                        float tolerance = 1.0e-6F) {
  ASSERT_EQ(tensor.numel(), expected.size());
  const auto* actual = tensor.data_as<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], tolerance) << "at index " << index;
  }
}

TEST(ElementwiseTest, AddAndMultiplyMatchExpectedValues) {
  const Tensor lhs = Tensor::from_values({5}, std::array{-2.0F, -0.0F, 1.5F, 3.0F, 7.0F});
  const Tensor rhs = Tensor::from_values({5}, std::array{4.0F, 2.0F, -2.0F, 0.5F, -1.0F});

  expect_values_near(add(lhs, rhs), std::array{2.0F, 2.0F, -0.5F, 3.5F, 6.0F});
  expect_values_near(multiply(lhs, rhs), std::array{-8.0F, -0.0F, -3.0F, 1.5F, -7.0F});
}

TEST(ElementwiseTest, ReluAndSiluHandleExtremeInputs) {
  const Tensor input =
      Tensor::from_values({7}, std::array{-100.0F, -2.0F, -0.0F, 0.0F, 1.0F, 2.0F, 100.0F});
  const Tensor relu_result = relu(input);
  expect_values_near(relu_result, std::array{0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 2.0F, 100.0F});

  const Tensor silu_result = silu(input);
  const auto* values = silu_result.data_as<float>();
  EXPECT_NEAR(values[0], 0.0F, 1.0e-6F);
  EXPECT_NEAR(values[1], -0.23840584F, 1.0e-6F);
  EXPECT_EQ(values[2], 0.0F);
  EXPECT_EQ(values[3], 0.0F);
  EXPECT_NEAR(values[4], 0.7310586F, 1.0e-6F);
  EXPECT_NEAR(values[5], 1.7615942F, 1.0e-6F);
  EXPECT_NEAR(values[6], 100.0F, 1.0e-5F);
}

TEST(ElementwiseTest, SupportsEmptyAndAliasedOutput) {
  const Tensor empty = Tensor::zeros({0, 3});
  EXPECT_EQ(add(empty, empty).numel(), 0U);
  EXPECT_EQ(relu(empty).numel(), 0U);

  Tensor lhs = Tensor::from_values({3}, std::array{1.0F, 2.0F, 3.0F});
  const Tensor rhs = Tensor::from_values({3}, std::array{4.0F, 5.0F, 6.0F});
  add_out(lhs, rhs, lhs);
  expect_values_near(lhs, std::array{5.0F, 7.0F, 9.0F});
}

TEST(ElementwiseTest, RejectsBroadcastingDTypeAndOutputErrors) {
  const Tensor lhs = Tensor::zeros({2, 3});
  const Tensor wrong_shape = Tensor::zeros({3});
  const Tensor half = Tensor::zeros({2, 3}, DType::kFloat16);
  Tensor output = Tensor::zeros({2, 3});
  Tensor wrong_output = Tensor::zeros({6});

  EXPECT_THROW(static_cast<void>(add(lhs, wrong_shape)), Error);
  EXPECT_THROW(add_out(lhs, lhs, wrong_output), Error);
  EXPECT_THROW(relu_out(half, output), Error);
}

TEST(ReductionTest, SumsAllElementsWithWideAccumulator) {
  const Tensor input = Tensor::from_values({5}, std::array{1.0e8F, 1.0F, -1.0e8F, 2.0F, 3.0F});
  const Tensor result = sum(input);
  EXPECT_EQ(result.shape(), Shape{});
  EXPECT_FLOAT_EQ(result.data_as<float>()[0], 6.0F);

  const Tensor empty = Tensor::zeros({0});
  EXPECT_FLOAT_EQ(sum(empty).data_as<float>()[0], 0.0F);
}

TEST(ReductionTest, ReducesLastDimensionWithOptionalRetention) {
  const Tensor input = Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, -1.0F, 4.0F, 0.5F});
  const Tensor reduced = sum_last_dim(input);
  const Tensor kept = sum_last_dim(input, true);

  EXPECT_EQ(reduced.shape(), (Shape{2}));
  EXPECT_EQ(kept.shape(), (Shape{2, 1}));
  expect_values_near(reduced, std::array{6.0F, 3.5F});
  expect_values_near(kept, std::array{6.0F, 3.5F});
}

TEST(ReductionTest, DefinesEmptyLastDimensionAsZeroSum) {
  const Tensor input = Tensor::zeros({3, 0});
  const Tensor result = sum_last_dim(input);
  expect_values_near(result, std::array{0.0F, 0.0F, 0.0F});
}

TEST(ReductionTest, ValidatesRankAndAliasing) {
  const Tensor scalar = Tensor::from_values({}, std::array{2.0F});
  EXPECT_THROW(static_cast<void>(sum_last_dim(scalar)), Error);

  const Tensor input = Tensor::from_values({2, 1}, std::array{1.0F, 2.0F});
  Tensor aliased_output = input.view({2});
  EXPECT_THROW(sum_last_dim_out(input, aliased_output), Error);
}

TEST(SoftmaxTest, ComputesStableRowWiseProbabilities) {
  const Tensor input =
      Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, 1000.0F, 1001.0F, 999.0F});
  const Tensor output = softmax(input);

  constexpr std::array expected{0.09003057F, 0.24472848F, 0.66524094F,
                                0.24472848F, 0.66524094F, 0.09003057F};
  expect_values_near(output, expected, 2.0e-6F);
  const auto* values = output.data_as<float>();
  EXPECT_NEAR(values[0] + values[1] + values[2], 1.0F, 1.0e-6F);
  EXPECT_NEAR(values[3] + values[4] + values[5], 1.0F, 1.0e-6F);
}

TEST(SoftmaxTest, SupportsInPlaceExecution) {
  Tensor values = Tensor::from_values({1, 3}, std::array{0.0F, 0.0F, 0.0F});
  softmax_out(values, values);
  expect_values_near(values, std::array{1.0F / 3.0F, 1.0F / 3.0F, 1.0F / 3.0F});
}

TEST(SoftmaxTest, DefinesNonFiniteRowsExplicitly) {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor positive_infinity =
      Tensor::from_values({1, 4}, std::array{infinity, 2.0F, infinity, -infinity});
  expect_values_near(softmax(positive_infinity), std::array{0.5F, 0.0F, 0.5F, 0.0F});

  const Tensor all_negative_infinity =
      Tensor::from_values({1, 2}, std::array{-infinity, -infinity});
  const Tensor negative_result = softmax(all_negative_infinity);
  EXPECT_TRUE(std::isnan(negative_result.data_as<float>()[0]));
  EXPECT_TRUE(std::isnan(negative_result.data_as<float>()[1]));

  const Tensor nan_input = Tensor::from_values({1, 2}, std::array{1.0F, nan});
  const Tensor nan_result = softmax(nan_input);
  EXPECT_TRUE(std::isnan(nan_result.data_as<float>()[0]));
  EXPECT_TRUE(std::isnan(nan_result.data_as<float>()[1]));
}

TEST(SoftmaxTest, RejectsScalarAndEmptyLastDimension) {
  EXPECT_THROW(static_cast<void>(softmax(Tensor::from_values({}, std::array{1.0F}))), Error);
  EXPECT_THROW(static_cast<void>(softmax(Tensor::zeros({2, 0}))), Error);
}

TEST(MatmulTest, MultipliesKnownMatrices) {
  const Tensor lhs = Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor rhs = Tensor::from_values({3, 2}, std::array{7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor output = matmul(lhs, rhs);

  EXPECT_EQ(output.shape(), (Shape{2, 2}));
  expect_values_near(output, std::array{58.0F, 64.0F, 139.0F, 154.0F});
}

TEST(MatmulTest, HandlesOddDimensions) {
  std::vector<float> lhs_values(15);
  std::vector<float> rhs_values(20);
  for (std::size_t index = 0; index < lhs_values.size(); ++index) {
    lhs_values[index] = static_cast<float>(index) * 0.25F - 1.0F;
  }
  for (std::size_t index = 0; index < rhs_values.size(); ++index) {
    rhs_values[index] = static_cast<float>(index % 7) * 0.2F - 0.4F;
  }

  const Tensor lhs = Tensor::from_values({3, 5}, lhs_values);
  const Tensor rhs = Tensor::from_values({5, 4}, rhs_values);
  const Tensor output = matmul(lhs, rhs);
  ASSERT_EQ(output.shape(), (Shape{3, 4}));

  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 4; ++column) {
      double expected = 0.0;
      for (std::size_t reduction = 0; reduction < 5; ++reduction) {
        expected += static_cast<double>(lhs_values[row * 5 + reduction]) *
                    static_cast<double>(rhs_values[reduction * 4 + column]);
      }
      EXPECT_NEAR(output.data_as<float>()[row * 4 + column], static_cast<float>(expected), 1.0e-6F);
    }
  }
}

TEST(MatmulTest, SupportsZeroInnerAndOuterDimensions) {
  const Tensor zero_inner = matmul(Tensor::zeros({2, 0}), Tensor::zeros({0, 3}));
  EXPECT_EQ(zero_inner.shape(), (Shape{2, 3}));
  expect_values_near(zero_inner, std::array{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});

  const Tensor zero_rows = matmul(Tensor::zeros({0, 5}), Tensor::zeros({5, 3}));
  EXPECT_EQ(zero_rows.shape(), (Shape{0, 3}));
  EXPECT_EQ(zero_rows.numel(), 0U);
}

TEST(MatmulTest, RejectsRankShapeAndAliasingErrors) {
  EXPECT_THROW(static_cast<void>(matmul(Tensor::zeros({2}), Tensor::zeros({2, 2}))), Error);
  EXPECT_THROW(static_cast<void>(matmul(Tensor::zeros({2, 3}), Tensor::zeros({4, 2}))), Error);

  Tensor lhs = Tensor::zeros({2, 2});
  const Tensor rhs = Tensor::zeros({2, 2});
  EXPECT_THROW(matmul_out(lhs, rhs, lhs), Error);
}

TEST(RmsNormTest, MatchesDefinitionAcrossRows) {
  const Tensor input = Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, -2.0F, 0.5F, 4.0F});
  const Tensor weight = Tensor::from_values({3}, std::array{1.0F, 0.5F, 2.0F});
  constexpr float kEpsilon = 1.0e-5F;
  const Tensor output = rms_norm(input, weight, kEpsilon);

  for (std::size_t row = 0; row < 2; ++row) {
    double square_sum = 0.0;
    for (std::size_t column = 0; column < 3; ++column) {
      const double value = input.data_as<float>()[row * 3 + column];
      square_sum += value * value;
    }
    const double scale = 1.0 / std::sqrt(square_sum / 3.0 + kEpsilon);
    for (std::size_t column = 0; column < 3; ++column) {
      const float expected = static_cast<float>(input.data_as<float>()[row * 3 + column] * scale *
                                                weight.data_as<float>()[column]);
      EXPECT_NEAR(output.data_as<float>()[row * 3 + column], expected, 1.0e-6F);
    }
  }
}

TEST(RmsNormTest, SupportsInputOutputAliasing) {
  Tensor input = Tensor::from_values({1, 2}, std::array{3.0F, 4.0F});
  const Tensor weight = Tensor::from_values({2}, std::array{1.0F, 1.0F});
  rms_norm_out(input, weight, 1.0e-5F, input);

  const float scale = 1.0F / std::sqrt(12.5F + 1.0e-5F);
  expect_values_near(input, std::array{3.0F * scale, 4.0F * scale});
}

TEST(RmsNormTest, ValidatesWeightShapeWidthAndEpsilon) {
  const Tensor input = Tensor::zeros({2, 3});
  const Tensor wrong_weight = Tensor::zeros({2});
  const Tensor matrix_weight = Tensor::zeros({1, 3});
  EXPECT_THROW(static_cast<void>(rms_norm(input, wrong_weight, 1.0e-5F)), Error);
  EXPECT_THROW(static_cast<void>(rms_norm(input, matrix_weight, 1.0e-5F)), Error);
  EXPECT_THROW(static_cast<void>(rms_norm(input, Tensor::zeros({3}), 0.0F)), Error);
  EXPECT_THROW(static_cast<void>(rms_norm(Tensor::zeros({2, 0}), Tensor::zeros({0}), 1.0e-5F)),
               Error);
}

TEST(SwiGluTest, ComputesSiluGateTimesUpProjection) {
  const Tensor gate = Tensor::from_values({4}, std::array{-2.0F, 0.0F, 1.0F, 3.0F});
  const Tensor up = Tensor::from_values({4}, std::array{0.5F, 2.0F, -1.0F, 4.0F});
  const Tensor output = swiglu(gate, up);

  constexpr std::array expected{-0.11920292F, 0.0F, -0.7310586F, 11.43089F};
  expect_values_near(output, expected, 2.0e-5F);
}

TEST(SwiGluTest, RejectsShapeMismatch) {
  EXPECT_THROW(static_cast<void>(swiglu(Tensor::zeros({2, 3}), Tensor::zeros({6}))), Error);
}

TEST(RopeTest, RotatesAdjacentPairsForEveryTokenAndHead) {
  const Tensor input = Tensor::from_values(
      {2, 2, 4}, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F,
                            12.0F, 13.0F, 14.0F, 15.0F, 16.0F});
  const Tensor cosine = Tensor::from_values({2, 2}, std::array{1.0F, 1.0F, 0.0F, 0.0F});
  const Tensor sine = Tensor::from_values({2, 2}, std::array{0.0F, 0.0F, 1.0F, 1.0F});

  const Tensor output = rope(input, cosine, sine);
  expect_values_near(output, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, -10.0F,
                                        9.0F, -12.0F, 11.0F, -14.0F, 13.0F, -16.0F, 15.0F});
}

TEST(RopeTest, SupportsOddWorkCountsEmptyTokensAndExactInPlaceExecution) {
  Tensor input = Tensor::random_uniform({3, 3, 6}, 9'001, -2.0F, 2.0F);
  std::array<float, 9> cosine_values{};
  std::array<float, 9> sine_values{};
  for (std::size_t index = 0; index < cosine_values.size(); ++index) {
    const float angle = static_cast<float>(index) * 0.17F;
    cosine_values[index] = std::cos(angle);
    sine_values[index] = std::sin(angle);
  }
  const Tensor cosine = Tensor::from_values({3, 3}, cosine_values);
  const Tensor sine = Tensor::from_values({3, 3}, sine_values);
  const Tensor expected = rope(input, cosine, sine);

  rope_out(input, cosine, sine, input);
  expect_values_near(input, std::span<const float>(expected.data_as<float>(), expected.numel()),
                     1.0e-6F);

  const Tensor empty = rope(Tensor::zeros({0, 3, 6}), Tensor::zeros({0, 3}), Tensor::zeros({0, 3}));
  EXPECT_EQ(empty.shape(), (Shape{0, 3, 6}));
  EXPECT_EQ(empty.numel(), 0U);
}

TEST(RopeTest, RejectsInvalidRankWidthCachesAndOutputShape) {
  EXPECT_THROW(
      static_cast<void>(rope(Tensor::zeros({4}), Tensor::zeros({4, 2}), Tensor::zeros({4, 2}))),
      Error);
  EXPECT_THROW(
      static_cast<void>(rope(Tensor::zeros({2, 5}), Tensor::zeros({2, 2}), Tensor::zeros({2, 2}))),
      Error);
  EXPECT_THROW(
      static_cast<void>(rope(Tensor::zeros({2, 0}), Tensor::zeros({2, 0}), Tensor::zeros({2, 0}))),
      Error);
  EXPECT_THROW(
      static_cast<void>(rope(Tensor::zeros({2, 4}), Tensor::zeros({1, 2}), Tensor::zeros({2, 2}))),
      Error);

  const Tensor input = Tensor::zeros({2, 4});
  Tensor wrong_output = Tensor::zeros({2, 2});
  EXPECT_THROW(rope_out(input, Tensor::zeros({2, 2}), Tensor::zeros({2, 2}), wrong_output), Error);
}

TEST(LinearTest, UsesInputByOutputFeatureWeightLayout) {
  const Tensor input = Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor weight =
      Tensor::from_values({3, 2}, std::array{7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  expect_values_near(linear(input, weight), std::array{58.0F, 64.0F, 139.0F, 154.0F});
  EXPECT_THROW(static_cast<void>(linear(input, Tensor::zeros({4, 2}))), Error);
}

} // namespace
} // namespace helix::cpu
