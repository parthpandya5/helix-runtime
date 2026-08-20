#include "helix/attention/cuda.hpp"
#include "helix/attention/reference.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>

namespace helix::attention::cuda {
namespace {

void expect_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected_host.shape());
  for (std::size_t index = 0; index < expected_host.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected_host.data_as<float>()[index], tolerance)
        << "at index " << index;
  }
}

class CudaAttentionTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!helix::cuda::runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    helix::cuda::set_device(0);
  }
};

TEST_F(CudaAttentionTest, MatchesKnownCpuCausalResultAndProbabilities) {
  const Tensor query_host = Tensor::from_values({1, 2, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F});
  const Tensor key_host = query_host.clone();
  const Tensor value_host = Tensor::from_values({1, 2, 2}, std::array{10.0F, 0.0F, 0.0F, 20.0F});
  Tensor expected_output = Tensor::empty({1, 2, 2});
  Tensor expected_probabilities = Tensor::empty({1, 2, 2});
  cpu::scaled_dot_product_attention_out(query_host, key_host, value_host, 0, expected_output,
                                        expected_probabilities);
  const Tensor query = query_host.to(Device::cuda(0));
  const Tensor key = key_host.to(Device::cuda(0));
  const Tensor value = value_host.to(Device::cuda(0));
  Tensor output = Tensor::empty({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::empty({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;

  scaled_dot_product_attention_out(query, key, value, 0, output, workspace, stream);
  stream.synchronize();
  expect_close(output, expected_output, 2.0e-5F);
  expect_close(workspace, expected_probabilities, 2.0e-6F);
}

TEST_F(CudaAttentionTest, MatchesCpuForDecodeAndOddNonTileAlignedShapes) {
  struct Problem {
    Dimension heads;
    Dimension queries;
    Dimension keys;
    Dimension head_dimension;
    Dimension value_dimension;
  };
  constexpr std::array kProblems{Problem{1, 1, 3, 1, 2}, Problem{2, 3, 5, 5, 7},
                                 Problem{3, 7, 11, 37, 41}};
  helix::cuda::Stream stream;
  for (const Problem problem : kProblems) {
    const std::size_t position_offset = static_cast<std::size_t>(problem.keys - problem.queries);
    const Tensor query_host = Tensor::random_uniform(
        {problem.heads, problem.queries, problem.head_dimension}, 8'001, -1.0F, 1.0F);
    const Tensor key_host = Tensor::random_uniform(
        {problem.heads, problem.keys, problem.head_dimension}, 8'002, -1.0F, 1.0F);
    const Tensor value_host = Tensor::random_uniform(
        {problem.heads, problem.keys, problem.value_dimension}, 8'003, -1.0F, 1.0F);
    const Tensor expected =
        cpu::scaled_dot_product_attention(query_host, key_host, value_host, position_offset);
    const Tensor query = query_host.to(Device::cuda(0));
    const Tensor key = key_host.to(Device::cuda(0));
    const Tensor value = value_host.to(Device::cuda(0));
    const Tensor actual = scaled_dot_product_attention(query, key, value, position_offset, stream);

    stream.synchronize();
    const float tolerance = 2.0e-5F + static_cast<float>(problem.head_dimension) * 1.0e-6F;
    expect_close(actual, expected, tolerance);
  }
}

TEST_F(CudaAttentionTest, AcceptsEmptyQueryWithoutLaunching) {
  const Tensor query = Tensor::zeros({2, 0, 5}, DType::kFloat32, Device::cuda(0));
  const Tensor key = Tensor::zeros({2, 7, 5}, DType::kFloat32, Device::cuda(0));
  const Tensor value = Tensor::zeros({2, 7, 3}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;
  const Tensor output = scaled_dot_product_attention(query, key, value, 7, stream);
  EXPECT_EQ(output.shape(), (Shape{2, 0, 3}));
  EXPECT_EQ(output.numel(), 0U);
  EXPECT_EQ(score_workspace_bytes(query, key), 0U);
}

TEST_F(CudaAttentionTest, RejectsInvalidShapesOffsetsDevicesAndAliasing) {
  const Tensor query = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  const Tensor key = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  const Tensor value = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;

  EXPECT_THROW(static_cast<void>(scaled_dot_product_attention(
                   Tensor::zeros({2, 2}, DType::kFloat32, Device::cuda(0)), key, value, 0, stream)),
               Error);
  EXPECT_THROW(
      static_cast<void>(scaled_dot_product_attention(
          query, Tensor::zeros({1, 2, 3}, DType::kFloat32, Device::cuda(0)), value, 0, stream)),
      Error);
  EXPECT_THROW(static_cast<void>(scaled_dot_product_attention(query, key, value, 1, stream)),
               Error);
  EXPECT_THROW(static_cast<void>(
                   scaled_dot_product_attention(Tensor::zeros({1, 2, 2}), key, value, 0, stream)),
               Error);

  Tensor output_alias = query;
  EXPECT_THROW(
      scaled_dot_product_attention_out(query, key, value, 0, output_alias, workspace, stream),
      Error);
  Tensor workspace_alias = query;
  EXPECT_THROW(
      scaled_dot_product_attention_out(query, key, value, 0, output, workspace_alias, stream),
      Error);
  EXPECT_THROW(scaled_dot_product_attention_out(query, key, value, 0, workspace, workspace, stream),
               Error);
}

} // namespace
} // namespace helix::attention::cuda
