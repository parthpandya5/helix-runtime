#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/transformer.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/memory.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace helix::cuda::kernels {
namespace {

void expect_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected_host.shape());
  for (std::size_t index = 0; index < expected_host.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected_host.data_as<float>()[index], tolerance)
        << "at index " << index;
  }
}

std::pair<Tensor, Tensor> make_rope_cache(Dimension tokens, Dimension head_dimension) {
  const auto elements =
      static_cast<std::size_t>(tokens) * static_cast<std::size_t>(head_dimension / 2);
  std::vector<float> cosine(elements);
  std::vector<float> sine(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const float angle = static_cast<float>(index) * 0.137F;
    cosine[index] = std::cos(angle);
    sine[index] = std::sin(angle);
  }
  return {Tensor::from_values({tokens, head_dimension / 2}, cosine),
          Tensor::from_values({tokens, head_dimension / 2}, sine)};
}

class CudaTransformerTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_F(CudaTransformerTest, RmsNormMatchesCpuAcrossOddWidthsAndSupportsInPlace) {
  constexpr std::array kShapes{std::array<Dimension, 2>{1, 1}, std::array<Dimension, 2>{3, 37},
                               std::array<Dimension, 2>{4, 513}};
  Stream stream;
  for (const auto shape : kShapes) {
    const Tensor input_host = Tensor::random_uniform({shape[0], shape[1]}, 2'001, -3.0F, 3.0F);
    const Tensor weight_host = Tensor::random_uniform({shape[1]}, 2'002, 0.25F, 1.75F);
    const Tensor expected = cpu::rms_norm(input_host, weight_host, 1.0e-5F);
    const Tensor weight = weight_host.to(Device::cuda(0));

    Tensor input = input_host.to(Device::cuda(0));
    const Tensor output = rms_norm(input, weight, 1.0e-5F, stream);
    stream.synchronize();
    expect_close(output, expected, 2.0e-5F);

    rms_norm_out(input, weight, 1.0e-5F, input, stream);
    stream.synchronize();
    expect_close(input, expected, 2.0e-5F);
  }
}

TEST_F(CudaTransformerTest, RmsNormAcceptsEmptyOuterDimension) {
  const Tensor input = Tensor::zeros({0, 37}, DType::kFloat32, Device::cuda(0));
  const Tensor weight = Tensor::zeros({37}, DType::kFloat32, Device::cuda(0));
  Stream stream;
  const Tensor output = rms_norm(input, weight, 1.0e-5F, stream);
  EXPECT_EQ(output.shape(), (Shape{0, 37}));
  EXPECT_EQ(output.numel(), 0U);
}

TEST_F(CudaTransformerTest, SwiGluMatchesCpuAtBoundaryAndNonBlockSizes) {
  constexpr std::array<Dimension, 5> kSizes{0, 1, 31, 257, 4'099};
  Stream stream;
  for (const Dimension size : kSizes) {
    const Tensor gate_host = Tensor::random_uniform({size}, 3'001, -10.0F, 10.0F);
    const Tensor up_host = Tensor::random_uniform({size}, 3'002, -2.0F, 2.0F);
    const Tensor expected = cpu::swiglu(gate_host, up_host);
    Tensor gate = gate_host.to(Device::cuda(0));
    const Tensor up = up_host.to(Device::cuda(0));

    swiglu_out(gate, up, gate, stream);
    stream.synchronize();
    expect_close(gate, expected, 3.0e-6F);
  }
}

TEST_F(CudaTransformerTest, HeadLayoutRoundTripsCpuOracleAcrossOddAndEmptyShapes) {
  struct Shape3 final {
    Dimension tokens;
    Dimension heads;
    Dimension head_dimension;
  };
  constexpr std::array kShapes{Shape3{0, 3, 5}, Shape3{1, 1, 1}, Shape3{3, 5, 7},
                               Shape3{17, 13, 37}, Shape3{257, 3, 5}};
  Stream stream;
  for (const Shape3 shape : kShapes) {
    const Tensor input_host =
        Tensor::random_uniform({shape.tokens, shape.heads, shape.head_dimension}, 3'501);
    const Tensor expected_head_major = cpu::token_heads_to_head_major(input_host);
    const Tensor input = input_host.to(Device::cuda(0));

    const Tensor head_major = token_heads_to_head_major(input, stream);
    stream.synchronize();
    expect_close(head_major, expected_head_major, 0.0F);

    const Tensor round_trip = head_major_to_token_heads(head_major, stream);
    stream.synchronize();
    expect_close(round_trip, input_host, 0.0F);
  }
}

TEST_F(CudaTransformerTest, HeadLayoutRejectsMetadataOverlapAndMovedStreamErrors) {
  Stream stream;
  const Tensor input = Tensor::zeros({2, 3, 5}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({3, 2, 5}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(token_heads_to_head_major_out(
                   Tensor::zeros({2, 15}, DType::kFloat32, Device::cuda(0)), output, stream),
               Error);
  EXPECT_THROW(token_heads_to_head_major_out(
                   Tensor::zeros({2, 3, 5}, DType::kUInt32, Device::cuda(0)), output, stream),
               Error);
  Tensor wrong_shape = Tensor::zeros({2, 3, 5}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(token_heads_to_head_major_out(input, wrong_shape, stream), Error);
  EXPECT_THROW(token_heads_to_head_major_out(Tensor::zeros({2, 3, 5}), output, stream), Error);

  Tensor exact_alias = Tensor::zeros({2, 2, 3}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(token_heads_to_head_major_out(exact_alias, exact_alias, stream), Error);

  const auto allocation = std::make_shared<Buffer>(1'536, 0);
  const Tensor partial_input = Tensor::from_mutable_cuda_buffer(
      {2, 2, 64}, DType::kFloat32, allocation->data(), 1'024, Device::cuda(0), allocation, 256);
  Tensor partial_output = Tensor::from_mutable_cuda_buffer(
      {2, 2, 64}, DType::kFloat32,
      static_cast<void*>(static_cast<std::byte*>(allocation->data()) + 512), 1'024, Device::cuda(0),
      allocation, 256);
  EXPECT_THROW(token_heads_to_head_major_out(partial_input, partial_output, stream), Error);
  EXPECT_THROW(head_major_to_token_heads_out(partial_input, partial_output, stream), Error);

  Stream moved_to = std::move(stream);
  EXPECT_THROW(token_heads_to_head_major_out(input, output, stream), Error);
  token_heads_to_head_major_out(input, output, moved_to);
  moved_to.synchronize();
}

TEST_F(CudaTransformerTest, RopeMatchesCpuForOddPairTaskCountsAndInPlace) {
  struct Shape3 {
    Dimension tokens;
    Dimension heads;
    Dimension head_dimension;
  };
  constexpr std::array kShapes{Shape3{1, 1, 2}, Shape3{3, 3, 6}, Shape3{5, 7, 34}, Shape3{0, 2, 6}};
  Stream stream;
  for (const Shape3 shape : kShapes) {
    const Tensor input_host =
        Tensor::random_uniform({shape.tokens, shape.heads, shape.head_dimension}, 4'001);
    auto [cosine_host, sine_host] = make_rope_cache(shape.tokens, shape.head_dimension);
    const Tensor expected = cpu::rope(input_host, cosine_host, sine_host);
    Tensor input = input_host.to(Device::cuda(0));
    const Tensor cosine = cosine_host.to(Device::cuda(0));
    const Tensor sine = sine_host.to(Device::cuda(0));

    rope_out(input, cosine, sine, input, stream);
    stream.synchronize();
    expect_close(input, expected, 2.0e-6F);
  }
}

TEST_F(CudaTransformerTest, LinearDelegatesToBothEducationalGemmPaths) {
  const Tensor input_host = Tensor::random_uniform({17, 19}, 5'001);
  const Tensor weight_host = Tensor::random_uniform({19, 23}, 5'002);
  const Tensor expected = cpu::linear(input_host, weight_host);
  const Tensor input = input_host.to(Device::cuda(0));
  const Tensor weight = weight_host.to(Device::cuda(0));
  Stream stream;

  for (const MatmulAlgorithm algorithm : {MatmulAlgorithm::kNaive, MatmulAlgorithm::kTiled}) {
    const Tensor output = linear(input, weight, stream, algorithm);
    stream.synchronize();
    expect_close(output, expected, 2.0e-5F);
  }
}

TEST_F(CudaTransformerTest, RejectsInvalidTransformerContractsBeforeLaunch) {
  Stream stream;
  const Tensor input = Tensor::zeros({2, 6}, DType::kFloat32, Device::cuda(0));
  const Tensor weight = Tensor::zeros({6}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({2, 6}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(rms_norm_out(input, Tensor::zeros({5}, DType::kFloat32, Device::cuda(0)), 1.0e-5F,
                            output, stream),
               Error);
  EXPECT_THROW(rms_norm_out(input, weight, 0.0F, output, stream), Error);
  const Tensor vector_input = Tensor::zeros({6}, DType::kFloat32, Device::cuda(0));
  Tensor aliased_weight = weight;
  EXPECT_THROW(rms_norm_out(vector_input, aliased_weight, 1.0e-5F, aliased_weight, stream), Error);

  EXPECT_THROW(
      swiglu_out(input, Tensor::zeros({12}, DType::kFloat32, Device::cuda(0)), output, stream),
      Error);

  EXPECT_THROW(rope_out(Tensor::zeros({2, 5}, DType::kFloat32, Device::cuda(0)),
                        Tensor::zeros({2, 2}, DType::kFloat32, Device::cuda(0)),
                        Tensor::zeros({2, 2}, DType::kFloat32, Device::cuda(0)), output, stream),
               Error);
  EXPECT_THROW(rope_out(input, Tensor::zeros({1, 3}, DType::kFloat32, Device::cuda(0)),
                        Tensor::zeros({2, 3}, DType::kFloat32, Device::cuda(0)), output, stream),
               Error);
  EXPECT_THROW(
      rope_out(Tensor::zeros({6}), Tensor::zeros({6, 3}), Tensor::zeros({6, 3}), output, stream),
      Error);

  const Tensor cpu_input = Tensor::zeros({2, 6});
  EXPECT_THROW(swiglu_out(cpu_input, cpu_input, output, stream), Error);
}

} // namespace
} // namespace helix::cuda::kernels
