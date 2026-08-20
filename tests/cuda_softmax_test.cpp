#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/softmax.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>

namespace helix::cuda::kernels {
namespace {

constexpr std::array kAlgorithms{SoftmaxAlgorithm::kNaive, SoftmaxAlgorithm::kSharedMemory};

void expect_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected_host.shape());
  for (std::size_t index = 0; index < expected_host.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected_host.data_as<float>()[index], tolerance)
        << "at index " << index;
  }
}

class CudaSoftmaxTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_F(CudaSoftmaxTest, EveryAlgorithmMatchesCpuAcrossOddWidths) {
  const std::array<Shape, 5> kShapes{Shape{0, 7}, Shape{1, 1}, Shape{3, 31}, Shape{5, 257},
                                     Shape{2, 1023}};
  Stream stream;

  for (const Shape& shape : kShapes) {
    const Tensor input_host = Tensor::random_uniform(shape, 801, -12.0F, 12.0F);
    const Tensor expected = cpu::softmax(input_host);
    const Tensor input = input_host.to(Device::cuda(0));
    for (const SoftmaxAlgorithm algorithm : kAlgorithms) {
      const Tensor actual = softmax(input, stream, algorithm);
      stream.synchronize();
      expect_close(actual, expected, 2.0e-5F);
    }
  }
}

TEST_F(CudaSoftmaxTest, RowsRemainNormalizedForLargeMagnitudeInputs) {
  const Tensor input_host = Tensor::random_uniform({4, 513}, 802, -80.0F, 80.0F);
  const Tensor input = input_host.to(Device::cuda(0));
  Stream stream;

  for (const SoftmaxAlgorithm algorithm : kAlgorithms) {
    const Tensor result = softmax(input, stream, algorithm);
    stream.synchronize();
    const Tensor host = result.to(Device::cpu());
    for (std::size_t row = 0; row < 4; ++row) {
      double probability_sum = 0.0;
      for (std::size_t column = 0; column < 513; ++column) {
        const float probability = host.data_as<float>()[row * 513 + column];
        EXPECT_GE(probability, 0.0F);
        EXPECT_LE(probability, 1.0F);
        probability_sum += probability;
      }
      EXPECT_NEAR(probability_sum, 1.0, 2.0e-5);
    }
  }
}

TEST_F(CudaSoftmaxTest, DefinesNonFiniteRowsLikeCpuOracle) {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const Tensor input_host = Tensor::from_values(
      {3, 4}, std::array{infinity, 2.0F, infinity, -infinity, -infinity, -infinity, -infinity,
                         -infinity, 1.0F, nan, 3.0F, 4.0F});
  const Tensor input = input_host.to(Device::cuda(0));
  Stream stream;

  for (const SoftmaxAlgorithm algorithm : kAlgorithms) {
    const Tensor device_result = softmax(input, stream, algorithm);
    stream.synchronize();
    const Tensor result = device_result.to(Device::cpu());
    EXPECT_FLOAT_EQ(result.data_as<float>()[0], 0.5F);
    EXPECT_FLOAT_EQ(result.data_as<float>()[1], 0.0F);
    EXPECT_FLOAT_EQ(result.data_as<float>()[2], 0.5F);
    EXPECT_FLOAT_EQ(result.data_as<float>()[3], 0.0F);
    for (std::size_t index = 4; index < result.numel(); ++index) {
      EXPECT_TRUE(std::isnan(result.data_as<float>()[index]));
    }
  }
}

TEST_F(CudaSoftmaxTest, SupportsExactInPlaceExecution) {
  const Tensor input_host = Tensor::random_uniform({3, 257}, 803, -4.0F, 4.0F);
  const Tensor expected = cpu::softmax(input_host);
  Stream stream;

  for (const SoftmaxAlgorithm algorithm : kAlgorithms) {
    Tensor values = input_host.to(Device::cuda(0));
    softmax_out(values, values, stream, algorithm);
    stream.synchronize();
    expect_close(values, expected, 2.0e-5F);
  }
}

TEST_F(CudaSoftmaxTest, RejectsInvalidContracts) {
  Stream stream;
  EXPECT_THROW(
      static_cast<void>(softmax(Tensor::zeros({}, DType::kFloat32, Device::cuda(0)), stream)),
      Error);
  EXPECT_THROW(
      static_cast<void>(softmax(Tensor::zeros({2, 0}, DType::kFloat32, Device::cuda(0)), stream)),
      Error);
  EXPECT_THROW(
      static_cast<void>(softmax(Tensor::zeros({2, 3}, DType::kFloat16, Device::cuda(0)), stream)),
      Error);

  const Tensor input = Tensor::zeros({2, 3}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({2, 3}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(softmax_out(input, output, stream, static_cast<SoftmaxAlgorithm>(999)), Error);

  const Tensor empty_input = Tensor::zeros({0, 3}, DType::kFloat32, Device::cuda(0));
  Tensor empty_output = Tensor::zeros({0, 3}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(softmax_out(empty_input, empty_output, stream, static_cast<SoftmaxAlgorithm>(999)),
               Error);
}

} // namespace
} // namespace helix::cuda::kernels
