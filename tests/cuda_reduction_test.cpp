#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/reduction.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

namespace helix::cuda::kernels {
namespace {

constexpr std::array kAlgorithms{ReductionAlgorithm::kNaive, ReductionAlgorithm::kSharedMemory,
                                 ReductionAlgorithm::kWarpShuffle};

void expect_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected_host.shape());
  for (std::size_t index = 0; index < expected_host.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected_host.data_as<float>()[index], tolerance)
        << "at index " << index;
  }
}

class CudaReductionTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_F(CudaReductionTest, EveryAlgorithmMatchesCpuAcrossOddWidths) {
  const std::array<Shape, 5> kShapes{Shape{2, 0}, Shape{1, 1}, Shape{3, 31}, Shape{5, 257},
                                     Shape{2, 1023}};
  Stream stream;

  for (const Shape& shape : kShapes) {
    const Tensor input_host = Tensor::random_uniform(shape, 701, -1.0F, 1.0F);
    const Tensor expected = cpu::sum_last_dim(input_host);
    const Tensor input = input_host.to(Device::cuda(0));
    const auto width = static_cast<std::size_t>(shape.back());
    const float tolerance = 1.0e-4F + static_cast<float>(width) * 2.0e-7F;
    for (const ReductionAlgorithm algorithm : kAlgorithms) {
      const Tensor actual = sum_last_dim(input, stream, false, algorithm);
      stream.synchronize();
      expect_close(actual, expected, tolerance);
    }
  }
}

TEST_F(CudaReductionTest, PreservesKeepDimensionShape) {
  const Tensor input_host = Tensor::random_uniform({2, 3, 17}, 702);
  const Tensor expected = cpu::sum_last_dim(input_host, true);
  const Tensor input = input_host.to(Device::cuda(0));
  Stream stream;

  for (const ReductionAlgorithm algorithm : kAlgorithms) {
    const Tensor actual = sum_last_dim(input, stream, true, algorithm);
    stream.synchronize();
    EXPECT_EQ(actual.shape(), (Shape{2, 3, 1}));
    expect_close(actual, expected, 2.0e-5F);
  }
}

TEST_F(CudaReductionTest, RejectsInvalidContractsAndAliasing) {
  Stream stream;
  const Tensor scalar = Tensor::zeros({}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(static_cast<void>(sum_last_dim(scalar, stream)), Error);

  const Tensor half = Tensor::zeros({2, 3}, DType::kFloat16, Device::cuda(0));
  EXPECT_THROW(static_cast<void>(sum_last_dim(half, stream)), Error);

  Tensor aliased = Tensor::zeros({4, 1}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(sum_last_dim_out(aliased, aliased, stream, true), Error);

  const Tensor input = Tensor::zeros({2, 3}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({2}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(sum_last_dim_out(input, output, stream, false, static_cast<ReductionAlgorithm>(999)),
               Error);

  const Tensor empty_input = Tensor::zeros({0, 3}, DType::kFloat32, Device::cuda(0));
  Tensor empty_output = Tensor::zeros({0}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(sum_last_dim_out(empty_input, empty_output, stream, false,
                                static_cast<ReductionAlgorithm>(999)),
               Error);
}

} // namespace
} // namespace helix::cuda::kernels
