#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/elementwise.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace helix::cuda::kernels {
namespace {

void expect_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual_host = actual_device.to(Device::cpu());
  ASSERT_EQ(actual_host.shape(), expected_host.shape());
  const auto* actual = actual_host.data_as<float>();
  const auto* expected = expected_host.data_as<float>();
  for (std::size_t index = 0; index < expected_host.numel(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], tolerance) << "at index " << index;
  }
}

class CudaElementwiseTest : public ::testing::TestWithParam<std::int64_t> {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_P(CudaElementwiseTest, AddMatchesCpuReference) {
  const Dimension elements = GetParam();
  const Tensor lhs_host = Tensor::random_uniform({elements}, 101);
  const Tensor rhs_host = Tensor::random_uniform({elements}, 202);
  const Tensor expected = cpu::add(lhs_host, rhs_host);
  const Tensor lhs = lhs_host.to(Device::cuda(0));
  const Tensor rhs = rhs_host.to(Device::cuda(0));
  Stream stream;

  const Tensor actual = add(lhs, rhs, stream);
  stream.synchronize();
  expect_close(actual, expected, 1.0e-6F);
}

TEST_P(CudaElementwiseTest, MultiplyMatchesCpuReference) {
  const Dimension elements = GetParam();
  const Tensor lhs_host = Tensor::random_uniform({elements}, 303);
  const Tensor rhs_host = Tensor::random_uniform({elements}, 404);
  const Tensor expected = cpu::multiply(lhs_host, rhs_host);
  const Tensor lhs = lhs_host.to(Device::cuda(0));
  const Tensor rhs = rhs_host.to(Device::cuda(0));
  Stream stream;

  const Tensor actual = multiply(lhs, rhs, stream);
  stream.synchronize();
  expect_close(actual, expected, 1.0e-6F);
}

TEST_P(CudaElementwiseTest, ReluMatchesCpuReference) {
  const Dimension elements = GetParam();
  const Tensor input_host = Tensor::random_uniform({elements}, 505, -4.0F, 4.0F);
  const Tensor expected = cpu::relu(input_host);
  const Tensor input = input_host.to(Device::cuda(0));
  Stream stream;

  const Tensor actual = relu(input, stream);
  stream.synchronize();
  expect_close(actual, expected, 0.0F);
}

TEST_P(CudaElementwiseTest, SiluMatchesCpuReference) {
  const Dimension elements = GetParam();
  const Tensor input_host = Tensor::random_uniform({elements}, 606, -10.0F, 10.0F);
  const Tensor expected = cpu::silu(input_host);
  const Tensor input = input_host.to(Device::cuda(0));
  Stream stream;

  const Tensor actual = silu(input, stream);
  stream.synchronize();
  expect_close(actual, expected, 2.0e-6F);
}

INSTANTIATE_TEST_SUITE_P(OddAndBoundarySizes, CudaElementwiseTest,
                         ::testing::Values(0, 1, 31, 257, 4099, 65'539));

class CudaElementwiseContractTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_F(CudaElementwiseContractTest, SupportsSafeInPlaceAliasing) {
  const Tensor lhs_host = Tensor::from_values({4}, std::array{1.0F, 2.0F, 3.0F, 4.0F});
  const Tensor rhs_host = Tensor::from_values({4}, std::array{5.0F, 6.0F, 7.0F, 8.0F});
  Tensor lhs = lhs_host.to(Device::cuda(0));
  const Tensor rhs = rhs_host.to(Device::cuda(0));
  Stream stream;

  add_out(lhs, rhs, lhs, stream);
  stream.synchronize();
  expect_close(lhs, cpu::add(lhs_host, rhs_host), 1.0e-6F);
}

TEST_F(CudaElementwiseContractTest, SiluHandlesExtremeFiniteValues) {
  const Tensor input_host =
      Tensor::from_values({5}, std::array{-100.0F, -20.0F, 0.0F, 20.0F, 100.0F});
  const Tensor input = input_host.to(Device::cuda(0));
  Stream stream;
  const Tensor actual = silu(input, stream);
  stream.synchronize();
  expect_close(actual, cpu::silu(input_host), 1.0e-5F);
}

TEST_F(CudaElementwiseContractTest, RejectsShapeAndDTypeMismatches) {
  const Tensor lhs = Tensor::zeros({7}, DType::kFloat32, Device::cuda(0));
  const Tensor wrong_shape = Tensor::zeros({8}, DType::kFloat32, Device::cuda(0));
  const Tensor half = Tensor::zeros({7}, DType::kFloat16, Device::cuda(0));
  Tensor output = Tensor::zeros({7}, DType::kFloat32, Device::cuda(0));
  Stream stream;

  EXPECT_THROW(add_out(lhs, wrong_shape, output, stream), Error);
  EXPECT_THROW(relu_out(half, output, stream), Error);
}

} // namespace
} // namespace helix::cuda::kernels
