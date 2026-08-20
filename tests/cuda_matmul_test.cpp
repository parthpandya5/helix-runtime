#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/matmul.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>

namespace helix::cuda::kernels {
namespace {

constexpr std::array kAlgorithms{MatmulAlgorithm::kNaive, MatmulAlgorithm::kTiled};

void expect_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected_host.shape());
  for (std::size_t index = 0; index < expected_host.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected_host.data_as<float>()[index], tolerance)
        << "at index " << index;
  }
}

class CudaMatmulTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_F(CudaMatmulTest, EveryAlgorithmMatchesKnownProduct) {
  const Tensor lhs_host =
      Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  const Tensor rhs_host =
      Tensor::from_values({3, 2}, std::array{7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
  const Tensor expected = cpu::matmul(lhs_host, rhs_host);
  const Tensor lhs = lhs_host.to(Device::cuda(0));
  const Tensor rhs = rhs_host.to(Device::cuda(0));
  Stream stream;

  for (const MatmulAlgorithm algorithm : kAlgorithms) {
    const Tensor actual = matmul(lhs, rhs, stream, algorithm);
    stream.synchronize();
    expect_close(actual, expected, 1.0e-5F);
  }
}

TEST_F(CudaMatmulTest, EveryAlgorithmMatchesCpuAcrossOddAndEdgeTiles) {
  struct Problem {
    Dimension rows;
    Dimension inner;
    Dimension columns;
  };
  constexpr std::array kProblems{Problem{1, 1, 1},    Problem{3, 5, 4}, Problem{17, 19, 23},
                                 Problem{31, 33, 29}, Problem{2, 0, 3}, Problem{0, 7, 5},
                                 Problem{4, 7, 0}};
  Stream stream;

  for (const Problem problem : kProblems) {
    const Tensor lhs_host = Tensor::random_uniform({problem.rows, problem.inner}, 1'001);
    const Tensor rhs_host = Tensor::random_uniform({problem.inner, problem.columns}, 1'002);
    const Tensor expected = cpu::matmul(lhs_host, rhs_host);
    const Tensor lhs = lhs_host.to(Device::cuda(0));
    const Tensor rhs = rhs_host.to(Device::cuda(0));
    const float tolerance = 1.0e-5F + static_cast<float>(problem.inner) * 3.0e-7F;
    for (const MatmulAlgorithm algorithm : kAlgorithms) {
      const Tensor actual = matmul(lhs, rhs, stream, algorithm);
      stream.synchronize();
      expect_close(actual, expected, tolerance);
    }
  }
}

TEST_F(CudaMatmulTest, RejectsInvalidShapesDTypesAliasingAndAlgorithms) {
  Stream stream;
  const Tensor vector = Tensor::zeros({4}, DType::kFloat32, Device::cuda(0));
  const Tensor matrix = Tensor::zeros({4, 4}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(static_cast<void>(matmul(vector, matrix, stream)), Error);

  const Tensor wrong_inner = Tensor::zeros({3, 4}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(static_cast<void>(matmul(
                   wrong_inner, Tensor::zeros({5, 2}, DType::kFloat32, Device::cuda(0)), stream)),
               Error);

  const Tensor half = Tensor::zeros({4, 4}, DType::kFloat16, Device::cuda(0));
  EXPECT_THROW(static_cast<void>(matmul(half, half, stream)), Error);

  Tensor aliased = Tensor::zeros({4, 4}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(matmul_out(aliased, matrix, aliased, stream), Error);
  EXPECT_THROW(matmul_out(matrix, aliased, aliased, stream), Error);

  Tensor output = Tensor::zeros({4, 4}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(matmul_out(matrix, matrix, output, stream, static_cast<MatmulAlgorithm>(999)),
               Error);

  const Tensor empty_lhs = Tensor::zeros({0, 4}, DType::kFloat32, Device::cuda(0));
  Tensor empty_output = Tensor::zeros({0, 4}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(
      matmul_out(empty_lhs, matrix, empty_output, stream, static_cast<MatmulAlgorithm>(999)),
      Error);
}

} // namespace
} // namespace helix::cuda::kernels
