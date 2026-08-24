#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/elementwise.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"
#include "helix/runtime/cuda/memory.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>

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

TEST_F(CudaElementwiseContractTest, DoubleBufferedPinnedPipelineHonorsEventDependencies) {
  constexpr std::size_t kChunkCount = 5;
  constexpr Dimension kElementsPerChunk = 257;
  constexpr std::size_t kChunkElements = static_cast<std::size_t>(kElementsPerChunk);
  constexpr std::size_t kChunkBytes = kChunkElements * sizeof(float);
  constexpr std::size_t kTotalElements = kChunkCount * kChunkElements;
  constexpr std::size_t kTotalBytes = kTotalElements * sizeof(float);
  PinnedHostBuffer host_input(kTotalBytes);
  PinnedHostBuffer host_output(kTotalBytes);
  auto* input_values = static_cast<float*>(host_input.data());
  for (std::size_t index = 0; index < kTotalElements; ++index) {
    input_values[index] = static_cast<float>(static_cast<int>(index % 257U) - 128) / 16.0F;
  }
  const Tensor expected =
      cpu::silu(Tensor::from_values({static_cast<Dimension>(kTotalElements)},
                                    std::span<const float>(input_values, kTotalElements)));

  std::array<Tensor, 2> device_input{
      Tensor::empty({kElementsPerChunk}, DType::kFloat32, Device::cuda(0)),
      Tensor::empty({kElementsPerChunk}, DType::kFloat32, Device::cuda(0))};
  std::array<Tensor, 2> device_output{
      Tensor::empty({kElementsPerChunk}, DType::kFloat32, Device::cuda(0)),
      Tensor::empty({kElementsPerChunk}, DType::kFloat32, Device::cuda(0))};
  std::array<Stream, 2> transfer_streams;
  Stream compute_stream;
  Stream completion_stream;
  Event start(cudaEventDisableTiming);
  std::array<Event, 2> input_ready{Event(cudaEventDisableTiming), Event(cudaEventDisableTiming)};
  std::array<Event, 2> compute_ready{Event(cudaEventDisableTiming), Event(cudaEventDisableTiming)};
  std::array<Event, 2> slot_done{Event(cudaEventDisableTiming), Event(cudaEventDisableTiming)};

  start.record(completion_stream);
  for (Stream& stream : transfer_streams) {
    stream.wait(start);
  }
  for (std::size_t chunk = 0; chunk < kChunkCount; ++chunk) {
    const std::size_t slot = chunk % device_input.size();
    const std::size_t byte_offset = chunk * kChunkBytes;
    copy_host_to_device_async(
        device_input[slot].data(),
        static_cast<const void*>(static_cast<const std::byte*>(host_input.data()) + byte_offset),
        kChunkBytes, transfer_streams[slot]);
    input_ready[slot].record(transfer_streams[slot]);
    compute_stream.wait(input_ready[slot]);
    silu_out(device_input[slot], device_output[slot], compute_stream);
    compute_ready[slot].record(compute_stream);
    transfer_streams[slot].wait(compute_ready[slot]);
    copy_device_to_host_async(
        static_cast<void*>(static_cast<std::byte*>(host_output.data()) + byte_offset),
        device_output[slot].data(), kChunkBytes, transfer_streams[slot]);
    slot_done[slot].record(transfer_streams[slot]);
  }
  for (const Event& event : slot_done) {
    completion_stream.wait(event);
  }
  completion_stream.synchronize();

  const auto* actual = static_cast<const float*>(host_output.data());
  const auto* reference = expected.data_as<float>();
  for (std::size_t index = 0; index < kTotalElements; ++index) {
    EXPECT_NEAR(actual[index], reference[index], 2.0e-6F) << "at index " << index;
  }
}

} // namespace
} // namespace helix::cuda::kernels
