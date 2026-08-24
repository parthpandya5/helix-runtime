#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"
#include "helix/runtime/cuda/memory.hpp"
#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace helix::cuda {
namespace {

static_assert(!std::is_copy_constructible_v<PinnedHostBuffer>);
static_assert(!std::is_copy_assignable_v<PinnedHostBuffer>);
static_assert(std::is_nothrow_move_constructible_v<PinnedHostBuffer>);
static_assert(std::is_nothrow_move_assignable_v<PinnedHostBuffer>);

class CudaRuntimeTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_F(CudaRuntimeTest, QueriesDeviceAndRestoresGuardedSelection) {
  const int original_device = current_device();
  const DeviceInfo info = query_device(original_device);
  EXPECT_EQ(info.index, original_device);
  EXPECT_FALSE(info.name.empty());
  EXPECT_GT(info.total_memory_bytes, 0U);
  EXPECT_GT(info.warp_size, 0);
  EXPECT_GT(info.max_threads_per_block, 0);

  {
    DeviceGuard guard(original_device);
    EXPECT_EQ(guard.previous_device(), original_device);
    EXPECT_EQ(guard.selected_device(), original_device);
  }
  EXPECT_EQ(current_device(), original_device);
  EXPECT_THROW(static_cast<void>(DeviceGuard(device_count())), Error);
}

TEST_F(CudaRuntimeTest, BufferRoundTripsBytesAndMovesOwnership) {
  constexpr std::size_t kElements = 257;
  std::vector<float> source(kElements);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<float>(index) * 0.25F;
  }
  std::vector<float> destination(kElements, 0.0F);

  Buffer original(source.size() * sizeof(float));
  ASSERT_NE(original.data(), nullptr);
  copy_host_to_device(original.data(), source.data(), original.size_bytes());

  Buffer moved(std::move(original));
  EXPECT_EQ(original.data(), nullptr);
  EXPECT_EQ(original.size_bytes(), 0U);
  copy_device_to_host(destination.data(), moved.data(), moved.size_bytes());
  EXPECT_EQ(destination, source);

  Buffer empty(0);
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.size_bytes(), 0U);
}

TEST_F(CudaRuntimeTest, StreamAndEventsOrderAsynchronousCopies) {
  constexpr std::size_t kBytes = 4099;
  std::vector<std::byte> source(kBytes, std::byte{0x5A});
  std::vector<std::byte> destination(kBytes, std::byte{0});
  Buffer buffer(kBytes);
  Stream stream;
  Event start;
  Event stop;

  start.record(stream);
  copy_host_to_device_async(buffer.data(), source.data(), kBytes, stream);
  copy_device_to_host_async(destination.data(), buffer.data(), kBytes, stream);
  stop.record(stream);
  stop.synchronize();

  EXPECT_TRUE(stream.query());
  EXPECT_TRUE(stop.query());
  EXPECT_EQ(destination, source);
  EXPECT_GE(elapsed_time_ms(start, stop), 0.0F);
}

TEST_F(CudaRuntimeTest, PinnedHostBufferMovesAndCompletesOddSizedAsyncRoundTrip) {
  constexpr std::size_t kBytes = 4099;
  PinnedHostBuffer source(kBytes);
  PinnedHostBuffer destination(kBytes);
  ASSERT_NE(source.data(), nullptr);
  ASSERT_NE(destination.data(), nullptr);
  auto* source_bytes = static_cast<std::byte*>(source.data());
  for (std::size_t index = 0; index < kBytes; ++index) {
    source_bytes[index] = static_cast<std::byte>((index * 37U + 11U) & 0xFFU);
  }

  void* const source_pointer = source.data();
  PinnedHostBuffer moved_source(std::move(source));
  EXPECT_EQ(source.data(), nullptr);
  EXPECT_EQ(source.size_bytes(), 0U);
  EXPECT_EQ(moved_source.data(), source_pointer);
  EXPECT_EQ(moved_source.size_bytes(), kBytes);

  PinnedHostBuffer assigned_source(17);
  assigned_source = std::move(moved_source);
  EXPECT_EQ(moved_source.data(), nullptr);
  EXPECT_EQ(moved_source.size_bytes(), 0U);
  EXPECT_EQ(assigned_source.data(), source_pointer);

  Buffer device(kBytes);
  Stream stream;
  copy_host_to_device_async(device.data(), assigned_source.data(), kBytes, stream);
  copy_device_to_host_async(destination.data(), device.data(), kBytes, stream);
  stream.synchronize();
  EXPECT_TRUE(std::equal(static_cast<const std::byte*>(assigned_source.data()),
                         static_cast<const std::byte*>(assigned_source.data()) + kBytes,
                         static_cast<const std::byte*>(destination.data())));

  PinnedHostBuffer empty(0);
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.size_bytes(), 0U);
}

TEST_F(CudaRuntimeTest, MoveOnlyStreamsAndEventsRetainNativeResources) {
  Stream original_stream;
  const cudaStream_t stream_handle = original_stream.native_handle();
  Stream moved_stream(std::move(original_stream));
  EXPECT_EQ(original_stream.native_handle(), nullptr);
  EXPECT_EQ(moved_stream.native_handle(), stream_handle);
  EXPECT_THROW(static_cast<void>(original_stream.query()), Error);

  Event original_event;
  const cudaEvent_t event_handle = original_event.native_handle();
  Event moved_event(std::move(original_event));
  EXPECT_EQ(original_event.native_handle(), nullptr);
  EXPECT_EQ(moved_event.native_handle(), event_handle);
  EXPECT_THROW(static_cast<void>(original_event.query()), Error);

  moved_event.record(moved_stream);
  moved_stream.wait(moved_event);
  moved_stream.synchronize();
}

TEST_F(CudaRuntimeTest, RejectsTimingDisabledEvents) {
  Stream stream;
  Event start(cudaEventDisableTiming);
  Event stop(cudaEventDisableTiming);
  start.record(stream);
  stop.record(stream);
  stop.synchronize();
  EXPECT_THROW(static_cast<void>(elapsed_time_ms(start, stop)), Error);
}

TEST_F(CudaRuntimeTest, TensorStorageTransfersAndClones) {
  const Tensor host =
      Tensor::from_values({2, 3}, std::array{1.0F, -2.0F, 3.5F, 4.0F, 0.25F, -8.0F});
  const Tensor device = host.to(Device::cuda(0));
  EXPECT_EQ(device.device(), Device::cuda(0));
  EXPECT_EQ(device.shape(), host.shape());

  const Tensor device_clone = device.clone();
  EXPECT_NE(device_clone.data(), device.data());
  const Tensor round_trip = device_clone.to(Device::cpu());
  const auto* actual = round_trip.data_as<float>();
  const auto* expected = host.data_as<float>();
  for (std::size_t index = 0; index < host.numel(); ++index) {
    EXPECT_FLOAT_EQ(actual[index], expected[index]);
  }

  const Tensor zeros = Tensor::zeros({17}, DType::kFloat32, Device::cuda(0));
  const Tensor host_zeros = zeros.to(Device::cpu());
  for (std::size_t index = 0; index < host_zeros.numel(); ++index) {
    EXPECT_FLOAT_EQ(host_zeros.data_as<float>()[index], 0.0F);
  }
}

} // namespace
} // namespace helix::cuda
