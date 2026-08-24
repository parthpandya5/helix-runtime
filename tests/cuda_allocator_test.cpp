#include "helix/core/error.hpp"
#include "helix/memory/cuda_memory_resource.hpp"
#include "helix/memory/cuda_tensor_storage.hpp"
#include "helix/memory/helix_allocator.hpp"
#include "helix/memory/stream_ordered_allocator.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"
#include "helix/runtime/cuda/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace helix::memory {
namespace {

class CudaAllocatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!cuda::runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    cuda::set_device(0);
  }
};

void CUDART_CB wait_for_host_gate(void* data) {
  auto* gate = static_cast<std::atomic<bool>*>(data);
  while (!gate->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

class GateRelease final {
public:
  explicit GateRelease(std::atomic<bool>& gate) : gate_(gate) {}
  ~GateRelease() {
    gate_.store(true, std::memory_order_release);
  }

  GateRelease(const GateRelease&) = delete;
  GateRelease& operator=(const GateRelease&) = delete;

private:
  std::atomic<bool>& gate_;
};

TEST_F(CudaAllocatorTest, ReusesAlignedStorageAndRoundTripsBytes) {
  constexpr std::size_t kSegmentBytes = 1U << 20;
  auto resource = std::make_shared<CudaMemoryResource>(0);
  HelixAllocator allocator(resource, AllocatorConfig{kSegmentBytes, 256, 4 * kSegmentBytes});
  const Allocation first = allocator.allocate(1'000);
  const Allocation second = allocator.allocate(777);

  EXPECT_EQ(first.device(), Device::cuda(0));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first.pointer()) % 256, 0U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second.pointer()) % 256, 0U);
  EXPECT_EQ(allocator.stats().underlying_allocations, 1U);

  cuda::set_device_bytes(first.pointer(), 0x3C, first.requested_bytes());
  std::vector<std::byte> host(first.requested_bytes());
  cuda::copy_device_to_host(host.data(), first.pointer(), host.size());
  EXPECT_TRUE(std::all_of(host.begin(), host.end(),
                          [](std::byte value) { return value == static_cast<std::byte>(0x3C); }));

  allocator.deallocate(first);
  const Allocation reused = allocator.allocate(1'000);
  EXPECT_EQ(reused.pointer(), first.pointer());
  EXPECT_NE(reused.id(), first.id());
  EXPECT_THROW(allocator.deallocate(first), Error);
  EXPECT_TRUE(allocator.owns(reused));

  allocator.deallocate(second);
  allocator.deallocate(reused);
  EXPECT_EQ(allocator.release_cached(), kSegmentBytes);
  EXPECT_EQ(allocator.stats().bytes_reserved, 0U);
}

TEST_F(CudaAllocatorTest, RejectsUnsupportedCudaAlignmentBeforeAllocation) {
  auto resource = std::make_shared<CudaMemoryResource>(0);
  HelixAllocator allocator(resource, AllocatorConfig{1U << 20, 256, 4U << 20});
  EXPECT_THROW(static_cast<void>(allocator.allocate(1'024, 512)), Error);
  EXPECT_EQ(allocator.stats().underlying_allocations, 0U);
  EXPECT_EQ(allocator.stats().bytes_reserved, 0U);
}

TEST_F(CudaAllocatorTest, StreamEventDefersReuseUntilPriorWorkCompletes) {
  constexpr std::size_t kSegmentBytes = 4'096;
  StreamOrderedAllocator allocator(0, AllocatorConfig{kSegmentBytes, 256, 4 * kSegmentBytes});
  const Allocation first = allocator.allocate(1'000);
  cuda::Stream release_stream;
  std::atomic<bool> gate{false};
  GateRelease gate_release(gate);
  HELIX_CUDA_CHECK_CONTEXT(
      cudaLaunchHostFunc(release_stream.native_handle(), wait_for_host_gate, &gate),
      "enqueueing allocator test host gate");
  allocator.defer_deallocate(first, release_stream);

  EXPECT_TRUE(allocator.is_pending(first));
  EXPECT_FALSE(allocator.owns(first));
  const Allocation second = allocator.allocate(1'000);
  EXPECT_NE(second.pointer(), first.pointer());
  EXPECT_EQ(allocator.reclaim_ready(), 0U);
  EXPECT_EQ(allocator.stats().pending_allocation_count, 1U);

  gate.store(true, std::memory_order_release);
  EXPECT_EQ(allocator.wait_and_reclaim_all(), 1U);
  const Allocation reused = allocator.allocate(1'000);
  EXPECT_EQ(reused.pointer(), first.pointer());

  allocator.defer_deallocate(second, release_stream);
  allocator.defer_deallocate(reused, release_stream);
  EXPECT_EQ(allocator.wait_and_reclaim_all(), 2U);
  EXPECT_EQ(allocator.release_cached(), kSegmentBytes);
}

TEST_F(CudaAllocatorTest, StreamBoundTensorDefersFinalViewAndReusesOnlyAfterCompletion) {
  constexpr std::size_t kSegmentBytes = 4'096;
  auto allocator = std::make_shared<StreamOrderedAllocator>(
      0, AllocatorConfig{kSegmentBytes, 256, 4 * kSegmentBytes});
  auto stream = std::make_shared<cuda::Stream>();
  std::optional<Tensor> tensor;
  tensor.emplace(allocate_cuda_tensor_on_stream({250}, DType::kFloat32, allocator, stream));
  void* first_pointer = tensor->data();
  ASSERT_NE(first_pointer, nullptr);
  EXPECT_EQ(tensor->device(), Device::cuda(0));
  EXPECT_EQ(tensor->nbytes(), 1'000U);
  EXPECT_EQ(tensor->allocated_bytes(), 1'024U);

  cuda::set_device_bytes_async(tensor->data(), 0x5A, tensor->nbytes(), *stream);
  stream->synchronize();
  std::vector<std::byte> host(tensor->nbytes());
  cuda::copy_device_to_host(host.data(), tensor->data(), host.size());
  EXPECT_TRUE(std::all_of(host.begin(), host.end(),
                          [](std::byte value) { return value == std::byte{0x5A}; }));

  std::optional<Tensor> view;
  view.emplace(tensor->view({10, 25}));
  tensor.reset();
  EXPECT_EQ(allocator->stats().client_allocation_count, 1U);
  EXPECT_EQ(allocator->stats().pending_allocation_count, 0U);

  std::atomic<bool> gate{false};
  GateRelease gate_release(gate);
  HELIX_CUDA_CHECK_CONTEXT(cudaLaunchHostFunc(stream->native_handle(), wait_for_host_gate, &gate),
                           "enqueueing stream-bound Tensor test host gate");
  view.reset();
  EXPECT_EQ(allocator->stats().client_allocation_count, 0U);
  EXPECT_EQ(allocator->stats().pending_allocation_count, 1U);

  std::optional<Tensor> second;
  second.emplace(allocate_cuda_tensor_on_stream({250}, DType::kFloat32, allocator, stream));
  EXPECT_NE(second->data(), first_pointer);
  EXPECT_EQ(allocator->reclaim_ready(), 0U);
  second.reset();
  EXPECT_EQ(allocator->stats().pending_allocation_count, 2U);

  gate.store(true, std::memory_order_release);
  EXPECT_EQ(allocator->wait_and_reclaim_all(), 2U);
  std::optional<Tensor> reused;
  reused.emplace(allocate_cuda_tensor_on_stream({250}, DType::kFloat32, allocator, stream));
  EXPECT_EQ(reused->data(), first_pointer);
  reused.reset();
  EXPECT_EQ(allocator->wait_and_reclaim_all(), 1U);
  EXPECT_EQ(allocator->release_cached(), kSegmentBytes);
}

TEST_F(CudaAllocatorTest, StreamBoundTensorRetainsAllocatorAndStreamUntilFinalOwner) {
  std::weak_ptr<StreamOrderedAllocator> allocator_lifetime;
  std::weak_ptr<cuda::Stream> stream_lifetime;
  std::optional<Tensor> tensor;
  {
    auto allocator =
        std::make_shared<StreamOrderedAllocator>(0, AllocatorConfig{4'096, 256, 16'384});
    auto stream = std::make_shared<cuda::Stream>();
    allocator_lifetime = allocator;
    stream_lifetime = stream;
    tensor.emplace(allocate_cuda_tensor_on_stream({17}, DType::kFloat32, allocator, stream));
  }

  EXPECT_FALSE(allocator_lifetime.expired());
  EXPECT_FALSE(stream_lifetime.expired());
  tensor.reset();
  EXPECT_TRUE(allocator_lifetime.expired());
  EXPECT_TRUE(stream_lifetime.expired());
}

TEST_F(CudaAllocatorTest, StreamBoundTensorValidatesContractsBeforeAllocation) {
  auto allocator = std::make_shared<StreamOrderedAllocator>(0, AllocatorConfig{4'096, 256, 16'384});
  auto stream = std::make_shared<cuda::Stream>();
  const Tensor empty =
      allocate_cuda_tensor_on_stream({2, 0, 3}, DType::kFloat32, allocator, stream);
  EXPECT_EQ(empty.device(), Device::cuda(0));
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.allocated_bytes(), 0U);
  EXPECT_EQ(allocator->stats().allocator.allocation_count, 0U);

  EXPECT_THROW(
      static_cast<void>(allocate_cuda_tensor_on_stream({4}, DType::kFloat32, nullptr, stream)),
      Error);
  EXPECT_THROW(
      static_cast<void>(allocate_cuda_tensor_on_stream({4}, DType::kFloat32, allocator, nullptr)),
      Error);
  EXPECT_THROW(
      static_cast<void>(allocate_cuda_tensor_on_stream({4}, DType::kFloat32, allocator, stream, 8)),
      Error);
  EXPECT_THROW(static_cast<void>(
                   allocate_cuda_tensor_on_stream({4}, DType::kFloat32, allocator, stream, 24)),
               Error);
  EXPECT_THROW(static_cast<void>(
                   allocate_cuda_tensor_on_stream({4}, DType::kFloat32, allocator, stream, 512)),
               Error);
  EXPECT_THROW(
      static_cast<void>(allocate_cuda_tensor_on_stream({-1}, DType::kFloat32, allocator, stream)),
      Error);
  EXPECT_THROW(static_cast<void>(allocate_cuda_tensor_on_stream(
                   {std::numeric_limits<Dimension>::max()}, DType::kFloat32, allocator, stream)),
               Error);

  auto moved_from_stream = std::make_shared<cuda::Stream>();
  cuda::Stream moved_stream(std::move(*moved_from_stream));
  EXPECT_THROW(static_cast<void>(allocate_cuda_tensor_on_stream({4}, DType::kFloat32, allocator,
                                                                moved_from_stream)),
               Error);

  const Allocation external = allocator->allocate(64);
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cuda_buffer(
                   {16}, DType::kFloat32, external.pointer(), external.allocated_bytes(),
                   Device::cuda(0), {}, external.alignment())),
               Error);
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cuda_buffer(
                   {17}, DType::kFloat32, external.pointer(), external.requested_bytes(),
                   Device::cuda(0), stream, external.alignment())),
               Error);
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cuda_buffer(
                   {16}, DType::kFloat32,
                   static_cast<void*>(static_cast<std::byte*>(external.pointer()) + 4),
                   external.allocated_bytes() - 4, Device::cuda(0), stream, external.alignment())),
               Error);
  allocator->deallocate_completed(external);
  EXPECT_EQ(allocator->stats().allocator.active_allocation_count, 0U);
}

} // namespace
} // namespace helix::memory
