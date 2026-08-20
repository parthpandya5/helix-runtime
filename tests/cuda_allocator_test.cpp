#include "helix/core/error.hpp"
#include "helix/memory/cuda_memory_resource.hpp"
#include "helix/memory/helix_allocator.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/memory.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
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

} // namespace
} // namespace helix::memory
