#include "helix/core/error.hpp"
#include "helix/memory/helix_allocator.hpp"
#include "helix/memory/system_memory_resource.hpp"
#include "helix/memory/tensor_storage.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace helix::memory {
namespace {

class TrackingResource final : public MemoryResource {
public:
  [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment) override {
    std::scoped_lock lock(mutex_);
    if (fail_allocations_) {
      throw_error(ErrorCode::kResourceExhausted, "injected upstream allocation failure");
    }
    void* pointer = ::operator new(bytes, std::align_val_t(alignment));
    live_.emplace(pointer, std::pair{bytes, alignment});
    ++allocation_calls_;
    return pointer;
  }

  void deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override {
    std::scoped_lock lock(mutex_);
    if (fail_deallocations_) {
      throw_error(ErrorCode::kRuntime, "injected upstream deallocation failure");
    }
    const auto iterator = live_.find(pointer);
    HELIX_CHECK(iterator != live_.end(), ErrorCode::kInvalidArgument,
                "tracking resource received an unknown pointer");
    HELIX_CHECK(iterator->second == (std::pair{bytes, alignment}), ErrorCode::kInvalidArgument,
                "tracking resource received inconsistent allocation metadata");
    live_.erase(iterator);
    ::operator delete(pointer, std::align_val_t(alignment));
    ++deallocation_calls_;
  }

  [[nodiscard]] Device device() const noexcept override {
    return Device::cpu();
  }

  void fail_allocations(bool enabled) {
    std::scoped_lock lock(mutex_);
    fail_allocations_ = enabled;
  }

  void fail_deallocations(bool enabled) {
    std::scoped_lock lock(mutex_);
    fail_deallocations_ = enabled;
  }

  [[nodiscard]] std::size_t allocation_calls() const {
    std::scoped_lock lock(mutex_);
    return allocation_calls_;
  }

  [[nodiscard]] std::size_t deallocation_calls() const {
    std::scoped_lock lock(mutex_);
    return deallocation_calls_;
  }

  [[nodiscard]] std::size_t live_allocations() const {
    std::scoped_lock lock(mutex_);
    return live_.size();
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<void*, std::pair<std::size_t, std::size_t>> live_;
  std::size_t allocation_calls_ = 0;
  std::size_t deallocation_calls_ = 0;
  bool fail_allocations_ = false;
  bool fail_deallocations_ = false;
};

class AliasingResource final : public MemoryResource {
public:
  explicit AliasingResource(void* storage) : storage_(storage) {}

  [[nodiscard]] void* allocate(std::size_t, std::size_t) override {
    return storage_;
  }

  void deallocate(void*, std::size_t, std::size_t) override {}

  [[nodiscard]] Device device() const noexcept override {
    return Device::cpu();
  }

private:
  void* storage_;
};

class CudaProbeResource final : public MemoryResource {
public:
  [[nodiscard]] void* allocate(std::size_t, std::size_t) override {
    ++allocation_calls_;
    return nullptr;
  }

  void deallocate(void*, std::size_t, std::size_t) override {
    ++deallocation_calls_;
  }

  [[nodiscard]] Device device() const noexcept override {
    return Device::cuda(0);
  }

  [[nodiscard]] std::size_t allocation_calls() const noexcept {
    return allocation_calls_;
  }

  [[nodiscard]] std::size_t deallocation_calls() const noexcept {
    return deallocation_calls_;
  }

private:
  std::size_t allocation_calls_ = 0;
  std::size_t deallocation_calls_ = 0;
};

AllocatorConfig config(std::size_t segment_bytes = 1024,
                       std::size_t maximum_reserved_bytes = 4096) {
  return AllocatorConfig{segment_bytes, 64, maximum_reserved_bytes};
}

TEST(HelixAllocatorTest, ValidatesConfigurationSizeAndAlignment) {
  auto resource = std::make_shared<SystemMemoryResource>();
  EXPECT_THROW(HelixAllocator(resource, AllocatorConfig{0, 64, 1024}), Error);
  EXPECT_THROW(HelixAllocator(resource, AllocatorConfig{1024, 48, 1024}), Error);
  EXPECT_THROW(HelixAllocator(resource, AllocatorConfig{1024, 64, 0}), Error);
  EXPECT_THROW(HelixAllocator(nullptr, config()), Error);

  HelixAllocator allocator(resource, config());
  EXPECT_THROW(static_cast<void>(allocator.allocate(0)), Error);
  EXPECT_THROW(static_cast<void>(allocator.allocate(1, 48)), Error);
  EXPECT_THROW(static_cast<void>(allocator.allocate(std::numeric_limits<std::size_t>::max())),
               Error);
  EXPECT_THROW(allocator.deallocate(Allocation{}), Error);
}

TEST(HelixAllocatorTest, SplitsRoundsReusesAndReportsExactStatistics) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>(), config());
  const Allocation first = allocator.allocate(100);
  const Allocation second = allocator.allocate(200);

  AllocatorStats stats = allocator.stats();
  EXPECT_EQ(stats.bytes_reserved, 1024U);
  EXPECT_EQ(stats.bytes_allocated, 384U);
  EXPECT_EQ(stats.bytes_requested, 300U);
  EXPECT_EQ(stats.bytes_free, 640U);
  EXPECT_EQ(stats.peak_bytes_allocated, 384U);
  EXPECT_EQ(stats.largest_free_block, 640U);
  EXPECT_EQ(stats.segment_count, 1U);
  EXPECT_EQ(stats.active_allocation_count, 2U);
  EXPECT_EQ(stats.allocation_count, 2U);
  EXPECT_EQ(stats.reuse_count, 1U);
  EXPECT_EQ(stats.underlying_allocations, 1U);
  EXPECT_DOUBLE_EQ(stats.external_fragmentation, 0.0);
  EXPECT_DOUBLE_EQ(stats.internal_fragmentation, 84.0 / 384.0);

  allocator.deallocate(first);
  stats = allocator.stats();
  EXPECT_EQ(stats.bytes_free, 768U);
  EXPECT_EQ(stats.largest_free_block, 640U);
  EXPECT_DOUBLE_EQ(stats.external_fragmentation, 1.0 - 640.0 / 768.0);
  const Allocation reused = allocator.allocate(64);
  EXPECT_EQ(reused.pointer(), first.pointer());
  EXPECT_NE(reused.id(), first.id());
  EXPECT_EQ(allocator.stats().reuse_count, 2U);
  EXPECT_TRUE(allocator.owns(reused));
  EXPECT_FALSE(allocator.owns(first));

  allocator.deallocate(second);
  allocator.deallocate(reused);
  stats = allocator.stats();
  EXPECT_EQ(stats.bytes_allocated, 0U);
  EXPECT_EQ(stats.bytes_requested, 0U);
  EXPECT_EQ(stats.bytes_free, 1024U);
  EXPECT_EQ(stats.largest_free_block, 1024U);
  EXPECT_DOUBLE_EQ(stats.external_fragmentation, 0.0);
  EXPECT_DOUBLE_EQ(stats.internal_fragmentation, 0.0);
}

TEST(HelixAllocatorTest, UsesBestFitWithDeterministicAddressTieBreaking) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>(), config());
  const Allocation small = allocator.allocate(100);
  const Allocation first_guard = allocator.allocate(1);
  const Allocation medium = allocator.allocate(200);
  const Allocation second_guard = allocator.allocate(1);
  allocator.deallocate(small);
  allocator.deallocate(medium);

  const Allocation exact_small = allocator.allocate(120);
  EXPECT_EQ(exact_small.pointer(), small.pointer());
  const Allocation best_medium = allocator.allocate(180);
  EXPECT_EQ(best_medium.pointer(), medium.pointer());

  allocator.deallocate(exact_small);
  allocator.deallocate(best_medium);
  allocator.deallocate(first_guard);
  allocator.deallocate(second_guard);
  EXPECT_EQ(allocator.stats().largest_free_block, 1024U);
}

TEST(HelixAllocatorTest, SatisfiesRequestedPowerOfTwoAlignments) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>(), config(4096, 8192));
  const Allocation aligned_64 = allocator.allocate(1, 64);
  const Allocation aligned_256 = allocator.allocate(1, 256);
  const Allocation aligned_1024 = allocator.allocate(1, 1024);

  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned_64.pointer()) % 64, 0U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned_256.pointer()) % 256, 0U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned_1024.pointer()) % 1024, 0U);
  EXPECT_EQ(aligned_64.allocated_bytes(), 64U);
  EXPECT_EQ(aligned_256.allocated_bytes(), 256U);
  EXPECT_EQ(aligned_1024.allocated_bytes(), 1024U);

  allocator.deallocate(aligned_256);
  allocator.deallocate(aligned_64);
  allocator.deallocate(aligned_1024);
  EXPECT_EQ(allocator.stats().largest_free_block, allocator.stats().bytes_reserved);
}

TEST(HelixAllocatorTest, DetectsDoubleFreeAndStaleHandleAbaReuse) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>(), config());
  const Allocation original = allocator.allocate(128);
  const Allocation stale_copy = original;
  allocator.deallocate(original);
  EXPECT_THROW(allocator.deallocate(stale_copy), Error);

  std::uint64_t previous_id = original.id();
  for (std::size_t iteration = 0; iteration < 1'024; ++iteration) {
    const Allocation replacement = allocator.allocate(128);
    EXPECT_EQ(replacement.pointer(), original.pointer());
    EXPECT_GT(replacement.id(), previous_id);
    EXPECT_FALSE(allocator.owns(stale_copy));
    if (iteration % 127 == 0) {
      EXPECT_THROW(allocator.deallocate(stale_copy), Error);
    }
    EXPECT_TRUE(allocator.owns(replacement));
    previous_id = replacement.id();
    allocator.deallocate(replacement);
  }
  EXPECT_THROW(allocator.deallocate(stale_copy), Error);
  EXPECT_EQ(allocator.stats().active_allocation_count, 0U);
}

TEST(HelixAllocatorTest, RejectsHandleFromAnotherAllocatorWithSameAddressAndLocalId) {
  alignas(64) std::byte shared_storage[1024];
  auto first_resource = std::make_shared<AliasingResource>(shared_storage);
  auto second_resource = std::make_shared<AliasingResource>(shared_storage);
  HelixAllocator first_allocator(first_resource, config());
  HelixAllocator second_allocator(second_resource, config());
  const Allocation first = first_allocator.allocate(64);
  const Allocation second = second_allocator.allocate(64);

  ASSERT_EQ(first.pointer(), second.pointer());
  ASSERT_EQ(first.id(), second.id());
  EXPECT_FALSE(first_allocator.owns(second));
  EXPECT_FALSE(second_allocator.owns(first));
  EXPECT_THROW(first_allocator.deallocate(second), Error);
  EXPECT_THROW(second_allocator.deallocate(first), Error);
  EXPECT_TRUE(first_allocator.owns(first));
  EXPECT_TRUE(second_allocator.owns(second));

  first_allocator.deallocate(first);
  second_allocator.deallocate(second);
}

TEST(HelixAllocatorTest, EnforcesReservationLimitTransactionally) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>(), config(256, 512));
  const Allocation first = allocator.allocate(200);
  const Allocation second = allocator.allocate(200);
  const AllocatorStats before_failure = allocator.stats();
  EXPECT_THROW(static_cast<void>(allocator.allocate(200)), Error);
  const AllocatorStats after_failure = allocator.stats();
  EXPECT_EQ(after_failure.bytes_reserved, before_failure.bytes_reserved);
  EXPECT_EQ(after_failure.bytes_allocated, before_failure.bytes_allocated);
  EXPECT_EQ(after_failure.active_allocation_count, before_failure.active_allocation_count);
  EXPECT_EQ(after_failure.underlying_allocations, 2U);

  allocator.deallocate(first);
  allocator.deallocate(second);
}

TEST(HelixAllocatorTest, GrowsSegmentForOversizedRequest) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>(), config(256, 4096));
  const Allocation allocation = allocator.allocate(300);
  EXPECT_EQ(allocation.allocated_bytes(), 320U);
  EXPECT_EQ(allocator.stats().bytes_reserved, 320U);
  allocator.deallocate(allocation);
}

TEST(HelixAllocatorTest, ReleasesOnlyCompletelyFreeSegments) {
  auto resource = std::make_shared<TrackingResource>();
  HelixAllocator allocator(resource, config(256, 1024));
  const Allocation first = allocator.allocate(200);
  const Allocation second = allocator.allocate(200);
  allocator.deallocate(first);

  EXPECT_EQ(allocator.release_cached(), 256U);
  AllocatorStats stats = allocator.stats();
  EXPECT_EQ(stats.segment_count, 1U);
  EXPECT_EQ(stats.bytes_reserved, 256U);
  EXPECT_EQ(stats.bytes_allocated, 256U);
  EXPECT_EQ(stats.underlying_deallocations, 1U);
  EXPECT_EQ(resource->live_allocations(), 1U);

  allocator.deallocate(second);
  EXPECT_EQ(allocator.release_cached(), 256U);
  EXPECT_EQ(allocator.stats().bytes_reserved, 0U);
  EXPECT_EQ(resource->live_allocations(), 0U);
}

TEST(HelixAllocatorTest, PreservesMetadataWhenExplicitReleaseFails) {
  auto resource = std::make_shared<TrackingResource>();
  HelixAllocator allocator(resource, config());
  const Allocation allocation = allocator.allocate(64);
  allocator.deallocate(allocation);
  resource->fail_deallocations(true);
  EXPECT_THROW(static_cast<void>(allocator.release_cached()), Error);
  EXPECT_EQ(allocator.stats().bytes_reserved, 1024U);
  EXPECT_EQ(resource->live_allocations(), 1U);

  resource->fail_deallocations(false);
  EXPECT_EQ(allocator.release_cached(), 1024U);
}

TEST(HelixAllocatorTest, UpstreamAllocationFailureLeavesStateUnchanged) {
  auto resource = std::make_shared<TrackingResource>();
  resource->fail_allocations(true);
  HelixAllocator allocator(resource, config());
  EXPECT_THROW(static_cast<void>(allocator.allocate(64)), Error);
  const AllocatorStats stats = allocator.stats();
  EXPECT_EQ(stats.bytes_reserved, 0U);
  EXPECT_EQ(stats.bytes_allocated, 0U);
  EXPECT_EQ(stats.allocation_count, 0U);
  EXPECT_EQ(stats.underlying_allocations, 0U);
}

TEST(HelixAllocatorTest, DestructorReleasesSegmentsEvenWithLiveHandles) {
  auto resource = std::make_shared<TrackingResource>();
  {
    HelixAllocator allocator(resource, config());
    const Allocation allocation = allocator.allocate(64);
    EXPECT_TRUE(allocator.owns(allocation));
    EXPECT_EQ(resource->live_allocations(), 1U);
  }
  EXPECT_EQ(resource->allocation_calls(), 1U);
  EXPECT_EQ(resource->deallocation_calls(), 1U);
  EXPECT_EQ(resource->live_allocations(), 0U);
}

TEST(HelixAllocatorTest, SerializesConcurrentAllocationAndFree) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>(), config(4096, 64 * 1024));
  constexpr std::size_t kThreads = 4;
  constexpr std::size_t kIterations = 500;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&allocator, thread] {
      for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
        const Allocation allocation = allocator.allocate(1 + ((iteration + thread) % 193));
        allocator.deallocate(allocation);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  const AllocatorStats stats = allocator.stats();
  EXPECT_EQ(stats.active_allocation_count, 0U);
  EXPECT_EQ(stats.bytes_allocated, 0U);
  EXPECT_EQ(stats.allocation_count, kThreads * kIterations);
  EXPECT_EQ(stats.largest_free_block, stats.bytes_reserved);
}

TEST(AllocatorTensorStorageTest, RetainsRoundedBlockUntilFinalViewAndThenReusesIt) {
  auto allocator =
      std::make_shared<HelixAllocator>(std::make_shared<SystemMemoryResource>(), config());
  std::optional<Tensor> view;
  void* first_pointer = nullptr;
  {
    Tensor tensor = allocate_cpu_tensor({3, 5}, DType::kFloat32, allocator, 128);
    first_pointer = tensor.data();
    ASSERT_NE(first_pointer, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first_pointer) % 128, 0U);
    EXPECT_EQ(tensor.nbytes(), 60U);
    EXPECT_EQ(tensor.allocated_bytes(), 128U);
    tensor.zero_();
    tensor.data_as<float>()[14] = 7.0F;
    view.emplace(tensor.view({15}));

    const AllocatorStats active = allocator->stats();
    EXPECT_EQ(active.bytes_requested, 60U);
    EXPECT_EQ(active.bytes_allocated, 128U);
    EXPECT_EQ(active.active_allocation_count, 1U);
    EXPECT_EQ(allocator->release_cached(), 0U);
  }

  ASSERT_TRUE(view.has_value());
  EXPECT_FLOAT_EQ(view->data_as<float>()[14], 7.0F);
  EXPECT_EQ(allocator->stats().active_allocation_count, 1U);
  view.reset();
  EXPECT_EQ(allocator->stats().active_allocation_count, 0U);

  Tensor reused = allocate_cpu_tensor({16}, DType::kFloat32, allocator, 128);
  EXPECT_EQ(reused.data(), first_pointer);
  EXPECT_EQ(allocator->stats().reuse_count, 1U);
  reused = Tensor::empty({0});
  EXPECT_EQ(allocator->stats().active_allocation_count, 0U);
}

TEST(AllocatorTensorStorageTest, TensorOwnershipKeepsAllocatorAliveUntilStorageExpires) {
  auto allocator =
      std::make_shared<HelixAllocator>(std::make_shared<SystemMemoryResource>(), config());
  std::weak_ptr<HelixAllocator> lifetime = allocator;
  std::optional<Tensor> tensor;
  tensor.emplace(allocate_cpu_tensor({17}, DType::kFloat32, allocator));
  allocator.reset();

  EXPECT_FALSE(lifetime.expired());
  EXPECT_EQ(tensor->allocated_bytes(), 128U);
  tensor.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST(AllocatorTensorStorageTest, ZeroAndRejectedRequestsCreateNoActiveAllocation) {
  auto allocator =
      std::make_shared<HelixAllocator>(std::make_shared<SystemMemoryResource>(), config());
  const Tensor empty = allocate_cpu_tensor({2, 0, 3}, DType::kFloat32, allocator, 128);
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.nbytes(), 0U);
  EXPECT_EQ(empty.allocated_bytes(), 0U);

  EXPECT_THROW(static_cast<void>(allocate_cpu_tensor({-1}, DType::kFloat32, allocator)), Error);
  EXPECT_THROW(static_cast<void>(allocate_cpu_tensor({std::numeric_limits<Dimension>::max()},
                                                     DType::kFloat32, allocator)),
               Error);
  EXPECT_THROW(static_cast<void>(allocate_cpu_tensor({4}, DType::kFloat32, allocator, 24)), Error);
  EXPECT_THROW(
      static_cast<void>(allocate_cpu_tensor({4}, DType::kFloat32, allocator, alignof(float))),
      Error);
  EXPECT_THROW(static_cast<void>(allocate_cpu_tensor({4}, DType::kFloat32, nullptr, 64)), Error);
  EXPECT_EQ(allocator->stats().allocation_count, 0U);
  EXPECT_EQ(allocator->stats().active_allocation_count, 0U);

  auto limited = std::make_shared<HelixAllocator>(std::make_shared<SystemMemoryResource>(),
                                                  AllocatorConfig{64, 64, 64});
  EXPECT_THROW(static_cast<void>(allocate_cpu_tensor({17}, DType::kFloat32, limited)), Error);
  EXPECT_EQ(limited->stats().allocation_count, 0U);
  EXPECT_EQ(limited->stats().bytes_reserved, 0U);

  auto cuda_resource = std::make_shared<CudaProbeResource>();
  auto cuda_allocator = std::make_shared<HelixAllocator>(cuda_resource, config());
  EXPECT_THROW(static_cast<void>(allocate_cpu_tensor({4}, DType::kFloat32, cuda_allocator)), Error);
  EXPECT_EQ(cuda_resource->allocation_calls(), 0U);
  EXPECT_EQ(cuda_resource->deallocation_calls(), 0U);
  EXPECT_EQ(cuda_allocator->stats().allocation_count, 0U);
}

} // namespace
} // namespace helix::memory
