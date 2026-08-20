#include "helix/core/error.hpp"
#include "helix/memory/deferred_allocator.hpp"
#include "helix/memory/system_memory_resource.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace helix::memory {
namespace {

struct CompletionState {
  bool ready = false;
  bool throw_on_query = false;
  bool throw_on_wait = false;
  std::size_t query_count = 0;
  std::size_t wait_count = 0;
};

class ManualCompletion final : public Completion {
public:
  explicit ManualCompletion(std::shared_ptr<CompletionState> state) : state_(std::move(state)) {}

  [[nodiscard]] bool ready() override {
    ++state_->query_count;
    if (state_->throw_on_query) {
      throw_error(ErrorCode::kRuntime, "injected completion query failure");
    }
    return state_->ready;
  }

  void wait() override {
    ++state_->wait_count;
    if (state_->throw_on_wait) {
      throw_error(ErrorCode::kRuntime, "injected completion wait failure");
    }
    state_->ready = true;
  }

private:
  std::shared_ptr<CompletionState> state_;
};

std::shared_ptr<HelixAllocator> make_allocator(std::size_t segment_bytes = 256) {
  return std::make_shared<HelixAllocator>(
      std::make_shared<SystemMemoryResource>(),
      AllocatorConfig{segment_bytes, 64, static_cast<std::size_t>(4 * 1024)});
}

TEST(DeferredAllocatorTest, PendingAllocationCannotBeReusedUntilCompletion) {
  auto base = make_allocator();
  DeferredAllocator allocator(base);
  const Allocation first = allocator.allocate(200);
  auto first_state = std::make_shared<CompletionState>();
  allocator.defer_deallocate(first, std::make_unique<ManualCompletion>(first_state));

  EXPECT_FALSE(allocator.owns(first));
  EXPECT_TRUE(allocator.is_pending(first));
  const Allocation second = allocator.allocate(200);
  EXPECT_NE(second.pointer(), first.pointer());
  EXPECT_EQ(base->stats().underlying_allocations, 2U);
  EXPECT_EQ(allocator.stats().pending_allocation_count, 1U);
  EXPECT_EQ(allocator.stats().pending_allocated_bytes, 256U);

  first_state->ready = true;
  const Allocation reused = allocator.allocate(200);
  EXPECT_EQ(reused.pointer(), first.pointer());
  EXPECT_EQ(allocator.stats().reclaimed_allocation_count, 1U);
  EXPECT_EQ(allocator.stats().client_allocation_count, 2U);

  auto ready = std::make_shared<CompletionState>();
  ready->ready = true;
  allocator.defer_deallocate(second, std::make_unique<ManualCompletion>(ready));
  allocator.defer_deallocate(reused, std::make_unique<ManualCompletion>(ready));
  EXPECT_EQ(allocator.reclaim_ready(), 2U);
}

TEST(DeferredAllocatorTest, WaitReclaimsAllAndAllowsCachedRelease) {
  auto base = make_allocator(512);
  DeferredAllocator allocator(base);
  const Allocation first = allocator.allocate(100);
  const Allocation second = allocator.allocate(100);
  auto first_state = std::make_shared<CompletionState>();
  auto second_state = std::make_shared<CompletionState>();
  allocator.defer_deallocate(first, std::make_unique<ManualCompletion>(first_state));
  allocator.defer_deallocate(second, std::make_unique<ManualCompletion>(second_state));

  EXPECT_EQ(allocator.wait_and_reclaim_all(), 2U);
  EXPECT_EQ(first_state->wait_count, 1U);
  EXPECT_EQ(second_state->wait_count, 1U);
  EXPECT_EQ(base->stats().bytes_allocated, 0U);
  EXPECT_EQ(allocator.release_cached(), 512U);
}

TEST(DeferredAllocatorTest, RejectsNullUnknownAndDuplicateDeferrals) {
  auto base = make_allocator();
  DeferredAllocator allocator(base);
  const Allocation allocation = allocator.allocate(64);
  EXPECT_THROW(allocator.defer_deallocate(allocation, nullptr), Error);
  EXPECT_THROW(allocator.defer_deallocate(Allocation{}, std::make_unique<ManualCompletion>(
                                                            std::make_shared<CompletionState>())),
               Error);

  auto state = std::make_shared<CompletionState>();
  allocator.defer_deallocate(allocation, std::make_unique<ManualCompletion>(state));
  EXPECT_THROW(allocator.defer_deallocate(allocation, std::make_unique<ManualCompletion>(state)),
               Error);
  state->ready = true;
  EXPECT_EQ(allocator.reclaim_ready(), 1U);
  EXPECT_THROW(allocator.defer_deallocate(allocation, std::make_unique<ManualCompletion>(state)),
               Error);
}

TEST(DeferredAllocatorTest, CompletionFailureLeavesPendingStateIntact) {
  auto base = make_allocator();
  DeferredAllocator allocator(base);
  const Allocation allocation = allocator.allocate(64);
  auto state = std::make_shared<CompletionState>();
  state->throw_on_query = true;
  allocator.defer_deallocate(allocation, std::make_unique<ManualCompletion>(state));

  EXPECT_THROW(static_cast<void>(allocator.reclaim_ready()), Error);
  EXPECT_TRUE(allocator.is_pending(allocation));
  EXPECT_TRUE(base->owns(allocation));
  EXPECT_EQ(allocator.stats().pending_allocation_count, 1U);

  state->throw_on_query = false;
  state->ready = true;
  EXPECT_EQ(allocator.reclaim_ready(), 1U);
  EXPECT_FALSE(base->owns(allocation));
}

TEST(DeferredAllocatorTest, DestructorWaitsForPendingCompletions) {
  auto base = make_allocator();
  auto state = std::make_shared<CompletionState>();
  {
    DeferredAllocator allocator(base);
    const Allocation allocation = allocator.allocate(64);
    allocator.defer_deallocate(allocation, std::make_unique<ManualCompletion>(state));
  }
  EXPECT_EQ(state->wait_count, 1U);
  EXPECT_EQ(base->stats().active_allocation_count, 0U);
  EXPECT_EQ(base->stats().bytes_allocated, 0U);
}

} // namespace
} // namespace helix::memory
