#include "helix/memory/deferred_allocator.hpp"

#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace helix::memory {
namespace {

std::string cleanup_message(const std::exception& exception) {
  std::ostringstream message;
  message << "DeferredAllocator failed while waiting during destruction: " << exception.what();
  return message.str();
}

} // namespace

struct DeferredAllocator::Impl {
  struct Pending {
    Allocation allocation;
    std::unique_ptr<Completion> completion;
  };

  explicit Impl(std::shared_ptr<HelixAllocator> base_allocator)
      : allocator(std::move(base_allocator)) {
    HELIX_CHECK(allocator != nullptr, ErrorCode::kInvalidArgument,
                "DeferredAllocator requires a non-null base allocator");
  }

  [[nodiscard]] bool pending_allocation(const Allocation& allocation) const {
    return std::any_of(pending.begin(), pending.end(), [&allocation](const Pending& item) {
      return item.allocation == allocation;
    });
  }

  std::size_t reclaim_ready_locked() {
    std::size_t reclaimed = 0;
    auto iterator = pending.begin();
    while (iterator != pending.end()) {
      if (!iterator->completion->ready()) {
        ++iterator;
        continue;
      }
      allocator->deallocate(iterator->allocation);
      iterator = pending.erase(iterator);
      ++reclaimed;
      ++reclaimed_allocation_count;
    }
    return reclaimed;
  }

  std::size_t wait_and_reclaim_all_locked() {
    std::size_t reclaimed = 0;
    auto iterator = pending.begin();
    while (iterator != pending.end()) {
      iterator->completion->wait();
      allocator->deallocate(iterator->allocation);
      iterator = pending.erase(iterator);
      ++reclaimed;
      ++reclaimed_allocation_count;
    }
    return reclaimed;
  }

  std::shared_ptr<HelixAllocator> allocator;
  mutable std::mutex mutex;
  std::vector<Pending> pending;
  std::uint64_t deferred_deallocation_count = 0;
  std::uint64_t reclaimed_allocation_count = 0;
};

DeferredAllocator::DeferredAllocator(std::shared_ptr<HelixAllocator> allocator)
    : impl_(std::make_unique<Impl>(std::move(allocator))) {}

DeferredAllocator::~DeferredAllocator() {
  std::scoped_lock lock(impl_->mutex);
  try {
    static_cast<void>(impl_->wait_and_reclaim_all_locked());
  } catch (const std::exception& exception) {
    HELIX_LOG_ERROR(cleanup_message(exception));
  } catch (...) {
    HELIX_LOG_ERROR("DeferredAllocator encountered an unknown completion failure during cleanup");
  }
}

Allocation DeferredAllocator::allocate(std::size_t bytes, std::size_t alignment) {
  std::scoped_lock lock(impl_->mutex);
  static_cast<void>(impl_->reclaim_ready_locked());
  return impl_->allocator->allocate(bytes, alignment);
}

void DeferredAllocator::deallocate_completed(const Allocation& allocation) {
  std::scoped_lock lock(impl_->mutex);
  HELIX_CHECK(!impl_->pending_allocation(allocation), ErrorCode::kInvalidArgument,
              "DeferredAllocator cannot immediately release a pending allocation");
  impl_->allocator->deallocate(allocation);
}

void DeferredAllocator::defer_deallocate(const Allocation& allocation,
                                         std::unique_ptr<Completion> completion) {
  std::scoped_lock lock(impl_->mutex);
  HELIX_CHECK(completion != nullptr, ErrorCode::kInvalidArgument,
              "DeferredAllocator requires a non-null completion");
  HELIX_CHECK(impl_->allocator->owns(allocation), ErrorCode::kInvalidArgument,
              "DeferredAllocator allocation is unknown or was already freed");
  HELIX_CHECK(!impl_->pending_allocation(allocation), ErrorCode::kInvalidArgument,
              "DeferredAllocator allocation is already pending reclamation");
  impl_->pending.push_back(Impl::Pending{allocation, std::move(completion)});
  ++impl_->deferred_deallocation_count;
}

std::size_t DeferredAllocator::reclaim_ready() {
  std::scoped_lock lock(impl_->mutex);
  return impl_->reclaim_ready_locked();
}

std::size_t DeferredAllocator::wait_and_reclaim_all() {
  std::scoped_lock lock(impl_->mutex);
  return impl_->wait_and_reclaim_all_locked();
}

std::size_t DeferredAllocator::release_cached() {
  std::scoped_lock lock(impl_->mutex);
  static_cast<void>(impl_->reclaim_ready_locked());
  return impl_->allocator->release_cached();
}

bool DeferredAllocator::owns(const Allocation& allocation) const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->allocator->owns(allocation) && !impl_->pending_allocation(allocation);
}

bool DeferredAllocator::is_pending(const Allocation& allocation) const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->pending_allocation(allocation);
}

DeferredAllocatorStats DeferredAllocator::stats() const {
  std::scoped_lock lock(impl_->mutex);
  DeferredAllocatorStats result;
  result.allocator = impl_->allocator->stats();
  result.pending_allocation_count = impl_->pending.size();
  result.deferred_deallocation_count = impl_->deferred_deallocation_count;
  result.reclaimed_allocation_count = impl_->reclaimed_allocation_count;
  for (const Impl::Pending& pending : impl_->pending) {
    result.pending_allocated_bytes += pending.allocation.allocated_bytes();
    result.pending_requested_bytes += pending.allocation.requested_bytes();
  }
  HELIX_ASSERT(result.pending_allocation_count <= result.allocator.active_allocation_count,
               "pending allocations exceed base active allocations");
  result.client_allocation_count =
      result.allocator.active_allocation_count - result.pending_allocation_count;
  return result;
}

Device DeferredAllocator::device() const noexcept {
  return impl_->allocator->device();
}

} // namespace helix::memory
