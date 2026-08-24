#pragma once

#include "helix/memory/completion.hpp"
#include "helix/memory/helix_allocator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace helix::memory {

struct DeferredAllocatorStats {
  AllocatorStats allocator;
  std::size_t client_allocation_count = 0;
  std::size_t pending_allocation_count = 0;
  std::size_t pending_allocated_bytes = 0;
  std::size_t pending_requested_bytes = 0;
  std::uint64_t deferred_deallocation_count = 0;
  std::uint64_t reclaimed_allocation_count = 0;
};

class DeferredAllocator final {
public:
  explicit DeferredAllocator(std::shared_ptr<HelixAllocator> allocator);
  ~DeferredAllocator();

  DeferredAllocator(const DeferredAllocator&) = delete;
  DeferredAllocator& operator=(const DeferredAllocator&) = delete;
  DeferredAllocator(DeferredAllocator&&) = delete;
  DeferredAllocator& operator=(DeferredAllocator&&) = delete;

  [[nodiscard]] Allocation allocate(std::size_t bytes, std::size_t alignment = 0);
  // Returns an allocation immediately after the caller has independently proved
  // that no asynchronous use remains. Pending allocations are rejected.
  void deallocate_completed(const Allocation& allocation);
  void defer_deallocate(const Allocation& allocation, std::unique_ptr<Completion> completion);
  [[nodiscard]] std::size_t reclaim_ready();
  [[nodiscard]] std::size_t wait_and_reclaim_all();
  [[nodiscard]] std::size_t release_cached();

  [[nodiscard]] bool owns(const Allocation& allocation) const;
  [[nodiscard]] bool is_pending(const Allocation& allocation) const;
  [[nodiscard]] DeferredAllocatorStats stats() const;
  [[nodiscard]] Device device() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace helix::memory
