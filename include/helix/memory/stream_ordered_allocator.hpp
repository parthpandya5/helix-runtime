#pragma once

#include "helix/memory/deferred_allocator.hpp"
#include "helix/runtime/cuda/stream.hpp"

namespace helix::memory {

class StreamOrderedAllocator final {
public:
  explicit StreamOrderedAllocator(int device_index, AllocatorConfig config = {});
  ~StreamOrderedAllocator();

  StreamOrderedAllocator(const StreamOrderedAllocator&) = delete;
  StreamOrderedAllocator& operator=(const StreamOrderedAllocator&) = delete;
  StreamOrderedAllocator(StreamOrderedAllocator&&) = delete;
  StreamOrderedAllocator& operator=(StreamOrderedAllocator&&) = delete;

  [[nodiscard]] Allocation allocate(std::size_t bytes, std::size_t alignment = 0);
  // Immediate release for callers that have already established completion.
  // Using this on an allocation with outstanding work is invalid.
  void deallocate_completed(const Allocation& allocation);
  void defer_deallocate(const Allocation& allocation, const cuda::Stream& release_stream);
  [[nodiscard]] std::size_t reclaim_ready();
  [[nodiscard]] std::size_t wait_and_reclaim_all();
  [[nodiscard]] std::size_t release_cached();
  [[nodiscard]] bool owns(const Allocation& allocation) const;
  [[nodiscard]] bool is_pending(const Allocation& allocation) const;
  [[nodiscard]] DeferredAllocatorStats stats() const;
  [[nodiscard]] Device device() const noexcept;

private:
  std::shared_ptr<HelixAllocator> allocator_;
  DeferredAllocator deferred_;
};

} // namespace helix::memory
