#include "helix/memory/stream_ordered_allocator.hpp"

#include "helix/core/error.hpp"
#include "helix/memory/cuda_memory_resource.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <cuda_runtime_api.h>
#include <memory>
#include <utility>

namespace helix::memory {
namespace {

class CudaEventCompletion final : public Completion {
public:
  explicit CudaEventCompletion(const cuda::Stream& stream) : event_(cudaEventDisableTiming) {
    event_.record(stream);
  }

  [[nodiscard]] bool ready() override {
    return event_.query();
  }

  void wait() override {
    event_.synchronize();
  }

private:
  cuda::Event event_;
};

std::shared_ptr<HelixAllocator> make_allocator(int device_index, AllocatorConfig config) {
  HELIX_CHECK(device_index >= 0, ErrorCode::kInvalidArgument,
              "StreamOrderedAllocator device index must be non-negative");
  return std::make_shared<HelixAllocator>(std::make_shared<CudaMemoryResource>(device_index),
                                          config);
}

} // namespace

StreamOrderedAllocator::StreamOrderedAllocator(int device_index, AllocatorConfig config)
    : allocator_(make_allocator(device_index, config)), deferred_(allocator_) {}

StreamOrderedAllocator::~StreamOrderedAllocator() = default;

Allocation StreamOrderedAllocator::allocate(std::size_t bytes, std::size_t alignment) {
  return deferred_.allocate(bytes, alignment);
}

void StreamOrderedAllocator::defer_deallocate(const Allocation& allocation,
                                              const cuda::Stream& release_stream) {
  HELIX_CHECK(release_stream.native_handle() != nullptr, ErrorCode::kInvalidArgument,
              "StreamOrderedAllocator cannot use a moved-from release stream");
  HELIX_CHECK(release_stream.device_index() == device().index(), ErrorCode::kInvalidArgument,
              "StreamOrderedAllocator release stream belongs to another device");
  HELIX_CHECK(allocation.device() == device(), ErrorCode::kInvalidArgument,
              "StreamOrderedAllocator allocation belongs to another device");
  cuda::DeviceGuard guard(device().index());
  deferred_.defer_deallocate(allocation, std::make_unique<CudaEventCompletion>(release_stream));
}

std::size_t StreamOrderedAllocator::reclaim_ready() {
  return deferred_.reclaim_ready();
}

std::size_t StreamOrderedAllocator::wait_and_reclaim_all() {
  return deferred_.wait_and_reclaim_all();
}

std::size_t StreamOrderedAllocator::release_cached() {
  return deferred_.release_cached();
}

bool StreamOrderedAllocator::owns(const Allocation& allocation) const {
  return deferred_.owns(allocation);
}

bool StreamOrderedAllocator::is_pending(const Allocation& allocation) const {
  return deferred_.is_pending(allocation);
}

DeferredAllocatorStats StreamOrderedAllocator::stats() const {
  return deferred_.stats();
}

Device StreamOrderedAllocator::device() const noexcept {
  return allocator_->device();
}

} // namespace helix::memory
