#include "helix/memory/cuda_tensor_storage.hpp"

#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>

namespace helix::memory {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

void log_release_failure(const Allocation& allocation, std::string_view stage,
                         std::string_view detail) noexcept {
  try {
    std::ostringstream message;
    message << "stream-bound CUDA Tensor could not release allocation " << allocation.id()
            << " while " << stage << ": " << detail;
    HELIX_LOG_ERROR(message.str());
  } catch (...) {
    // Storage destruction must not throw, including if formatting fails.
  }
}

class CudaTensorAllocationLifetime final {
public:
  CudaTensorAllocationLifetime(std::shared_ptr<StreamOrderedAllocator> allocator,
                               std::shared_ptr<cuda::Stream> stream, Allocation allocation) noexcept
      : stream_(std::move(stream)), allocator_(std::move(allocator)), allocation_(allocation) {}

  ~CudaTensorAllocationLifetime() noexcept {
    if (!armed_) {
      release_completed("rolling back Tensor construction");
      return;
    }

    try {
      allocator_->defer_deallocate(allocation_, *stream_);
      return;
    } catch (const std::exception& exception) {
      log_release_failure(allocation_, "recording its release event", exception.what());
    } catch (...) {
      log_release_failure(allocation_, "recording its release event", "unknown failure");
    }

    try {
      stream_->synchronize();
      release_completed("falling back after stream synchronization");
    } catch (const std::exception& exception) {
      log_release_failure(allocation_, "synchronizing its fallback release", exception.what());
    } catch (...) {
      log_release_failure(allocation_, "synchronizing its fallback release", "unknown failure");
    }
  }

  CudaTensorAllocationLifetime(const CudaTensorAllocationLifetime&) = delete;
  CudaTensorAllocationLifetime& operator=(const CudaTensorAllocationLifetime&) = delete;

  void arm() noexcept {
    armed_ = true;
  }

private:
  void release_completed(std::string_view stage) noexcept {
    try {
      allocator_->deallocate_completed(allocation_);
    } catch (const std::exception& exception) {
      log_release_failure(allocation_, stage, exception.what());
    } catch (...) {
      log_release_failure(allocation_, stage, "unknown deallocation failure");
    }
  }

  // Declared before allocator_ so reverse member destruction keeps the stream
  // alive while a last-owned allocator drains its recorded completion event.
  std::shared_ptr<cuda::Stream> stream_;
  std::shared_ptr<StreamOrderedAllocator> allocator_;
  Allocation allocation_;
  bool armed_ = false;
};

} // namespace

Tensor allocate_cuda_tensor_on_stream(Shape shape, DType dtype,
                                      std::shared_ptr<StreamOrderedAllocator> allocator,
                                      std::shared_ptr<cuda::Stream> usage_stream,
                                      std::size_t alignment) {
  HELIX_CHECK(allocator != nullptr, ErrorCode::kInvalidArgument,
              "allocate_cuda_tensor_on_stream requires a non-null allocator");
  HELIX_CHECK(usage_stream != nullptr && usage_stream->native_handle() != nullptr,
              ErrorCode::kInvalidArgument,
              "allocate_cuda_tensor_on_stream requires a valid non-null stream");
  HELIX_CHECK(is_power_of_two(alignment) && alignment >= alignof(std::max_align_t) &&
                  alignment <= kCudaTensorMaximumAlignment,
              ErrorCode::kInvalidArgument,
              "stream-bound CUDA Tensor alignment must be a power of two between "
              "alignof(max_align_t) and 256 bytes");
  HELIX_CHECK(allocator->device().is_cuda(), ErrorCode::kInvalidArgument,
              "allocate_cuda_tensor_on_stream requires a CUDA allocator");
  HELIX_CHECK(allocator->device().index() == usage_stream->device_index(),
              ErrorCode::kInvalidArgument,
              "stream-bound CUDA Tensor allocator and stream devices differ");
  const std::size_t logical_bytes = checked_nbytes(shape, dtype);
  if (logical_bytes == 0) {
    return Tensor::from_mutable_cuda_buffer(std::move(shape), dtype, nullptr, 0,
                                            allocator->device(), {}, alignment);
  }

  const Allocation allocation = allocator->allocate(logical_bytes, alignment);
  std::shared_ptr<CudaTensorAllocationLifetime> lifetime;
  try {
    lifetime = std::make_shared<CudaTensorAllocationLifetime>(allocator, usage_stream, allocation);
  } catch (...) {
    allocator->deallocate_completed(allocation);
    throw;
  }

  Tensor result = Tensor::from_mutable_cuda_buffer(
      std::move(shape), dtype, allocation.pointer(), allocation.allocated_bytes(),
      allocator->device(), lifetime, allocation.alignment());
  lifetime->arm();
  return result;
}

} // namespace helix::memory
