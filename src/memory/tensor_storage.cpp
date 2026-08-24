#include "helix/memory/tensor_storage.hpp"

#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <cstddef>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace helix::memory {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

void log_deallocation_failure(const Allocation& allocation, std::string_view detail) noexcept {
  try {
    std::ostringstream message;
    message << "allocator-backed CPU Tensor could not return allocation " << allocation.id()
            << " during destruction: " << detail;
    HELIX_LOG_ERROR(message.str());
  } catch (...) {
    // Destruction must not throw, including when formatting a cleanup error fails.
  }
}

class AllocationLifetime final {
public:
  AllocationLifetime(std::shared_ptr<HelixAllocator> allocator, Allocation allocation) noexcept
      : allocator_(std::move(allocator)), allocation_(allocation) {}

  ~AllocationLifetime() noexcept {
    try {
      allocator_->deallocate(allocation_);
    } catch (const std::exception& exception) {
      log_deallocation_failure(allocation_, exception.what());
    } catch (...) {
      log_deallocation_failure(allocation_, "unknown deallocation failure");
    }
  }

  AllocationLifetime(const AllocationLifetime&) = delete;
  AllocationLifetime& operator=(const AllocationLifetime&) = delete;

private:
  std::shared_ptr<HelixAllocator> allocator_;
  Allocation allocation_;
};

} // namespace

Tensor allocate_cpu_tensor(Shape shape, DType dtype, std::shared_ptr<HelixAllocator> allocator,
                           std::size_t alignment) {
  HELIX_CHECK(allocator != nullptr, ErrorCode::kInvalidArgument,
              "allocate_cpu_tensor requires a non-null allocator");
  const std::size_t logical_bytes = checked_nbytes(shape, dtype);
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "allocator-backed Tensor alignment must be a non-zero power of two");
  HELIX_CHECK(alignment >= alignof(std::max_align_t), ErrorCode::kInvalidArgument,
              "allocator-backed Tensor alignment must be at least alignof(max_align_t)");
  HELIX_CHECK(allocator->device().is_cpu(), ErrorCode::kUnsupported,
              "allocate_cpu_tensor requires a CPU HelixAllocator; CUDA Tensor pooling requires "
              "complete multi-stream use tracking");

  if (logical_bytes == 0) {
    return Tensor::from_mutable_cpu_buffer(std::move(shape), dtype, nullptr, 0, {}, alignment);
  }

  const Allocation allocation = allocator->allocate(logical_bytes, alignment);
  std::shared_ptr<AllocationLifetime> lifetime;
  try {
    lifetime = std::make_shared<AllocationLifetime>(allocator, allocation);
  } catch (...) {
    allocator->deallocate(allocation);
    throw;
  }

  return Tensor::from_mutable_cpu_buffer(std::move(shape), dtype, allocation.pointer(),
                                         allocation.allocated_bytes(), std::move(lifetime),
                                         allocation.alignment());
}

} // namespace helix::memory
