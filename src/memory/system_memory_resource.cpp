#include "helix/memory/system_memory_resource.hpp"

#include "helix/core/error.hpp"

#include <cstddef>
#include <new>

namespace helix::memory {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

void* SystemMemoryResource::allocate(std::size_t bytes, std::size_t alignment) {
  HELIX_CHECK(bytes != 0, ErrorCode::kInvalidArgument,
              "SystemMemoryResource cannot allocate zero bytes");
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "SystemMemoryResource alignment must be a non-zero power of two");
  try {
    return ::operator new(bytes, std::align_val_t(alignment));
  } catch (const std::bad_alloc&) {
    throw_error(ErrorCode::kResourceExhausted, "SystemMemoryResource aligned allocation failed");
  }
}

void SystemMemoryResource::deallocate(void* pointer, [[maybe_unused]] std::size_t bytes,
                                      std::size_t alignment) {
  HELIX_CHECK(pointer != nullptr, ErrorCode::kInvalidArgument,
              "SystemMemoryResource cannot deallocate a null pointer");
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "SystemMemoryResource alignment must be a non-zero power of two");
  ::operator delete(pointer, std::align_val_t(alignment));
}

Device SystemMemoryResource::device() const noexcept {
  return Device::cpu();
}

} // namespace helix::memory
