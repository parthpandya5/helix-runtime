#include "helix/tensor/storage.hpp"

#include "helix/core/config.hpp"
#include "helix/core/error.hpp"

#if HELIX_ENABLE_CUDA
#include "helix/runtime/cuda/memory.hpp"
#endif

#include <cstdint>
#include <new>
#include <sstream>

namespace helix {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

Storage::Storage(void* data, std::size_t size_bytes, Device device, std::size_t alignment) noexcept
    : data_(data), size_bytes_(size_bytes), device_(device), alignment_(alignment) {}

Storage::~Storage() {
  if (data_ == nullptr) {
    return;
  }
  if (device_.is_cpu()) {
    ::operator delete(data_, std::align_val_t(alignment_));
    return;
  }
#if HELIX_ENABLE_CUDA
  cuda::free_device_bytes_noexcept(data_, device_.index());
#endif
}

std::shared_ptr<Storage> Storage::allocate(std::size_t size_bytes, Device device,
                                           std::size_t alignment) {
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "storage alignment must be a non-zero power of two");
  HELIX_CHECK(alignment >= alignof(std::max_align_t), ErrorCode::kInvalidArgument,
              "storage alignment must be at least alignof(max_align_t)");
  void* pointer = nullptr;
  if (device.is_cpu()) {
    if (size_bytes != 0) {
      try {
        pointer = ::operator new(size_bytes, std::align_val_t(alignment));
      } catch (const std::bad_alloc&) {
        std::ostringstream message;
        message << "CPU allocation of " << size_bytes << " bytes with " << alignment
                << "-byte alignment failed";
        throw_error(ErrorCode::kResourceExhausted, message.str());
      }
    }
  } else {
#if HELIX_ENABLE_CUDA
    pointer = cuda::allocate_device_bytes(size_bytes, device.index());
    if (pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
      cuda::free_device_bytes_noexcept(pointer, device.index());
      std::ostringstream message;
      message << "CUDA allocation did not satisfy requested " << alignment << "-byte alignment";
      throw_error(ErrorCode::kRuntime, message.str());
    }
#else
    throw_error(ErrorCode::kUnsupported,
                "CUDA storage requested in a build configured with HELIX_ENABLE_CUDA=OFF");
#endif
  }

  return std::shared_ptr<Storage>(new Storage(pointer, size_bytes, device, alignment));
}

} // namespace helix
