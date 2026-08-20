#include "helix/tensor/storage.hpp"

#include "helix/core/error.hpp"

#include <cstdint>
#include <new>
#include <sstream>
#include <utility>

namespace helix {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

void Storage::AlignedDeleter::operator()(std::byte* pointer) const noexcept {
  if (pointer != nullptr) {
    ::operator delete(pointer, std::align_val_t(alignment));
  }
}

Storage::Storage(AlignedPointer data, std::size_t size_bytes, Device device,
                 std::size_t alignment) noexcept
    : data_(std::move(data)), size_bytes_(size_bytes), device_(device), alignment_(alignment) {}

std::shared_ptr<Storage> Storage::allocate(std::size_t size_bytes, Device device,
                                           std::size_t alignment) {
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "storage alignment must be a non-zero power of two");
  HELIX_CHECK(alignment >= alignof(std::max_align_t), ErrorCode::kInvalidArgument,
              "storage alignment must be at least alignof(max_align_t)");
  HELIX_CHECK(device.is_cpu(), ErrorCode::kUnsupported,
              "CUDA storage is unavailable until the CUDA runtime layer is enabled");

  std::byte* pointer = nullptr;
  if (size_bytes != 0) {
    try {
      pointer = static_cast<std::byte*>(::operator new(size_bytes, std::align_val_t(alignment)));
    } catch (const std::bad_alloc&) {
      std::ostringstream message;
      message << "CPU allocation of " << size_bytes << " bytes with " << alignment
              << "-byte alignment failed";
      throw_error(ErrorCode::kResourceExhausted, message.str());
    }
  }

  AlignedPointer data(pointer, AlignedDeleter{alignment});
  return std::shared_ptr<Storage>(new Storage(std::move(data), size_bytes, device, alignment));
}

} // namespace helix
