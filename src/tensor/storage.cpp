#include "helix/tensor/storage.hpp"

#include "helix/core/config.hpp"
#include "helix/core/error.hpp"

#if HELIX_ENABLE_CUDA
#include "helix/runtime/cuda/memory.hpp"
#endif

#include <cstdint>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace helix {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

void validate_external_cpu_storage(const void* data, std::size_t size_bytes,
                                   const std::shared_ptr<const void>& owner, std::size_t alignment,
                                   std::string_view mutability) {
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "storage alignment must be a non-zero power of two");
  HELIX_CHECK(alignment >= alignof(std::max_align_t), ErrorCode::kInvalidArgument,
              "storage alignment must be at least alignof(max_align_t)");
  HELIX_CHECK((size_bytes == 0) == (data == nullptr), ErrorCode::kInvalidArgument,
              std::string(mutability) +
                  " storage requires a null pointer exactly when its size is zero");
  HELIX_CHECK(size_bytes == 0 || owner != nullptr, ErrorCode::kInvalidArgument,
              "non-empty " + std::string(mutability) + " storage requires a lifetime owner");
  if (data != nullptr) {
    HELIX_CHECK(
        reinterpret_cast<std::uintptr_t>(data) % alignment == 0, ErrorCode::kInvalidArgument,
        std::string(mutability) + " storage pointer does not satisfy its declared alignment");
  }
}

#if HELIX_ENABLE_CUDA
void validate_external_cuda_storage(const void* data, std::size_t size_bytes, Device device,
                                    const std::shared_ptr<const void>& owner,
                                    std::size_t alignment) {
  HELIX_CHECK(device.is_cuda(), ErrorCode::kInvalidArgument,
              "mutable CUDA storage requires a CUDA device");
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "storage alignment must be a non-zero power of two");
  HELIX_CHECK(alignment >= alignof(std::max_align_t), ErrorCode::kInvalidArgument,
              "storage alignment must be at least alignof(max_align_t)");
  HELIX_CHECK((size_bytes == 0) == (data == nullptr), ErrorCode::kInvalidArgument,
              "mutable CUDA storage requires a null pointer exactly when its size is zero");
  HELIX_CHECK(size_bytes == 0 || owner != nullptr, ErrorCode::kInvalidArgument,
              "non-empty mutable CUDA storage requires a lifetime owner");
  if (data != nullptr) {
    HELIX_CHECK(reinterpret_cast<std::uintptr_t>(data) % alignment == 0,
                ErrorCode::kInvalidArgument,
                "mutable CUDA storage pointer does not satisfy its declared alignment");
  }
}
#endif

} // namespace

Storage::Storage(const void* data, std::size_t size_bytes, Device device, std::size_t alignment,
                 bool read_only, ReleaseKind release_kind,
                 std::shared_ptr<const void> external_owner) noexcept
    : data_(data), size_bytes_(size_bytes), device_(device), alignment_(alignment),
      read_only_(read_only), release_kind_(release_kind),
      external_owner_(std::move(external_owner)) {}

Storage::~Storage() {
  if (data_ == nullptr) {
    return;
  }
  if (release_kind_ == ReleaseKind::kAlignedCpu) {
    ::operator delete(const_cast<void*>(data_), std::align_val_t(alignment_));
    return;
  }
#if HELIX_ENABLE_CUDA
  if (release_kind_ == ReleaseKind::kCuda) {
    cuda::free_device_bytes_noexcept(const_cast<void*>(data_), device_.index());
  }
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

  const ReleaseKind release_kind = device.is_cpu() ? ReleaseKind::kAlignedCpu : ReleaseKind::kCuda;
  return std::shared_ptr<Storage>(
      new Storage(pointer, size_bytes, device, alignment, false, release_kind));
}

std::shared_ptr<Storage> Storage::wrap_read_only_cpu(const void* data, std::size_t size_bytes,
                                                     std::shared_ptr<const void> owner,
                                                     std::size_t alignment) {
  validate_external_cpu_storage(data, size_bytes, owner, alignment, "read-only");
  return std::shared_ptr<Storage>(new Storage(data, size_bytes, Device::cpu(), alignment, true,
                                              ReleaseKind::kExternal, std::move(owner)));
}

std::shared_ptr<Storage> Storage::wrap_mutable_cpu(void* data, std::size_t size_bytes,
                                                   std::shared_ptr<void> owner,
                                                   std::size_t alignment) {
  std::shared_ptr<const void> immutable_owner = std::move(owner);
  validate_external_cpu_storage(data, size_bytes, immutable_owner, alignment, "mutable");
  return std::shared_ptr<Storage>(new Storage(data, size_bytes, Device::cpu(), alignment, false,
                                              ReleaseKind::kExternal, std::move(immutable_owner)));
}

std::shared_ptr<Storage> Storage::wrap_mutable_cuda(void* data, std::size_t size_bytes,
                                                    Device device, std::shared_ptr<void> owner,
                                                    std::size_t alignment) {
#if HELIX_ENABLE_CUDA
  std::shared_ptr<const void> immutable_owner = std::move(owner);
  validate_external_cuda_storage(data, size_bytes, device, immutable_owner, alignment);
  return std::shared_ptr<Storage>(new Storage(data, size_bytes, device, alignment, false,
                                              ReleaseKind::kExternal, std::move(immutable_owner)));
#else
  static_cast<void>(data);
  static_cast<void>(size_bytes);
  static_cast<void>(device);
  static_cast<void>(owner);
  static_cast<void>(alignment);
  throw_error(ErrorCode::kUnsupported, "external CUDA storage requires a CUDA-enabled build");
#endif
}

void* Storage::mutable_data() {
  HELIX_CHECK(!read_only_, ErrorCode::kInvalidArgument,
              "mutable access requested from read-only storage");
  return const_cast<void*>(data_);
}

} // namespace helix
