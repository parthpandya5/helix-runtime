#include "helix/runtime/cuda/memory.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"
#include "helix/runtime/cuda/stream.hpp"

#include <cuda_runtime_api.h>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace helix::cuda {
namespace {

void validate_copy_arguments(void* destination, const void* source, std::size_t size_bytes,
                             std::string_view operation) {
  if (size_bytes == 0) {
    return;
  }
  HELIX_CHECK(destination != nullptr, ErrorCode::kInvalidArgument,
              std::string(operation) + ": destination must not be null for a non-empty copy");
  HELIX_CHECK(source != nullptr, ErrorCode::kInvalidArgument,
              std::string(operation) + ": source must not be null for a non-empty copy");
}

std::string copy_context(std::string_view direction, std::size_t size_bytes) {
  std::ostringstream context;
  context << "copying " << size_bytes << " bytes " << direction;
  return context.str();
}

} // namespace

void* allocate_device_bytes(std::size_t size_bytes, int device_index) {
  DeviceGuard guard(device_index);
  if (size_bytes == 0) {
    return nullptr;
  }
  void* pointer = nullptr;
  std::ostringstream context;
  context << "allocating " << size_bytes << " bytes on CUDA device " << device_index;
  HELIX_CUDA_CHECK_CONTEXT(cudaMalloc(&pointer, size_bytes), context.str());
  return pointer;
}

void free_device_bytes_noexcept(void* pointer, int device_index) noexcept {
  if (pointer == nullptr) {
    return;
  }

  int previous_device = -1;
  const cudaError_t get_result = cudaGetDevice(&previous_device);
  if (get_result == cudaSuccess && previous_device != device_index) {
    const cudaError_t set_result = cudaSetDevice(device_index);
    if (set_result != cudaSuccess) {
      report_cleanup_error(set_result, "selecting allocation's CUDA device before cudaFree");
    }
  } else if (get_result != cudaSuccess) {
    report_cleanup_error(get_result, "querying CUDA device before cudaFree");
  }

  report_cleanup_error(cudaFree(pointer), "freeing CUDA allocation");
  if (get_result == cudaSuccess && previous_device != device_index) {
    report_cleanup_error(cudaSetDevice(previous_device), "restoring CUDA device after cudaFree");
  }
}

void copy_host_to_device(void* destination, const void* source, std::size_t size_bytes) {
  validate_copy_arguments(destination, source, size_bytes, "copy_host_to_device");
  if (size_bytes == 0) {
    return;
  }
  HELIX_CUDA_CHECK_CONTEXT(cudaMemcpy(destination, source, size_bytes, cudaMemcpyHostToDevice),
                           copy_context("host to device", size_bytes));
}

void copy_device_to_host(void* destination, const void* source, std::size_t size_bytes) {
  validate_copy_arguments(destination, source, size_bytes, "copy_device_to_host");
  if (size_bytes == 0) {
    return;
  }
  HELIX_CUDA_CHECK_CONTEXT(cudaMemcpy(destination, source, size_bytes, cudaMemcpyDeviceToHost),
                           copy_context("device to host", size_bytes));
}

void copy_device_to_device(void* destination, const void* source, std::size_t size_bytes) {
  validate_copy_arguments(destination, source, size_bytes, "copy_device_to_device");
  if (size_bytes == 0) {
    return;
  }
  HELIX_CUDA_CHECK_CONTEXT(cudaMemcpy(destination, source, size_bytes, cudaMemcpyDeviceToDevice),
                           copy_context("device to device", size_bytes));
}

void copy_device_to_device_peer(void* destination, int destination_device, const void* source,
                                int source_device, std::size_t size_bytes) {
  validate_copy_arguments(destination, source, size_bytes, "copy_device_to_device_peer");
  if (size_bytes == 0) {
    return;
  }
  std::ostringstream context;
  context << "copying " << size_bytes << " bytes from CUDA device " << source_device
          << " to CUDA device " << destination_device;
  HELIX_CUDA_CHECK_CONTEXT(
      cudaMemcpyPeer(destination, destination_device, source, source_device, size_bytes),
      context.str());
}

void set_device_bytes(void* destination, int byte_value, std::size_t size_bytes) {
  if (size_bytes == 0) {
    return;
  }
  HELIX_CHECK(destination != nullptr, ErrorCode::kInvalidArgument,
              "set_device_bytes destination must not be null for a non-empty operation");
  std::ostringstream context;
  context << "setting " << size_bytes << " device bytes";
  HELIX_CUDA_CHECK_CONTEXT(cudaMemset(destination, byte_value, size_bytes), context.str());
}

void copy_host_to_device_async(void* destination, const void* source, std::size_t size_bytes,
                               const Stream& stream) {
  validate_copy_arguments(destination, source, size_bytes, "copy_host_to_device_async");
  if (size_bytes == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  HELIX_CUDA_CHECK_CONTEXT(cudaMemcpyAsync(destination, source, size_bytes, cudaMemcpyHostToDevice,
                                           stream.native_handle()),
                           copy_context("host to device asynchronously", size_bytes));
}

void copy_device_to_host_async(void* destination, const void* source, std::size_t size_bytes,
                               const Stream& stream) {
  validate_copy_arguments(destination, source, size_bytes, "copy_device_to_host_async");
  if (size_bytes == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  HELIX_CUDA_CHECK_CONTEXT(cudaMemcpyAsync(destination, source, size_bytes, cudaMemcpyDeviceToHost,
                                           stream.native_handle()),
                           copy_context("device to host asynchronously", size_bytes));
}

void copy_device_to_device_async(void* destination, const void* source, std::size_t size_bytes,
                                 const Stream& stream) {
  validate_copy_arguments(destination, source, size_bytes, "copy_device_to_device_async");
  if (size_bytes == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  HELIX_CUDA_CHECK_CONTEXT(cudaMemcpyAsync(destination, source, size_bytes,
                                           cudaMemcpyDeviceToDevice, stream.native_handle()),
                           copy_context("device to device asynchronously", size_bytes));
}

void set_device_bytes_async(void* destination, int byte_value, std::size_t size_bytes,
                            const Stream& stream) {
  if (size_bytes == 0) {
    return;
  }
  HELIX_CHECK(destination != nullptr, ErrorCode::kInvalidArgument,
              "set_device_bytes_async destination must not be null for non-empty work");
  DeviceGuard guard(stream.device_index());
  std::ostringstream context;
  context << "setting " << size_bytes << " device bytes asynchronously";
  HELIX_CUDA_CHECK_CONTEXT(
      cudaMemsetAsync(destination, byte_value, size_bytes, stream.native_handle()), context.str());
}

Buffer::Buffer(std::size_t size_bytes, int device_index)
    : size_bytes_(size_bytes), device_index_(device_index < 0 ? current_device() : device_index) {
  pointer_ = allocate_device_bytes(size_bytes_, device_index_);
}

Buffer::~Buffer() {
  destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)),
      device_index_(std::exchange(other.device_index_, -1)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    destroy();
    pointer_ = std::exchange(other.pointer_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
    device_index_ = std::exchange(other.device_index_, -1);
  }
  return *this;
}

void* Buffer::release() noexcept {
  size_bytes_ = 0;
  device_index_ = -1;
  return std::exchange(pointer_, nullptr);
}

void Buffer::destroy() noexcept {
  void* pointer = std::exchange(pointer_, nullptr);
  const int device_index = std::exchange(device_index_, -1);
  size_bytes_ = 0;
  free_device_bytes_noexcept(pointer, device_index);
}

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) : size_bytes_(size_bytes) {
  if (size_bytes_ == 0) {
    return;
  }
  std::ostringstream context;
  context << "allocating " << size_bytes_ << " bytes of portable pinned host memory";
  HELIX_CUDA_CHECK_CONTEXT(cudaHostAlloc(&pointer_, size_bytes_, cudaHostAllocPortable),
                           context.str());
}

PinnedHostBuffer::~PinnedHostBuffer() {
  destroy();
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
  if (this != &other) {
    destroy();
    pointer_ = std::exchange(other.pointer_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
  }
  return *this;
}

void PinnedHostBuffer::destroy() noexcept {
  void* pointer = std::exchange(pointer_, nullptr);
  size_bytes_ = 0;
  if (pointer != nullptr) {
    report_cleanup_error(cudaFreeHost(pointer), "freeing portable pinned host memory");
  }
}

} // namespace helix::cuda
