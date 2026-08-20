#include "helix/memory/cuda_memory_resource.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <sstream>

namespace helix::memory {
namespace {

constexpr std::size_t kCudaGuaranteedAlignment = 256;

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

} // namespace

CudaMemoryResource::CudaMemoryResource(int device_index) : device_index_(device_index) {
  HELIX_CHECK(device_index >= 0, ErrorCode::kInvalidArgument,
              "CudaMemoryResource device index must be non-negative");
}

void* CudaMemoryResource::allocate(std::size_t bytes, std::size_t alignment) {
  HELIX_CHECK(bytes != 0, ErrorCode::kInvalidArgument,
              "CudaMemoryResource cannot allocate zero bytes");
  HELIX_CHECK(is_power_of_two(alignment) && alignment <= kCudaGuaranteedAlignment,
              ErrorCode::kInvalidArgument,
              "CudaMemoryResource alignment must be a power of two no greater than 256 bytes");
  cuda::DeviceGuard guard(device_index_);
  void* pointer = nullptr;
  std::ostringstream context;
  context << "allocating " << bytes << " bytes on CUDA device " << device_index_;
  HELIX_CUDA_CHECK_CONTEXT(cudaMalloc(&pointer, bytes), context.str());
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address % alignment != 0) {
    const cudaError_t cleanup_result = cudaFree(pointer);
    if (cleanup_result != cudaSuccess) {
      cuda::report_cleanup_error(cleanup_result,
                                 "releasing unexpectedly misaligned CUDA allocation");
    }
    throw_error(ErrorCode::kRuntime,
                "cudaMalloc returned a pointer that violates the requested alignment");
  }
  return pointer;
}

void CudaMemoryResource::deallocate(void* pointer, [[maybe_unused]] std::size_t bytes,
                                    [[maybe_unused]] std::size_t alignment) {
  HELIX_CHECK(pointer != nullptr, ErrorCode::kInvalidArgument,
              "CudaMemoryResource cannot deallocate a null pointer");
  cuda::DeviceGuard guard(device_index_);
  std::ostringstream context;
  context << "releasing allocation on CUDA device " << device_index_;
  HELIX_CUDA_CHECK_CONTEXT(cudaFree(pointer), context.str());
}

Device CudaMemoryResource::device() const noexcept {
  return Device::cuda(device_index_);
}

} // namespace helix::memory
