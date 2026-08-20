#include "helix/runtime/cuda/device.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <cuda_runtime_api.h>
#include <sstream>

namespace helix::cuda {
namespace {

void validate_device_index(int device_index) {
  HELIX_CHECK(device_index >= 0, ErrorCode::kInvalidArgument,
              "CUDA device index must be non-negative");
  const int count = device_count();
  if (device_index >= count) {
    std::ostringstream message;
    message << "CUDA device index " << device_index << " is out of range for " << count
            << " visible device(s)";
    throw_error(ErrorCode::kOutOfRange, message.str());
  }
}

} // namespace

bool runtime_available() noexcept {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int device_count() {
  int count = 0;
  HELIX_CUDA_CHECK_CONTEXT(cudaGetDeviceCount(&count), "querying visible CUDA devices");
  return count;
}

int current_device() {
  int device_index = -1;
  HELIX_CUDA_CHECK_CONTEXT(cudaGetDevice(&device_index), "querying the current CUDA device");
  return device_index;
}

void set_device(int device_index) {
  validate_device_index(device_index);
  std::ostringstream context;
  context << "selecting CUDA device " << device_index;
  HELIX_CUDA_CHECK_CONTEXT(cudaSetDevice(device_index), context.str());
}

void synchronize_device() {
  HELIX_CUDA_CHECK_CONTEXT(cudaDeviceSynchronize(), "synchronizing the current CUDA device");
}

DeviceInfo query_device(int device_index) {
  validate_device_index(device_index);
  cudaDeviceProp properties{};
  std::ostringstream context;
  context << "querying properties for CUDA device " << device_index;
  HELIX_CUDA_CHECK_CONTEXT(cudaGetDeviceProperties(&properties, device_index), context.str());
  return DeviceInfo{device_index,
                    properties.name,
                    properties.totalGlobalMem,
                    properties.major,
                    properties.minor,
                    properties.multiProcessorCount,
                    properties.warpSize,
                    properties.maxThreadsPerBlock};
}

DeviceGuard::DeviceGuard(int device_index)
    : previous_device_(current_device()), selected_device_(device_index),
      changed_device_(previous_device_ != selected_device_) {
  if (changed_device_) {
    set_device(selected_device_);
  }
}

DeviceGuard::~DeviceGuard() {
  if (changed_device_) {
    report_cleanup_error(cudaSetDevice(previous_device_), "restoring previous CUDA device");
  }
}

} // namespace helix::cuda
