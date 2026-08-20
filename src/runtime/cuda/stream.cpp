#include "helix/runtime/cuda/stream.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"
#include "helix/runtime/cuda/event.hpp"

#include <utility>

namespace helix::cuda {

Stream::Stream(unsigned int flags) : device_index_(current_device()) {
  HELIX_CUDA_CHECK_CONTEXT(cudaStreamCreateWithFlags(&stream_, flags), "creating CUDA stream");
}

Stream::Stream(int priority, unsigned int flags) : device_index_(current_device()) {
  HELIX_CUDA_CHECK_CONTEXT(cudaStreamCreateWithPriority(&stream_, flags, priority),
                           "creating priority CUDA stream");
}

Stream::~Stream() {
  destroy();
}

Stream::Stream(Stream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)),
      device_index_(std::exchange(other.device_index_, -1)) {}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    destroy();
    stream_ = std::exchange(other.stream_, nullptr);
    device_index_ = std::exchange(other.device_index_, -1);
  }
  return *this;
}

bool Stream::query() const {
  HELIX_CHECK(stream_ != nullptr, ErrorCode::kInvalidArgument,
              "cannot query a moved-from CUDA stream");
  DeviceGuard guard(device_index_);
  const cudaError_t result = cudaStreamQuery(stream_);
  if (result == cudaSuccess) {
    return true;
  }
  if (result == cudaErrorNotReady) {
    return false;
  }
  check(result, "cudaStreamQuery", "querying CUDA stream completion");
  return false;
}

void Stream::synchronize() const {
  HELIX_CHECK(stream_ != nullptr, ErrorCode::kInvalidArgument,
              "cannot synchronize a moved-from CUDA stream");
  DeviceGuard guard(device_index_);
  HELIX_CUDA_CHECK_CONTEXT(cudaStreamSynchronize(stream_), "synchronizing CUDA stream");
}

void Stream::wait(const Event& event) const {
  HELIX_CHECK(stream_ != nullptr, ErrorCode::kInvalidArgument,
              "cannot enqueue a wait on a moved-from CUDA stream");
  HELIX_CHECK(event.native_handle() != nullptr, ErrorCode::kInvalidArgument,
              "CUDA stream cannot wait on a moved-from event");
  HELIX_CHECK(event.device_index() == device_index_, ErrorCode::kInvalidArgument,
              "CUDA stream cannot wait on an event created on another device");
  DeviceGuard guard(device_index_);
  HELIX_CUDA_CHECK_CONTEXT(cudaStreamWaitEvent(stream_, event.native_handle(), 0),
                           "waiting for CUDA event on stream");
}

void Stream::destroy() noexcept {
  if (stream_ == nullptr) {
    return;
  }
  const cudaStream_t stream = std::exchange(stream_, nullptr);
  const int owning_device = std::exchange(device_index_, -1);
  int previous_device = -1;
  const cudaError_t get_result = cudaGetDevice(&previous_device);
  if (get_result == cudaSuccess && previous_device != owning_device) {
    const cudaError_t set_result = cudaSetDevice(owning_device);
    if (set_result != cudaSuccess) {
      report_cleanup_error(set_result, "selecting stream's CUDA device for destruction");
    }
  } else if (get_result != cudaSuccess) {
    report_cleanup_error(get_result, "querying CUDA device during stream destruction");
  }

  report_cleanup_error(cudaStreamDestroy(stream), "destroying CUDA stream");
  if (get_result == cudaSuccess && previous_device != owning_device) {
    report_cleanup_error(cudaSetDevice(previous_device),
                         "restoring CUDA device after stream destruction");
  }
}

} // namespace helix::cuda
