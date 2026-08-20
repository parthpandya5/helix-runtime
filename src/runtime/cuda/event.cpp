#include "helix/runtime/cuda/event.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"
#include "helix/runtime/cuda/stream.hpp"

#include <utility>

namespace helix::cuda {

Event::Event(unsigned int flags) : device_index_(current_device()), flags_(flags) {
  HELIX_CUDA_CHECK_CONTEXT(cudaEventCreateWithFlags(&event_, flags), "creating CUDA event");
}

Event::~Event() {
  destroy();
}

Event::Event(Event&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)),
      device_index_(std::exchange(other.device_index_, -1)),
      flags_(std::exchange(other.flags_, cudaEventDefault)) {}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    destroy();
    event_ = std::exchange(other.event_, nullptr);
    device_index_ = std::exchange(other.device_index_, -1);
    flags_ = std::exchange(other.flags_, cudaEventDefault);
  }
  return *this;
}

bool Event::query() const {
  HELIX_CHECK(event_ != nullptr, ErrorCode::kInvalidArgument,
              "cannot query a moved-from CUDA event");
  DeviceGuard guard(device_index_);
  const cudaError_t result = cudaEventQuery(event_);
  if (result == cudaSuccess) {
    return true;
  }
  if (result == cudaErrorNotReady) {
    return false;
  }
  check(result, "cudaEventQuery", "querying CUDA event completion");
  return false;
}

void Event::record(const Stream& stream) const {
  HELIX_CHECK(event_ != nullptr, ErrorCode::kInvalidArgument,
              "cannot record a moved-from CUDA event");
  HELIX_CHECK(stream.native_handle() != nullptr, ErrorCode::kInvalidArgument,
              "cannot record an event on a moved-from CUDA stream");
  HELIX_CHECK(stream.device_index() == device_index_, ErrorCode::kInvalidArgument,
              "CUDA event and stream must belong to the same device");
  DeviceGuard guard(device_index_);
  HELIX_CUDA_CHECK_CONTEXT(cudaEventRecord(event_, stream.native_handle()), "recording CUDA event");
}

void Event::synchronize() const {
  HELIX_CHECK(event_ != nullptr, ErrorCode::kInvalidArgument,
              "cannot synchronize a moved-from CUDA event");
  DeviceGuard guard(device_index_);
  HELIX_CUDA_CHECK_CONTEXT(cudaEventSynchronize(event_), "synchronizing CUDA event");
}

void Event::destroy() noexcept {
  if (event_ == nullptr) {
    return;
  }
  const cudaEvent_t event = std::exchange(event_, nullptr);
  const int owning_device = std::exchange(device_index_, -1);
  flags_ = cudaEventDefault;

  int previous_device = -1;
  const cudaError_t get_result = cudaGetDevice(&previous_device);
  if (get_result == cudaSuccess && previous_device != owning_device) {
    const cudaError_t set_result = cudaSetDevice(owning_device);
    if (set_result != cudaSuccess) {
      report_cleanup_error(set_result, "selecting event's CUDA device for destruction");
    }
  } else if (get_result != cudaSuccess) {
    report_cleanup_error(get_result, "querying CUDA device during event destruction");
  }

  report_cleanup_error(cudaEventDestroy(event), "destroying CUDA event");
  if (get_result == cudaSuccess && previous_device != owning_device) {
    report_cleanup_error(cudaSetDevice(previous_device),
                         "restoring CUDA device after event destruction");
  }
}

float elapsed_time_ms(const Event& start, const Event& stop) {
  HELIX_CHECK(start.native_handle() != nullptr && stop.native_handle() != nullptr,
              ErrorCode::kInvalidArgument, "CUDA timing events must not be moved-from");
  HELIX_CHECK(start.device_index() == stop.device_index(), ErrorCode::kInvalidArgument,
              "CUDA timing events must belong to the same device");
  HELIX_CHECK((start.flags() & cudaEventDisableTiming) == 0 &&
                  (stop.flags() & cudaEventDisableTiming) == 0,
              ErrorCode::kInvalidArgument, "CUDA elapsed time requires timing-enabled events");
  DeviceGuard guard(start.device_index());
  float milliseconds = 0.0F;
  HELIX_CUDA_CHECK_CONTEXT(
      cudaEventElapsedTime(&milliseconds, start.native_handle(), stop.native_handle()),
      "measuring elapsed CUDA event time");
  return milliseconds;
}

} // namespace helix::cuda
