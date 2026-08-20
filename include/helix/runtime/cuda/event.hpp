#pragma once

#include <cuda_runtime_api.h>

namespace helix::cuda {

class Stream;

class Event final {
public:
  explicit Event(unsigned int flags = cudaEventDefault);
  ~Event();

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;

  [[nodiscard]] cudaEvent_t native_handle() const noexcept {
    return event_;
  }
  [[nodiscard]] int device_index() const noexcept {
    return device_index_;
  }
  [[nodiscard]] unsigned int flags() const noexcept {
    return flags_;
  }
  [[nodiscard]] bool query() const;
  void record(const Stream& stream) const;
  void synchronize() const;

private:
  void destroy() noexcept;

  cudaEvent_t event_{nullptr};
  int device_index_{-1};
  unsigned int flags_{cudaEventDefault};
};

[[nodiscard]] float elapsed_time_ms(const Event& start, const Event& stop);

} // namespace helix::cuda
