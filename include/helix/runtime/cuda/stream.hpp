#pragma once

#include <cuda_runtime_api.h>

namespace helix::cuda {

class Event;

class Stream final {
public:
  explicit Stream(unsigned int flags = cudaStreamNonBlocking);
  Stream(int priority, unsigned int flags = cudaStreamNonBlocking);
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  [[nodiscard]] cudaStream_t native_handle() const noexcept {
    return stream_;
  }
  [[nodiscard]] int device_index() const noexcept {
    return device_index_;
  }
  [[nodiscard]] bool query() const;
  void synchronize() const;
  void wait(const Event& event) const;

private:
  void destroy() noexcept;

  cudaStream_t stream_{nullptr};
  int device_index_{-1};
};

} // namespace helix::cuda
