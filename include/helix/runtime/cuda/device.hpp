#pragma once

#include <cstddef>
#include <string>

namespace helix::cuda {

struct DeviceInfo {
  int index;
  std::string name;
  std::size_t total_memory_bytes;
  int compute_capability_major;
  int compute_capability_minor;
  int multiprocessor_count;
  int warp_size;
  int max_threads_per_block;
};

[[nodiscard]] bool runtime_available() noexcept;
[[nodiscard]] int device_count();
[[nodiscard]] int current_device();
void set_device(int device_index);
void synchronize_device();
[[nodiscard]] DeviceInfo query_device(int device_index);

class DeviceGuard final {
public:
  explicit DeviceGuard(int device_index);
  ~DeviceGuard();

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  DeviceGuard(DeviceGuard&&) = delete;
  DeviceGuard& operator=(DeviceGuard&&) = delete;

  [[nodiscard]] int previous_device() const noexcept {
    return previous_device_;
  }
  [[nodiscard]] int selected_device() const noexcept {
    return selected_device_;
  }

private:
  int previous_device_;
  int selected_device_;
  bool changed_device_;
};

} // namespace helix::cuda
