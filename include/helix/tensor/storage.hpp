#pragma once

#include "helix/core/device.hpp"

#include <cstddef>
#include <memory>

namespace helix {

inline constexpr std::size_t kDefaultStorageAlignment = 64;

class Storage final {
public:
  [[nodiscard]] static std::shared_ptr<Storage>
  allocate(std::size_t size_bytes, Device device, std::size_t alignment = kDefaultStorageAlignment);

  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

  [[nodiscard]] void* data() noexcept {
    return data_;
  }
  [[nodiscard]] const void* data() const noexcept {
    return data_;
  }
  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return size_bytes_;
  }
  [[nodiscard]] std::size_t alignment() const noexcept {
    return alignment_;
  }
  [[nodiscard]] Device device() const noexcept {
    return device_;
  }

private:
  Storage(void* data, std::size_t size_bytes, Device device, std::size_t alignment) noexcept;

  void* data_;
  std::size_t size_bytes_;
  Device device_;
  std::size_t alignment_;
};

} // namespace helix
