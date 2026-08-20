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

  ~Storage() = default;

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

  [[nodiscard]] void* data() noexcept {
    return data_.get();
  }
  [[nodiscard]] const void* data() const noexcept {
    return data_.get();
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
  struct AlignedDeleter {
    std::size_t alignment{kDefaultStorageAlignment};
    void operator()(std::byte* pointer) const noexcept;
  };

  using AlignedPointer = std::unique_ptr<std::byte, AlignedDeleter>;

  Storage(AlignedPointer data, std::size_t size_bytes, Device device,
          std::size_t alignment) noexcept;

  AlignedPointer data_;
  std::size_t size_bytes_;
  Device device_;
  std::size_t alignment_;
};

} // namespace helix
