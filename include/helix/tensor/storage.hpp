#pragma once

#include "helix/core/device.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace helix {

inline constexpr std::size_t kDefaultStorageAlignment = 64;

class Storage final {
public:
  [[nodiscard]] static std::shared_ptr<Storage>
  allocate(std::size_t size_bytes, Device device, std::size_t alignment = kDefaultStorageAlignment);
  [[nodiscard]] static std::shared_ptr<Storage>
  wrap_read_only_cpu(const void* data, std::size_t size_bytes, std::shared_ptr<const void> owner,
                     std::size_t alignment = kDefaultStorageAlignment);

  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

  [[nodiscard]] const void* data() const noexcept {
    return data_;
  }
  [[nodiscard]] void* mutable_data();
  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return size_bytes_;
  }
  [[nodiscard]] std::size_t alignment() const noexcept {
    return alignment_;
  }
  [[nodiscard]] Device device() const noexcept {
    return device_;
  }
  [[nodiscard]] bool is_read_only() const noexcept {
    return read_only_;
  }

private:
  enum class ReleaseKind : std::uint8_t {
    kAlignedCpu,
    kCuda,
    kExternal,
  };

  Storage(const void* data, std::size_t size_bytes, Device device, std::size_t alignment,
          bool read_only, ReleaseKind release_kind,
          std::shared_ptr<const void> external_owner = {}) noexcept;

  const void* data_;
  std::size_t size_bytes_;
  Device device_;
  std::size_t alignment_;
  bool read_only_;
  ReleaseKind release_kind_;
  std::shared_ptr<const void> external_owner_;
};

} // namespace helix
