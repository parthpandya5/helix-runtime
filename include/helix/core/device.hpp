#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

namespace helix {

enum class DeviceType : std::uint8_t {
  kCpu,
  kCuda,
};

[[nodiscard]] std::string_view device_type_name(DeviceType type);

class Device final {
public:
  explicit Device(DeviceType type = DeviceType::kCpu, std::int32_t index = 0);

  [[nodiscard]] static Device cpu();
  [[nodiscard]] static Device cuda(std::int32_t index = 0);

  [[nodiscard]] DeviceType type() const noexcept {
    return type_;
  }
  [[nodiscard]] std::int32_t index() const noexcept {
    return index_;
  }
  [[nodiscard]] bool is_cpu() const noexcept {
    return type_ == DeviceType::kCpu;
  }
  [[nodiscard]] bool is_cuda() const noexcept {
    return type_ == DeviceType::kCuda;
  }
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const Device&, const Device&) = default;

private:
  DeviceType type_;
  std::int32_t index_;
};

std::ostream& operator<<(std::ostream& stream, const Device& device);

} // namespace helix
