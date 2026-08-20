#include "helix/core/device.hpp"

#include "helix/core/error.hpp"

#include <ostream>
#include <sstream>

namespace helix {

std::string_view device_type_name(DeviceType type) {
  switch (type) {
  case DeviceType::kCpu:
    return "cpu";
  case DeviceType::kCuda:
    return "cuda";
  }
  throw_error(ErrorCode::kInvalidArgument, "unknown device type");
}

Device::Device(DeviceType type, std::int32_t index) : type_(type), index_(index) {
  HELIX_CHECK(index >= 0, ErrorCode::kInvalidArgument, "device index must be non-negative");
  HELIX_CHECK(type != DeviceType::kCpu || index == 0, ErrorCode::kInvalidArgument,
              "CPU device index must be zero");
  static_cast<void>(device_type_name(type));
}

Device Device::cpu() {
  return Device(DeviceType::kCpu, 0);
}

Device Device::cuda(std::int32_t index) {
  return Device(DeviceType::kCuda, index);
}

std::string Device::to_string() const {
  std::ostringstream stream;
  stream << device_type_name(type_);
  if (type_ == DeviceType::kCuda) {
    stream << ':' << index_;
  }
  return stream.str();
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  return stream << device.to_string();
}

} // namespace helix
