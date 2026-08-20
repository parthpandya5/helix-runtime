#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace helix {

enum class DType : std::uint8_t {
  kFloat32,
  kFloat16,
};

[[nodiscard]] std::string_view dtype_name(DType dtype);
[[nodiscard]] std::size_t dtype_size(DType dtype);
[[nodiscard]] bool dtype_is_floating_point(DType dtype);

} // namespace helix
