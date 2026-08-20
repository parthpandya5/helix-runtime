#pragma once

#include "helix/core/dtype.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace helix {

using Dimension = std::int64_t;
using Shape = std::vector<Dimension>;
using Strides = std::vector<Dimension>;

[[nodiscard]] std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                                           std::string_view context);
[[nodiscard]] std::size_t checked_numel(std::span<const Dimension> shape);
[[nodiscard]] std::size_t checked_nbytes(std::span<const Dimension> shape, DType dtype);
[[nodiscard]] Strides contiguous_strides(std::span<const Dimension> shape);
[[nodiscard]] bool has_contiguous_strides(std::span<const Dimension> shape,
                                          std::span<const Dimension> strides);
[[nodiscard]] std::string shape_string(std::span<const Dimension> shape);

} // namespace helix
