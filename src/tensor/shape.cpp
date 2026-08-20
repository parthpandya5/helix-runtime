#include "helix/tensor/shape.hpp"

#include "helix/core/error.hpp"

#include <limits>
#include <sstream>

namespace helix {
namespace {

std::size_t dimension_to_size(Dimension dimension, std::size_t index) {
  if (dimension < 0) {
    std::ostringstream message;
    message << "shape dimension " << index << " is negative: " << dimension;
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }

  const auto unsigned_dimension = static_cast<std::uint64_t>(dimension);
  if (unsigned_dimension > std::numeric_limits<std::size_t>::max()) {
    std::ostringstream message;
    message << "shape dimension " << index << " does not fit size_t: " << dimension;
    throw_error(ErrorCode::kOutOfRange, message.str());
  }
  return static_cast<std::size_t>(unsigned_dimension);
}

} // namespace

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, std::string_view context) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    std::ostringstream message;
    message << context << " overflow while multiplying " << lhs << " by " << rhs;
    throw_error(ErrorCode::kOutOfRange, message.str());
  }
  return lhs * rhs;
}

std::size_t checked_numel(std::span<const Dimension> shape) {
  std::size_t elements = 1;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    const auto extent = dimension_to_size(shape[index], index);
    elements = checked_multiply(elements, extent, "tensor element count");
  }
  return elements;
}

std::size_t checked_nbytes(std::span<const Dimension> shape, DType dtype) {
  return checked_multiply(checked_numel(shape), dtype_size(dtype), "tensor byte count");
}

Strides contiguous_strides(std::span<const Dimension> shape) {
  Strides strides(shape.size());
  std::size_t running_stride = 1;
  for (std::size_t reverse_index = shape.size(); reverse_index > 0; --reverse_index) {
    const auto index = reverse_index - 1;
    if (running_stride > static_cast<std::size_t>(std::numeric_limits<Dimension>::max())) {
      throw_error(ErrorCode::kOutOfRange, "contiguous stride does not fit Dimension");
    }
    strides[index] = static_cast<Dimension>(running_stride);
    const auto extent = dimension_to_size(shape[index], index);
    running_stride = checked_multiply(running_stride, extent, "contiguous stride");
  }
  return strides;
}

bool has_contiguous_strides(std::span<const Dimension> shape, std::span<const Dimension> strides) {
  if (shape.size() != strides.size()) {
    return false;
  }
  return contiguous_strides(shape) == Strides(strides.begin(), strides.end());
}

std::string shape_string(std::span<const Dimension> shape) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      stream << ", ";
    }
    stream << shape[index];
  }
  stream << ']';
  return stream.str();
}

} // namespace helix
