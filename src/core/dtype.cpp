#include "helix/core/dtype.hpp"

#include "helix/core/error.hpp"

namespace helix {

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return "float32";
  case DType::kFloat16:
    return "float16";
  }
  throw_error(ErrorCode::kInvalidArgument, "unknown dtype");
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
    return 4;
  case DType::kFloat16:
    return 2;
  }
  throw_error(ErrorCode::kInvalidArgument, "unknown dtype");
}

bool dtype_is_floating_point(DType dtype) {
  switch (dtype) {
  case DType::kFloat32:
  case DType::kFloat16:
    return true;
  }
  throw_error(ErrorCode::kInvalidArgument, "unknown dtype");
}

} // namespace helix
