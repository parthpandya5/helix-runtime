#pragma once

#include "helix/core/config.hpp"

#include <string_view>

namespace helix {

struct BuildInfo {
  std::string_view version;
  bool cuda_enabled;
};

[[nodiscard]] constexpr BuildInfo build_info() noexcept {
  return BuildInfo{HELIX_VERSION_STRING, HELIX_ENABLE_CUDA != 0};
}

} // namespace helix
