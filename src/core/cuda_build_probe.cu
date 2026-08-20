#include "helix/core/config.hpp"

static_assert(HELIX_ENABLE_CUDA == 1, "CUDA probe must only compile in a CUDA build");

namespace helix::detail {

// This translation unit makes Milestone 0 CUDA validation an actual CUDA compile,
// rather than merely locating the toolkit. Runtime wrappers begin in Milestone 3.
int cuda_compilation_probe() noexcept {
  return __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__;
}

} // namespace helix::detail
