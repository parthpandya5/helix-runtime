#pragma once

#include "helix/tensor/tensor.hpp"

#include <cstddef>

namespace helix::model::detail {

struct RopeCache final {
  Tensor cosine;
  Tensor sine;
};

[[nodiscard]] RopeCache make_rope_cache(std::size_t tokens, std::size_t head_dimension,
                                        float theta, std::size_t position_offset = 0);

} // namespace helix::model::detail
