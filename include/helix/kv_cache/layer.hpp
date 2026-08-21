#pragma once

#include "helix/tensor/tensor.hpp"

namespace helix::kv_cache {

struct KvLayer final {
  Tensor keys;
  Tensor values;
};

} // namespace helix::kv_cache
