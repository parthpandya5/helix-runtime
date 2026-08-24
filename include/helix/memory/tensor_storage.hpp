#pragma once

#include "helix/memory/helix_allocator.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>
#include <memory>

namespace helix::memory {

// Allocates mutable CPU Tensor storage from a shared HelixAllocator. The returned
// Tensor and all of its views retain the allocator and return the block to its
// cache when the final shared Storage owner is destroyed.
[[nodiscard]] Tensor allocate_cpu_tensor(Shape shape, DType dtype,
                                         std::shared_ptr<HelixAllocator> allocator,
                                         std::size_t alignment = kDefaultStorageAlignment);

} // namespace helix::memory
