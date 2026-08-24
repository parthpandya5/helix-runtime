#pragma once

#include "helix/memory/stream_ordered_allocator.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>
#include <memory>

namespace helix::memory {

inline constexpr std::size_t kCudaTensorMaximumAlignment = 256;

// Allocates a mutable CUDA Tensor from a shared caching allocator. Every
// asynchronous use must be ordered on usage_stream before the final Tensor/view
// owner is destroyed. Final destruction records a release event on that stream;
// the allocator will not reuse the block until the event completes.
//
// This is an explicit single-stream ownership contract, not arbitrary record-
// stream tracking. Raw-pointer or other-stream use without a dependency into
// usage_stream is a caller error.
[[nodiscard]] Tensor
allocate_cuda_tensor_on_stream(Shape shape, DType dtype,
                               std::shared_ptr<StreamOrderedAllocator> allocator,
                               std::shared_ptr<cuda::Stream> usage_stream,
                               std::size_t alignment = kCudaTensorMaximumAlignment);

} // namespace helix::memory
