#pragma once

#include "helix/runtime/cuda/stream.hpp"
#include "helix/tensor/tensor.hpp"

namespace helix::cuda::kernels {

// Gathers rows from a Float32 [vocabulary, width] table using UInt32 [tokens]
// indices. The launch is asynchronous. Metadata errors throw before launch;
// device-resident IDs outside the vocabulary produce a NaN output row without
// reading outside table storage.
void embedding_lookup_out(const Tensor& token_ids, const Tensor& table, Tensor& output,
                          const Stream& stream);
[[nodiscard]] Tensor embedding_lookup(const Tensor& token_ids, const Tensor& table,
                                      const Stream& stream);

} // namespace helix::cuda::kernels
