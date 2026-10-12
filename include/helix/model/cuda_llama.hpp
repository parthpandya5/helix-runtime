#pragma once

#include "helix/attention/cuda_paged.hpp"
#include "helix/model/llama.hpp"
#include "helix/runtime/cuda/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace helix::model {

// Transactional one-token decode artifacts. The device-resident key/value
// tensors are [layers, heads, head_dimension] and remain uncommitted; a cache
// owner appends them only after this result is returned successfully. Logits
// are [1, vocabulary].
struct CudaLlamaDecodeResult final {
  Tensor token_keys;
  Tensor token_values;
  Tensor logits;
};

// Synchronous Float32 CUDA correctness baseline for full causal prefill. The
// model owns a deep CUDA copy of all weights. Every kernel issued by
// forward_device uses the caller's stream, and that path synchronizes before
// returning so temporary tensor storage cannot be reclaimed while kernels still
// use it. The host-input overload performs blocking input/cache transfers first.
//
// One-token decode can read caller-owned paged K/V history without compacting
// it. This baseline still intentionally omits cache ownership/mutation,
// batching, allocator integration, graph capture, and overlap. Those require
// explicit long-lived workspace and stream-aware ownership rather than hidden
// lifetime assumptions.
class CudaLlamaModel final {
public:
  CudaLlamaModel(LlamaConfig config, const LlamaWeights& cpu_weights, int device_index = 0);

  CudaLlamaModel(const CudaLlamaModel&) = delete;
  CudaLlamaModel& operator=(const CudaLlamaModel&) = delete;
  CudaLlamaModel(CudaLlamaModel&&) noexcept = default;
  CudaLlamaModel& operator=(CudaLlamaModel&&) noexcept = default;

  [[nodiscard]] const LlamaConfig& config() const noexcept;
  [[nodiscard]] const LlamaWeights& weights() const noexcept;
  [[nodiscard]] int device_index() const noexcept;

  // Validates host token IDs, synchronously transfers IDs and RoPE tables, runs
  // prefill, and returns device-resident [tokens, vocabulary] logits.
  [[nodiscard]] Tensor forward(std::span<const std::uint32_t> token_ids,
                               const helix::cuda::Stream& stream) const;

  // Preloaded-input path for benchmarking and runtime composition. token_ids is
  // CUDA UInt32 [tokens]; cosine and sine are CUDA Float32
  // [tokens, head_dimension/2]. Device-resident token values cannot be checked
  // on the host without synchronization: out-of-vocabulary IDs are converted
  // by the embedding kernel into NaN rows, never out-of-bounds table reads.
  [[nodiscard]] Tensor forward_device(const Tensor& token_ids, const Tensor& cosine,
                                      const Tensor& sine, const helix::cuda::Stream& stream) const;

  // Synchronously transfers one validated host token and its position-specific
  // RoPE row before executing paged decode. Each history span contains exactly
  // one CUDA Float32 [physical_blocks, heads, block_tokens, head_dimension]
  // tensor per model layer. All layers share block_table and history length.
  [[nodiscard]] CudaLlamaDecodeResult decode_one_paged(
      std::uint32_t token_id, std::size_t position, std::span<const Tensor> layer_key_blocks,
      std::span<const Tensor> layer_value_blocks,
      const attention::cuda::PagedBlockTable& block_table, const helix::cuda::Stream& stream) const;

  // Preloaded one-token path for runtime composition and benchmarking. token_id
  // is CUDA UInt32 [1]; cosine/sine are CUDA Float32 [1,
  // head_dimension/2] for history_tokens. The call is synchronous, so all
  // caller-owned history/table inputs need remain alive only through return.
  // Device-resident token values cannot be synchronously rejected without a
  // readback; an out-of-vocabulary ID therefore propagates the embedding
  // kernel's NaN sentinel without an out-of-bounds read.
  [[nodiscard]] CudaLlamaDecodeResult
  decode_one_paged_device(const Tensor& token_id, const Tensor& cosine, const Tensor& sine,
                          std::span<const Tensor> layer_key_blocks,
                          std::span<const Tensor> layer_value_blocks,
                          const attention::cuda::PagedBlockTable& block_table,
                          std::size_t history_tokens, const helix::cuda::Stream& stream) const;

private:
  LlamaConfig config_;
  int device_index_;
  LlamaWeights weights_;
};

} // namespace helix::model
