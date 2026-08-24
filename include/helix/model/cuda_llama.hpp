#pragma once

#include "helix/model/llama.hpp"
#include "helix/runtime/cuda/stream.hpp"

#include <cstdint>
#include <span>

namespace helix::model {

// Synchronous Float32 CUDA correctness baseline for full causal prefill. The
// model owns a deep CUDA copy of all weights. Every operation is submitted to
// the caller's stream, and forward synchronizes that stream before returning so
// temporary tensor storage cannot be reclaimed while kernels still use it.
//
// This baseline intentionally does not provide KV-backed decode, batching,
// allocator integration, graph capture, or overlap. Those require explicit
// long-lived workspace and stream-aware ownership rather than hidden lifetime
// assumptions.
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

private:
  LlamaConfig config_;
  int device_index_;
  LlamaWeights weights_;
};

} // namespace helix::model
