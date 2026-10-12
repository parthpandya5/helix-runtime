#pragma once

#include "helix/kv_cache/cuda_paged.hpp"
#include "helix/model/cuda_llama.hpp"
#include "helix/model/generation.hpp"
#include "helix/runtime/cuda/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace helix::model {

// Correctness-first, single-sequence CUDA generation session. The session owns
// its model, stream, and paged K/V pool. Model execution and cache append are
// synchronous; greedy selection explicitly copies one logits row to the CPU.
// Prompt prefill currently advances through the one-token paged decode path.
// The class is deliberately neither movable nor thread-safe.
class CudaLlamaGenerator final {
public:
  explicit CudaLlamaGenerator(CudaLlamaModel model, std::size_t block_tokens = 16);

  CudaLlamaGenerator(const CudaLlamaGenerator&) = delete;
  CudaLlamaGenerator& operator=(const CudaLlamaGenerator&) = delete;
  CudaLlamaGenerator(CudaLlamaGenerator&&) = delete;
  CudaLlamaGenerator& operator=(CudaLlamaGenerator&&) = delete;

  [[nodiscard]] Tensor prefill(std::span<const std::uint32_t> prompt_tokens);
  [[nodiscard]] Tensor decode(std::uint32_t token_id);
  [[nodiscard]] GenerationResult generate(std::span<const std::uint32_t> prompt_tokens,
                                          GenerationOptions options);

  [[nodiscard]] bool has_prefilled() const noexcept;
  [[nodiscard]] std::size_t cached_token_count() const;
  [[nodiscard]] kv_cache::PagedKvCacheStats cache_stats() const;
  [[nodiscard]] std::size_t block_tokens() const noexcept;
  [[nodiscard]] const CudaLlamaModel& model() const noexcept;
  [[nodiscard]] const helix::cuda::Stream& stream() const noexcept;

private:
  CudaLlamaModel model_;
  kv_cache::CudaPagedKvCache cache_;
  helix::cuda::Stream stream_;
  kv_cache::CudaPagedSequenceHandle sequence_;
  std::size_t block_tokens_ = 0;
  bool has_prefilled_ = false;
};

} // namespace helix::model
