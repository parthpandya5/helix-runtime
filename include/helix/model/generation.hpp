#pragma once

#include "helix/model/llama.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace helix::model {

enum class GenerationStopReason {
  kMaxNewTokens,
  kEosToken,
  kContextLimit,
};

[[nodiscard]] std::string_view generation_stop_reason_name(GenerationStopReason reason);

struct GenerationOptions {
  std::size_t max_new_tokens = 0;
  std::optional<std::uint32_t> eos_token;
};

struct GenerationResult {
  std::vector<std::uint32_t> generated_tokens;
  GenerationStopReason stop_reason = GenerationStopReason::kMaxNewTokens;
  std::size_t cached_token_count = 0;
};

// Selects the lowest token ID among equal finite maxima in the last logits row.
[[nodiscard]] std::uint32_t select_greedy_token(const Tensor& logits);

// A correctness-first, single-sequence CPU generation session. It owns both the
// model and its contiguous KV cache, so neither can dangle independently. The
// class is deliberately neither movable nor thread-safe.
class CpuLlamaGenerator final {
public:
  explicit CpuLlamaGenerator(CpuLlamaModel model);

  CpuLlamaGenerator(const CpuLlamaGenerator&) = delete;
  CpuLlamaGenerator& operator=(const CpuLlamaGenerator&) = delete;
  CpuLlamaGenerator(CpuLlamaGenerator&&) = delete;
  CpuLlamaGenerator& operator=(CpuLlamaGenerator&&) = delete;

  [[nodiscard]] Tensor prefill(std::span<const std::uint32_t> prompt_tokens);
  [[nodiscard]] Tensor decode(std::uint32_t token_id);
  [[nodiscard]] GenerationResult generate(std::span<const std::uint32_t> prompt_tokens,
                                          GenerationOptions options);

  [[nodiscard]] bool has_prefilled() const noexcept;
  [[nodiscard]] std::size_t cached_token_count() const;
  [[nodiscard]] kv_cache::ContiguousKvCacheStats cache_stats() const;
  [[nodiscard]] const CpuLlamaModel& model() const noexcept;

private:
  CpuLlamaModel model_;
  kv_cache::ContiguousKvCache cache_;
  kv_cache::SequenceHandle sequence_;
  bool has_prefilled_ = false;
};

} // namespace helix::model
