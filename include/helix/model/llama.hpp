#pragma once

#include "helix/kv_cache/contiguous.hpp"
#include "helix/tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace helix::model {

struct LlamaConfig {
  std::size_t vocabulary_size = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t layer_count = 0;
  std::size_t attention_head_count = 0;
  std::size_t max_sequence_length = 0;
  float rms_norm_epsilon = 1.0e-5F;
  float rope_theta = 10'000.0F;

  void validate() const;
  [[nodiscard]] std::size_t head_dimension() const;
};

struct LlamaLayerWeights final {
  LlamaLayerWeights(Tensor attention_norm, Tensor query, Tensor key, Tensor value,
                    Tensor attention_output, Tensor ffn_norm, Tensor gate, Tensor up, Tensor down);

  LlamaLayerWeights(const LlamaLayerWeights&) = delete;
  LlamaLayerWeights& operator=(const LlamaLayerWeights&) = delete;
  LlamaLayerWeights(LlamaLayerWeights&&) noexcept = default;
  LlamaLayerWeights& operator=(LlamaLayerWeights&&) noexcept = default;

  Tensor attention_norm_weight;
  Tensor query_weight;
  Tensor key_weight;
  Tensor value_weight;
  Tensor attention_output_weight;
  Tensor ffn_norm_weight;
  Tensor gate_weight;
  Tensor up_weight;
  Tensor down_weight;
};

class LlamaWeights final {
public:
  LlamaWeights(Tensor token_embedding, std::vector<LlamaLayerWeights> layers,
               Tensor final_norm_weight, Tensor lm_head_weight);

  LlamaWeights(const LlamaWeights&) = delete;
  LlamaWeights& operator=(const LlamaWeights&) = delete;
  LlamaWeights(LlamaWeights&&) noexcept = default;
  LlamaWeights& operator=(LlamaWeights&&) noexcept = default;

  void validate(const LlamaConfig& config) const;

  [[nodiscard]] const Tensor& token_embedding() const noexcept;
  [[nodiscard]] std::span<const LlamaLayerWeights> layers() const noexcept;
  [[nodiscard]] const Tensor& final_norm_weight() const noexcept;
  [[nodiscard]] const Tensor& lm_head_weight() const noexcept;

private:
  Tensor token_embedding_;
  std::vector<LlamaLayerWeights> layers_;
  Tensor final_norm_weight_;
  Tensor lm_head_weight_;
};

// Opt-in correctness/debug trace. Layer states are captured after each complete
// attention-plus-MLP block. The ordinary forward path does not retain them.
struct CpuLlamaForwardTrace final {
  std::vector<Tensor> layer_hidden_states;
  Tensor final_normalized_state;
  Tensor logits;
};

// Full-prefill artifacts for populating a KV cache. Each layer tensor is
// head-major [heads, tokens, head_dimension]. Keys have RoPE applied; values do
// not. Logits contain every prompt position [tokens, vocabulary].
struct CpuLlamaPrefillResult final {
  std::vector<Tensor> layer_keys;
  std::vector<Tensor> layer_values;
  Tensor logits;
};

// One-token decode is computed transactionally against existing history. The
// caller appends token_keys/token_values [layers, heads, head_dimension] only
// after this method succeeds.
struct CpuLlamaDecodeResult final {
  Tensor token_keys;
  Tensor token_values;
  Tensor logits;
};

// Float32 CPU baseline for one precisely scoped Llama-style architecture:
// bias-free projections, equal Q/K/V head counts, full adjacent-pair RoPE,
// pre-attention/pre-MLP RMSNorm, SwiGLU, residual connections, and an explicit
// untied LM head. It performs full causal prefill and does not use a KV cache.
class CpuLlamaModel final {
public:
  CpuLlamaModel(LlamaConfig config, LlamaWeights weights);

  CpuLlamaModel(const CpuLlamaModel&) = delete;
  CpuLlamaModel& operator=(const CpuLlamaModel&) = delete;
  CpuLlamaModel(CpuLlamaModel&&) noexcept = default;
  CpuLlamaModel& operator=(CpuLlamaModel&&) noexcept = default;

  [[nodiscard]] const LlamaConfig& config() const noexcept;
  [[nodiscard]] const LlamaWeights& weights() const noexcept;
  [[nodiscard]] Tensor forward(std::span<const std::uint32_t> token_ids) const;
  [[nodiscard]] CpuLlamaForwardTrace
  forward_with_trace(std::span<const std::uint32_t> token_ids) const;
  [[nodiscard]] CpuLlamaPrefillResult
  prefill_with_kv(std::span<const std::uint32_t> token_ids) const;
  [[nodiscard]] CpuLlamaDecodeResult decode_one(std::uint32_t token_id, std::size_t position,
                                                const kv_cache::ContiguousKvCache& cache,
                                                const kv_cache::SequenceHandle& sequence) const;

private:
  LlamaConfig config_;
  LlamaWeights weights_;
};

} // namespace helix::model
