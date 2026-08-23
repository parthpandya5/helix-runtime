#include "helix/attention/reference.hpp"
#include "helix/core/error.hpp"
#include "helix/kernels/cpu/reference.hpp"
#include "helix/model/llama.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace helix::model {
namespace {

Dimension dimension(std::size_t value) {
  return static_cast<Dimension>(value);
}

Tensor embed_tokens(std::span<const std::uint32_t> token_ids, const Tensor& embedding,
                    std::size_t hidden_size, std::size_t vocabulary_size) {
  Tensor output = Tensor::empty({dimension(token_ids.size()), dimension(hidden_size)});
  const auto* source = embedding.data_as<float>();
  auto* destination = output.data_as<float>();
  for (std::size_t token = 0; token < token_ids.size(); ++token) {
    const std::size_t id = token_ids[token];
    HELIX_CHECK(id < vocabulary_size, ErrorCode::kOutOfRange,
                "model::CpuLlamaModel::forward: token ID is outside the vocabulary");
    std::copy_n(source + id * hidden_size, hidden_size, destination + token * hidden_size);
  }
  return output;
}

struct RopeCache {
  Tensor cosine;
  Tensor sine;
};

RopeCache make_rope_cache(std::size_t tokens, std::size_t head_dimension, float theta) {
  const std::size_t pairs = head_dimension / 2;
  Tensor cosine = Tensor::empty({dimension(tokens), dimension(pairs)});
  Tensor sine = Tensor::empty({dimension(tokens), dimension(pairs)});
  auto* cosine_values = cosine.data_as<float>();
  auto* sine_values = sine.data_as<float>();
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const double exponent =
          -2.0 * static_cast<double>(pair) / static_cast<double>(head_dimension);
      const double frequency = std::pow(static_cast<double>(theta), exponent);
      const double angle = static_cast<double>(token) * frequency;
      cosine_values[token * pairs + pair] = static_cast<float>(std::cos(angle));
      sine_values[token * pairs + pair] = static_cast<float>(std::sin(angle));
    }
  }
  return RopeCache{std::move(cosine), std::move(sine)};
}

Tensor token_major_to_head_major(const Tensor& input, std::size_t tokens, std::size_t heads,
                                 std::size_t head_dimension) {
  Tensor output = Tensor::empty({dimension(heads), dimension(tokens), dimension(head_dimension)});
  const auto* source = input.data_as<float>();
  auto* destination = output.data_as<float>();
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < heads; ++head) {
      for (std::size_t coordinate = 0; coordinate < head_dimension; ++coordinate) {
        destination[(head * tokens + token) * head_dimension + coordinate] =
            source[(token * heads + head) * head_dimension + coordinate];
      }
    }
  }
  return output;
}

Tensor head_major_to_token_major(const Tensor& input, std::size_t tokens, std::size_t heads,
                                 std::size_t head_dimension) {
  Tensor output = Tensor::empty({dimension(tokens), dimension(heads * head_dimension)});
  const auto* source = input.data_as<float>();
  auto* destination = output.data_as<float>();
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t token = 0; token < tokens; ++token) {
      for (std::size_t coordinate = 0; coordinate < head_dimension; ++coordinate) {
        destination[token * heads * head_dimension + head * head_dimension + coordinate] =
            source[(head * tokens + token) * head_dimension + coordinate];
      }
    }
  }
  return output;
}

CpuLlamaForwardTrace execute_forward(const LlamaConfig& config, const LlamaWeights& weights,
                                     std::span<const std::uint32_t> token_ids,
                                     bool capture_layer_states) {
  HELIX_CHECK(!token_ids.empty(), ErrorCode::kInvalidArgument,
              "model::CpuLlamaModel::forward: token sequence must not be empty");
  HELIX_CHECK(token_ids.size() <= config.max_sequence_length, ErrorCode::kOutOfRange,
              "model::CpuLlamaModel::forward: token sequence exceeds configured maximum");

  const std::size_t tokens = token_ids.size();
  const std::size_t heads = config.attention_head_count;
  const std::size_t head_dimension = config.hidden_size / heads;
  const RopeCache rope = make_rope_cache(tokens, head_dimension, config.rope_theta);
  Tensor hidden = embed_tokens(token_ids, weights.token_embedding(), config.hidden_size,
                               config.vocabulary_size);
  std::vector<Tensor> layer_hidden_states;
  if (capture_layer_states) {
    layer_hidden_states.reserve(config.layer_count);
  }

  for (const LlamaLayerWeights& layer : weights.layers()) {
    const Tensor attention_input =
        cpu::rms_norm(hidden, layer.attention_norm_weight, config.rms_norm_epsilon);
    const Tensor query_projection = cpu::linear(attention_input, layer.query_weight);
    const Tensor key_projection = cpu::linear(attention_input, layer.key_weight);
    const Tensor value_projection = cpu::linear(attention_input, layer.value_weight);

    const Tensor query_token_heads =
        query_projection.reshape({dimension(tokens), dimension(heads), dimension(head_dimension)});
    const Tensor key_token_heads =
        key_projection.reshape({dimension(tokens), dimension(heads), dimension(head_dimension)});
    const Tensor value_token_heads =
        value_projection.reshape({dimension(tokens), dimension(heads), dimension(head_dimension)});
    const Tensor rotated_query = cpu::rope(query_token_heads, rope.cosine, rope.sine);
    const Tensor rotated_key = cpu::rope(key_token_heads, rope.cosine, rope.sine);

    const Tensor query = token_major_to_head_major(rotated_query, tokens, heads, head_dimension);
    const Tensor key = token_major_to_head_major(rotated_key, tokens, heads, head_dimension);
    const Tensor value =
        token_major_to_head_major(value_token_heads, tokens, heads, head_dimension);
    const Tensor attention_context =
        attention::cpu::scaled_dot_product_attention(query, key, value);
    const Tensor merged_attention =
        head_major_to_token_major(attention_context, tokens, heads, head_dimension);
    const Tensor attention_output = cpu::linear(merged_attention, layer.attention_output_weight);
    hidden = cpu::add(hidden, attention_output);

    const Tensor ffn_input = cpu::rms_norm(hidden, layer.ffn_norm_weight, config.rms_norm_epsilon);
    const Tensor gate = cpu::linear(ffn_input, layer.gate_weight);
    const Tensor up = cpu::linear(ffn_input, layer.up_weight);
    const Tensor activated = cpu::swiglu(gate, up);
    const Tensor ffn_output = cpu::linear(activated, layer.down_weight);
    hidden = cpu::add(hidden, ffn_output);
    if (capture_layer_states) {
      // Operations are out of place, so this shallow handle pins an immutable
      // stage allocation without copying its payload.
      layer_hidden_states.push_back(hidden);
    }
  }

  Tensor normalized = cpu::rms_norm(hidden, weights.final_norm_weight(), config.rms_norm_epsilon);
  Tensor logits = cpu::linear(normalized, weights.lm_head_weight());
  return CpuLlamaForwardTrace{std::move(layer_hidden_states), std::move(normalized),
                              std::move(logits)};
}

} // namespace

Tensor CpuLlamaModel::forward(std::span<const std::uint32_t> token_ids) const {
  CpuLlamaForwardTrace result = execute_forward(config_, weights_, token_ids, false);
  return std::move(result.logits);
}

CpuLlamaForwardTrace
CpuLlamaModel::forward_with_trace(std::span<const std::uint32_t> token_ids) const {
  return execute_forward(config_, weights_, token_ids, true);
}

} // namespace helix::model
