#include "helix/attention/reference.hpp"
#include "helix/core/error.hpp"
#include "helix/kernels/cpu/reference.hpp"
#include "helix/model/llama.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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

RopeCache make_rope_cache(std::size_t tokens, std::size_t head_dimension, float theta,
                          std::size_t position_offset = 0) {
  HELIX_CHECK(position_offset <= std::numeric_limits<std::size_t>::max() - tokens,
              ErrorCode::kOutOfRange, "model::CpuLlamaModel: RoPE position range overflows");
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
      const double angle = static_cast<double>(position_offset + token) * frequency;
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

struct ForwardArtifacts {
  std::vector<Tensor> layer_hidden_states;
  std::vector<Tensor> layer_keys;
  std::vector<Tensor> layer_values;
  Tensor final_normalized_state;
  Tensor logits;
};

ForwardArtifacts execute_forward(const LlamaConfig& config, const LlamaWeights& weights,
                                 std::span<const std::uint32_t> token_ids,
                                 bool capture_layer_states, bool capture_kv) {
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
  std::vector<Tensor> layer_keys;
  std::vector<Tensor> layer_values;
  if (capture_kv) {
    layer_keys.reserve(config.layer_count);
    layer_values.reserve(config.layer_count);
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
    if (capture_kv) {
      layer_keys.push_back(key);
      layer_values.push_back(value);
    }
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
  return ForwardArtifacts{std::move(layer_hidden_states), std::move(layer_keys),
                          std::move(layer_values), std::move(normalized), std::move(logits)};
}

} // namespace

Tensor CpuLlamaModel::forward(std::span<const std::uint32_t> token_ids) const {
  ForwardArtifacts result = execute_forward(config_, weights_, token_ids, false, false);
  return std::move(result.logits);
}

CpuLlamaForwardTrace
CpuLlamaModel::forward_with_trace(std::span<const std::uint32_t> token_ids) const {
  ForwardArtifacts result = execute_forward(config_, weights_, token_ids, true, false);
  return CpuLlamaForwardTrace{std::move(result.layer_hidden_states),
                              std::move(result.final_normalized_state), std::move(result.logits)};
}

CpuLlamaPrefillResult
CpuLlamaModel::prefill_with_kv(std::span<const std::uint32_t> token_ids) const {
  ForwardArtifacts result = execute_forward(config_, weights_, token_ids, false, true);
  return CpuLlamaPrefillResult{std::move(result.layer_keys), std::move(result.layer_values),
                               std::move(result.logits)};
}

CpuLlamaDecodeResult CpuLlamaModel::decode_one(std::uint32_t token_id, std::size_t position,
                                               const kv_cache::ContiguousKvCache& cache,
                                               const kv_cache::SequenceHandle& sequence) const {
  HELIX_CHECK(position < config_.max_sequence_length, ErrorCode::kOutOfRange,
              "model::CpuLlamaModel::decode_one: position exceeds configured maximum");
  HELIX_CHECK(cache.length(sequence) == position, ErrorCode::kInvalidArgument,
              "model::CpuLlamaModel::decode_one: cache length must equal decode position");
  const kv_cache::ContiguousKvCacheConfig& cache_config = cache.config();
  HELIX_CHECK(cache_config.layers == config_.layer_count &&
                  cache_config.kv_heads == config_.attention_head_count &&
                  cache_config.head_dimension == config_.head_dimension(),
              ErrorCode::kInvalidArgument,
              "model::CpuLlamaModel::decode_one: cache geometry does not match the model");

  const std::size_t layers = config_.layer_count;
  const std::size_t heads = config_.attention_head_count;
  const std::size_t head_dimension = config_.hidden_size / heads;
  const std::array<std::uint32_t, 1> input_ids{token_id};
  const RopeCache rope = make_rope_cache(1, head_dimension, config_.rope_theta, position);
  Tensor hidden = embed_tokens(input_ids, weights_.token_embedding(), config_.hidden_size,
                               config_.vocabulary_size);
  Tensor token_keys =
      Tensor::empty({dimension(layers), dimension(heads), dimension(head_dimension)});
  Tensor token_values =
      Tensor::empty({dimension(layers), dimension(heads), dimension(head_dimension)});
  float* token_key_values = token_keys.data_as<float>();
  float* token_value_values = token_values.data_as<float>();
  const std::size_t layer_elements = heads * head_dimension;

  for (std::size_t layer_index = 0; layer_index < layers; ++layer_index) {
    const LlamaLayerWeights& layer = weights_.layers()[layer_index];
    const Tensor attention_input =
        cpu::rms_norm(hidden, layer.attention_norm_weight, config_.rms_norm_epsilon);
    const Tensor query_projection = cpu::linear(attention_input, layer.query_weight);
    const Tensor key_projection = cpu::linear(attention_input, layer.key_weight);
    const Tensor value_projection = cpu::linear(attention_input, layer.value_weight);
    const Tensor query_token_heads =
        query_projection.reshape({1, dimension(heads), dimension(head_dimension)});
    const Tensor key_token_heads =
        key_projection.reshape({1, dimension(heads), dimension(head_dimension)});
    const Tensor value_token_heads =
        value_projection.reshape({1, dimension(heads), dimension(head_dimension)});
    const Tensor rotated_query = cpu::rope(query_token_heads, rope.cosine, rope.sine);
    const Tensor rotated_key = cpu::rope(key_token_heads, rope.cosine, rope.sine);
    const Tensor query = token_major_to_head_major(rotated_query, 1, heads, head_dimension);
    const Tensor current_key = token_major_to_head_major(rotated_key, 1, heads, head_dimension);
    const Tensor current_value =
        token_major_to_head_major(value_token_heads, 1, heads, head_dimension);

    Tensor combined_key =
        Tensor::empty({dimension(heads), dimension(position + 1), dimension(head_dimension)});
    Tensor combined_value = Tensor::empty(combined_key.shape());
    cache.copy_layer_prefix_out(sequence, layer_index, position, combined_key, combined_value);
    float* combined_key_values = combined_key.data_as<float>();
    float* combined_value_values = combined_value.data_as<float>();
    const float* current_key_values = current_key.data_as<float>();
    const float* current_value_values = current_value.data_as<float>();
    const std::size_t combined_head_elements = (position + 1) * head_dimension;
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t destination_offset =
          head * combined_head_elements + position * head_dimension;
      std::copy_n(current_key_values + head * head_dimension, head_dimension,
                  combined_key_values + destination_offset);
      std::copy_n(current_value_values + head * head_dimension, head_dimension,
                  combined_value_values + destination_offset);
    }
    const Tensor attention_context =
        attention::cpu::scaled_dot_product_attention(query, combined_key, combined_value, position);
    const Tensor merged_attention =
        head_major_to_token_major(attention_context, 1, heads, head_dimension);
    const Tensor attention_output = cpu::linear(merged_attention, layer.attention_output_weight);
    hidden = cpu::add(hidden, attention_output);

    const Tensor ffn_input = cpu::rms_norm(hidden, layer.ffn_norm_weight, config_.rms_norm_epsilon);
    const Tensor gate = cpu::linear(ffn_input, layer.gate_weight);
    const Tensor up = cpu::linear(ffn_input, layer.up_weight);
    const Tensor activated = cpu::swiglu(gate, up);
    hidden = cpu::add(hidden, cpu::linear(activated, layer.down_weight));

    std::copy_n(current_key.data_as<float>(), layer_elements,
                token_key_values + layer_index * layer_elements);
    std::copy_n(current_value.data_as<float>(), layer_elements,
                token_value_values + layer_index * layer_elements);
  }

  const Tensor normalized =
      cpu::rms_norm(hidden, weights_.final_norm_weight(), config_.rms_norm_epsilon);
  Tensor logits = cpu::linear(normalized, weights_.lm_head_weight());
  return CpuLlamaDecodeResult{std::move(token_keys), std::move(token_values), std::move(logits)};
}

} // namespace helix::model
