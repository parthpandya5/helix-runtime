#include "helix/core/error.hpp"
#include "helix/model/llama.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace helix::model {
namespace {

struct RawLayerWeights {
  std::vector<float> attention_norm;
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> attention_output;
  std::vector<float> ffn_norm;
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> down;
};

struct RawWeights {
  std::vector<float> token_embedding;
  std::vector<RawLayerWeights> layers;
  std::vector<float> final_norm;
  std::vector<float> lm_head;
};

LlamaConfig tiny_config() {
  return LlamaConfig{6, 4, 3, 2, 2, 8, 1.0e-5F, 10'000.0F};
}

std::vector<float> patterned(std::size_t count, std::size_t seed, float scale = 0.035F) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto bucket = static_cast<int>((index * 7 + seed * 5) % 19) - 9;
    values[index] = static_cast<float>(bucket) * scale;
  }
  return values;
}

std::vector<float> norm_weights(std::size_t count, std::size_t seed) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = 0.85F + 0.025F * static_cast<float>((index + seed) % 7);
  }
  return values;
}

RawWeights make_raw_weights(const LlamaConfig& config) {
  RawWeights weights;
  weights.token_embedding = patterned(config.vocabulary_size * config.hidden_size, 1, 0.045F);
  weights.layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::size_t seed = 10 + layer * 11;
    weights.layers.push_back(
        RawLayerWeights{norm_weights(config.hidden_size, seed),
                        patterned(config.hidden_size * config.hidden_size, seed + 1),
                        patterned(config.hidden_size * config.hidden_size, seed + 2),
                        patterned(config.hidden_size * config.hidden_size, seed + 3),
                        patterned(config.hidden_size * config.hidden_size, seed + 4),
                        norm_weights(config.hidden_size, seed + 5),
                        patterned(config.hidden_size * config.intermediate_size, seed + 6),
                        patterned(config.hidden_size * config.intermediate_size, seed + 7),
                        patterned(config.intermediate_size * config.hidden_size, seed + 8)});
  }
  weights.final_norm = norm_weights(config.hidden_size, 101);
  weights.lm_head = patterned(config.hidden_size * config.vocabulary_size, 102, 0.04F);
  return weights;
}

LlamaLayerWeights materialize_layer(const LlamaConfig& config, const RawLayerWeights& raw) {
  const auto hidden = static_cast<Dimension>(config.hidden_size);
  const auto intermediate = static_cast<Dimension>(config.intermediate_size);
  return LlamaLayerWeights(Tensor::from_values({hidden}, raw.attention_norm),
                           Tensor::from_values({hidden, hidden}, raw.query),
                           Tensor::from_values({hidden, hidden}, raw.key),
                           Tensor::from_values({hidden, hidden}, raw.value),
                           Tensor::from_values({hidden, hidden}, raw.attention_output),
                           Tensor::from_values({hidden}, raw.ffn_norm),
                           Tensor::from_values({hidden, intermediate}, raw.gate),
                           Tensor::from_values({hidden, intermediate}, raw.up),
                           Tensor::from_values({intermediate, hidden}, raw.down));
}

LlamaWeights materialize_weights(const LlamaConfig& config, const RawWeights& raw) {
  const auto vocabulary = static_cast<Dimension>(config.vocabulary_size);
  const auto hidden = static_cast<Dimension>(config.hidden_size);
  std::vector<LlamaLayerWeights> layers;
  layers.reserve(raw.layers.size());
  for (const RawLayerWeights& layer : raw.layers) {
    layers.push_back(materialize_layer(config, layer));
  }
  return LlamaWeights(Tensor::from_values({vocabulary, hidden}, raw.token_embedding),
                      std::move(layers), Tensor::from_values({hidden}, raw.final_norm),
                      Tensor::from_values({hidden, vocabulary}, raw.lm_head));
}

std::vector<float> rms_norm(const std::vector<float>& input, std::size_t rows, std::size_t width,
                            const std::vector<float>& weight, float epsilon) {
  std::vector<float> output(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    double squares = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      const double value = input[row * width + column];
      squares += value * value;
    }
    const double inverse = 1.0 / std::sqrt(squares / static_cast<double>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      output[row * width + column] =
          static_cast<float>(static_cast<double>(input[row * width + column]) * inverse *
                             static_cast<double>(weight[column]));
    }
  }
  return output;
}

std::vector<float> linear(const std::vector<float>& input, std::size_t rows,
                          std::size_t input_width, const std::vector<float>& weight,
                          std::size_t output_width) {
  std::vector<float> output(rows * output_width);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < output_width; ++column) {
      double accumulator = 0.0;
      for (std::size_t reduction = 0; reduction < input_width; ++reduction) {
        accumulator += static_cast<double>(input[row * input_width + reduction]) *
                       static_cast<double>(weight[reduction * output_width + column]);
      }
      output[row * output_width + column] = static_cast<float>(accumulator);
    }
  }
  return output;
}

void add_in_place(std::vector<float>& destination, const std::vector<float>& increment) {
  if (destination.size() != increment.size()) {
    throw std::logic_error("oracle residual operands must have equal sizes");
  }
  for (std::size_t index = 0; index < destination.size(); ++index) {
    destination[index] += increment[index];
  }
}

void apply_rope(std::vector<float>& projection, std::size_t tokens, std::size_t heads,
                std::size_t head_dimension, float theta) {
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t base = (token * heads + head) * head_dimension;
      for (std::size_t pair = 0; pair < head_dimension / 2; ++pair) {
        const double exponent =
            -2.0 * static_cast<double>(pair) / static_cast<double>(head_dimension);
        const double angle =
            static_cast<double>(token) * std::pow(static_cast<double>(theta), exponent);
        const float cosine = static_cast<float>(std::cos(angle));
        const float sine = static_cast<float>(std::sin(angle));
        const float first = projection[base + 2 * pair];
        const float second = projection[base + 2 * pair + 1];
        projection[base + 2 * pair] = first * cosine - second * sine;
        projection[base + 2 * pair + 1] = first * sine + second * cosine;
      }
    }
  }
}

std::vector<float> causal_attention(const std::vector<float>& query, const std::vector<float>& key,
                                    const std::vector<float>& value, std::size_t tokens,
                                    std::size_t heads, std::size_t head_dimension) {
  std::vector<float> output(tokens * heads * head_dimension);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dimension));
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t head = 0; head < heads; ++head) {
      std::vector<float> scores(token + 1);
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t key_token = 0; key_token <= token; ++key_token) {
        double dot = 0.0;
        for (std::size_t coordinate = 0; coordinate < head_dimension; ++coordinate) {
          dot += static_cast<double>(query[(token * heads + head) * head_dimension + coordinate]) *
                 static_cast<double>(key[(key_token * heads + head) * head_dimension + coordinate]);
        }
        scores[key_token] = static_cast<float>(dot * scale);
        maximum = std::max(maximum, scores[key_token]);
      }
      double denominator = 0.0;
      for (const float score : scores) {
        denominator += std::exp(static_cast<double>(score - maximum));
      }
      for (float& score : scores) {
        score = static_cast<float>(std::exp(static_cast<double>(score - maximum)) / denominator);
      }
      for (std::size_t coordinate = 0; coordinate < head_dimension; ++coordinate) {
        double accumulator = 0.0;
        for (std::size_t key_token = 0; key_token <= token; ++key_token) {
          accumulator +=
              static_cast<double>(scores[key_token]) *
              static_cast<double>(value[(key_token * heads + head) * head_dimension + coordinate]);
        }
        output[(token * heads + head) * head_dimension + coordinate] =
            static_cast<float>(accumulator);
      }
    }
  }
  return output;
}

float stable_silu(float value) {
  const double wide = value;
  const double sigmoid =
      value >= 0.0F ? 1.0 / (1.0 + std::exp(-wide)) : std::exp(wide) / (1.0 + std::exp(wide));
  return static_cast<float>(wide * sigmoid);
}

struct OracleTrace {
  std::vector<std::vector<float>> layer_hidden_states;
  std::vector<float> final_normalized_state;
  std::vector<float> logits;
};

OracleTrace oracle_forward(const LlamaConfig& config, const RawWeights& weights,
                           std::span<const std::uint32_t> token_ids) {
  const std::size_t tokens = token_ids.size();
  const std::size_t hidden = config.hidden_size;
  const std::size_t heads = config.attention_head_count;
  const std::size_t head_dimension = hidden / heads;
  std::vector<float> states(tokens * hidden);
  std::vector<std::vector<float>> layer_hidden_states;
  layer_hidden_states.reserve(config.layer_count);
  for (std::size_t token = 0; token < tokens; ++token) {
    std::copy_n(weights.token_embedding.data() + token_ids[token] * hidden, hidden,
                states.data() + token * hidden);
  }

  for (const RawLayerWeights& layer : weights.layers) {
    const std::vector<float> attention_input =
        rms_norm(states, tokens, hidden, layer.attention_norm, config.rms_norm_epsilon);
    std::vector<float> query = linear(attention_input, tokens, hidden, layer.query, hidden);
    std::vector<float> key = linear(attention_input, tokens, hidden, layer.key, hidden);
    const std::vector<float> value = linear(attention_input, tokens, hidden, layer.value, hidden);
    apply_rope(query, tokens, heads, head_dimension, config.rope_theta);
    apply_rope(key, tokens, heads, head_dimension, config.rope_theta);
    const std::vector<float> context =
        causal_attention(query, key, value, tokens, heads, head_dimension);
    add_in_place(states, linear(context, tokens, hidden, layer.attention_output, hidden));

    const std::vector<float> ffn_input =
        rms_norm(states, tokens, hidden, layer.ffn_norm, config.rms_norm_epsilon);
    const std::vector<float> gate =
        linear(ffn_input, tokens, hidden, layer.gate, config.intermediate_size);
    const std::vector<float> up =
        linear(ffn_input, tokens, hidden, layer.up, config.intermediate_size);
    std::vector<float> activated(gate.size());
    for (std::size_t index = 0; index < gate.size(); ++index) {
      activated[index] = stable_silu(gate[index]) * up[index];
    }
    add_in_place(states, linear(activated, tokens, config.intermediate_size, layer.down, hidden));
    layer_hidden_states.push_back(states);
  }

  std::vector<float> normalized =
      rms_norm(states, tokens, hidden, weights.final_norm, config.rms_norm_epsilon);
  std::vector<float> logits =
      linear(normalized, tokens, hidden, weights.lm_head, config.vocabulary_size);
  return OracleTrace{std::move(layer_hidden_states), std::move(normalized), std::move(logits)};
}

void expect_tensor_near(const Tensor& actual, const std::vector<float>& expected, float tolerance) {
  ASSERT_EQ(actual.numel(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected[index], tolerance)
        << "at element " << index;
  }
}

TEST(LlamaConfigTest, ValidatesNarrowArchitectureContract) {
  LlamaConfig config = tiny_config();
  EXPECT_NO_THROW(config.validate());
  EXPECT_EQ(config.head_dimension(), 2U);

  config.vocabulary_size = 0;
  EXPECT_THROW(config.validate(), Error);
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    config = tiny_config();
    config.vocabulary_size =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 2;
    EXPECT_THROW(config.validate(), Error);
  }
  config = tiny_config();
  config.hidden_size = 5;
  EXPECT_THROW(config.validate(), Error);
  config = tiny_config();
  config.hidden_size = 6;
  EXPECT_THROW(config.validate(), Error);
  config = tiny_config();
  config.layer_count = 0;
  EXPECT_THROW(config.validate(), Error);
  config = tiny_config();
  config.max_sequence_length = 0;
  EXPECT_THROW(config.validate(), Error);
  config = tiny_config();
  config.rms_norm_epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(config.validate(), Error);
  config = tiny_config();
  config.rope_theta = 0.0F;
  EXPECT_THROW(config.validate(), Error);
}

TEST(LlamaWeightsTest, ValidatesEveryLayerAndTopLevelShape) {
  const LlamaConfig config = tiny_config();
  const RawWeights raw = make_raw_weights(config);
  LlamaWeights weights = materialize_weights(config, raw);
  EXPECT_NO_THROW(weights.validate(config));
  EXPECT_EQ(weights.layers().size(), 2U);

  LlamaWeights no_layers(Tensor::zeros({6, 4}), {}, Tensor::zeros({4}), Tensor::zeros({4, 6}));
  EXPECT_THROW(no_layers.validate(config), Error);

  std::vector<LlamaLayerWeights> embedding_layers;
  embedding_layers.push_back(materialize_layer(config, raw.layers[0]));
  embedding_layers.push_back(materialize_layer(config, raw.layers[1]));
  LlamaWeights wrong_embedding(Tensor::zeros({6, 4}, DType::kFloat16), std::move(embedding_layers),
                               Tensor::zeros({4}), Tensor::zeros({4, 6}));
  EXPECT_THROW(wrong_embedding.validate(config), Error);

  LlamaLayerWeights wrong_query(Tensor::zeros({4}), Tensor::zeros({4, 5}), Tensor::zeros({4, 4}),
                                Tensor::zeros({4, 4}), Tensor::zeros({4, 4}), Tensor::zeros({4}),
                                Tensor::zeros({4, 3}), Tensor::zeros({4, 3}),
                                Tensor::zeros({3, 4}));
  std::vector<LlamaLayerWeights> invalid_layers;
  invalid_layers.push_back(std::move(wrong_query));
  invalid_layers.push_back(materialize_layer(config, raw.layers[1]));
  LlamaWeights wrong_layer(Tensor::zeros({6, 4}), std::move(invalid_layers), Tensor::zeros({4}),
                           Tensor::zeros({4, 6}));
  EXPECT_THROW(wrong_layer.validate(config), Error);
}

TEST(LlamaWeightsTest, TransfersEveryTensorDeeplyAndValidatesTheExpectedDevice) {
  const LlamaConfig config = tiny_config();
  const RawWeights raw = make_raw_weights(config);
  LlamaWeights source = materialize_weights(config, raw);

  LlamaWeights copied = source.to(Device::cpu());
  EXPECT_NO_THROW(copied.validate(config, Device::cpu()));
  EXPECT_THROW(source.validate(config, Device::cuda(0)), Error);
  EXPECT_NE(copied.token_embedding().data(), source.token_embedding().data());
  ASSERT_EQ(copied.layers().size(), source.layers().size());
  EXPECT_NE(copied.layers()[0].query_weight.data(), source.layers()[0].query_weight.data());
  for (std::size_t index = 0; index < source.token_embedding().numel(); ++index) {
    EXPECT_FLOAT_EQ(copied.token_embedding().data_as<float>()[index],
                    source.token_embedding().data_as<float>()[index]);
  }

  Tensor mutable_copy = copied.token_embedding();
  const float original = source.token_embedding().data_as<float>()[0];
  mutable_copy.data_as<float>()[0] += 1.0F;
  EXPECT_FLOAT_EQ(source.token_embedding().data_as<float>()[0], original);
  EXPECT_NE(copied.token_embedding().data_as<float>()[0], original);
}

TEST(CpuLlamaModelTest, MatchesIndependentOracleAcrossFullCausalPrefill) {
  const LlamaConfig config = tiny_config();
  const RawWeights raw = make_raw_weights(config);
  const std::vector<std::uint32_t> token_ids{2, 5, 1};
  const OracleTrace expected = oracle_forward(config, raw, token_ids);
  CpuLlamaModel model(config, materialize_weights(config, raw));

  const CpuLlamaForwardTrace trace = model.forward_with_trace(token_ids);
  ASSERT_EQ(trace.layer_hidden_states.size(), expected.layer_hidden_states.size());
  for (std::size_t layer = 0; layer < expected.layer_hidden_states.size(); ++layer) {
    EXPECT_EQ(trace.layer_hidden_states[layer].shape(), (Shape{3, 4}));
    expect_tensor_near(trace.layer_hidden_states[layer], expected.layer_hidden_states[layer],
                       5.0e-5F);
  }
  EXPECT_EQ(trace.final_normalized_state.shape(), (Shape{3, 4}));
  expect_tensor_near(trace.final_normalized_state, expected.final_normalized_state, 5.0e-5F);
  ASSERT_EQ(trace.logits.shape(), (Shape{3, 6}));
  expect_tensor_near(trace.logits, expected.logits, 5.0e-5F);

  const Tensor repeat = model.forward(token_ids);
  for (std::size_t index = 0; index < expected.logits.size(); ++index) {
    EXPECT_FLOAT_EQ(repeat.data_as<float>()[index], trace.logits.data_as<float>()[index]);
  }
  EXPECT_EQ(model.config().head_dimension(), 2U);
  EXPECT_EQ(model.weights().layers().size(), 2U);
}

TEST(CpuLlamaModelTest, RejectsEmptyOverlengthAndOutOfVocabularyInputs) {
  const LlamaConfig config = tiny_config();
  const RawWeights raw = make_raw_weights(config);
  CpuLlamaModel model(config, materialize_weights(config, raw));

  EXPECT_THROW(static_cast<void>(model.forward(std::span<const std::uint32_t>{})), Error);
  const std::vector<std::uint32_t> overlength(config.max_sequence_length + 1, 0);
  EXPECT_THROW(static_cast<void>(model.forward(overlength)), Error);
  EXPECT_THROW(static_cast<void>(model.forward(std::vector<std::uint32_t>{0, 6})), Error);

  const Tensor boundary = model.forward(std::vector<std::uint32_t>(config.max_sequence_length, 5));
  EXPECT_EQ(boundary.shape(), (Shape{8, 6}));
}

static_assert(!std::is_copy_constructible_v<LlamaWeights>);
static_assert(!std::is_copy_constructible_v<CpuLlamaModel>);

} // namespace
} // namespace helix::model
