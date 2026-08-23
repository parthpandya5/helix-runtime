#include "helix/model/llama.hpp"

#include "helix/core/error.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace helix::model {
namespace {

Dimension model_dimension(std::size_t value, std::string_view name) {
  HELIX_CHECK(value <= static_cast<std::size_t>(std::numeric_limits<Dimension>::max()),
              ErrorCode::kInvalidArgument,
              "model::LlamaConfig: " + std::string(name) + " exceeds tensor dimension range");
  return static_cast<Dimension>(value);
}

void require_weight(const Tensor& tensor, const Shape& expected, std::string_view name) {
  HELIX_CHECK(tensor.device().is_cpu(), ErrorCode::kInvalidArgument,
              "model::LlamaWeights: " + std::string(name) + " must be on CPU");
  HELIX_CHECK(tensor.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              "model::LlamaWeights: " + std::string(name) + " must have dtype float32");
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              "model::LlamaWeights: " + std::string(name) + " must be contiguous");
  if (tensor.shape() != expected) {
    std::ostringstream message;
    message << "model::LlamaWeights: " << name << " shape " << shape_string(tensor.shape())
            << " does not match expected " << shape_string(expected);
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
}

} // namespace

void LlamaConfig::validate() const {
  HELIX_CHECK(vocabulary_size != 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: vocabulary size must be positive");
  HELIX_CHECK(hidden_size != 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: hidden size must be positive");
  HELIX_CHECK(intermediate_size != 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: intermediate size must be positive");
  HELIX_CHECK(layer_count != 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: layer count must be positive");
  HELIX_CHECK(attention_head_count != 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: attention head count must be positive");
  HELIX_CHECK(max_sequence_length != 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: maximum sequence length must be positive");
  HELIX_CHECK(hidden_size % attention_head_count == 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: hidden size must be divisible by attention head count");
  const std::size_t dimension = hidden_size / attention_head_count;
  HELIX_CHECK(dimension % 2 == 0, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: head dimension must be even for full RoPE");
  HELIX_CHECK(std::isfinite(rms_norm_epsilon) && rms_norm_epsilon > 0.0F,
              ErrorCode::kInvalidArgument,
              "model::LlamaConfig: RMSNorm epsilon must be finite and positive");
  HELIX_CHECK(std::isfinite(rope_theta) && rope_theta > 0.0F, ErrorCode::kInvalidArgument,
              "model::LlamaConfig: RoPE theta must be finite and positive");
  static_cast<void>(model_dimension(vocabulary_size, "vocabulary size"));
  static_cast<void>(model_dimension(hidden_size, "hidden size"));
  static_cast<void>(model_dimension(intermediate_size, "intermediate size"));
  static_cast<void>(model_dimension(max_sequence_length, "maximum sequence length"));
}

std::size_t LlamaConfig::head_dimension() const {
  validate();
  return hidden_size / attention_head_count;
}

LlamaLayerWeights::LlamaLayerWeights(Tensor attention_norm, Tensor query, Tensor key, Tensor value,
                                     Tensor attention_output, Tensor ffn_norm, Tensor gate,
                                     Tensor up, Tensor down)
    : attention_norm_weight(std::move(attention_norm)), query_weight(std::move(query)),
      key_weight(std::move(key)), value_weight(std::move(value)),
      attention_output_weight(std::move(attention_output)), ffn_norm_weight(std::move(ffn_norm)),
      gate_weight(std::move(gate)), up_weight(std::move(up)), down_weight(std::move(down)) {}

LlamaWeights::LlamaWeights(Tensor token_embedding, std::vector<LlamaLayerWeights> layers,
                           Tensor final_norm_weight, Tensor lm_head_weight)
    : token_embedding_(std::move(token_embedding)), layers_(std::move(layers)),
      final_norm_weight_(std::move(final_norm_weight)), lm_head_weight_(std::move(lm_head_weight)) {
}

void LlamaWeights::validate(const LlamaConfig& config) const {
  config.validate();
  HELIX_CHECK(layers_.size() == config.layer_count, ErrorCode::kInvalidArgument,
              "model::LlamaWeights: layer count does not match configuration");

  const Dimension vocabulary = model_dimension(config.vocabulary_size, "vocabulary size");
  const Dimension hidden = model_dimension(config.hidden_size, "hidden size");
  const Dimension intermediate = model_dimension(config.intermediate_size, "intermediate size");
  require_weight(token_embedding_, Shape{vocabulary, hidden}, "token embedding");
  require_weight(final_norm_weight_, Shape{hidden}, "final norm weight");
  require_weight(lm_head_weight_, Shape{hidden, vocabulary}, "LM head weight");

  for (std::size_t index = 0; index < layers_.size(); ++index) {
    const LlamaLayerWeights& layer = layers_[index];
    const std::string prefix = "layer " + std::to_string(index) + ' ';
    require_weight(layer.attention_norm_weight, Shape{hidden}, prefix + "attention norm");
    require_weight(layer.query_weight, Shape{hidden, hidden}, prefix + "query weight");
    require_weight(layer.key_weight, Shape{hidden, hidden}, prefix + "key weight");
    require_weight(layer.value_weight, Shape{hidden, hidden}, prefix + "value weight");
    require_weight(layer.attention_output_weight, Shape{hidden, hidden},
                   prefix + "attention output weight");
    require_weight(layer.ffn_norm_weight, Shape{hidden}, prefix + "FFN norm");
    require_weight(layer.gate_weight, Shape{hidden, intermediate}, prefix + "gate weight");
    require_weight(layer.up_weight, Shape{hidden, intermediate}, prefix + "up weight");
    require_weight(layer.down_weight, Shape{intermediate, hidden}, prefix + "down weight");
  }
}

const Tensor& LlamaWeights::token_embedding() const noexcept {
  return token_embedding_;
}

std::span<const LlamaLayerWeights> LlamaWeights::layers() const noexcept {
  return layers_;
}

const Tensor& LlamaWeights::final_norm_weight() const noexcept {
  return final_norm_weight_;
}

const Tensor& LlamaWeights::lm_head_weight() const noexcept {
  return lm_head_weight_;
}

CpuLlamaModel::CpuLlamaModel(LlamaConfig config, LlamaWeights weights)
    : config_(std::move(config)), weights_(std::move(weights)) {
  config_.validate();
  weights_.validate(config_);
}

const LlamaConfig& CpuLlamaModel::config() const noexcept {
  return config_;
}

const LlamaWeights& CpuLlamaModel::weights() const noexcept {
  return weights_;
}

} // namespace helix::model
