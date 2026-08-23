#include "helix/core/error.hpp"
#include "helix/model/generation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace helix::model {
namespace {

Dimension dimension(std::size_t value) {
  return static_cast<Dimension>(value);
}

kv_cache::ContiguousKvCacheConfig cache_config(const CpuLlamaModel& model) {
  const LlamaConfig& config = model.config();
  return kv_cache::ContiguousKvCacheConfig{config.layer_count, config.attention_head_count,
                                           config.head_dimension()};
}

void pack_prefill_token(const CpuLlamaPrefillResult& prefill, std::size_t token,
                        std::size_t token_count, std::size_t heads, std::size_t head_dimension,
                        Tensor& token_keys, Tensor& token_values) {
  const std::size_t layer_elements = heads * head_dimension;
  float* keys_destination = token_keys.data_as<float>();
  float* values_destination = token_values.data_as<float>();
  for (std::size_t layer = 0; layer < prefill.layer_keys.size(); ++layer) {
    const float* keys_source = prefill.layer_keys[layer].data_as<float>();
    const float* values_source = prefill.layer_values[layer].data_as<float>();
    for (std::size_t head = 0; head < heads; ++head) {
      const std::size_t source_offset = (head * token_count + token) * head_dimension;
      const std::size_t destination_offset = layer * layer_elements + head * head_dimension;
      std::copy_n(keys_source + source_offset, head_dimension,
                  keys_destination + destination_offset);
      std::copy_n(values_source + source_offset, head_dimension,
                  values_destination + destination_offset);
    }
  }
}

void validate_options(const GenerationOptions& options, const LlamaConfig& config) {
  if (options.eos_token.has_value()) {
    HELIX_CHECK(*options.eos_token < config.vocabulary_size, ErrorCode::kOutOfRange,
                "model::CpuLlamaGenerator::generate: EOS token is outside the vocabulary");
  }
}

} // namespace

std::string_view generation_stop_reason_name(GenerationStopReason reason) {
  switch (reason) {
  case GenerationStopReason::kMaxNewTokens:
    return "max_new_tokens";
  case GenerationStopReason::kEosToken:
    return "eos_token";
  case GenerationStopReason::kContextLimit:
    return "context_limit";
  }
  throw_error(ErrorCode::kInvalidArgument, "model::generation: unknown stop reason");
}

std::uint32_t select_greedy_token(const Tensor& logits) {
  HELIX_CHECK(logits.device().is_cpu(), ErrorCode::kInvalidArgument,
              "model::select_greedy_token: logits must be on CPU");
  HELIX_CHECK(logits.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              "model::select_greedy_token: logits must have dtype float32");
  HELIX_CHECK(logits.is_contiguous(), ErrorCode::kInvalidArgument,
              "model::select_greedy_token: logits must be contiguous");
  HELIX_CHECK(logits.rank() == 2 && logits.shape()[0] > 0 && logits.shape()[1] > 0,
              ErrorCode::kInvalidArgument,
              "model::select_greedy_token: logits must have nonempty [tokens, vocabulary] shape");
  const auto vocabulary_size = static_cast<std::uint64_t>(logits.shape()[1]);
  constexpr std::uint64_t kMaximumVocabulary =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  HELIX_CHECK(vocabulary_size <= kMaximumVocabulary, ErrorCode::kOutOfRange,
              "model::select_greedy_token: vocabulary exceeds TokenId range");

  const std::size_t rows = static_cast<std::size_t>(logits.shape()[0]);
  const std::size_t columns = static_cast<std::size_t>(logits.shape()[1]);
  const float* row = logits.data_as<float>() + (rows - 1) * columns;
  HELIX_CHECK(std::isfinite(row[0]), ErrorCode::kRuntime,
              "model::select_greedy_token: logits contain a non-finite value");
  std::size_t best_index = 0;
  float best_value = row[0];
  for (std::size_t column = 1; column < columns; ++column) {
    HELIX_CHECK(std::isfinite(row[column]), ErrorCode::kRuntime,
                "model::select_greedy_token: logits contain a non-finite value");
    if (row[column] > best_value) {
      best_value = row[column];
      best_index = column;
    }
  }
  return static_cast<std::uint32_t>(best_index);
}

CpuLlamaGenerator::CpuLlamaGenerator(CpuLlamaModel model)
    : model_(std::move(model)), cache_(cache_config(model_)),
      sequence_(cache_.create_sequence(model_.config().max_sequence_length)) {}

Tensor CpuLlamaGenerator::prefill(std::span<const std::uint32_t> prompt_tokens) {
  HELIX_CHECK(!has_prefilled_, ErrorCode::kInvalidArgument,
              "model::CpuLlamaGenerator::prefill: session was already prefilled");
  CpuLlamaPrefillResult result = model_.prefill_with_kv(prompt_tokens);
  const LlamaConfig& config = model_.config();
  HELIX_ASSERT(result.layer_keys.size() == config.layer_count &&
                   result.layer_values.size() == config.layer_count,
               "prefill KV artifact count must match model layers");

  Tensor token_keys =
      Tensor::empty({dimension(config.layer_count), dimension(config.attention_head_count),
                     dimension(config.head_dimension())});
  Tensor token_values = Tensor::empty(token_keys.shape());
  for (std::size_t token = 0; token < prompt_tokens.size(); ++token) {
    pack_prefill_token(result, token, prompt_tokens.size(), config.attention_head_count,
                       config.head_dimension(), token_keys, token_values);
    cache_.append(sequence_, token_keys, token_values);
  }
  has_prefilled_ = true;
  return std::move(result.logits);
}

Tensor CpuLlamaGenerator::decode(std::uint32_t token_id) {
  HELIX_CHECK(has_prefilled_, ErrorCode::kInvalidArgument,
              "model::CpuLlamaGenerator::decode: prefill must complete first");
  const std::size_t position = cache_.length(sequence_);
  HELIX_CHECK(position < model_.config().max_sequence_length, ErrorCode::kResourceExhausted,
              "model::CpuLlamaGenerator::decode: context capacity is exhausted");
  CpuLlamaDecodeResult result = model_.decode_one(token_id, position, cache_, sequence_);
  cache_.append(sequence_, result.token_keys, result.token_values);
  return std::move(result.logits);
}

GenerationResult CpuLlamaGenerator::generate(std::span<const std::uint32_t> prompt_tokens,
                                             GenerationOptions options) {
  validate_options(options, model_.config());
  Tensor logits = prefill(prompt_tokens);
  GenerationResult result;
  result.generated_tokens.reserve(
      std::min(options.max_new_tokens, model_.config().max_sequence_length));
  if (options.max_new_tokens == 0) {
    result.cached_token_count = cache_.length(sequence_);
    return result;
  }

  while (result.generated_tokens.size() < options.max_new_tokens) {
    const std::uint32_t token = select_greedy_token(logits);
    result.generated_tokens.push_back(token);
    if (options.eos_token.has_value() && token == *options.eos_token) {
      result.stop_reason = GenerationStopReason::kEosToken;
      break;
    }
    if (result.generated_tokens.size() == options.max_new_tokens) {
      break;
    }
    if (cache_.length(sequence_) == model_.config().max_sequence_length) {
      result.stop_reason = GenerationStopReason::kContextLimit;
      break;
    }
    logits = decode(token);
  }
  result.cached_token_count = cache_.length(sequence_);
  return result;
}

bool CpuLlamaGenerator::has_prefilled() const noexcept {
  return has_prefilled_;
}

std::size_t CpuLlamaGenerator::cached_token_count() const {
  return cache_.length(sequence_);
}

kv_cache::ContiguousKvCacheStats CpuLlamaGenerator::cache_stats() const {
  return cache_.stats();
}

const CpuLlamaModel& CpuLlamaGenerator::model() const noexcept {
  return model_;
}

} // namespace helix::model
