#include "helix/model/cuda_generation.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

namespace helix::model {
namespace {

kv_cache::PagedKvCacheConfig cache_config(const CudaLlamaModel& model, std::size_t block_tokens) {
  HELIX_CHECK(block_tokens != 0, ErrorCode::kInvalidArgument,
              "model::CudaLlamaGenerator: KV block token count must be positive");
  const LlamaConfig& config = model.config();
  const std::size_t physical_blocks = 1 + (config.max_sequence_length - 1) / block_tokens;
  return kv_cache::PagedKvCacheConfig{config.layer_count, config.attention_head_count,
                                      config.head_dimension(), block_tokens, physical_blocks};
}

helix::cuda::Stream make_stream(int device_index) {
  helix::cuda::DeviceGuard guard(device_index);
  return helix::cuda::Stream{};
}

void validate_prompt(std::span<const std::uint32_t> prompt_tokens, const LlamaConfig& config) {
  HELIX_CHECK(!prompt_tokens.empty(), ErrorCode::kInvalidArgument,
              "model::CudaLlamaGenerator::prefill: prompt must not be empty");
  HELIX_CHECK(prompt_tokens.size() <= config.max_sequence_length, ErrorCode::kOutOfRange,
              "model::CudaLlamaGenerator::prefill: prompt exceeds context capacity");
  for (const std::uint32_t token : prompt_tokens) {
    HELIX_CHECK(token < config.vocabulary_size, ErrorCode::kOutOfRange,
                "model::CudaLlamaGenerator::prefill: prompt token is outside the vocabulary");
  }
}

void validate_options(const GenerationOptions& options, const LlamaConfig& config) {
  if (options.eos_token.has_value()) {
    HELIX_CHECK(*options.eos_token < config.vocabulary_size, ErrorCode::kOutOfRange,
                "model::CudaLlamaGenerator::generate: EOS token is outside the vocabulary");
  }
}

void release_if_present_noexcept(kv_cache::CudaPagedKvCache& cache,
                                 const kv_cache::CudaPagedSequenceHandle& sequence) noexcept {
  try {
    if (cache.contains(sequence)) {
      cache.release_sequence(sequence);
    }
  } catch (...) {
    // Preserve the model/cache exception that caused staged prefill to unwind.
    // A valid private staging handle should always be releasable.
  }
}

} // namespace

CudaLlamaGenerator::CudaLlamaGenerator(CudaLlamaModel model, std::size_t block_tokens)
    : model_(std::move(model)), cache_(cache_config(model_, block_tokens), model_.device_index()),
      stream_(make_stream(model_.device_index())), sequence_(cache_.create_sequence()),
      block_tokens_(block_tokens) {}

Tensor CudaLlamaGenerator::prefill(std::span<const std::uint32_t> prompt_tokens) {
  HELIX_CHECK(!has_prefilled_, ErrorCode::kInvalidArgument,
              "model::CudaLlamaGenerator::prefill: session was already prefilled");
  validate_prompt(prompt_tokens, model_.config());

  // Build the prompt in a private table. The session's original empty table is
  // retained until every model step and page commit succeeds.
  const kv_cache::CudaPagedSequenceHandle staging = cache_.create_sequence();
  std::optional<Tensor> logits;
  try {
    for (std::size_t position = 0; position < prompt_tokens.size(); ++position) {
      CudaLlamaDecodeResult decoded = [&] {
        auto view = cache_.read_sequence(staging);
        HELIX_ASSERT(view.length() == position,
                     "CUDA generation staging history disagrees with prompt position");
        try {
          return model_.decode_one_paged(prompt_tokens[position], position, view.layer_key_blocks(),
                                         view.layer_value_blocks(), view.block_table(), stream_);
        } catch (...) {
          view.poison_after_consumer_failure();
          throw;
        }
      }();
      cache_.append(staging, decoded.token_keys, decoded.token_values, stream_);
      logits.emplace(std::move(decoded.logits));
    }
  } catch (...) {
    release_if_present_noexcept(cache_, staging);
    throw;
  }

  HELIX_ASSERT(logits.has_value(), "validated CUDA generation prompt produced no logits");
  try {
    cache_.release_sequence(sequence_);
  } catch (...) {
    release_if_present_noexcept(cache_, staging);
    throw;
  }
  sequence_ = staging;
  has_prefilled_ = true;
  return std::move(*logits);
}

Tensor CudaLlamaGenerator::decode(std::uint32_t token_id) {
  HELIX_CHECK(has_prefilled_, ErrorCode::kInvalidArgument,
              "model::CudaLlamaGenerator::decode: prefill must complete first");
  HELIX_CHECK(token_id < model_.config().vocabulary_size, ErrorCode::kOutOfRange,
              "model::CudaLlamaGenerator::decode: token is outside the vocabulary");
  const std::size_t position = cache_.length(sequence_);
  HELIX_CHECK(position < model_.config().max_sequence_length, ErrorCode::kResourceExhausted,
              "model::CudaLlamaGenerator::decode: context capacity is exhausted");

  CudaLlamaDecodeResult decoded = [&] {
    auto view = cache_.read_sequence(sequence_);
    HELIX_ASSERT(view.length() == position,
                 "CUDA generation history changed while its decode lease was held");
    try {
      return model_.decode_one_paged(token_id, position, view.layer_key_blocks(),
                                     view.layer_value_blocks(), view.block_table(), stream_);
    } catch (...) {
      view.poison_after_consumer_failure();
      throw;
    }
  }();
  cache_.append(sequence_, decoded.token_keys, decoded.token_values, stream_);
  return std::move(decoded.logits);
}

GenerationResult CudaLlamaGenerator::generate(std::span<const std::uint32_t> prompt_tokens,
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
    const Tensor host_logits = logits.to(Device::cpu());
    const std::uint32_t token = select_greedy_token(host_logits);
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

bool CudaLlamaGenerator::has_prefilled() const noexcept {
  return has_prefilled_;
}

std::size_t CudaLlamaGenerator::cached_token_count() const {
  return cache_.length(sequence_);
}

kv_cache::PagedKvCacheStats CudaLlamaGenerator::cache_stats() const {
  return cache_.stats();
}

std::size_t CudaLlamaGenerator::block_tokens() const noexcept {
  return block_tokens_;
}

const CudaLlamaModel& CudaLlamaGenerator::model() const noexcept {
  return model_;
}

const helix::cuda::Stream& CudaLlamaGenerator::stream() const noexcept {
  return stream_;
}

} // namespace helix::model
