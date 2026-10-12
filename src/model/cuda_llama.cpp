#include "helix/model/cuda_llama.hpp"

#include "helix/attention/cuda.hpp"
#include "helix/attention/cuda_paged.hpp"
#include "helix/core/error.hpp"
#include "helix/kernels/cuda/elementwise.hpp"
#include "helix/kernels/cuda/embedding.hpp"
#include "helix/kernels/cuda/transformer.hpp"
#include "helix/runtime/cuda/memory.hpp"
#include "llama_detail.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace helix::model {
namespace {

Dimension dimension(std::size_t value) {
  return static_cast<Dimension>(value);
}

Device model_device(int device_index) {
  return Device::cuda(device_index);
}

LlamaWeights transfer_weights(const LlamaConfig& config, const LlamaWeights& cpu_weights,
                              int device_index) {
  config.validate();
  cpu_weights.validate(config, Device::cpu());
  const Device target = model_device(device_index);
  LlamaWeights transferred = cpu_weights.to(target);
  transferred.validate(config, target);
  return transferred;
}

void validate_stream(const helix::cuda::Stream& stream, int device_index,
                     std::string_view operation) {
  HELIX_CHECK(stream.native_handle() != nullptr, ErrorCode::kInvalidArgument,
              std::string(operation) + ": stream must not be moved-from");
  HELIX_CHECK(stream.device_index() == device_index, ErrorCode::kInvalidArgument,
              std::string(operation) + ": stream must belong to the model device");
}

std::string operation_error(std::string_view operation, std::string_view requirement) {
  return std::string(operation) + ": " + std::string(requirement);
}

void validate_host_token_ids(std::span<const std::uint32_t> token_ids, const LlamaConfig& config,
                             std::string_view operation) {
  HELIX_CHECK(!token_ids.empty(), ErrorCode::kInvalidArgument,
              operation_error(operation, "token sequence must not be empty"));
  HELIX_CHECK(token_ids.size() <= config.max_sequence_length, ErrorCode::kOutOfRange,
              operation_error(operation, "token sequence exceeds configured maximum"));
  for (const std::uint32_t token_id : token_ids) {
    HELIX_CHECK(
        static_cast<std::uint64_t>(token_id) < static_cast<std::uint64_t>(config.vocabulary_size),
        ErrorCode::kOutOfRange, operation_error(operation, "token ID is outside the vocabulary"));
  }
}

void validate_device_inputs(const Tensor& token_ids, const Tensor& cosine, const Tensor& sine,
                            const LlamaConfig& config, int device_index,
                            const helix::cuda::Stream& stream, std::string_view operation) {
  validate_stream(stream, device_index, operation);
  const Device expected_device = model_device(device_index);
  HELIX_CHECK(token_ids.device() == expected_device, ErrorCode::kInvalidArgument,
              operation_error(operation, "token IDs must be on the model device"));
  HELIX_CHECK(token_ids.dtype() == DType::kUInt32, ErrorCode::kInvalidArgument,
              operation_error(operation, "token IDs must have dtype uint32"));
  HELIX_CHECK(token_ids.is_contiguous() && token_ids.rank() == 1, ErrorCode::kInvalidArgument,
              operation_error(operation, "token IDs must be contiguous and rank one"));
  HELIX_CHECK(token_ids.numel() != 0, ErrorCode::kInvalidArgument,
              operation_error(operation, "token sequence must not be empty"));
  HELIX_CHECK(token_ids.numel() <= config.max_sequence_length, ErrorCode::kOutOfRange,
              operation_error(operation, "token sequence exceeds configured maximum"));

  const Shape expected_cache_shape{token_ids.shape()[0], dimension(config.head_dimension() / 2)};
  const std::array<const Tensor*, 2> caches{&cosine, &sine};
  for (const Tensor* cache : caches) {
    HELIX_CHECK(cache->device() == expected_device, ErrorCode::kInvalidArgument,
                operation_error(operation, "RoPE cache must be on the model device"));
    HELIX_CHECK(cache->dtype() == DType::kFloat32 && cache->is_contiguous(),
                ErrorCode::kInvalidArgument,
                operation_error(operation, "RoPE cache must be contiguous float32"));
    HELIX_CHECK(cache->shape() == expected_cache_shape, ErrorCode::kInvalidArgument,
                operation_error(operation, "RoPE cache shape is invalid"));
  }
}

struct PagedHistoryDimensions final {
  std::size_t physical_blocks = 0;
  std::size_t block_tokens = 0;
};

void require_cuda_float_history(const Tensor& tensor, std::string_view name, Device device,
                                std::string_view operation) {
  HELIX_CHECK(tensor.device() == device, ErrorCode::kInvalidArgument,
              operation_error(operation, std::string(name) + " must be on the model device"));
  HELIX_CHECK(
      tensor.dtype() == DType::kFloat32 && tensor.is_contiguous(), ErrorCode::kInvalidArgument,
      operation_error(operation, std::string(name) + " must be contiguous with dtype float32"));
  HELIX_CHECK(tensor.rank() == 4, ErrorCode::kInvalidArgument,
              operation_error(operation, std::string(name) + " must have rank four"));
}

PagedHistoryDimensions validate_paged_history(std::span<const Tensor> layer_key_blocks,
                                              std::span<const Tensor> layer_value_blocks,
                                              const attention::cuda::PagedBlockTable& block_table,
                                              std::size_t history_tokens, const LlamaConfig& config,
                                              int device_index, std::string_view operation) {
  HELIX_CHECK(history_tokens < config.max_sequence_length, ErrorCode::kOutOfRange,
              operation_error(operation, "history length leaves no position to decode"));
  HELIX_CHECK(layer_key_blocks.size() == config.layer_count &&
                  layer_value_blocks.size() == config.layer_count,
              ErrorCode::kInvalidArgument,
              operation_error(operation, "history must provide one key/value tensor per layer"));
  HELIX_CHECK(block_table.valid(), ErrorCode::kInvalidArgument,
              operation_error(operation, "block table must be valid"));
  HELIX_CHECK(block_table.device_index() == device_index, ErrorCode::kInvalidArgument,
              operation_error(operation, "block table must be on the model device"));

  const Device device = model_device(device_index);
  const Dimension expected_heads = dimension(config.attention_head_count);
  const Dimension expected_width = dimension(config.head_dimension());
  PagedHistoryDimensions dimensions;
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const Tensor& keys = layer_key_blocks[layer];
    const Tensor& values = layer_value_blocks[layer];
    const std::string key_name = "layer " + std::to_string(layer) + " key blocks";
    const std::string value_name = "layer " + std::to_string(layer) + " value blocks";
    require_cuda_float_history(keys, key_name, device, operation);
    require_cuda_float_history(values, value_name, device, operation);
    HELIX_CHECK(keys.shape()[0] > 0 && keys.shape()[1] == expected_heads && keys.shape()[2] > 0 &&
                    keys.shape()[3] == expected_width,
                ErrorCode::kInvalidArgument,
                operation_error(operation, key_name + " have invalid page/head/token geometry"));
    HELIX_CHECK(values.shape() == keys.shape(), ErrorCode::kInvalidArgument,
                operation_error(operation, value_name + " must match key-block geometry"));

    const auto physical_blocks = static_cast<std::size_t>(keys.shape()[0]);
    const auto block_tokens = static_cast<std::size_t>(keys.shape()[2]);
    if (layer == 0) {
      dimensions = PagedHistoryDimensions{physical_blocks, block_tokens};
    } else {
      HELIX_CHECK(
          physical_blocks == dimensions.physical_blocks && block_tokens == dimensions.block_tokens,
          ErrorCode::kInvalidArgument,
          operation_error(operation, "all layers must share physical-page and token geometry"));
    }
  }

  HELIX_CHECK(block_table.physical_block_count() == dimensions.physical_blocks,
              ErrorCode::kInvalidArgument,
              operation_error(operation, "block-table capacity must match packed physical pages"));
  const std::size_t required_logical_blocks =
      history_tokens == 0 ? 0 : 1 + (history_tokens - 1) / dimensions.block_tokens;
  HELIX_CHECK(block_table.logical_block_count() == required_logical_blocks,
              ErrorCode::kInvalidArgument,
              operation_error(operation, "block table must exactly cover history length"));
  return dimensions;
}

void synchronize_before_unwind(const helix::cuda::Stream& stream) noexcept {
  try {
    stream.synchronize();
  } catch (...) {
    // Preserve the original model-execution failure while preventing a second
    // cleanup exception from escaping.
  }
}

} // namespace

CudaLlamaModel::CudaLlamaModel(LlamaConfig config, const LlamaWeights& cpu_weights,
                               int device_index)
    : config_(std::move(config)), device_index_(device_index),
      weights_(transfer_weights(config_, cpu_weights, device_index_)) {}

const LlamaConfig& CudaLlamaModel::config() const noexcept {
  return config_;
}

const LlamaWeights& CudaLlamaModel::weights() const noexcept {
  return weights_;
}

int CudaLlamaModel::device_index() const noexcept {
  return device_index_;
}

Tensor CudaLlamaModel::forward(std::span<const std::uint32_t> token_ids,
                               const helix::cuda::Stream& stream) const {
  constexpr std::string_view kOperation = "model::CudaLlamaModel::forward";
  validate_stream(stream, device_index_, kOperation);
  validate_host_token_ids(token_ids, config_, kOperation);
  const std::size_t tokens = token_ids.size();
  const detail::RopeCache host_rope =
      detail::make_rope_cache(tokens, config_.head_dimension(), config_.rope_theta);
  const Device device = model_device(device_index_);
  const Tensor device_ids =
      Tensor::from_values({dimension(tokens)}, token_ids, model_device(device_index_));
  const Tensor cosine = host_rope.cosine.to(device);
  const Tensor sine = host_rope.sine.to(device);
  return forward_device(device_ids, cosine, sine, stream);
}

Tensor CudaLlamaModel::forward_device(const Tensor& token_ids, const Tensor& cosine,
                                      const Tensor& sine, const helix::cuda::Stream& stream) const {
  constexpr std::string_view kOperation = "model::CudaLlamaModel::forward_device";
  validate_device_inputs(token_ids, cosine, sine, config_, device_index_, stream, kOperation);
  const Device device = model_device(device_index_);
  const std::size_t tokens = token_ids.numel();
  const std::size_t heads = config_.attention_head_count;
  const std::size_t head_dimension = config_.head_dimension();

  try {
    Tensor hidden =
        helix::cuda::kernels::embedding_lookup(token_ids, weights_.token_embedding(), stream);
    for (const LlamaLayerWeights& layer : weights_.layers()) {
      const Tensor attention_input = helix::cuda::kernels::rms_norm(
          hidden, layer.attention_norm_weight, config_.rms_norm_epsilon, stream);
      const Tensor query_projection =
          helix::cuda::kernels::linear(attention_input, layer.query_weight, stream);
      const Tensor key_projection =
          helix::cuda::kernels::linear(attention_input, layer.key_weight, stream);
      const Tensor value_projection =
          helix::cuda::kernels::linear(attention_input, layer.value_weight, stream);

      const Shape token_head_shape{dimension(tokens), dimension(heads), dimension(head_dimension)};
      const Tensor query_token_heads = query_projection.reshape(token_head_shape);
      const Tensor key_token_heads = key_projection.reshape(token_head_shape);
      const Tensor value_token_heads = value_projection.reshape(token_head_shape);
      const Tensor rotated_query =
          helix::cuda::kernels::rope(query_token_heads, cosine, sine, stream);
      const Tensor rotated_key = helix::cuda::kernels::rope(key_token_heads, cosine, sine, stream);

      const Tensor query = helix::cuda::kernels::token_heads_to_head_major(rotated_query, stream);
      const Tensor key = helix::cuda::kernels::token_heads_to_head_major(rotated_key, stream);
      const Tensor value =
          helix::cuda::kernels::token_heads_to_head_major(value_token_heads, stream);
      Tensor attention_context =
          Tensor::empty({dimension(heads), dimension(tokens), dimension(head_dimension)},
                        DType::kFloat32, device);
      Tensor score_workspace = Tensor::empty(
          {dimension(heads), dimension(tokens), dimension(tokens)}, DType::kFloat32, device);
      attention::cuda::scaled_dot_product_attention_out(query, key, value, 0, attention_context,
                                                        score_workspace, stream);
      const Tensor attention_token_heads =
          helix::cuda::kernels::head_major_to_token_heads(attention_context, stream);
      const Tensor merged_attention =
          attention_token_heads.reshape({dimension(tokens), dimension(heads * head_dimension)});
      const Tensor attention_output =
          helix::cuda::kernels::linear(merged_attention, layer.attention_output_weight, stream);
      const Tensor attention_hidden = helix::cuda::kernels::add(hidden, attention_output, stream);

      const Tensor ffn_input = helix::cuda::kernels::rms_norm(
          attention_hidden, layer.ffn_norm_weight, config_.rms_norm_epsilon, stream);
      const Tensor gate = helix::cuda::kernels::linear(ffn_input, layer.gate_weight, stream);
      const Tensor up = helix::cuda::kernels::linear(ffn_input, layer.up_weight, stream);
      const Tensor activated = helix::cuda::kernels::swiglu(gate, up, stream);
      const Tensor ffn_output = helix::cuda::kernels::linear(activated, layer.down_weight, stream);
      Tensor next_hidden = helix::cuda::kernels::add(attention_hidden, ffn_output, stream);

      // This synchronization is the baseline's explicit lifetime boundary:
      // every layer-local input, output, and attention scratch allocation may
      // be reclaimed safely after it completes.
      stream.synchronize();
      hidden = std::move(next_hidden);
    }

    const Tensor normalized = helix::cuda::kernels::rms_norm(hidden, weights_.final_norm_weight(),
                                                             config_.rms_norm_epsilon, stream);
    Tensor logits = helix::cuda::kernels::linear(normalized, weights_.lm_head_weight(), stream);
    stream.synchronize();
    return logits;
  } catch (...) {
    // If a later launch or allocation fails after earlier asynchronous work was
    // accepted, make a best effort to drain the stream before local storage is
    // destroyed. Preserve the original exception as the primary diagnosis.
    synchronize_before_unwind(stream);
    throw;
  }
}

CudaLlamaDecodeResult CudaLlamaModel::decode_one_paged(
    std::uint32_t token_id, std::size_t position, std::span<const Tensor> layer_key_blocks,
    std::span<const Tensor> layer_value_blocks, const attention::cuda::PagedBlockTable& block_table,
    const helix::cuda::Stream& stream) const {
  constexpr std::string_view kOperation = "model::CudaLlamaModel::decode_one_paged";
  validate_stream(stream, device_index_, kOperation);
  const std::array<std::uint32_t, 1> token_ids{token_id};
  validate_host_token_ids(token_ids, config_, kOperation);
  static_cast<void>(validate_paged_history(layer_key_blocks, layer_value_blocks, block_table,
                                           position, config_, device_index_, kOperation));

  const detail::RopeCache host_rope =
      detail::make_rope_cache(1, config_.head_dimension(), config_.rope_theta, position);
  const Device device = model_device(device_index_);
  const Tensor device_id = Tensor::from_values({1}, token_ids, device);
  const Tensor cosine = host_rope.cosine.to(device);
  const Tensor sine = host_rope.sine.to(device);
  return decode_one_paged_device(device_id, cosine, sine, layer_key_blocks, layer_value_blocks,
                                 block_table, position, stream);
}

CudaLlamaDecodeResult CudaLlamaModel::decode_one_paged_device(
    const Tensor& token_id, const Tensor& cosine, const Tensor& sine,
    std::span<const Tensor> layer_key_blocks, std::span<const Tensor> layer_value_blocks,
    const attention::cuda::PagedBlockTable& block_table, std::size_t history_tokens,
    const helix::cuda::Stream& stream) const {
  constexpr std::string_view kOperation = "model::CudaLlamaModel::decode_one_paged_device";
  validate_device_inputs(token_id, cosine, sine, config_, device_index_, stream, kOperation);
  HELIX_CHECK(token_id.numel() == 1, ErrorCode::kInvalidArgument,
              operation_error(kOperation, "decode requires exactly one token ID"));
  static_cast<void>(validate_paged_history(layer_key_blocks, layer_value_blocks, block_table,
                                           history_tokens, config_, device_index_, kOperation));

  const Device device = model_device(device_index_);
  const std::size_t layers = config_.layer_count;
  const std::size_t heads = config_.attention_head_count;
  const std::size_t head_dimension = config_.head_dimension();
  const std::size_t layer_elements = heads * head_dimension;
  const std::size_t layer_bytes =
      checked_nbytes(Shape{dimension(heads), dimension(head_dimension)}, DType::kFloat32);

  try {
    Tensor hidden =
        helix::cuda::kernels::embedding_lookup(token_id, weights_.token_embedding(), stream);
    Tensor token_keys = Tensor::empty(
        {dimension(layers), dimension(heads), dimension(head_dimension)}, DType::kFloat32, device);
    Tensor token_values = Tensor::empty(token_keys.shape(), DType::kFloat32, device);
    for (std::size_t layer_index = 0; layer_index < layers; ++layer_index) {
      const LlamaLayerWeights& layer = weights_.layers()[layer_index];
      const Tensor attention_input = helix::cuda::kernels::rms_norm(
          hidden, layer.attention_norm_weight, config_.rms_norm_epsilon, stream);
      const Tensor query_projection =
          helix::cuda::kernels::linear(attention_input, layer.query_weight, stream);
      const Tensor key_projection =
          helix::cuda::kernels::linear(attention_input, layer.key_weight, stream);
      const Tensor value_projection =
          helix::cuda::kernels::linear(attention_input, layer.value_weight, stream);

      const Shape token_head_shape{1, dimension(heads), dimension(head_dimension)};
      const Tensor query_token_heads = query_projection.reshape(token_head_shape);
      const Tensor key_token_heads = key_projection.reshape(token_head_shape);
      const Tensor value_token_heads = value_projection.reshape(token_head_shape);
      const Tensor rotated_query =
          helix::cuda::kernels::rope(query_token_heads, cosine, sine, stream);
      const Tensor rotated_key = helix::cuda::kernels::rope(key_token_heads, cosine, sine, stream);

      const Tensor query = helix::cuda::kernels::token_heads_to_head_major(rotated_query, stream);
      const Tensor current_key =
          helix::cuda::kernels::token_heads_to_head_major(rotated_key, stream);
      const Tensor current_value =
          helix::cuda::kernels::token_heads_to_head_major(value_token_heads, stream);
      Tensor attention_context =
          Tensor::empty({dimension(heads), 1, dimension(head_dimension)}, DType::kFloat32, device);
      Tensor score_workspace = Tensor::empty({dimension(heads), 1, dimension(history_tokens + 1)},
                                             DType::kFloat32, device);
      attention::cuda::paged_scaled_dot_product_attention_with_tail_out(
          query, layer_key_blocks[layer_index], layer_value_blocks[layer_index], block_table,
          history_tokens, current_key, current_value, attention_context, score_workspace, stream);

      auto* key_destination =
          static_cast<std::byte*>(token_keys.data()) + layer_index * layer_bytes;
      auto* value_destination =
          static_cast<std::byte*>(token_values.data()) + layer_index * layer_bytes;
      helix::cuda::copy_device_to_device_async(key_destination, current_key.data(), layer_bytes,
                                               stream);
      helix::cuda::copy_device_to_device_async(value_destination, current_value.data(), layer_bytes,
                                               stream);

      const Tensor attention_token_heads =
          helix::cuda::kernels::head_major_to_token_heads(attention_context, stream);
      const Tensor merged_attention = attention_token_heads.reshape({1, dimension(layer_elements)});
      const Tensor attention_output =
          helix::cuda::kernels::linear(merged_attention, layer.attention_output_weight, stream);
      const Tensor attention_hidden = helix::cuda::kernels::add(hidden, attention_output, stream);

      const Tensor ffn_input = helix::cuda::kernels::rms_norm(
          attention_hidden, layer.ffn_norm_weight, config_.rms_norm_epsilon, stream);
      const Tensor gate = helix::cuda::kernels::linear(ffn_input, layer.gate_weight, stream);
      const Tensor up = helix::cuda::kernels::linear(ffn_input, layer.up_weight, stream);
      const Tensor activated = helix::cuda::kernels::swiglu(gate, up, stream);
      const Tensor ffn_output = helix::cuda::kernels::linear(activated, layer.down_weight, stream);
      Tensor next_hidden = helix::cuda::kernels::add(attention_hidden, ffn_output, stream);

      // Layer-local tensors and score scratch are direct allocations. This
      // barrier is the baseline lifetime boundary for their asynchronous use.
      stream.synchronize();
      hidden = std::move(next_hidden);
    }

    const Tensor normalized = helix::cuda::kernels::rms_norm(hidden, weights_.final_norm_weight(),
                                                             config_.rms_norm_epsilon, stream);
    Tensor logits = helix::cuda::kernels::linear(normalized, weights_.lm_head_weight(), stream);
    stream.synchronize();
    return CudaLlamaDecodeResult{std::move(token_keys), std::move(token_values), std::move(logits)};
  } catch (...) {
    synchronize_before_unwind(stream);
    throw;
  }
}

} // namespace helix::model
