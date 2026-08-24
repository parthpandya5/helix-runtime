#include "helix/model/cuda_llama.hpp"

#include "helix/attention/cuda.hpp"
#include "helix/core/error.hpp"
#include "helix/kernels/cuda/elementwise.hpp"
#include "helix/kernels/cuda/embedding.hpp"
#include "helix/kernels/cuda/transformer.hpp"
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

void validate_host_token_ids(std::span<const std::uint32_t> token_ids, const LlamaConfig& config) {
  HELIX_CHECK(!token_ids.empty(), ErrorCode::kInvalidArgument,
              "model::CudaLlamaModel::forward: token sequence must not be empty");
  HELIX_CHECK(token_ids.size() <= config.max_sequence_length, ErrorCode::kOutOfRange,
              "model::CudaLlamaModel::forward: token sequence exceeds configured maximum");
  for (const std::uint32_t token_id : token_ids) {
    HELIX_CHECK(static_cast<std::uint64_t>(token_id) <
                    static_cast<std::uint64_t>(config.vocabulary_size),
                ErrorCode::kOutOfRange,
                "model::CudaLlamaModel::forward: token ID is outside the vocabulary");
  }
}

void validate_device_inputs(const Tensor& token_ids, const Tensor& cosine, const Tensor& sine,
                            const LlamaConfig& config, int device_index,
                            const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation = "model::CudaLlamaModel::forward_device";
  validate_stream(stream, device_index, kOperation);
  const Device expected_device = model_device(device_index);
  HELIX_CHECK(token_ids.device() == expected_device, ErrorCode::kInvalidArgument,
              "model::CudaLlamaModel::forward_device: token IDs must be on the model device");
  HELIX_CHECK(token_ids.dtype() == DType::kUInt32, ErrorCode::kInvalidArgument,
              "model::CudaLlamaModel::forward_device: token IDs must have dtype uint32");
  HELIX_CHECK(token_ids.is_contiguous() && token_ids.rank() == 1, ErrorCode::kInvalidArgument,
              "model::CudaLlamaModel::forward_device: token IDs must be contiguous and rank one");
  HELIX_CHECK(token_ids.numel() != 0, ErrorCode::kInvalidArgument,
              "model::CudaLlamaModel::forward_device: token sequence must not be empty");
  HELIX_CHECK(token_ids.numel() <= config.max_sequence_length, ErrorCode::kOutOfRange,
              "model::CudaLlamaModel::forward_device: token sequence exceeds configured maximum");

  const Shape expected_cache_shape{token_ids.shape()[0], dimension(config.head_dimension() / 2)};
  const std::array<const Tensor*, 2> caches{&cosine, &sine};
  for (const Tensor* cache : caches) {
    HELIX_CHECK(cache->device() == expected_device, ErrorCode::kInvalidArgument,
                "model::CudaLlamaModel::forward_device: RoPE cache must be on the model device");
    HELIX_CHECK(cache->dtype() == DType::kFloat32 && cache->is_contiguous(),
                ErrorCode::kInvalidArgument,
                "model::CudaLlamaModel::forward_device: RoPE cache must be contiguous float32");
    HELIX_CHECK(cache->shape() == expected_cache_shape, ErrorCode::kInvalidArgument,
                "model::CudaLlamaModel::forward_device: RoPE cache shape is invalid");
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
  validate_stream(stream, device_index_, "model::CudaLlamaModel::forward");
  validate_host_token_ids(token_ids, config_);
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
  validate_device_inputs(token_ids, cosine, sine, config_, device_index_, stream);
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
    try {
      stream.synchronize();
    } catch (...) {
    }
    throw;
  }
}

} // namespace helix::model
