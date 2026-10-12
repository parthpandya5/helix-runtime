#include "helix/attention/cuda_paged.hpp"
#include "helix/core/error.hpp"
#include "helix/kernels/cuda/softmax.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>
#include <sstream>
#include <string>
#include <string_view>

namespace helix::attention::cuda {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65'535;

struct PagedAttentionDimensions {
  std::size_t physical_blocks;
  std::size_t logical_blocks;
  std::size_t heads;
  std::size_t queries;
  std::size_t keys;
  std::size_t block_tokens;
  std::size_t key_dimension;
  std::size_t value_dimension;
  std::size_t history_keys;
  std::size_t tail_keys;
};

__global__ void paged_attention_scores_kernel(const float* query, const float* key_blocks,
                                              const std::uint32_t* block_table, float* scores,
                                              std::size_t tasks, std::size_t physical_blocks,
                                              std::size_t heads, std::size_t queries,
                                              std::size_t keys, std::size_t block_tokens,
                                              std::size_t head_dimension,
                                              std::size_t query_position_offset, float scale) {
  std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; task < tasks; task += stride) {
    const std::size_t key_index = task % keys;
    const std::size_t query_and_head = task / keys;
    const std::size_t query_index = query_and_head % queries;
    const std::size_t head = query_and_head / queries;
    if (key_index > query_position_offset + query_index) {
      scores[task] = -CUDART_INF_F;
      continue;
    }

    const std::size_t logical_block = key_index / block_tokens;
    const std::size_t token_in_block = key_index % block_tokens;
    const std::size_t physical_block = static_cast<std::size_t>(block_table[logical_block]);
    if (physical_block >= physical_blocks) {
      scores[task] = CUDART_NAN_F;
      continue;
    }

    const std::size_t query_offset = (head * queries + query_index) * head_dimension;
    const std::size_t key_offset =
        (((physical_block * heads + head) * block_tokens + token_in_block) * head_dimension);
    float dot_product = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      dot_product += query[query_offset + dimension] * key_blocks[key_offset + dimension];
    }
    scores[task] = dot_product * scale;
  }
}

__global__ void paged_attention_values_kernel(const float* probabilities, const float* value_blocks,
                                              const std::uint32_t* block_table, float* output,
                                              std::size_t tasks, std::size_t physical_blocks,
                                              std::size_t heads, std::size_t queries,
                                              std::size_t keys, std::size_t block_tokens,
                                              std::size_t query_position_offset,
                                              std::size_t value_dimension) {
  std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; task < tasks; task += stride) {
    const std::size_t dimension = task % value_dimension;
    const std::size_t query_and_head = task / value_dimension;
    const std::size_t query_index = query_and_head % queries;
    const std::size_t head = query_and_head / queries;
    const std::size_t probability_offset = (head * queries + query_index) * keys;
    const std::size_t visible_keys = query_position_offset + query_index + 1;
    float accumulator = 0.0F;
    for (std::size_t key_index = 0; key_index < visible_keys; ++key_index) {
      const std::size_t logical_block = key_index / block_tokens;
      const std::size_t token_in_block = key_index % block_tokens;
      const std::size_t physical_block = static_cast<std::size_t>(block_table[logical_block]);
      if (physical_block >= physical_blocks) {
        accumulator = CUDART_NAN_F;
        break;
      }
      const std::size_t value_offset =
          (((physical_block * heads + head) * block_tokens + token_in_block) * value_dimension) +
          dimension;
      accumulator += probabilities[probability_offset + key_index] * value_blocks[value_offset];
    }
    output[task] = accumulator;
  }
}

__global__ void paged_tail_attention_scores_kernel(
    const float* query, const float* key_blocks, const float* tail_key,
    const std::uint32_t* block_table, float* scores, std::size_t tasks, std::size_t physical_blocks,
    std::size_t heads, std::size_t queries, std::size_t history_keys, std::size_t keys,
    std::size_t block_tokens, std::size_t head_dimension, float scale) {
  std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; task < tasks; task += stride) {
    const std::size_t key_index = task % keys;
    const std::size_t query_and_head = task / keys;
    const std::size_t query_index = query_and_head % queries;
    const std::size_t head = query_and_head / queries;
    if (key_index > history_keys + query_index) {
      scores[task] = -CUDART_INF_F;
      continue;
    }

    const std::size_t query_offset = (head * queries + query_index) * head_dimension;
    const float* key_values = nullptr;
    std::size_t key_offset = 0;
    if (key_index < history_keys) {
      const std::size_t logical_block = key_index / block_tokens;
      const std::size_t token_in_block = key_index % block_tokens;
      const std::size_t physical_block = static_cast<std::size_t>(block_table[logical_block]);
      if (physical_block >= physical_blocks) {
        scores[task] = CUDART_NAN_F;
        continue;
      }
      key_values = key_blocks;
      key_offset =
          (((physical_block * heads + head) * block_tokens + token_in_block) * head_dimension);
    } else {
      const std::size_t tail_index = key_index - history_keys;
      key_values = tail_key;
      key_offset = (head * queries + tail_index) * head_dimension;
    }

    float dot_product = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      dot_product += query[query_offset + dimension] * key_values[key_offset + dimension];
    }
    scores[task] = dot_product * scale;
  }
}

__global__ void paged_tail_attention_values_kernel(
    const float* probabilities, const float* value_blocks, const float* tail_value,
    const std::uint32_t* block_table, float* output, std::size_t tasks, std::size_t physical_blocks,
    std::size_t heads, std::size_t queries, std::size_t history_keys, std::size_t keys,
    std::size_t block_tokens, std::size_t value_dimension) {
  std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; task < tasks; task += stride) {
    const std::size_t dimension = task % value_dimension;
    const std::size_t query_and_head = task / value_dimension;
    const std::size_t query_index = query_and_head % queries;
    const std::size_t head = query_and_head / queries;
    const std::size_t probability_offset = (head * queries + query_index) * keys;
    const std::size_t visible_keys = history_keys + query_index + 1;
    float accumulator = 0.0F;
    for (std::size_t key_index = 0; key_index < visible_keys; ++key_index) {
      const float* value_values = nullptr;
      std::size_t value_offset = 0;
      if (key_index < history_keys) {
        const std::size_t logical_block = key_index / block_tokens;
        const std::size_t token_in_block = key_index % block_tokens;
        const std::size_t physical_block = static_cast<std::size_t>(block_table[logical_block]);
        if (physical_block >= physical_blocks) {
          accumulator = CUDART_NAN_F;
          break;
        }
        value_values = value_blocks;
        value_offset =
            (((physical_block * heads + head) * block_tokens + token_in_block) * value_dimension) +
            dimension;
      } else {
        const std::size_t tail_index = key_index - history_keys;
        value_values = tail_value;
        value_offset = (head * queries + tail_index) * value_dimension + dimension;
      }
      accumulator += probabilities[probability_offset + key_index] * value_values[value_offset];
    }
    output[task] = accumulator;
  }
}

std::string tensor_error(std::string_view operation, std::string_view name,
                         std::string_view requirement) {
  std::ostringstream message;
  message << operation << ": " << name << ' ' << requirement;
  return message.str();
}

void require_cuda_float_tensor(const Tensor& tensor, std::string_view name,
                               std::string_view operation) {
  HELIX_CHECK(tensor.device().is_cuda(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on CUDA"));
  HELIX_CHECK(tensor.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must have dtype float32"));
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be contiguous"));
}

PagedAttentionDimensions validate_query_key_and_table(const Tensor& query, const Tensor& key_blocks,
                                                      const PagedBlockTable& block_table,
                                                      std::size_t kv_tokens,
                                                      bool allow_empty_history,
                                                      std::string_view operation) {
  require_cuda_float_tensor(query, "query", operation);
  require_cuda_float_tensor(key_blocks, "key blocks", operation);
  HELIX_CHECK(query.rank() == 3, ErrorCode::kInvalidArgument,
              tensor_error(operation, "query", "must have rank three"));
  HELIX_CHECK(key_blocks.rank() == 4, ErrorCode::kInvalidArgument,
              tensor_error(operation, "key blocks", "must have rank four"));
  HELIX_CHECK(key_blocks.shape()[0] > 0 && key_blocks.shape()[1] > 0 && key_blocks.shape()[2] > 0 &&
                  key_blocks.shape()[3] > 0,
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "physical block, head, token, and key dimensions",
                           "must be positive"));
  HELIX_CHECK(
      query.shape()[0] == key_blocks.shape()[1] && query.shape()[2] == key_blocks.shape()[3],
      ErrorCode::kInvalidArgument,
      tensor_error(operation, "query and key blocks", "must have matching heads and width"));
  HELIX_CHECK(query.device() == key_blocks.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and key blocks", "must be on the same device"));
  HELIX_CHECK(block_table.valid(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "block table", "must own a valid device mapping"));
  HELIX_CHECK(block_table.device_index() == query.device().index(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "block table", "must be on the input device"));

  const auto physical_blocks = static_cast<std::size_t>(key_blocks.shape()[0]);
  const auto heads = static_cast<std::size_t>(key_blocks.shape()[1]);
  const auto block_tokens = static_cast<std::size_t>(key_blocks.shape()[2]);
  const auto key_dimension = static_cast<std::size_t>(key_blocks.shape()[3]);
  HELIX_CHECK(block_table.physical_block_count() == physical_blocks, ErrorCode::kInvalidArgument,
              tensor_error(operation, "block table capacity",
                           "must equal the packed physical block count"));
  HELIX_CHECK(allow_empty_history || kv_tokens != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "KV token count", "must be positive"));
  HELIX_CHECK(kv_tokens <= static_cast<std::size_t>(std::numeric_limits<Dimension>::max()),
              ErrorCode::kOutOfRange,
              tensor_error(operation, "KV token count", "exceeds the Tensor shape range"));
  const std::size_t required_logical_blocks =
      kv_tokens == 0 ? 0 : 1 + (kv_tokens - 1) / block_tokens;
  HELIX_CHECK(block_table.logical_block_count() == required_logical_blocks,
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "block table length",
                           "must exactly cover the logical KV token range"));

  return PagedAttentionDimensions{physical_blocks, required_logical_blocks,
                                  heads,           static_cast<std::size_t>(query.shape()[1]),
                                  kv_tokens,       block_tokens,
                                  key_dimension,   0,
                                  kv_tokens,       0};
}

void validate_value_blocks(const Tensor& query, const Tensor& key_blocks,
                           const Tensor& value_blocks, PagedAttentionDimensions& dimensions,
                           std::string_view operation) {
  require_cuda_float_tensor(value_blocks, "value blocks", operation);
  HELIX_CHECK(value_blocks.rank() == 4, ErrorCode::kInvalidArgument,
              tensor_error(operation, "value blocks", "must have rank four"));
  HELIX_CHECK(value_blocks.shape()[0] == key_blocks.shape()[0] &&
                  value_blocks.shape()[1] == key_blocks.shape()[1] &&
                  value_blocks.shape()[2] == key_blocks.shape()[2],
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "key and value blocks",
                           "must have equal physical-block, head, and token geometry"));
  HELIX_CHECK(value_blocks.shape()[3] > 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "value dimension", "must be positive"));
  HELIX_CHECK(value_blocks.device() == query.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "value blocks", "must be on the input device"));
  dimensions.value_dimension = static_cast<std::size_t>(value_blocks.shape()[3]);
}

PagedAttentionDimensions validate_inputs(const Tensor& query, const Tensor& key_blocks,
                                         const Tensor& value_blocks,
                                         const PagedBlockTable& block_table, std::size_t kv_tokens,
                                         std::string_view operation) {
  PagedAttentionDimensions dimensions =
      validate_query_key_and_table(query, key_blocks, block_table, kv_tokens, false, operation);
  validate_value_blocks(query, key_blocks, value_blocks, dimensions, operation);
  return dimensions;
}

PagedAttentionDimensions
validate_tail_inputs(const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
                     const PagedBlockTable& block_table, std::size_t history_tokens,
                     const Tensor& tail_key, const Tensor& tail_value, std::string_view operation) {
  PagedAttentionDimensions dimensions =
      validate_query_key_and_table(query, key_blocks, block_table, history_tokens, true, operation);
  validate_value_blocks(query, key_blocks, value_blocks, dimensions, operation);
  require_cuda_float_tensor(tail_key, "tail key", operation);
  require_cuda_float_tensor(tail_value, "tail value", operation);
  HELIX_CHECK(tail_key.rank() == 3 && tail_value.rank() == 3, ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail key and value", "must have rank three"));
  HELIX_CHECK(tail_key.shape() == query.shape(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and tail key", "must have equal shapes"));
  HELIX_CHECK(tail_value.shape()[0] == query.shape()[0] &&
                  tail_value.shape()[1] == query.shape()[1] &&
                  tail_value.shape()[2] == value_blocks.shape()[3],
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail value", "must match query rows and value width"));
  HELIX_CHECK(tail_key.device() == query.device() && tail_value.device() == query.device(),
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail key and value", "must be on the input device"));

  const std::size_t tail_tokens = dimensions.queries;
  HELIX_CHECK(tail_tokens != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail token count", "must be positive"));
  constexpr auto kMaximumDimension =
      static_cast<std::size_t>(std::numeric_limits<Dimension>::max());
  HELIX_CHECK(history_tokens <= kMaximumDimension - tail_tokens, ErrorCode::kResourceExhausted,
              tensor_error(operation, "history plus tail token count", "overflows shape range"));
  dimensions.history_keys = history_tokens;
  dimensions.tail_keys = tail_tokens;
  dimensions.keys = history_tokens + tail_tokens;
  return dimensions;
}

void validate_offset(const PagedAttentionDimensions& dimensions, std::size_t query_position_offset,
                     std::string_view operation) {
  HELIX_CHECK(query_position_offset <= dimensions.keys, ErrorCode::kInvalidArgument,
              tensor_error(operation, "query position offset", "exceeds the KV token count"));
  HELIX_CHECK(dimensions.queries <= dimensions.keys - query_position_offset,
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "query range", "extends beyond the KV token count"));
}

void require_shape(const Tensor& tensor, const Shape& expected, std::string_view name,
                   std::string_view operation) {
  if (tensor.shape() != expected) {
    std::ostringstream message;
    message << operation << ": " << name << " shape " << shape_string(tensor.shape())
            << " does not match expected shape " << shape_string(expected);
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
}

void require_no_alias(const Tensor& tensor, const Tensor& output, std::string_view tensor_name,
                      std::string_view output_name, std::string_view operation) {
  HELIX_CHECK(!tensor.overlaps_logical_range(output), ErrorCode::kInvalidArgument,
              tensor_error(operation, output_name,
                           std::string("must not alias ") + std::string(tensor_name)));
}

unsigned int block_count(std::size_t work_items) {
  HELIX_ASSERT(work_items != 0, "cannot select a CUDA paged-attention grid for zero work items");
  const std::size_t required_blocks =
      1 + (work_items - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  const std::size_t blocks = std::min(required_blocks, kMaximumBlocks);
  HELIX_ASSERT(blocks <= std::numeric_limits<unsigned int>::max(),
               "CUDA paged-attention grid dimension does not fit unsigned int");
  return static_cast<unsigned int>(blocks);
}

void check_launch(std::string_view kernel_name) {
  std::ostringstream context;
  context << "launching " << kernel_name;
  HELIX_CUDA_CHECK_CONTEXT(cudaGetLastError(), context.str());
}

void synchronize_before_unwind(const helix::cuda::Stream& stream) noexcept {
  try {
    stream.synchronize();
  } catch (...) {
    // Preserve the operation's original exception. Cleanup below remains
    // nonthrowing even when the CUDA context cannot be drained.
  }
}

} // namespace

Shape paged_score_workspace_shape(const Tensor& query, const Tensor& key_blocks,
                                  const PagedBlockTable& block_table, std::size_t kv_tokens) {
  constexpr std::string_view kOperation = "attention::cuda::paged_score_workspace_shape";
  const PagedAttentionDimensions dimensions =
      validate_query_key_and_table(query, key_blocks, block_table, kv_tokens, false, kOperation);
  return Shape{static_cast<Dimension>(dimensions.heads), static_cast<Dimension>(dimensions.queries),
               static_cast<Dimension>(dimensions.keys)};
}

std::size_t paged_score_workspace_bytes(const Tensor& query, const Tensor& key_blocks,
                                        const PagedBlockTable& block_table, std::size_t kv_tokens) {
  return checked_nbytes(paged_score_workspace_shape(query, key_blocks, block_table, kv_tokens),
                        DType::kFloat32);
}

Shape paged_tail_score_workspace_shape(const Tensor& query, const Tensor& key_blocks,
                                       const Tensor& value_blocks,
                                       const PagedBlockTable& block_table,
                                       std::size_t history_tokens, const Tensor& tail_key,
                                       const Tensor& tail_value) {
  constexpr std::string_view kOperation = "attention::cuda::paged_tail_score_workspace_shape";
  const PagedAttentionDimensions dimensions =
      validate_tail_inputs(query, key_blocks, value_blocks, block_table, history_tokens, tail_key,
                           tail_value, kOperation);
  return Shape{static_cast<Dimension>(dimensions.heads), static_cast<Dimension>(dimensions.queries),
               static_cast<Dimension>(dimensions.keys)};
}

std::size_t paged_tail_score_workspace_bytes(const Tensor& query, const Tensor& key_blocks,
                                             const Tensor& value_blocks,
                                             const PagedBlockTable& block_table,
                                             std::size_t history_tokens, const Tensor& tail_key,
                                             const Tensor& tail_value) {
  return checked_nbytes(paged_tail_score_workspace_shape(query, key_blocks, value_blocks,
                                                         block_table, history_tokens, tail_key,
                                                         tail_value),
                        DType::kFloat32);
}

void paged_scaled_dot_product_attention_out(
    const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
    const PagedBlockTable& block_table, std::size_t kv_tokens, std::size_t query_position_offset,
    Tensor& output, Tensor& score_workspace, const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation = "attention::cuda::paged_scaled_dot_product_attention_out";
  const PagedAttentionDimensions dimensions =
      validate_inputs(query, key_blocks, value_blocks, block_table, kv_tokens, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  require_cuda_float_tensor(output, "output", kOperation);
  require_cuda_float_tensor(score_workspace, "score workspace", kOperation);
  require_shape(output,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.value_dimension)},
                "output", kOperation);
  require_shape(score_workspace,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.keys)},
                "score workspace", kOperation);
  HELIX_CHECK(output.device() == query.device() && score_workspace.device() == query.device(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "output and workspace", "must be on the input device"));
  HELIX_CHECK(stream.native_handle() != nullptr && stream.device_index() == query.device().index(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "stream", "must be valid and on the input device"));
  require_no_alias(query, output, "query", "output", kOperation);
  require_no_alias(key_blocks, output, "key blocks", "output", kOperation);
  require_no_alias(value_blocks, output, "value blocks", "output", kOperation);
  require_no_alias(query, score_workspace, "query", "score workspace", kOperation);
  require_no_alias(key_blocks, score_workspace, "key blocks", "score workspace", kOperation);
  require_no_alias(value_blocks, score_workspace, "value blocks", "score workspace", kOperation);
  require_no_alias(score_workspace, output, "score workspace", "output", kOperation);

  const std::size_t score_tasks = score_workspace.numel();
  const std::size_t output_tasks = output.numel();
  if (score_tasks == 0) {
    HELIX_ASSERT(output_tasks == 0, "empty paged-attention scores must imply empty output");
    return;
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(dimensions.key_dimension));
  helix::cuda::DeviceGuard guard(stream.device_index());
  paged_attention_scores_kernel<<<block_count(score_tasks), kThreadsPerBlock, 0,
                                  stream.native_handle()>>>(
      query.data_as<float>(), key_blocks.data_as<float>(), block_table.data(),
      score_workspace.data_as<float>(), score_tasks, dimensions.physical_blocks, dimensions.heads,
      dimensions.queries, dimensions.keys, dimensions.block_tokens, dimensions.key_dimension,
      query_position_offset, scale);
  check_launch("paged_attention_scores_kernel");

  helix::cuda::kernels::softmax_out(score_workspace, score_workspace, stream,
                                    helix::cuda::kernels::SoftmaxAlgorithm::kSharedMemory);

  paged_attention_values_kernel<<<block_count(output_tasks), kThreadsPerBlock, 0,
                                  stream.native_handle()>>>(
      score_workspace.data_as<float>(), value_blocks.data_as<float>(), block_table.data(),
      output.data_as<float>(), output_tasks, dimensions.physical_blocks, dimensions.heads,
      dimensions.queries, dimensions.keys, dimensions.block_tokens, query_position_offset,
      dimensions.value_dimension);
  check_launch("paged_attention_values_kernel");
}

void paged_scaled_dot_product_attention_with_tail_out(
    const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
    const PagedBlockTable& block_table, std::size_t history_tokens, const Tensor& tail_key,
    const Tensor& tail_value, Tensor& output, Tensor& score_workspace,
    const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation =
      "attention::cuda::paged_scaled_dot_product_attention_with_tail_out";
  const PagedAttentionDimensions dimensions =
      validate_tail_inputs(query, key_blocks, value_blocks, block_table, history_tokens, tail_key,
                           tail_value, kOperation);
  validate_offset(dimensions, history_tokens, kOperation);
  require_cuda_float_tensor(output, "output", kOperation);
  require_cuda_float_tensor(score_workspace, "score workspace", kOperation);
  require_shape(output,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.value_dimension)},
                "output", kOperation);
  require_shape(score_workspace,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.keys)},
                "score workspace", kOperation);
  HELIX_CHECK(output.device() == query.device() && score_workspace.device() == query.device(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "output and workspace", "must be on the input device"));
  HELIX_CHECK(stream.native_handle() != nullptr && stream.device_index() == query.device().index(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "stream", "must be valid and on the input device"));
  require_no_alias(query, output, "query", "output", kOperation);
  require_no_alias(key_blocks, output, "key blocks", "output", kOperation);
  require_no_alias(value_blocks, output, "value blocks", "output", kOperation);
  require_no_alias(tail_key, output, "tail key", "output", kOperation);
  require_no_alias(tail_value, output, "tail value", "output", kOperation);
  require_no_alias(query, score_workspace, "query", "score workspace", kOperation);
  require_no_alias(key_blocks, score_workspace, "key blocks", "score workspace", kOperation);
  require_no_alias(value_blocks, score_workspace, "value blocks", "score workspace", kOperation);
  require_no_alias(tail_key, score_workspace, "tail key", "score workspace", kOperation);
  require_no_alias(tail_value, score_workspace, "tail value", "score workspace", kOperation);
  require_no_alias(score_workspace, output, "score workspace", "output", kOperation);

  const std::size_t score_tasks = score_workspace.numel();
  const std::size_t output_tasks = output.numel();
  HELIX_ASSERT(score_tasks != 0 && output_tasks != 0,
               "validated paged-tail attention must contain work");
  const float scale = 1.0F / std::sqrt(static_cast<float>(dimensions.key_dimension));
  helix::cuda::DeviceGuard guard(stream.device_index());
  paged_tail_attention_scores_kernel<<<block_count(score_tasks), kThreadsPerBlock, 0,
                                       stream.native_handle()>>>(
      query.data_as<float>(), key_blocks.data_as<float>(), tail_key.data_as<float>(),
      block_table.data(), score_workspace.data_as<float>(), score_tasks, dimensions.physical_blocks,
      dimensions.heads, dimensions.queries, dimensions.history_keys, dimensions.keys,
      dimensions.block_tokens, dimensions.key_dimension, scale);
  check_launch("paged_tail_attention_scores_kernel");

  helix::cuda::kernels::softmax_out(score_workspace, score_workspace, stream,
                                    helix::cuda::kernels::SoftmaxAlgorithm::kSharedMemory);

  paged_tail_attention_values_kernel<<<block_count(output_tasks), kThreadsPerBlock, 0,
                                       stream.native_handle()>>>(
      score_workspace.data_as<float>(), value_blocks.data_as<float>(), tail_value.data_as<float>(),
      block_table.data(), output.data_as<float>(), output_tasks, dimensions.physical_blocks,
      dimensions.heads, dimensions.queries, dimensions.history_keys, dimensions.keys,
      dimensions.block_tokens, dimensions.value_dimension);
  check_launch("paged_tail_attention_values_kernel");
}

Tensor paged_scaled_dot_product_attention(const Tensor& query, const Tensor& key_blocks,
                                          const Tensor& value_blocks,
                                          const PagedBlockTable& block_table, std::size_t kv_tokens,
                                          std::size_t query_position_offset,
                                          const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation = "attention::cuda::paged_scaled_dot_product_attention";
  const PagedAttentionDimensions dimensions =
      validate_inputs(query, key_blocks, value_blocks, block_table, kv_tokens, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  Tensor output = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                 static_cast<Dimension>(dimensions.queries),
                                 static_cast<Dimension>(dimensions.value_dimension)},
                                DType::kFloat32, query.device());
  Tensor score_workspace = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                          static_cast<Dimension>(dimensions.queries),
                                          static_cast<Dimension>(dimensions.keys)},
                                         DType::kFloat32, query.device());
  try {
    paged_scaled_dot_product_attention_out(query, key_blocks, value_blocks, block_table, kv_tokens,
                                           query_position_offset, output, score_workspace, stream);
    if (score_workspace.numel() != 0) {
      stream.synchronize();
    }
  } catch (...) {
    synchronize_before_unwind(stream);
    throw;
  }
  return output;
}

Tensor paged_scaled_dot_product_attention_with_tail(
    const Tensor& query, const Tensor& key_blocks, const Tensor& value_blocks,
    const PagedBlockTable& block_table, std::size_t history_tokens, const Tensor& tail_key,
    const Tensor& tail_value, const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation =
      "attention::cuda::paged_scaled_dot_product_attention_with_tail";
  const PagedAttentionDimensions dimensions =
      validate_tail_inputs(query, key_blocks, value_blocks, block_table, history_tokens, tail_key,
                           tail_value, kOperation);
  Tensor output = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                 static_cast<Dimension>(dimensions.queries),
                                 static_cast<Dimension>(dimensions.value_dimension)},
                                DType::kFloat32, query.device());
  Tensor score_workspace = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                          static_cast<Dimension>(dimensions.queries),
                                          static_cast<Dimension>(dimensions.keys)},
                                         DType::kFloat32, query.device());
  try {
    paged_scaled_dot_product_attention_with_tail_out(query, key_blocks, value_blocks, block_table,
                                                     history_tokens, tail_key, tail_value, output,
                                                     score_workspace, stream);
    stream.synchronize();
  } catch (...) {
    synchronize_before_unwind(stream);
    throw;
  }
  return output;
}

} // namespace helix::attention::cuda
