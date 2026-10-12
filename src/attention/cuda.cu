#include "helix/attention/cuda.hpp"
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

struct AttentionDimensions {
  std::size_t heads;
  std::size_t queries;
  std::size_t keys;
  std::size_t head_dimension;
  std::size_t value_dimension;
};

__global__ void attention_scores_kernel(const float* query, const float* key, float* scores,
                                        std::size_t tasks, std::size_t queries, std::size_t keys,
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

    const std::size_t query_offset = (head * queries + query_index) * head_dimension;
    const std::size_t key_offset = (head * keys + key_index) * head_dimension;
    float dot_product = 0.0F;
    for (std::size_t dimension = 0; dimension < head_dimension; ++dimension) {
      dot_product += query[query_offset + dimension] * key[key_offset + dimension];
    }
    scores[task] = dot_product * scale;
  }
}

__global__ void attention_values_kernel(const float* probabilities, const float* value,
                                        float* output, std::size_t tasks, std::size_t queries,
                                        std::size_t keys, std::size_t query_position_offset,
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
      const std::size_t value_offset = (head * keys + key_index) * value_dimension + dimension;
      accumulator += probabilities[probability_offset + key_index] * value[value_offset];
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

AttentionDimensions validate_inputs(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::string_view operation) {
  require_cuda_float_tensor(query, "query", operation);
  require_cuda_float_tensor(key, "key", operation);
  require_cuda_float_tensor(value, "value", operation);
  HELIX_CHECK(query.rank() == 3 && key.rank() == 3 && value.rank() == 3,
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "query, key, and value", "must have rank three"));
  HELIX_CHECK(query.shape()[0] == key.shape()[0] && query.shape()[0] == value.shape()[0],
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "query, key, and value", "must have equal head counts"));
  HELIX_CHECK(query.shape()[2] == key.shape()[2], ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and key", "must have equal head dimensions"));
  HELIX_CHECK(key.shape()[1] == value.shape()[1], ErrorCode::kInvalidArgument,
              tensor_error(operation, "key and value", "must have equal token counts"));
  HELIX_CHECK(query.shape()[2] > 0 && key.shape()[1] > 0 && value.shape()[2] > 0,
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "head, KV-token, and value dimensions", "must be positive"));
  HELIX_CHECK(query.device() == key.device() && query.device() == value.device(),
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "query, key, and value", "must be on the same device"));
  return AttentionDimensions{
      static_cast<std::size_t>(query.shape()[0]), static_cast<std::size_t>(query.shape()[1]),
      static_cast<std::size_t>(key.shape()[1]), static_cast<std::size_t>(query.shape()[2]),
      static_cast<std::size_t>(value.shape()[2])};
}

void validate_offset(const AttentionDimensions& dimensions, std::size_t query_position_offset,
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
  HELIX_ASSERT(work_items != 0, "cannot select a CUDA attention grid for zero work items");
  const std::size_t required_blocks =
      1 + (work_items - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  const std::size_t blocks = std::min(required_blocks, kMaximumBlocks);
  HELIX_ASSERT(blocks <= std::numeric_limits<unsigned int>::max(),
               "CUDA attention grid dimension does not fit unsigned int");
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

Shape score_workspace_shape(const Tensor& query, const Tensor& key) {
  constexpr std::string_view kOperation = "attention::cuda::score_workspace_shape";
  require_cuda_float_tensor(query, "query", kOperation);
  require_cuda_float_tensor(key, "key", kOperation);
  HELIX_CHECK(query.rank() == 3 && key.rank() == 3, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "query and key", "must have rank three"));
  HELIX_CHECK(query.shape()[0] == key.shape()[0] && query.shape()[2] == key.shape()[2],
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "query and key", "must have matching heads and width"));
  HELIX_CHECK(query.shape()[2] > 0 && key.shape()[1] > 0, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "head dimension and KV token count", "must be positive"));
  HELIX_CHECK(query.device() == key.device(), ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "query and key", "must be on the same device"));
  return Shape{query.shape()[0], query.shape()[1], key.shape()[1]};
}

std::size_t score_workspace_bytes(const Tensor& query, const Tensor& key) {
  return checked_nbytes(score_workspace_shape(query, key), DType::kFloat32);
}

void scaled_dot_product_attention_out(const Tensor& query, const Tensor& key, const Tensor& value,
                                      std::size_t query_position_offset, Tensor& output,
                                      Tensor& score_workspace, const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation = "attention::cuda::scaled_dot_product_attention_out";
  const AttentionDimensions dimensions = validate_inputs(query, key, value, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  require_cuda_float_tensor(output, "output", kOperation);
  require_cuda_float_tensor(score_workspace, "score workspace", kOperation);
  require_shape(output, Shape{query.shape()[0], query.shape()[1], value.shape()[2]}, "output",
                kOperation);
  require_shape(score_workspace, Shape{query.shape()[0], query.shape()[1], key.shape()[1]},
                "score workspace", kOperation);
  HELIX_CHECK(output.device() == query.device() && score_workspace.device() == query.device(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "output and workspace", "must be on the input device"));
  HELIX_CHECK(stream.native_handle() != nullptr && stream.device_index() == query.device().index(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "stream", "must be valid and on the input device"));
  require_no_alias(query, output, "query", "output", kOperation);
  require_no_alias(key, output, "key", "output", kOperation);
  require_no_alias(value, output, "value", "output", kOperation);
  require_no_alias(query, score_workspace, "query", "score workspace", kOperation);
  require_no_alias(key, score_workspace, "key", "score workspace", kOperation);
  require_no_alias(value, score_workspace, "value", "score workspace", kOperation);
  require_no_alias(score_workspace, output, "score workspace", "output", kOperation);

  const std::size_t score_tasks = score_workspace.numel();
  const std::size_t output_tasks = output.numel();
  if (score_tasks == 0) {
    HELIX_ASSERT(output_tasks == 0, "empty attention scores must imply empty output");
    return;
  }

  const float scale = 1.0F / std::sqrt(static_cast<float>(dimensions.head_dimension));
  helix::cuda::DeviceGuard guard(stream.device_index());
  attention_scores_kernel<<<block_count(score_tasks), kThreadsPerBlock, 0,
                            stream.native_handle()>>>(
      query.data_as<float>(), key.data_as<float>(), score_workspace.data_as<float>(), score_tasks,
      dimensions.queries, dimensions.keys, dimensions.head_dimension, query_position_offset, scale);
  check_launch("attention_scores_kernel");

  helix::cuda::kernels::softmax_out(score_workspace, score_workspace, stream,
                                    helix::cuda::kernels::SoftmaxAlgorithm::kSharedMemory);

  attention_values_kernel<<<block_count(output_tasks), kThreadsPerBlock, 0,
                            stream.native_handle()>>>(
      score_workspace.data_as<float>(), value.data_as<float>(), output.data_as<float>(),
      output_tasks, dimensions.queries, dimensions.keys, query_position_offset,
      dimensions.value_dimension);
  check_launch("attention_values_kernel");
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::size_t query_position_offset,
                                    const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation = "attention::cuda::scaled_dot_product_attention";
  const AttentionDimensions dimensions = validate_inputs(query, key, value, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  Tensor output = Tensor::empty({query.shape()[0], query.shape()[1], value.shape()[2]},
                                DType::kFloat32, query.device());
  Tensor score_workspace = Tensor::empty({query.shape()[0], query.shape()[1], key.shape()[1]},
                                         DType::kFloat32, query.device());
  try {
    scaled_dot_product_attention_out(query, key, value, query_position_offset, output,
                                     score_workspace, stream);
    if (score_workspace.numel() != 0) {
      stream.synchronize();
    }
  } catch (...) {
    synchronize_before_unwind(stream);
    throw;
  }
  return output;
}

} // namespace helix::attention::cuda
