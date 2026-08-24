#include "helix/attention/cuda_paged.hpp"
#include "helix/core/error.hpp"
#include "helix/kernels/cuda/softmax.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

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
                                              std::size_t value_dimension) {
  std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; task < tasks; task += stride) {
    const std::size_t dimension = task % value_dimension;
    const std::size_t query_and_head = task / value_dimension;
    const std::size_t query_index = query_and_head % queries;
    const std::size_t head = query_and_head / queries;
    const std::size_t probability_offset = (head * queries + query_index) * keys;
    float accumulator = 0.0F;
    for (std::size_t key_index = 0; key_index < keys; ++key_index) {
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
  HELIX_CHECK(kv_tokens != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "KV token count", "must be positive"));
  HELIX_CHECK(kv_tokens <= static_cast<std::size_t>(std::numeric_limits<Dimension>::max()),
              ErrorCode::kOutOfRange,
              tensor_error(operation, "KV token count", "exceeds the Tensor shape range"));
  const std::size_t required_logical_blocks = 1 + (kv_tokens - 1) / block_tokens;
  HELIX_CHECK(block_table.logical_block_count() == required_logical_blocks,
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "block table length",
                           "must exactly cover the logical KV token range"));

  return PagedAttentionDimensions{physical_blocks, required_logical_blocks,
                                  heads,           static_cast<std::size_t>(query.shape()[1]),
                                  kv_tokens,       block_tokens,
                                  key_dimension,   0};
}

PagedAttentionDimensions validate_inputs(const Tensor& query, const Tensor& key_blocks,
                                         const Tensor& value_blocks,
                                         const PagedBlockTable& block_table, std::size_t kv_tokens,
                                         std::string_view operation) {
  PagedAttentionDimensions dimensions =
      validate_query_key_and_table(query, key_blocks, block_table, kv_tokens, operation);
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
  HELIX_CHECK(tensor.nbytes() == 0 || tensor.data() != output.data(), ErrorCode::kInvalidArgument,
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

} // namespace

PagedBlockTable::PagedBlockTable(std::span<const std::uint32_t> physical_blocks,
                                 std::size_t physical_block_count, int device_index)
    : logical_block_count_(physical_blocks.size()), physical_block_count_(physical_block_count),
      device_index_(device_index) {
  constexpr std::string_view kOperation = "attention::cuda::PagedBlockTable";
  HELIX_CHECK(!physical_blocks.empty(), ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "physical block IDs", "must not be empty"));
  HELIX_CHECK(physical_block_count != 0 &&
                  physical_block_count <= std::numeric_limits<std::uint32_t>::max(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "physical block capacity",
                           "must fit the uint32 block-ID representation"));
  HELIX_CHECK(device_index >= 0, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "device index", "must be non-negative"));
  HELIX_CHECK(physical_blocks.size() <=
                  std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t),
              ErrorCode::kResourceExhausted,
              tensor_error(kOperation, "block table byte count", "overflows size_t"));
  for (const std::uint32_t physical_block : physical_blocks) {
    HELIX_CHECK(static_cast<std::size_t>(physical_block) < physical_block_count,
                ErrorCode::kOutOfRange,
                tensor_error(kOperation, "physical block ID", "exceeds the packed block capacity"));
  }

  device_blocks_.emplace(size_bytes(), device_index_);
  helix::cuda::DeviceGuard guard(device_index_);
  helix::cuda::copy_host_to_device(device_blocks_->data(), physical_blocks.data(), size_bytes());
}

PagedBlockTable::PagedBlockTable(PagedBlockTable&& other) noexcept
    : device_blocks_(std::move(other.device_blocks_)),
      logical_block_count_(std::exchange(other.logical_block_count_, 0)),
      physical_block_count_(std::exchange(other.physical_block_count_, 0)),
      device_index_(std::exchange(other.device_index_, -1)) {
  other.device_blocks_.reset();
}

PagedBlockTable& PagedBlockTable::operator=(PagedBlockTable&& other) noexcept {
  if (this != &other) {
    device_blocks_ = std::move(other.device_blocks_);
    logical_block_count_ = std::exchange(other.logical_block_count_, 0);
    physical_block_count_ = std::exchange(other.physical_block_count_, 0);
    device_index_ = std::exchange(other.device_index_, -1);
    other.device_blocks_.reset();
  }
  return *this;
}

bool PagedBlockTable::valid() const noexcept {
  return device_blocks_.has_value() && device_blocks_->data() != nullptr &&
         logical_block_count_ != 0 && physical_block_count_ != 0 && device_index_ >= 0;
}

const std::uint32_t* PagedBlockTable::data() const noexcept {
  return device_blocks_.has_value() ? static_cast<const std::uint32_t*>(device_blocks_->data())
                                    : nullptr;
}

Shape paged_score_workspace_shape(const Tensor& query, const Tensor& key_blocks,
                                  const PagedBlockTable& block_table, std::size_t kv_tokens) {
  constexpr std::string_view kOperation = "attention::cuda::paged_score_workspace_shape";
  const PagedAttentionDimensions dimensions =
      validate_query_key_and_table(query, key_blocks, block_table, kv_tokens, kOperation);
  return Shape{static_cast<Dimension>(dimensions.heads), static_cast<Dimension>(dimensions.queries),
               static_cast<Dimension>(dimensions.keys)};
}

std::size_t paged_score_workspace_bytes(const Tensor& query, const Tensor& key_blocks,
                                        const PagedBlockTable& block_table, std::size_t kv_tokens) {
  return checked_nbytes(paged_score_workspace_shape(query, key_blocks, block_table, kv_tokens),
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
      dimensions.queries, dimensions.keys, dimensions.block_tokens, dimensions.value_dimension);
  check_launch("paged_attention_values_kernel");
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
  paged_scaled_dot_product_attention_out(query, key_blocks, value_blocks, block_table, kv_tokens,
                                         query_position_offset, output, score_workspace, stream);
  if (score_workspace.numel() != 0) {
    stream.synchronize();
  }
  return output;
}

} // namespace helix::attention::cuda
