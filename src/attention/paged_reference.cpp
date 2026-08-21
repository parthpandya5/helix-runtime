#include "helix/attention/paged_reference.hpp"

#include "helix/core/error.hpp"
#include "helix/kernels/cpu/reference.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

namespace helix::attention::cpu {
namespace {

struct PagedAttentionDimensions {
  std::size_t heads;
  std::size_t queries;
  std::size_t keys;
  std::size_t head_dimension;
  std::size_t logical_blocks;
};

std::string tensor_error(std::string_view operation, std::string_view name,
                         std::string_view requirement) {
  std::ostringstream message;
  message << operation << ": " << name << ' ' << requirement;
  return message.str();
}

void require_cpu_float_tensor(const Tensor& tensor, std::string_view name,
                              std::string_view operation) {
  HELIX_CHECK(tensor.device().is_cpu(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on CPU"));
  HELIX_CHECK(tensor.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must have dtype float32"));
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be contiguous"));
}

PagedAttentionDimensions validate_inputs(const Tensor& query,
                                         const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                         std::string_view operation) {
  require_cpu_float_tensor(query, "query", operation);
  HELIX_CHECK(query.rank() == 3, ErrorCode::kInvalidArgument,
              tensor_error(operation, "query", "must have rank three"));
  const std::size_t keys = kv_layer.length();
  const std::size_t heads = kv_layer.kv_heads();
  const std::size_t head_dimension = kv_layer.head_dimension();
  HELIX_CHECK(keys != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "KV token count", "must be positive"));
  HELIX_CHECK(query.shape()[0] == static_cast<Dimension>(heads), ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and KV cache", "must have equal head counts"));
  HELIX_CHECK(query.shape()[2] == static_cast<Dimension>(head_dimension),
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and KV cache", "must have equal head dimensions"));
  return PagedAttentionDimensions{heads, static_cast<std::size_t>(query.shape()[1]), keys,
                                  head_dimension, kv_layer.logical_block_count()};
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

} // namespace

Shape paged_score_workspace_shape(const Tensor& query,
                                  const kv_cache::PagedKvCache::LayerReadView& kv_layer) {
  constexpr std::string_view kOperation = "attention::cpu::paged_score_workspace_shape";
  const PagedAttentionDimensions dimensions = validate_inputs(query, kv_layer, kOperation);
  return Shape{static_cast<Dimension>(dimensions.heads), static_cast<Dimension>(dimensions.queries),
               static_cast<Dimension>(dimensions.keys)};
}

std::size_t paged_score_workspace_bytes(const Tensor& query,
                                        const kv_cache::PagedKvCache::LayerReadView& kv_layer) {
  return checked_nbytes(paged_score_workspace_shape(query, kv_layer), DType::kFloat32);
}

void paged_scaled_dot_product_attention_out(const Tensor& query,
                                            const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                            std::size_t query_position_offset, Tensor& output,
                                            Tensor& score_workspace) {
  constexpr std::string_view kOperation = "attention::cpu::paged_scaled_dot_product_attention_out";
  const PagedAttentionDimensions dimensions = validate_inputs(query, kv_layer, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  require_cpu_float_tensor(output, "output", kOperation);
  require_cpu_float_tensor(score_workspace, "score workspace", kOperation);
  require_shape(output,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.head_dimension)},
                "output", kOperation);
  require_shape(score_workspace,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.keys)},
                "score workspace", kOperation);
  require_no_alias(query, output, "query", "output", kOperation);
  require_no_alias(query, score_workspace, "query", "score workspace", kOperation);
  require_no_alias(score_workspace, output, "score workspace", "output", kOperation);

  if (dimensions.queries == 0) {
    return;
  }

  const auto* query_values = query.data_as<float>();
  auto* scores = score_workspace.data_as<float>();
  auto* output_values = output.data_as<float>();
  const double scale = 1.0 / std::sqrt(static_cast<double>(dimensions.head_dimension));

  for (std::size_t head = 0; head < dimensions.heads; ++head) {
    std::size_t key_start = 0;
    for (std::size_t logical_block = 0; logical_block < dimensions.logical_blocks;
         ++logical_block) {
      const std::size_t tokens_in_block = kv_layer.block_token_count(logical_block);
      const std::span<const float> keys = kv_layer.key_block(logical_block, head);
      HELIX_ASSERT(keys.size() == tokens_in_block * dimensions.head_dimension,
                   "paged attention key span has an inconsistent size");
      for (std::size_t query_index = 0; query_index < dimensions.queries; ++query_index) {
        const std::size_t query_offset =
            (head * dimensions.queries + query_index) * dimensions.head_dimension;
        const std::size_t score_offset =
            (head * dimensions.queries + query_index) * dimensions.keys;
        const std::size_t query_position = query_position_offset + query_index;
        for (std::size_t token = 0; token < tokens_in_block; ++token) {
          const std::size_t key_index = key_start + token;
          if (key_index > query_position) {
            scores[score_offset + key_index] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const std::size_t key_offset = token * dimensions.head_dimension;
          double dot_product = 0.0;
          for (std::size_t dimension = 0; dimension < dimensions.head_dimension; ++dimension) {
            dot_product += static_cast<double>(query_values[query_offset + dimension]) *
                           static_cast<double>(keys[key_offset + dimension]);
          }
          scores[score_offset + key_index] = static_cast<float>(dot_product * scale);
        }
      }
      key_start += tokens_in_block;
    }
    HELIX_ASSERT(key_start == dimensions.keys,
                 "paged attention block traversal did not cover the KV sequence");
  }

  helix::cpu::softmax_out(score_workspace, score_workspace);

  for (std::size_t head = 0; head < dimensions.heads; ++head) {
    for (std::size_t query_index = 0; query_index < dimensions.queries; ++query_index) {
      const std::size_t score_offset = (head * dimensions.queries + query_index) * dimensions.keys;
      const std::size_t output_offset =
          (head * dimensions.queries + query_index) * dimensions.head_dimension;
      for (std::size_t dimension = 0; dimension < dimensions.head_dimension; ++dimension) {
        double accumulator = 0.0;
        std::size_t value_start = 0;
        for (std::size_t logical_block = 0; logical_block < dimensions.logical_blocks;
             ++logical_block) {
          const std::size_t tokens_in_block = kv_layer.block_token_count(logical_block);
          const std::span<const float> values = kv_layer.value_block(logical_block, head);
          HELIX_ASSERT(values.size() == tokens_in_block * dimensions.head_dimension,
                       "paged attention value span has an inconsistent size");
          for (std::size_t token = 0; token < tokens_in_block; ++token) {
            const std::size_t key_index = value_start + token;
            const std::size_t value_offset = token * dimensions.head_dimension + dimension;
            accumulator += static_cast<double>(scores[score_offset + key_index]) *
                           static_cast<double>(values[value_offset]);
          }
          value_start += tokens_in_block;
        }
        HELIX_ASSERT(value_start == dimensions.keys,
                     "paged attention value traversal did not cover the KV sequence");
        output_values[output_offset + dimension] = static_cast<float>(accumulator);
      }
    }
  }
}

Tensor paged_scaled_dot_product_attention(const Tensor& query,
                                          const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                          std::size_t query_position_offset) {
  constexpr std::string_view kOperation = "attention::cpu::paged_scaled_dot_product_attention";
  const PagedAttentionDimensions dimensions = validate_inputs(query, kv_layer, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  Tensor output = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                 static_cast<Dimension>(dimensions.queries),
                                 static_cast<Dimension>(dimensions.head_dimension)});
  Tensor workspace = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                    static_cast<Dimension>(dimensions.queries),
                                    static_cast<Dimension>(dimensions.keys)});
  paged_scaled_dot_product_attention_out(query, kv_layer, query_position_offset, output, workspace);
  return output;
}

} // namespace helix::attention::cpu
