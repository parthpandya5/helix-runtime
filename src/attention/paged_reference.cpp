#include "helix/attention/paged_reference.hpp"

#include "helix/core/error.hpp"
#include "helix/kernels/cpu/reference.hpp"

#include <algorithm>
#include <array>
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
  std::size_t history_keys;
  std::size_t tail_keys;
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

PagedAttentionDimensions
validate_query_and_history(const Tensor& query,
                           const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                           std::string_view operation) {
  require_cpu_float_tensor(query, "query", operation);
  HELIX_CHECK(query.rank() == 3, ErrorCode::kInvalidArgument,
              tensor_error(operation, "query", "must have rank three"));
  const std::size_t history_keys = kv_layer.length();
  const std::size_t heads = kv_layer.kv_heads();
  const std::size_t head_dimension = kv_layer.head_dimension();
  HELIX_CHECK(query.shape()[0] == static_cast<Dimension>(heads), ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and KV cache", "must have equal head counts"));
  HELIX_CHECK(query.shape()[2] == static_cast<Dimension>(head_dimension),
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and KV cache", "must have equal head dimensions"));
  return PagedAttentionDimensions{heads,
                                  static_cast<std::size_t>(query.shape()[1]),
                                  history_keys,
                                  0,
                                  history_keys,
                                  head_dimension,
                                  kv_layer.logical_block_count()};
}

PagedAttentionDimensions validate_inputs(const Tensor& query,
                                         const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                         std::string_view operation) {
  PagedAttentionDimensions dimensions = validate_query_and_history(query, kv_layer, operation);
  HELIX_CHECK(dimensions.history_keys != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "KV token count", "must be positive"));
  return dimensions;
}

PagedAttentionDimensions validate_tail_inputs(const Tensor& query,
                                              const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                              const Tensor& tail_key, const Tensor& tail_value,
                                              std::string_view operation) {
  PagedAttentionDimensions dimensions = validate_query_and_history(query, kv_layer, operation);
  require_cpu_float_tensor(tail_key, "tail key", operation);
  require_cpu_float_tensor(tail_value, "tail value", operation);
  HELIX_CHECK(tail_key.rank() == 3 && tail_value.rank() == 3, ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail key and value", "must have rank three"));
  HELIX_CHECK(tail_key.shape() == tail_value.shape(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail key and value", "must have equal shapes"));
  HELIX_CHECK(query.shape() == tail_key.shape(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "query and tail", "must have equal shapes"));
  HELIX_CHECK(tail_key.shape()[0] == static_cast<Dimension>(dimensions.heads) &&
                  tail_key.shape()[2] == static_cast<Dimension>(dimensions.head_dimension),
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail and KV cache", "must have equal geometry"));
  dimensions.tail_keys = static_cast<std::size_t>(tail_key.shape()[1]);
  HELIX_CHECK(dimensions.tail_keys != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "tail token count", "must be positive"));
  constexpr auto kMaximumDimension =
      static_cast<std::size_t>(std::numeric_limits<Dimension>::max());
  HELIX_CHECK(dimensions.history_keys <= kMaximumDimension - dimensions.tail_keys,
              ErrorCode::kResourceExhausted,
              tensor_error(operation, "history plus tail token count", "overflows shape range"));
  dimensions.keys = dimensions.history_keys + dimensions.tail_keys;
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

Shape paged_tail_score_workspace_shape(const Tensor& query,
                                       const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                       const Tensor& tail_key, const Tensor& tail_value) {
  constexpr std::string_view kOperation = "attention::cpu::paged_tail_score_workspace_shape";
  const PagedAttentionDimensions dimensions =
      validate_tail_inputs(query, kv_layer, tail_key, tail_value, kOperation);
  return Shape{static_cast<Dimension>(dimensions.heads), static_cast<Dimension>(dimensions.queries),
               static_cast<Dimension>(dimensions.keys)};
}

std::size_t paged_tail_score_workspace_bytes(const Tensor& query,
                                             const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                             const Tensor& tail_key, const Tensor& tail_value) {
  return checked_nbytes(paged_tail_score_workspace_shape(query, kv_layer, tail_key, tail_value),
                        DType::kFloat32);
}

namespace {

enum class ValueTraversal {
  kDimensionMajor,
  kDimensionTiled,
};

void paged_scaled_dot_product_attention_out_impl(
    const Tensor& query, const kv_cache::PagedKvCache::LayerReadView& kv_layer,
    const Tensor* tail_key, const Tensor* tail_value, std::size_t query_position_offset,
    Tensor& output, Tensor& score_workspace, ValueTraversal value_traversal,
    std::string_view operation) {
  HELIX_ASSERT((tail_key == nullptr) == (tail_value == nullptr),
               "paged attention tail pointers must be both null or both nonnull");
  const bool has_tail = tail_key != nullptr;
  const PagedAttentionDimensions dimensions =
      has_tail ? validate_tail_inputs(query, kv_layer, *tail_key, *tail_value, operation)
               : validate_inputs(query, kv_layer, operation);
  validate_offset(dimensions, query_position_offset, operation);
  require_cpu_float_tensor(output, "output", operation);
  require_cpu_float_tensor(score_workspace, "score workspace", operation);
  require_shape(output,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.head_dimension)},
                "output", operation);
  require_shape(score_workspace,
                Shape{static_cast<Dimension>(dimensions.heads),
                      static_cast<Dimension>(dimensions.queries),
                      static_cast<Dimension>(dimensions.keys)},
                "score workspace", operation);
  require_no_alias(query, output, "query", "output", operation);
  require_no_alias(query, score_workspace, "query", "score workspace", operation);
  require_no_alias(score_workspace, output, "score workspace", "output", operation);
  if (has_tail) {
    require_no_alias(*tail_key, output, "tail key", "output", operation);
    require_no_alias(*tail_value, output, "tail value", "output", operation);
    require_no_alias(*tail_key, score_workspace, "tail key", "score workspace", operation);
    require_no_alias(*tail_value, score_workspace, "tail value", "score workspace", operation);
  }

  if (dimensions.queries == 0) {
    return;
  }

  const auto* query_values = query.data_as<float>();
  const float* tail_key_values = has_tail ? tail_key->data_as<float>() : nullptr;
  const float* tail_value_values = has_tail ? tail_value->data_as<float>() : nullptr;
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
    HELIX_ASSERT(key_start == dimensions.history_keys,
                 "paged attention block traversal did not cover the cached history");
    if (has_tail) {
      for (std::size_t query_index = 0; query_index < dimensions.queries; ++query_index) {
        const std::size_t query_offset =
            (head * dimensions.queries + query_index) * dimensions.head_dimension;
        const std::size_t score_offset =
            (head * dimensions.queries + query_index) * dimensions.keys;
        const std::size_t query_position = query_position_offset + query_index;
        for (std::size_t token = 0; token < dimensions.tail_keys; ++token) {
          const std::size_t key_index = dimensions.history_keys + token;
          if (key_index > query_position) {
            scores[score_offset + key_index] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const std::size_t key_offset =
              (head * dimensions.tail_keys + token) * dimensions.head_dimension;
          double dot_product = 0.0;
          for (std::size_t dimension = 0; dimension < dimensions.head_dimension; ++dimension) {
            dot_product += static_cast<double>(query_values[query_offset + dimension]) *
                           static_cast<double>(tail_key_values[key_offset + dimension]);
          }
          scores[score_offset + key_index] = static_cast<float>(dot_product * scale);
        }
      }
    }
  }

  helix::cpu::softmax_out(score_workspace, score_workspace);

  // Keep the accumulator allocation-free while traversing values in their
  // contiguous [token, head_dimension] order.  A dimension-at-a-time loop
  // would reacquire every physical block and read each token with a wide
  // stride.  Fixed-size tiles amortize block-table traversal without making
  // head_dimension part of the stack footprint.
  constexpr std::size_t kValueDimensionTile = 32;
  for (std::size_t head = 0; head < dimensions.heads; ++head) {
    for (std::size_t query_index = 0; query_index < dimensions.queries; ++query_index) {
      const std::size_t score_offset = (head * dimensions.queries + query_index) * dimensions.keys;
      const std::size_t output_offset =
          (head * dimensions.queries + query_index) * dimensions.head_dimension;
      const std::size_t visible_keys = query_position_offset + query_index + 1;
      if (value_traversal == ValueTraversal::kDimensionMajor) {
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
              if (key_index >= visible_keys) {
                continue;
              }
              const std::size_t value_offset = token * dimensions.head_dimension + dimension;
              accumulator += static_cast<double>(scores[score_offset + key_index]) *
                             static_cast<double>(values[value_offset]);
            }
            value_start += tokens_in_block;
          }
          HELIX_ASSERT(value_start == dimensions.history_keys,
                       "paged attention value traversal did not cover the cached history");
          if (has_tail) {
            for (std::size_t token = 0; token < dimensions.tail_keys; ++token) {
              const std::size_t key_index = dimensions.history_keys + token;
              if (key_index >= visible_keys) {
                continue;
              }
              const std::size_t value_offset =
                  (head * dimensions.tail_keys + token) * dimensions.head_dimension + dimension;
              accumulator += static_cast<double>(scores[score_offset + key_index]) *
                             static_cast<double>(tail_value_values[value_offset]);
            }
          }
          output_values[output_offset + dimension] = static_cast<float>(accumulator);
        }
        continue;
      }

      for (std::size_t dimension_start = 0; dimension_start < dimensions.head_dimension;
           dimension_start += kValueDimensionTile) {
        const std::size_t tile_dimensions =
            std::min(kValueDimensionTile, dimensions.head_dimension - dimension_start);
        std::array<double, kValueDimensionTile> accumulators{};
        std::size_t value_start = 0;
        for (std::size_t logical_block = 0; logical_block < dimensions.logical_blocks;
             ++logical_block) {
          const std::size_t tokens_in_block = kv_layer.block_token_count(logical_block);
          const std::span<const float> values = kv_layer.value_block(logical_block, head);
          HELIX_ASSERT(values.size() == tokens_in_block * dimensions.head_dimension,
                       "paged attention value span has an inconsistent size");
          for (std::size_t token = 0; token < tokens_in_block; ++token) {
            const std::size_t key_index = value_start + token;
            if (key_index >= visible_keys) {
              continue;
            }
            const std::size_t value_offset = token * dimensions.head_dimension + dimension_start;
            const double probability = static_cast<double>(scores[score_offset + key_index]);
            for (std::size_t tile_dimension = 0; tile_dimension < tile_dimensions;
                 ++tile_dimension) {
              accumulators[tile_dimension] +=
                  probability * static_cast<double>(values[value_offset + tile_dimension]);
            }
          }
          value_start += tokens_in_block;
        }
        HELIX_ASSERT(value_start == dimensions.history_keys,
                     "paged attention value traversal did not cover the cached history");
        if (has_tail) {
          for (std::size_t token = 0; token < dimensions.tail_keys; ++token) {
            const std::size_t key_index = dimensions.history_keys + token;
            if (key_index >= visible_keys) {
              continue;
            }
            const std::size_t value_offset =
                (head * dimensions.tail_keys + token) * dimensions.head_dimension + dimension_start;
            const double probability = static_cast<double>(scores[score_offset + key_index]);
            for (std::size_t tile_dimension = 0; tile_dimension < tile_dimensions;
                 ++tile_dimension) {
              accumulators[tile_dimension] +=
                  probability *
                  static_cast<double>(tail_value_values[value_offset + tile_dimension]);
            }
          }
        }
        for (std::size_t tile_dimension = 0; tile_dimension < tile_dimensions; ++tile_dimension) {
          output_values[output_offset + dimension_start + tile_dimension] =
              static_cast<float>(accumulators[tile_dimension]);
        }
      }
    }
  }
}

} // namespace

void paged_scaled_dot_product_attention_out(const Tensor& query,
                                            const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                            std::size_t query_position_offset, Tensor& output,
                                            Tensor& score_workspace) {
  constexpr std::string_view kOperation = "attention::cpu::paged_scaled_dot_product_attention_out";
  paged_scaled_dot_product_attention_out_impl(query, kv_layer, nullptr, nullptr,
                                              query_position_offset, output, score_workspace,
                                              ValueTraversal::kDimensionTiled, kOperation);
}

void paged_scaled_dot_product_attention_with_tail_out(
    const Tensor& query, const kv_cache::PagedKvCache::LayerReadView& kv_layer,
    const Tensor& tail_key, const Tensor& tail_value, Tensor& output, Tensor& score_workspace) {
  constexpr std::string_view kOperation =
      "attention::cpu::paged_scaled_dot_product_attention_with_tail_out";
  paged_scaled_dot_product_attention_out_impl(query, kv_layer, &tail_key, &tail_value,
                                              kv_layer.length(), output, score_workspace,
                                              ValueTraversal::kDimensionTiled, kOperation);
}

void paged_scaled_dot_product_attention_dimension_major_out(
    const Tensor& query, const kv_cache::PagedKvCache::LayerReadView& kv_layer,
    std::size_t query_position_offset, Tensor& output, Tensor& score_workspace) {
  constexpr std::string_view kOperation =
      "attention::cpu::paged_scaled_dot_product_attention_dimension_major_out";
  paged_scaled_dot_product_attention_out_impl(query, kv_layer, nullptr, nullptr,
                                              query_position_offset, output, score_workspace,
                                              ValueTraversal::kDimensionMajor, kOperation);
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

Tensor
paged_scaled_dot_product_attention_with_tail(const Tensor& query,
                                             const kv_cache::PagedKvCache::LayerReadView& kv_layer,
                                             const Tensor& tail_key, const Tensor& tail_value) {
  constexpr std::string_view kOperation =
      "attention::cpu::paged_scaled_dot_product_attention_with_tail";
  const PagedAttentionDimensions dimensions =
      validate_tail_inputs(query, kv_layer, tail_key, tail_value, kOperation);
  Tensor output = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                 static_cast<Dimension>(dimensions.queries),
                                 static_cast<Dimension>(dimensions.head_dimension)});
  Tensor workspace = Tensor::empty({static_cast<Dimension>(dimensions.heads),
                                    static_cast<Dimension>(dimensions.queries),
                                    static_cast<Dimension>(dimensions.keys)});
  paged_scaled_dot_product_attention_with_tail_out(query, kv_layer, tail_key, tail_value, output,
                                                   workspace);
  return output;
}

} // namespace helix::attention::cpu
