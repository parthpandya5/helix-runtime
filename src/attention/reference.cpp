#include "helix/attention/reference.hpp"

#include "helix/core/error.hpp"
#include "helix/kernels/cpu/reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace helix::attention::cpu {
namespace {

struct AttentionDimensions {
  std::size_t heads;
  std::size_t queries;
  std::size_t keys;
  std::size_t head_dimension;
  std::size_t value_dimension;
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

AttentionDimensions validate_inputs(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::string_view operation) {
  require_cpu_float_tensor(query, "query", operation);
  require_cpu_float_tensor(key, "key", operation);
  require_cpu_float_tensor(value, "value", operation);
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
  HELIX_CHECK(query.shape()[2] > 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "head dimension", "must be positive"));
  HELIX_CHECK(key.shape()[1] > 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "KV token count", "must be positive"));
  HELIX_CHECK(value.shape()[2] > 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "value dimension", "must be positive"));
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

} // namespace

Shape score_workspace_shape(const Tensor& query, const Tensor& key) {
  constexpr std::string_view kOperation = "attention::cpu::score_workspace_shape";
  require_cpu_float_tensor(query, "query", kOperation);
  require_cpu_float_tensor(key, "key", kOperation);
  HELIX_CHECK(query.rank() == 3 && key.rank() == 3, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "query and key", "must have rank three"));
  HELIX_CHECK(query.shape()[0] == key.shape()[0], ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "query and key", "must have equal head counts"));
  HELIX_CHECK(query.shape()[2] == key.shape()[2], ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "query and key", "must have equal head dimensions"));
  HELIX_CHECK(query.shape()[2] > 0 && key.shape()[1] > 0, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "head dimension and KV token count", "must be positive"));
  return Shape{query.shape()[0], query.shape()[1], key.shape()[1]};
}

std::size_t score_workspace_bytes(const Tensor& query, const Tensor& key) {
  return checked_nbytes(score_workspace_shape(query, key), DType::kFloat32);
}

namespace {

enum class ValueTraversal {
  kDimensionMajor,
  kDimensionTiled,
};

void scaled_dot_product_attention_out_impl(const Tensor& query, const Tensor& key,
                                           const Tensor& value, std::size_t query_position_offset,
                                           Tensor& output, Tensor& score_workspace,
                                           ValueTraversal value_traversal) {
  constexpr std::string_view kOperation = "attention::cpu::scaled_dot_product_attention_out";
  const AttentionDimensions dimensions = validate_inputs(query, key, value, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  require_cpu_float_tensor(output, "output", kOperation);
  require_cpu_float_tensor(score_workspace, "score workspace", kOperation);
  require_shape(output, Shape{query.shape()[0], query.shape()[1], value.shape()[2]}, "output",
                kOperation);
  require_shape(score_workspace, Shape{query.shape()[0], query.shape()[1], key.shape()[1]},
                "score workspace", kOperation);
  require_no_alias(query, output, "query", "output", kOperation);
  require_no_alias(key, output, "key", "output", kOperation);
  require_no_alias(value, output, "value", "output", kOperation);
  require_no_alias(query, score_workspace, "query", "score workspace", kOperation);
  require_no_alias(key, score_workspace, "key", "score workspace", kOperation);
  require_no_alias(value, score_workspace, "value", "score workspace", kOperation);
  require_no_alias(score_workspace, output, "score workspace", "output", kOperation);

  if (dimensions.heads == 0 || dimensions.queries == 0) {
    return;
  }

  const auto* query_values = query.data_as<float>();
  const auto* key_values = key.data_as<float>();
  const auto* value_values = value.data_as<float>();
  auto* scores = score_workspace.data_as<float>();
  auto* output_values = output.data_as<float>();
  const double scale = 1.0 / std::sqrt(static_cast<double>(dimensions.head_dimension));

  for (std::size_t head = 0; head < dimensions.heads; ++head) {
    for (std::size_t query_index = 0; query_index < dimensions.queries; ++query_index) {
      const std::size_t query_offset =
          (head * dimensions.queries + query_index) * dimensions.head_dimension;
      const std::size_t score_offset = (head * dimensions.queries + query_index) * dimensions.keys;
      const std::size_t query_position = query_position_offset + query_index;
      for (std::size_t key_index = 0; key_index < dimensions.keys; ++key_index) {
        if (key_index > query_position) {
          scores[score_offset + key_index] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const std::size_t key_offset =
            (head * dimensions.keys + key_index) * dimensions.head_dimension;
        double dot_product = 0.0;
        for (std::size_t dimension = 0; dimension < dimensions.head_dimension; ++dimension) {
          dot_product += static_cast<double>(query_values[query_offset + dimension]) *
                         static_cast<double>(key_values[key_offset + dimension]);
        }
        scores[score_offset + key_index] = static_cast<float>(dot_product * scale);
      }
    }
  }

  helix::cpu::softmax_out(score_workspace, score_workspace);

  constexpr std::size_t kValueDimensionTile = 32;
  for (std::size_t head = 0; head < dimensions.heads; ++head) {
    for (std::size_t query_index = 0; query_index < dimensions.queries; ++query_index) {
      const std::size_t score_offset = (head * dimensions.queries + query_index) * dimensions.keys;
      const std::size_t output_offset =
          (head * dimensions.queries + query_index) * dimensions.value_dimension;
      const std::size_t visible_keys = query_position_offset + query_index + 1;
      if (value_traversal == ValueTraversal::kDimensionMajor) {
        for (std::size_t dimension = 0; dimension < dimensions.value_dimension; ++dimension) {
          double accumulator = 0.0;
          for (std::size_t key_index = 0; key_index < visible_keys; ++key_index) {
            const std::size_t value_offset =
                (head * dimensions.keys + key_index) * dimensions.value_dimension;
            accumulator += static_cast<double>(scores[score_offset + key_index]) *
                           static_cast<double>(value_values[value_offset + dimension]);
          }
          output_values[output_offset + dimension] = static_cast<float>(accumulator);
        }
        continue;
      }

      // Accumulate a fixed-size coordinate tile while each row of V is hot.
      // This preserves each coordinate's KV accumulation order and avoids a
      // head-dimension-sized dynamic allocation.
      for (std::size_t dimension_start = 0; dimension_start < dimensions.value_dimension;
           dimension_start += kValueDimensionTile) {
        const std::size_t tile_dimensions =
            std::min(kValueDimensionTile, dimensions.value_dimension - dimension_start);
        std::array<double, kValueDimensionTile> accumulators{};
        for (std::size_t key_index = 0; key_index < visible_keys; ++key_index) {
          const std::size_t value_offset =
              (head * dimensions.keys + key_index) * dimensions.value_dimension + dimension_start;
          const double probability = static_cast<double>(scores[score_offset + key_index]);
          for (std::size_t tile_dimension = 0; tile_dimension < tile_dimensions; ++tile_dimension) {
            accumulators[tile_dimension] +=
                probability * static_cast<double>(value_values[value_offset + tile_dimension]);
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

void scaled_dot_product_attention_out(const Tensor& query, const Tensor& key, const Tensor& value,
                                      std::size_t query_position_offset, Tensor& output,
                                      Tensor& score_workspace) {
  scaled_dot_product_attention_out_impl(query, key, value, query_position_offset, output,
                                        score_workspace, ValueTraversal::kDimensionTiled);
}

void scaled_dot_product_attention_dimension_major_out(const Tensor& query, const Tensor& key,
                                                      const Tensor& value,
                                                      std::size_t query_position_offset,
                                                      Tensor& output, Tensor& score_workspace) {
  scaled_dot_product_attention_out_impl(query, key, value, query_position_offset, output,
                                        score_workspace, ValueTraversal::kDimensionMajor);
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    std::size_t query_position_offset) {
  constexpr std::string_view kOperation = "attention::cpu::scaled_dot_product_attention";
  const AttentionDimensions dimensions = validate_inputs(query, key, value, kOperation);
  validate_offset(dimensions, query_position_offset, kOperation);
  Tensor output = Tensor::empty({query.shape()[0], query.shape()[1], value.shape()[2]},
                                DType::kFloat32, Device::cpu());
  Tensor score_workspace = Tensor::empty({query.shape()[0], query.shape()[1], key.shape()[1]},
                                         DType::kFloat32, Device::cpu());
  scaled_dot_product_attention_out(query, key, value, query_position_offset, output,
                                   score_workspace);
  return output;
}

} // namespace helix::attention::cpu
