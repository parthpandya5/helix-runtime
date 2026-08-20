#include "helix/kernels/cpu/reference.hpp"

#include "helix/core/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace helix::cpu {
namespace {

std::string tensor_contract_message(std::string_view operation, std::string_view tensor_name,
                                    std::string_view requirement) {
  std::ostringstream message;
  message << operation << ": " << tensor_name << ' ' << requirement;
  return message.str();
}

void require_reference_tensor(const Tensor& tensor, std::string_view tensor_name,
                              std::string_view operation) {
  HELIX_CHECK(tensor.device().is_cpu(), ErrorCode::kInvalidArgument,
              tensor_contract_message(operation, tensor_name, "must be on CPU"));
  HELIX_CHECK(tensor.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              tensor_contract_message(operation, tensor_name, "must have dtype float32"));
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              tensor_contract_message(operation, tensor_name, "must be contiguous"));
}

void require_shape(const Tensor& tensor, const Shape& expected, std::string_view tensor_name,
                   std::string_view operation) {
  if (tensor.shape() != expected) {
    std::ostringstream message;
    message << operation << ": " << tensor_name << " shape " << shape_string(tensor.shape())
            << " does not match expected shape " << shape_string(expected);
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
}

void require_same_shape(const Tensor& lhs, const Tensor& rhs, std::string_view operation) {
  if (lhs.shape() != rhs.shape()) {
    std::ostringstream message;
    message << operation << ": input shapes differ: " << shape_string(lhs.shape()) << " and "
            << shape_string(rhs.shape()) << "; broadcasting is not supported";
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
}

void require_output(const Tensor& output, const Shape& expected_shape, std::string_view operation) {
  require_reference_tensor(output, "output", operation);
  require_shape(output, expected_shape, "output", operation);
}

template <typename Function>
void unary_out(const Tensor& input, Tensor& output, std::string_view operation,
               Function&& function) {
  require_reference_tensor(input, "input", operation);
  require_output(output, input.shape(), operation);

  const auto* input_values = input.data_as<float>();
  auto* output_values = output.data_as<float>();
  for (std::size_t index = 0; index < input.numel(); ++index) {
    output_values[index] = function(input_values[index]);
  }
}

template <typename Function>
void binary_out(const Tensor& lhs, const Tensor& rhs, Tensor& output, std::string_view operation,
                Function&& function) {
  require_reference_tensor(lhs, "lhs", operation);
  require_reference_tensor(rhs, "rhs", operation);
  require_same_shape(lhs, rhs, operation);
  require_output(output, lhs.shape(), operation);

  const auto* lhs_values = lhs.data_as<float>();
  const auto* rhs_values = rhs.data_as<float>();
  auto* output_values = output.data_as<float>();
  for (std::size_t index = 0; index < lhs.numel(); ++index) {
    output_values[index] = function(lhs_values[index], rhs_values[index]);
  }
}

float stable_silu(float value) {
  const double wide_value = static_cast<double>(value);
  double sigmoid = 0.0;
  if (value >= 0.0F) {
    sigmoid = 1.0 / (1.0 + std::exp(-wide_value));
  } else {
    const double exponential = std::exp(wide_value);
    sigmoid = exponential / (1.0 + exponential);
  }
  return static_cast<float>(wide_value * sigmoid);
}

Shape reduced_last_dim_shape(const Tensor& input, bool keep_dim, std::string_view operation) {
  HELIX_CHECK(input.rank() != 0, ErrorCode::kInvalidArgument,
              tensor_contract_message(operation, "input", "must have rank at least one"));
  Shape result = input.shape();
  if (keep_dim) {
    result.back() = 1;
  } else {
    result.pop_back();
  }
  return result;
}

void require_distinct_output(const Tensor& input, const Tensor& output, std::string_view operation,
                             std::string_view input_name) {
  if (input.nbytes() != 0 && input.data() == output.data()) {
    std::ostringstream message;
    message << operation << ": output must not alias " << input_name;
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
}

} // namespace

void add_out(const Tensor& lhs, const Tensor& rhs, Tensor& output) {
  binary_out(lhs, rhs, output, "cpu::add_out",
             [](float lhs_value, float rhs_value) { return lhs_value + rhs_value; });
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
  Tensor output = Tensor::empty(lhs.shape(), DType::kFloat32, Device::cpu());
  add_out(lhs, rhs, output);
  return output;
}

void multiply_out(const Tensor& lhs, const Tensor& rhs, Tensor& output) {
  binary_out(lhs, rhs, output, "cpu::multiply_out",
             [](float lhs_value, float rhs_value) { return lhs_value * rhs_value; });
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs) {
  Tensor output = Tensor::empty(lhs.shape(), DType::kFloat32, Device::cpu());
  multiply_out(lhs, rhs, output);
  return output;
}

void relu_out(const Tensor& input, Tensor& output) {
  unary_out(input, output, "cpu::relu_out",
            [](float value) { return value > 0.0F ? value : 0.0F; });
}

Tensor relu(const Tensor& input) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, Device::cpu());
  relu_out(input, output);
  return output;
}

void silu_out(const Tensor& input, Tensor& output) {
  unary_out(input, output, "cpu::silu_out", [](float value) { return stable_silu(value); });
}

Tensor silu(const Tensor& input) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, Device::cpu());
  silu_out(input, output);
  return output;
}

void sum_out(const Tensor& input, Tensor& output) {
  constexpr std::string_view kOperation = "cpu::sum_out";
  require_reference_tensor(input, "input", kOperation);
  require_output(output, Shape{}, kOperation);
  require_distinct_output(input, output, kOperation, "input");

  const auto* input_values = input.data_as<float>();
  double accumulator = 0.0;
  for (std::size_t index = 0; index < input.numel(); ++index) {
    accumulator += static_cast<double>(input_values[index]);
  }
  output.data_as<float>()[0] = static_cast<float>(accumulator);
}

Tensor sum(const Tensor& input) {
  Tensor output = Tensor::empty({}, DType::kFloat32, Device::cpu());
  sum_out(input, output);
  return output;
}

void sum_last_dim_out(const Tensor& input, Tensor& output, bool keep_dim) {
  constexpr std::string_view kOperation = "cpu::sum_last_dim_out";
  require_reference_tensor(input, "input", kOperation);
  const Shape output_shape = reduced_last_dim_shape(input, keep_dim, kOperation);
  require_output(output, output_shape, kOperation);
  require_distinct_output(input, output, kOperation, "input");

  const auto width = static_cast<std::size_t>(input.shape().back());
  const auto rows = output.numel();
  const auto* input_values = input.data_as<float>();
  auto* output_values = output.data_as<float>();
  for (std::size_t row = 0; row < rows; ++row) {
    double accumulator = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      accumulator += static_cast<double>(input_values[row * width + column]);
    }
    output_values[row] = static_cast<float>(accumulator);
  }
}

Tensor sum_last_dim(const Tensor& input, bool keep_dim) {
  const Shape output_shape = reduced_last_dim_shape(input, keep_dim, "cpu::sum_last_dim");
  Tensor output = Tensor::empty(output_shape, DType::kFloat32, Device::cpu());
  sum_last_dim_out(input, output, keep_dim);
  return output;
}

void softmax_out(const Tensor& input, Tensor& output) {
  constexpr std::string_view kOperation = "cpu::softmax_out";
  require_reference_tensor(input, "input", kOperation);
  require_output(output, input.shape(), kOperation);
  HELIX_CHECK(input.rank() != 0, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "input", "must have rank at least one"));

  const auto width = static_cast<std::size_t>(input.shape().back());
  HELIX_CHECK(width != 0, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "last dimension", "must be non-zero"));
  const auto rows = input.numel() / width;
  const auto* input_values = input.data_as<float>();
  auto* output_values = output.data_as<float>();

  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t row_offset = row * width;
    bool contains_nan = false;
    std::size_t positive_infinities = 0;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t column = 0; column < width; ++column) {
      const float value = input_values[row_offset + column];
      contains_nan = contains_nan || std::isnan(value);
      positive_infinities += static_cast<std::size_t>(std::isinf(value) && value > 0.0F);
      maximum = std::max(maximum, value);
    }

    if (contains_nan || maximum == -std::numeric_limits<float>::infinity()) {
      std::fill_n(output_values + row_offset, width, std::numeric_limits<float>::quiet_NaN());
      continue;
    }

    if (positive_infinities != 0) {
      const float probability = 1.0F / static_cast<float>(positive_infinities);
      for (std::size_t column = 0; column < width; ++column) {
        output_values[row_offset + column] =
            input_values[row_offset + column] == std::numeric_limits<float>::infinity()
                ? probability
                : 0.0F;
      }
      continue;
    }

    double denominator = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      denominator += std::exp(static_cast<double>(input_values[row_offset + column] - maximum));
    }
    for (std::size_t column = 0; column < width; ++column) {
      const double numerator =
          std::exp(static_cast<double>(input_values[row_offset + column] - maximum));
      output_values[row_offset + column] = static_cast<float>(numerator / denominator);
    }
  }
}

Tensor softmax(const Tensor& input) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, Device::cpu());
  softmax_out(input, output);
  return output;
}

void matmul_out(const Tensor& lhs, const Tensor& rhs, Tensor& output) {
  constexpr std::string_view kOperation = "cpu::matmul_out";
  require_reference_tensor(lhs, "lhs", kOperation);
  require_reference_tensor(rhs, "rhs", kOperation);
  HELIX_CHECK(lhs.rank() == 2 && rhs.rank() == 2, ErrorCode::kInvalidArgument,
              "cpu::matmul_out requires rank-two inputs");
  HELIX_CHECK(lhs.shape()[1] == rhs.shape()[0], ErrorCode::kInvalidArgument,
              "cpu::matmul_out inner dimensions do not match");

  const Shape output_shape{lhs.shape()[0], rhs.shape()[1]};
  require_output(output, output_shape, kOperation);
  require_distinct_output(lhs, output, kOperation, "lhs");
  require_distinct_output(rhs, output, kOperation, "rhs");

  const auto rows = static_cast<std::size_t>(lhs.shape()[0]);
  const auto inner = static_cast<std::size_t>(lhs.shape()[1]);
  const auto columns = static_cast<std::size_t>(rhs.shape()[1]);
  if (rows == 0 || inner == 0 || columns == 0) {
    output.zero_();
    return;
  }

  const auto* lhs_values = lhs.data_as<float>();
  const auto* rhs_values = rhs.data_as<float>();
  auto* output_values = output.data_as<float>();
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      double accumulator = 0.0;
      for (std::size_t reduction = 0; reduction < inner; ++reduction) {
        accumulator += static_cast<double>(lhs_values[row * inner + reduction]) *
                       static_cast<double>(rhs_values[reduction * columns + column]);
      }
      output_values[row * columns + column] = static_cast<float>(accumulator);
    }
  }
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
  HELIX_CHECK(lhs.rank() == 2 && rhs.rank() == 2, ErrorCode::kInvalidArgument,
              "cpu::matmul requires rank-two inputs");
  Tensor output = Tensor::empty({lhs.shape()[0], rhs.shape()[1]}, DType::kFloat32, Device::cpu());
  matmul_out(lhs, rhs, output);
  return output;
}

void rms_norm_out(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output) {
  constexpr std::string_view kOperation = "cpu::rms_norm_out";
  require_reference_tensor(input, "input", kOperation);
  require_reference_tensor(weight, "weight", kOperation);
  require_output(output, input.shape(), kOperation);
  HELIX_CHECK(input.rank() != 0, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "input", "must have rank at least one"));
  HELIX_CHECK(weight.rank() == 1, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "weight", "must have rank one"));
  HELIX_CHECK(weight.shape()[0] == input.shape().back(), ErrorCode::kInvalidArgument,
              "cpu::rms_norm_out weight size must equal the input's last dimension");
  HELIX_CHECK(std::isfinite(epsilon) && epsilon > 0.0F, ErrorCode::kInvalidArgument,
              "cpu::rms_norm_out epsilon must be finite and positive");
  require_distinct_output(weight, output, kOperation, "weight");

  const auto width = static_cast<std::size_t>(input.shape().back());
  HELIX_CHECK(width != 0, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "last dimension", "must be non-zero"));
  const auto rows = input.numel() / width;
  const auto* input_values = input.data_as<float>();
  const auto* weights = weight.data_as<float>();
  auto* output_values = output.data_as<float>();

  for (std::size_t row = 0; row < rows; ++row) {
    const auto row_offset = row * width;
    double sum_of_squares = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      const double value = static_cast<double>(input_values[row_offset + column]);
      sum_of_squares += value * value;
    }
    const double inverse_rms =
        1.0 / std::sqrt(sum_of_squares / static_cast<double>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      output_values[row_offset + column] =
          static_cast<float>(static_cast<double>(input_values[row_offset + column]) * inverse_rms *
                             static_cast<double>(weights[column]));
    }
  }
}

Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, Device::cpu());
  rms_norm_out(input, weight, epsilon, output);
  return output;
}

void swiglu_out(const Tensor& gate, const Tensor& up, Tensor& output) {
  binary_out(gate, up, output, "cpu::swiglu_out",
             [](float gate_value, float up_value) { return stable_silu(gate_value) * up_value; });
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
  Tensor output = Tensor::empty(gate.shape(), DType::kFloat32, Device::cpu());
  swiglu_out(gate, up, output);
  return output;
}

void rope_out(const Tensor& input, const Tensor& cosine, const Tensor& sine, Tensor& output) {
  constexpr std::string_view kOperation = "cpu::rope_out";
  require_reference_tensor(input, "input", kOperation);
  require_reference_tensor(cosine, "cosine", kOperation);
  require_reference_tensor(sine, "sine", kOperation);
  require_output(output, input.shape(), kOperation);
  HELIX_CHECK(input.rank() >= 2, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "input", "must have rank at least two"));

  const auto tokens = static_cast<std::size_t>(input.shape().front());
  const auto head_dimension = static_cast<std::size_t>(input.shape().back());
  HELIX_CHECK(head_dimension != 0 && head_dimension % 2 == 0, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "last dimension", "must be positive and even"));
  const std::size_t pairs = head_dimension / 2;
  const Shape cache_shape{input.shape().front(), static_cast<Dimension>(pairs)};
  require_shape(cosine, cache_shape, "cosine", kOperation);
  require_shape(sine, cache_shape, "sine", kOperation);
  require_distinct_output(cosine, output, kOperation, "cosine");
  require_distinct_output(sine, output, kOperation, "sine");

  if (tokens == 0 || input.numel() == 0) {
    return;
  }
  const std::size_t vectors_per_token = (input.numel() / tokens) / head_dimension;
  const auto* input_values = input.data_as<float>();
  const auto* cosine_values = cosine.data_as<float>();
  const auto* sine_values = sine.data_as<float>();
  auto* output_values = output.data_as<float>();

  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t vector = 0; vector < vectors_per_token; ++vector) {
      const std::size_t vector_offset = (token * vectors_per_token + vector) * head_dimension;
      const std::size_t cache_offset = token * pairs;
      for (std::size_t pair = 0; pair < pairs; ++pair) {
        const float first = input_values[vector_offset + 2 * pair];
        const float second = input_values[vector_offset + 2 * pair + 1];
        const float pair_cosine = cosine_values[cache_offset + pair];
        const float pair_sine = sine_values[cache_offset + pair];
        output_values[vector_offset + 2 * pair] = first * pair_cosine - second * pair_sine;
        output_values[vector_offset + 2 * pair + 1] = first * pair_sine + second * pair_cosine;
      }
    }
  }
}

Tensor rope(const Tensor& input, const Tensor& cosine, const Tensor& sine) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, Device::cpu());
  rope_out(input, cosine, sine, output);
  return output;
}

void linear_out(const Tensor& input, const Tensor& weight, Tensor& output) {
  constexpr std::string_view kOperation = "cpu::linear_out";
  require_reference_tensor(input, "input", kOperation);
  require_reference_tensor(weight, "weight", kOperation);
  HELIX_CHECK(input.rank() == 2, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "input", "must have rank two"));
  HELIX_CHECK(weight.rank() == 2, ErrorCode::kInvalidArgument,
              tensor_contract_message(kOperation, "weight", "must have rank two"));
  HELIX_CHECK(input.shape()[1] == weight.shape()[0], ErrorCode::kInvalidArgument,
              "cpu::linear_out input width must equal weight input features");
  require_output(output, Shape{input.shape()[0], weight.shape()[1]}, kOperation);
  matmul_out(input, weight, output);
}

Tensor linear(const Tensor& input, const Tensor& weight) {
  HELIX_CHECK(input.rank() == 2 && weight.rank() == 2, ErrorCode::kInvalidArgument,
              "cpu::linear requires rank-two input and weight");
  Tensor output =
      Tensor::empty({input.shape()[0], weight.shape()[1]}, DType::kFloat32, Device::cpu());
  linear_out(input, weight, output);
  return output;
}

} // namespace helix::cpu
