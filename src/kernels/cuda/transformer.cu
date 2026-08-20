#include "helix/core/error.hpp"
#include "helix/kernels/cuda/transformer.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace helix::cuda::kernels {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kWarpSize = 32;
constexpr unsigned int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr std::size_t kMaximumBlocks = 65'535;

__device__ float warp_sum(float value) {
  constexpr unsigned int kActiveMask = 0xFFFF'FFFFU;
  for (int offset = static_cast<int>(kWarpSize / 2); offset > 0; offset /= 2) {
    value += __shfl_down_sync(kActiveMask, value, offset);
  }
  return value;
}

__global__ void rms_norm_kernel(const float* input, const float* weight, float epsilon,
                                float* output, std::size_t rows, std::size_t width) {
  __shared__ float warp_sums[kWarpsPerBlock];
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp = threadIdx.x / kWarpSize;

  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const std::size_t row_offset = row * width;
    float sum_of_squares = 0.0F;
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
      const float value = input[row_offset + column];
      sum_of_squares += value * value;
    }
    sum_of_squares = warp_sum(sum_of_squares);
    if (lane == 0) {
      warp_sums[warp] = sum_of_squares;
    }
    __syncthreads();

    if (warp == 0) {
      float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
      block_sum = warp_sum(block_sum);
      if (lane == 0) {
        warp_sums[0] = rsqrtf(block_sum / static_cast<float>(width) + epsilon);
      }
    }
    __syncthreads();

    const float inverse_rms = warp_sums[0];
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
      output[row_offset + column] = input[row_offset + column] * inverse_rms * weight[column];
    }
    __syncthreads();
  }
}

__device__ float stable_silu(float value) {
  if (value >= 0.0F) {
    return value / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return value * exponential / (1.0F + exponential);
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* output,
                              std::size_t elements) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; index < elements; index += stride) {
    output[index] = stable_silu(gate[index]) * up[index];
  }
}

__global__ void rope_kernel(const float* input, const float* cosine, const float* sine,
                            float* output, std::size_t pair_tasks, std::size_t vectors_per_token,
                            std::size_t pairs) {
  std::size_t task = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t head_dimension = pairs * 2;
  for (; task < pair_tasks; task += stride) {
    const std::size_t pair = task % pairs;
    const std::size_t vector_index = task / pairs;
    const std::size_t token = vector_index / vectors_per_token;
    const std::size_t input_offset = vector_index * head_dimension + pair * 2;
    const std::size_t cache_offset = token * pairs + pair;
    const float first = input[input_offset];
    const float second = input[input_offset + 1];
    const float pair_cosine = cosine[cache_offset];
    const float pair_sine = sine[cache_offset];
    output[input_offset] = first * pair_cosine - second * pair_sine;
    output[input_offset + 1] = first * pair_sine + second * pair_cosine;
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

void require_same_device(const Tensor& tensor, const Tensor& reference, std::string_view name,
                         std::string_view operation) {
  HELIX_CHECK(tensor.device() == reference.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on the input device"));
}

void require_stream(const Tensor& input, const Stream& stream, std::string_view operation) {
  HELIX_CHECK(stream.native_handle() != nullptr, ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must not be moved-from"));
  HELIX_CHECK(stream.device_index() == input.device().index(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must belong to the input device"));
}

void require_no_alias(const Tensor& input, const Tensor& output, std::string_view input_name,
                      std::string_view operation) {
  HELIX_CHECK(
      input.nbytes() == 0 || input.data() != output.data(), ErrorCode::kInvalidArgument,
      tensor_error(operation, "output", std::string("must not alias ") + std::string(input_name)));
}

unsigned int block_count(std::size_t work_items, bool block_per_item = false) {
  HELIX_ASSERT(work_items != 0, "cannot select a CUDA grid for zero work items");
  std::size_t required_blocks = work_items;
  if (!block_per_item) {
    required_blocks = 1 + (work_items - 1) / static_cast<std::size_t>(kThreadsPerBlock);
  }
  const std::size_t blocks = std::min(required_blocks, kMaximumBlocks);
  HELIX_ASSERT(blocks <= std::numeric_limits<unsigned int>::max(),
               "CUDA grid dimension does not fit unsigned int");
  return static_cast<unsigned int>(blocks);
}

void check_launch(std::string_view kernel_name) {
  std::ostringstream context;
  context << "launching " << kernel_name;
  HELIX_CUDA_CHECK_CONTEXT(cudaGetLastError(), context.str());
}

} // namespace

void rms_norm_out(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output,
                  const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::rms_norm_out";
  require_cuda_float_tensor(input, "input", kOperation);
  require_cuda_float_tensor(weight, "weight", kOperation);
  require_cuda_float_tensor(output, "output", kOperation);
  HELIX_CHECK(input.rank() != 0, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "input", "must have rank at least one"));
  HELIX_CHECK(weight.rank() == 1 && weight.shape()[0] == input.shape().back(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "weight", "must match the input's last dimension"));
  HELIX_CHECK(output.shape() == input.shape(), ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "output", "must have the input shape"));
  require_same_device(weight, input, "weight", kOperation);
  require_same_device(output, input, "output", kOperation);
  require_stream(input, stream, kOperation);
  HELIX_CHECK(std::isfinite(epsilon) && epsilon > 0.0F, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "epsilon", "must be finite and positive"));
  require_no_alias(weight, output, "weight", kOperation);

  const auto width = static_cast<std::size_t>(input.shape().back());
  HELIX_CHECK(width != 0, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "last dimension", "must be non-zero"));
  const std::size_t rows = input.numel() / width;
  if (rows == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  rms_norm_kernel<<<block_count(rows, true), kThreadsPerBlock, 0, stream.native_handle()>>>(
      input.data_as<float>(), weight.data_as<float>(), epsilon, output.data_as<float>(), rows,
      width);
  check_launch("rms_norm_kernel");
}

Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon, const Stream& stream) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, input.device());
  rms_norm_out(input, weight, epsilon, output, stream);
  return output;
}

void swiglu_out(const Tensor& gate, const Tensor& up, Tensor& output, const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::swiglu_out";
  require_cuda_float_tensor(gate, "gate", kOperation);
  require_cuda_float_tensor(up, "up", kOperation);
  require_cuda_float_tensor(output, "output", kOperation);
  HELIX_CHECK(up.shape() == gate.shape() && output.shape() == gate.shape(),
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "gate, up, and output", "must have equal shapes"));
  require_same_device(up, gate, "up", kOperation);
  require_same_device(output, gate, "output", kOperation);
  require_stream(gate, stream, kOperation);
  if (gate.numel() == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  swiglu_kernel<<<block_count(gate.numel()), kThreadsPerBlock, 0, stream.native_handle()>>>(
      gate.data_as<float>(), up.data_as<float>(), output.data_as<float>(), gate.numel());
  check_launch("swiglu_kernel");
}

Tensor swiglu(const Tensor& gate, const Tensor& up, const Stream& stream) {
  Tensor output = Tensor::empty(gate.shape(), DType::kFloat32, gate.device());
  swiglu_out(gate, up, output, stream);
  return output;
}

void rope_out(const Tensor& input, const Tensor& cosine, const Tensor& sine, Tensor& output,
              const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::rope_out";
  require_cuda_float_tensor(input, "input", kOperation);
  require_cuda_float_tensor(cosine, "cosine", kOperation);
  require_cuda_float_tensor(sine, "sine", kOperation);
  require_cuda_float_tensor(output, "output", kOperation);
  HELIX_CHECK(input.rank() >= 2, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "input", "must have rank at least two"));
  const auto tokens = static_cast<std::size_t>(input.shape().front());
  const auto head_dimension = static_cast<std::size_t>(input.shape().back());
  HELIX_CHECK(head_dimension != 0 && head_dimension % 2 == 0, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "last dimension", "must be positive and even"));
  const std::size_t pairs = head_dimension / 2;
  const Shape cache_shape{input.shape().front(), static_cast<Dimension>(pairs)};
  HELIX_CHECK(cosine.shape() == cache_shape && sine.shape() == cache_shape,
              ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "cosine and sine", "must have shape [tokens, head_dim/2]"));
  HELIX_CHECK(output.shape() == input.shape(), ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "output", "must have the input shape"));
  require_same_device(cosine, input, "cosine", kOperation);
  require_same_device(sine, input, "sine", kOperation);
  require_same_device(output, input, "output", kOperation);
  require_stream(input, stream, kOperation);
  require_no_alias(cosine, output, "cosine", kOperation);
  require_no_alias(sine, output, "sine", kOperation);

  const std::size_t pair_tasks = input.numel() / 2;
  if (pair_tasks == 0) {
    return;
  }
  HELIX_ASSERT(tokens != 0, "nonempty RoPE input must contain a token");
  const std::size_t vectors_per_token = (input.numel() / tokens) / head_dimension;
  HELIX_ASSERT(vectors_per_token != 0, "nonempty RoPE input must contain a vector per token");
  DeviceGuard guard(stream.device_index());
  rope_kernel<<<block_count(pair_tasks), kThreadsPerBlock, 0, stream.native_handle()>>>(
      input.data_as<float>(), cosine.data_as<float>(), sine.data_as<float>(),
      output.data_as<float>(), pair_tasks, vectors_per_token, pairs);
  check_launch("rope_kernel");
}

Tensor rope(const Tensor& input, const Tensor& cosine, const Tensor& sine, const Stream& stream) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, input.device());
  rope_out(input, cosine, sine, output, stream);
  return output;
}

void linear_out(const Tensor& input, const Tensor& weight, Tensor& output, const Stream& stream,
                MatmulAlgorithm algorithm) {
  matmul_out(input, weight, output, stream, algorithm);
}

Tensor linear(const Tensor& input, const Tensor& weight, const Stream& stream,
              MatmulAlgorithm algorithm) {
  return matmul(input, weight, stream, algorithm);
}

} // namespace helix::cuda::kernels
