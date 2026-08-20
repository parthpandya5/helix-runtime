#include "helix/core/error.hpp"
#include "helix/kernels/cuda/elementwise.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <sstream>
#include <string_view>

namespace helix::cuda::kernels {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65'535;

__global__ void add_kernel(const float* lhs, const float* rhs, float* output,
                           std::size_t elements) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; index < elements; index += stride) {
    output[index] = lhs[index] + rhs[index];
  }
}

__global__ void multiply_kernel(const float* lhs, const float* rhs, float* output,
                                std::size_t elements) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; index < elements; index += stride) {
    output[index] = lhs[index] * rhs[index];
  }
}

__global__ void relu_kernel(const float* input, float* output, std::size_t elements) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; index < elements; index += stride) {
    output[index] = input[index] > 0.0F ? input[index] : 0.0F;
  }
}

__device__ float stable_silu(float value) {
  if (value >= 0.0F) {
    return value / (1.0F + expf(-value));
  }
  const float exponential = expf(value);
  return value * exponential / (1.0F + exponential);
}

__global__ void silu_kernel(const float* input, float* output, std::size_t elements) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; index < elements; index += stride) {
    output[index] = stable_silu(input[index]);
  }
}

std::string tensor_error(std::string_view operation, std::string_view name,
                         std::string_view requirement) {
  std::ostringstream message;
  message << operation << ": " << name << ' ' << requirement;
  return message.str();
}

void require_cuda_tensor(const Tensor& tensor, std::string_view name, std::string_view operation) {
  HELIX_CHECK(tensor.device().is_cuda(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on CUDA"));
  HELIX_CHECK(tensor.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must have dtype float32"));
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be contiguous"));
}

void require_stream_device(const Tensor& tensor, const Stream& stream, std::string_view operation) {
  HELIX_CHECK(tensor.device().index() == stream.device_index(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must belong to the tensor's CUDA device"));
}

void require_output(const Tensor& input, const Tensor& output, const Stream& stream,
                    std::string_view operation) {
  require_cuda_tensor(output, "output", operation);
  HELIX_CHECK(output.shape() == input.shape(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must have the input shape"));
  HELIX_CHECK(output.device() == input.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must be on the input device"));
  require_stream_device(output, stream, operation);
}

void require_binary(const Tensor& lhs, const Tensor& rhs, const Tensor& output,
                    const Stream& stream, std::string_view operation) {
  require_cuda_tensor(lhs, "lhs", operation);
  require_cuda_tensor(rhs, "rhs", operation);
  HELIX_CHECK(
      lhs.shape() == rhs.shape(), ErrorCode::kInvalidArgument,
      tensor_error(operation, "inputs", "must have equal shapes; broadcasting is unsupported"));
  HELIX_CHECK(lhs.device() == rhs.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "inputs", "must be on the same CUDA device"));
  require_output(lhs, output, stream, operation);
}

unsigned int block_count(std::size_t elements) {
  HELIX_ASSERT(elements != 0, "cannot launch an elementwise kernel for zero elements");
  const std::size_t required_blocks =
      1 + (elements - 1) / static_cast<std::size_t>(kThreadsPerBlock);
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

void add_out(const Tensor& lhs, const Tensor& rhs, Tensor& output, const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::add_out";
  require_binary(lhs, rhs, output, stream, kOperation);
  if (lhs.numel() == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  add_kernel<<<block_count(lhs.numel()), kThreadsPerBlock, 0, stream.native_handle()>>>(
      lhs.data_as<float>(), rhs.data_as<float>(), output.data_as<float>(), lhs.numel());
  check_launch("add_kernel");
}

Tensor add(const Tensor& lhs, const Tensor& rhs, const Stream& stream) {
  Tensor output = Tensor::empty(lhs.shape(), DType::kFloat32, lhs.device());
  add_out(lhs, rhs, output, stream);
  return output;
}

void multiply_out(const Tensor& lhs, const Tensor& rhs, Tensor& output, const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::multiply_out";
  require_binary(lhs, rhs, output, stream, kOperation);
  if (lhs.numel() == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  multiply_kernel<<<block_count(lhs.numel()), kThreadsPerBlock, 0, stream.native_handle()>>>(
      lhs.data_as<float>(), rhs.data_as<float>(), output.data_as<float>(), lhs.numel());
  check_launch("multiply_kernel");
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs, const Stream& stream) {
  Tensor output = Tensor::empty(lhs.shape(), DType::kFloat32, lhs.device());
  multiply_out(lhs, rhs, output, stream);
  return output;
}

void relu_out(const Tensor& input, Tensor& output, const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::relu_out";
  require_cuda_tensor(input, "input", kOperation);
  require_output(input, output, stream, kOperation);
  if (input.numel() == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  relu_kernel<<<block_count(input.numel()), kThreadsPerBlock, 0, stream.native_handle()>>>(
      input.data_as<float>(), output.data_as<float>(), input.numel());
  check_launch("relu_kernel");
}

Tensor relu(const Tensor& input, const Stream& stream) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, input.device());
  relu_out(input, output, stream);
  return output;
}

void silu_out(const Tensor& input, Tensor& output, const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::silu_out";
  require_cuda_tensor(input, "input", kOperation);
  require_output(input, output, stream, kOperation);
  if (input.numel() == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());
  silu_kernel<<<block_count(input.numel()), kThreadsPerBlock, 0, stream.native_handle()>>>(
      input.data_as<float>(), output.data_as<float>(), input.numel());
  check_launch("silu_kernel");
}

Tensor silu(const Tensor& input, const Stream& stream) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, input.device());
  silu_out(input, output, stream);
  return output;
}

} // namespace helix::cuda::kernels
