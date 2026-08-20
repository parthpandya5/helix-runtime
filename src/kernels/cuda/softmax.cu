#include "helix/core/error.hpp"
#include "helix/kernels/cuda/softmax.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>
#include <sstream>
#include <string_view>

namespace helix::cuda::kernels {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65'535;

__global__ void softmax_naive_kernel(const float* input, float* output, std::size_t rows,
                                     std::size_t width) {
  std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; row < rows; row += row_stride) {
    const std::size_t row_offset = row * width;
    bool contains_nan = false;
    std::size_t positive_infinities = 0;
    float maximum = -CUDART_INF_F;
    for (std::size_t column = 0; column < width; ++column) {
      const float value = input[row_offset + column];
      contains_nan = contains_nan || isnan(value);
      positive_infinities += static_cast<std::size_t>(isinf(value) && value > 0.0F);
      maximum = fmaxf(maximum, value);
    }

    if (contains_nan || maximum == -CUDART_INF_F) {
      for (std::size_t column = 0; column < width; ++column) {
        output[row_offset + column] = CUDART_NAN_F;
      }
      continue;
    }
    if (positive_infinities != 0) {
      const float probability = 1.0F / static_cast<float>(positive_infinities);
      for (std::size_t column = 0; column < width; ++column) {
        output[row_offset + column] =
            input[row_offset + column] == CUDART_INF_F ? probability : 0.0F;
      }
      continue;
    }

    float denominator = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      denominator += expf(input[row_offset + column] - maximum);
    }
    for (std::size_t column = 0; column < width; ++column) {
      output[row_offset + column] = expf(input[row_offset + column] - maximum) / denominator;
    }
  }
}

__global__ void softmax_shared_kernel(const float* input, float* output, std::size_t rows,
                                      std::size_t width) {
  __shared__ float shared_values[kThreadsPerBlock];
  __shared__ std::size_t shared_nan_counts[kThreadsPerBlock];
  __shared__ std::size_t shared_infinity_counts[kThreadsPerBlock];

  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const std::size_t row_offset = row * width;
    float local_maximum = -CUDART_INF_F;
    std::size_t local_nan_count = 0;
    std::size_t local_infinity_count = 0;
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
      const float value = input[row_offset + column];
      local_nan_count += static_cast<std::size_t>(isnan(value));
      local_infinity_count += static_cast<std::size_t>(isinf(value) && value > 0.0F);
      local_maximum = fmaxf(local_maximum, value);
    }
    shared_values[threadIdx.x] = local_maximum;
    shared_nan_counts[threadIdx.x] = local_nan_count;
    shared_infinity_counts[threadIdx.x] = local_infinity_count;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
      if (threadIdx.x < stride) {
        shared_values[threadIdx.x] =
            fmaxf(shared_values[threadIdx.x], shared_values[threadIdx.x + stride]);
        shared_nan_counts[threadIdx.x] += shared_nan_counts[threadIdx.x + stride];
        shared_infinity_counts[threadIdx.x] += shared_infinity_counts[threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float maximum = shared_values[0];
    const std::size_t nan_count = shared_nan_counts[0];
    const std::size_t positive_infinities = shared_infinity_counts[0];
    if (nan_count != 0 || maximum == -CUDART_INF_F) {
      for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
        output[row_offset + column] = CUDART_NAN_F;
      }
      __syncthreads();
      continue;
    }
    if (positive_infinities != 0) {
      const float probability = 1.0F / static_cast<float>(positive_infinities);
      for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
        output[row_offset + column] =
            input[row_offset + column] == CUDART_INF_F ? probability : 0.0F;
      }
      __syncthreads();
      continue;
    }

    float local_sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
      local_sum += expf(input[row_offset + column] - maximum);
    }
    shared_values[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
      if (threadIdx.x < stride) {
        shared_values[threadIdx.x] += shared_values[threadIdx.x + stride];
      }
      __syncthreads();
    }

    const float denominator = shared_values[0];
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
      output[row_offset + column] = expf(input[row_offset + column] - maximum) / denominator;
    }
    __syncthreads();
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

std::size_t validate(const Tensor& input, const Tensor& output, const Stream& stream,
                     std::string_view operation) {
  require_cuda_float_tensor(input, "input", operation);
  require_cuda_float_tensor(output, "output", operation);
  HELIX_CHECK(input.rank() != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "input", "must have rank at least one"));
  const std::size_t width = static_cast<std::size_t>(input.shape().back());
  HELIX_CHECK(width != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "last dimension", "must be non-zero"));
  HELIX_CHECK(output.shape() == input.shape(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must have the input shape"));
  HELIX_CHECK(output.device() == input.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must be on the input device"));
  HELIX_CHECK(stream.device_index() == input.device().index(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must belong to the input device"));
  return width;
}

unsigned int block_count(std::size_t rows, bool one_thread_per_row) {
  HELIX_ASSERT(rows != 0, "cannot select a CUDA grid for zero rows");
  std::size_t required_blocks = rows;
  if (one_thread_per_row) {
    required_blocks = 1 + (rows - 1) / static_cast<std::size_t>(kThreadsPerBlock);
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

void softmax_out(const Tensor& input, Tensor& output, const Stream& stream,
                 SoftmaxAlgorithm algorithm) {
  constexpr std::string_view kOperation = "cuda::kernels::softmax_out";
  const std::size_t width = validate(input, output, stream, kOperation);
  HELIX_CHECK(algorithm == SoftmaxAlgorithm::kNaive || algorithm == SoftmaxAlgorithm::kSharedMemory,
              ErrorCode::kInvalidArgument, "cuda::kernels::softmax_out: unknown algorithm");
  const std::size_t rows = input.numel() / width;
  if (rows == 0) {
    return;
  }
  DeviceGuard guard(stream.device_index());

  switch (algorithm) {
  case SoftmaxAlgorithm::kNaive:
    softmax_naive_kernel<<<block_count(rows, true), kThreadsPerBlock, 0, stream.native_handle()>>>(
        input.data_as<float>(), output.data_as<float>(), rows, width);
    check_launch("softmax_naive_kernel");
    return;
  case SoftmaxAlgorithm::kSharedMemory:
    softmax_shared_kernel<<<block_count(rows, false), kThreadsPerBlock, 0,
                            stream.native_handle()>>>(input.data_as<float>(),
                                                      output.data_as<float>(), rows, width);
    check_launch("softmax_shared_kernel");
    return;
  }
  HELIX_ASSERT(false, "validated softmax algorithm was not dispatched");
}

Tensor softmax(const Tensor& input, const Stream& stream, SoftmaxAlgorithm algorithm) {
  Tensor output = Tensor::empty(input.shape(), DType::kFloat32, input.device());
  softmax_out(input, output, stream, algorithm);
  return output;
}

} // namespace helix::cuda::kernels
