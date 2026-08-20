#include "helix/core/error.hpp"
#include "helix/kernels/cuda/reduction.hpp"
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
constexpr unsigned int kWarpSize = 32;
constexpr std::size_t kMaximumBlocks = 65'535;

__global__ void sum_last_dim_naive_kernel(const float* input, float* output, std::size_t rows,
                                          std::size_t width) {
  std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; row < rows; row += row_stride) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      sum += input[row * width + column];
    }
    output[row] = sum;
  }
}

__global__ void sum_last_dim_shared_kernel(const float* input, float* output, std::size_t rows,
                                           std::size_t width) {
  extern __shared__ float partial_sums[];
  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float thread_sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
      thread_sum += input[row * width + column];
    }
    partial_sums[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
      if (threadIdx.x < stride) {
        partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      output[row] = partial_sums[0];
    }
    __syncthreads();
  }
}

__device__ float warp_sum(float value) {
  constexpr unsigned int kActiveMask = 0xFFFF'FFFFU;
  for (int offset = static_cast<int>(kWarpSize / 2); offset > 0; offset /= 2) {
    value += __shfl_down_sync(kActiveMask, value, offset);
  }
  return value;
}

__global__ void sum_last_dim_warp_kernel(const float* input, float* output, std::size_t rows,
                                         std::size_t width) {
  __shared__ float warp_sums[kThreadsPerBlock / kWarpSize];
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp = threadIdx.x / kWarpSize;
  constexpr unsigned int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

  for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    float thread_sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < width; column += blockDim.x) {
      thread_sum += input[row * width + column];
    }
    thread_sum = warp_sum(thread_sum);
    if (lane == 0) {
      warp_sums[warp] = thread_sum;
    }
    __syncthreads();

    if (warp == 0) {
      float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
      block_sum = warp_sum(block_sum);
      if (lane == 0) {
        output[row] = block_sum;
      }
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

Shape reduced_shape(const Tensor& input, bool keep_dim, std::string_view operation) {
  HELIX_CHECK(input.rank() != 0, ErrorCode::kInvalidArgument,
              tensor_error(operation, "input", "must have rank at least one"));
  Shape shape = input.shape();
  if (keep_dim) {
    shape.back() = 1;
  } else {
    shape.pop_back();
  }
  return shape;
}

void validate(const Tensor& input, const Tensor& output, const Stream& stream, bool keep_dim,
              std::string_view operation) {
  require_cuda_float_tensor(input, "input", operation);
  require_cuda_float_tensor(output, "output", operation);
  HELIX_CHECK(output.shape() == reduced_shape(input, keep_dim, operation),
              ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "has an invalid reduced shape"));
  HELIX_CHECK(output.device() == input.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must be on the input device"));
  HELIX_CHECK(stream.device_index() == input.device().index(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must belong to the input device"));
  HELIX_CHECK(input.nbytes() == 0 || input.data() != output.data(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must not alias input"));
}

unsigned int block_count(std::size_t work_items, bool one_thread_per_item) {
  HELIX_ASSERT(work_items != 0, "cannot select a CUDA grid for zero work items");
  std::size_t required_blocks = work_items;
  if (one_thread_per_item) {
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

void sum_last_dim_out(const Tensor& input, Tensor& output, const Stream& stream, bool keep_dim,
                      ReductionAlgorithm algorithm) {
  constexpr std::string_view kOperation = "cuda::kernels::sum_last_dim_out";
  validate(input, output, stream, keep_dim, kOperation);
  HELIX_CHECK(algorithm == ReductionAlgorithm::kNaive ||
                  algorithm == ReductionAlgorithm::kSharedMemory ||
                  algorithm == ReductionAlgorithm::kWarpShuffle,
              ErrorCode::kInvalidArgument, "cuda::kernels::sum_last_dim_out: unknown algorithm");
  const std::size_t rows = output.numel();
  if (rows == 0) {
    return;
  }
  const std::size_t width = static_cast<std::size_t>(input.shape().back());
  DeviceGuard guard(stream.device_index());

  switch (algorithm) {
  case ReductionAlgorithm::kNaive:
    sum_last_dim_naive_kernel<<<block_count(rows, true), kThreadsPerBlock, 0,
                                stream.native_handle()>>>(input.data_as<float>(),
                                                          output.data_as<float>(), rows, width);
    check_launch("sum_last_dim_naive_kernel");
    return;
  case ReductionAlgorithm::kSharedMemory:
    sum_last_dim_shared_kernel<<<block_count(rows, false), kThreadsPerBlock,
                                 kThreadsPerBlock * sizeof(float), stream.native_handle()>>>(
        input.data_as<float>(), output.data_as<float>(), rows, width);
    check_launch("sum_last_dim_shared_kernel");
    return;
  case ReductionAlgorithm::kWarpShuffle:
    sum_last_dim_warp_kernel<<<block_count(rows, false), kThreadsPerBlock, 0,
                               stream.native_handle()>>>(input.data_as<float>(),
                                                         output.data_as<float>(), rows, width);
    check_launch("sum_last_dim_warp_kernel");
    return;
  }
  HELIX_ASSERT(false, "validated reduction algorithm was not dispatched");
}

Tensor sum_last_dim(const Tensor& input, const Stream& stream, bool keep_dim,
                    ReductionAlgorithm algorithm) {
  const Shape shape = reduced_shape(input, keep_dim, "cuda::kernels::sum_last_dim");
  Tensor output = Tensor::empty(shape, DType::kFloat32, input.device());
  sum_last_dim_out(input, output, stream, keep_dim, algorithm);
  return output;
}

} // namespace helix::cuda::kernels
