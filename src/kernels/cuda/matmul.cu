#include "helix/core/error.hpp"
#include "helix/kernels/cuda/matmul.hpp"
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

constexpr unsigned int kTile = 16;
constexpr std::size_t kMaximumGridExtent = 65'535;

__global__ void matmul_naive_kernel(const float* lhs, const float* rhs, float* output,
                                    std::size_t rows, std::size_t inner, std::size_t columns) {
  std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t row_stride = static_cast<std::size_t>(gridDim.y) * blockDim.y;
  for (; row < rows; row += row_stride) {
    std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t column_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (; column < columns; column += column_stride) {
      float accumulator = 0.0F;
      for (std::size_t reduction = 0; reduction < inner; ++reduction) {
        accumulator += lhs[row * inner + reduction] * rhs[reduction * columns + column];
      }
      output[row * columns + column] = accumulator;
    }
  }
}

__global__ void matmul_tiled_kernel(const float* lhs, const float* rhs, float* output,
                                    std::size_t rows, std::size_t inner, std::size_t columns,
                                    std::size_t row_tiles, std::size_t column_tiles) {
  __shared__ float lhs_tile[kTile][kTile];
  __shared__ float rhs_tile[kTile][kTile];
  const std::size_t inner_tiles = inner == 0 ? 0 : 1 + (inner - 1) / kTile;

  for (std::size_t row_tile = blockIdx.y; row_tile < row_tiles; row_tile += gridDim.y) {
    for (std::size_t column_tile = blockIdx.x; column_tile < column_tiles;
         column_tile += gridDim.x) {
      const std::size_t row = row_tile * kTile + threadIdx.y;
      const std::size_t column = column_tile * kTile + threadIdx.x;
      float accumulator = 0.0F;

      for (std::size_t reduction_tile = 0; reduction_tile < inner_tiles; ++reduction_tile) {
        const std::size_t lhs_column = reduction_tile * kTile + threadIdx.x;
        const std::size_t rhs_row = reduction_tile * kTile + threadIdx.y;
        lhs_tile[threadIdx.y][threadIdx.x] =
            row < rows && lhs_column < inner ? lhs[row * inner + lhs_column] : 0.0F;
        rhs_tile[threadIdx.y][threadIdx.x] =
            rhs_row < inner && column < columns ? rhs[rhs_row * columns + column] : 0.0F;
        __syncthreads();

#pragma unroll
        for (unsigned int reduction = 0; reduction < kTile; ++reduction) {
          accumulator += lhs_tile[threadIdx.y][reduction] * rhs_tile[reduction][threadIdx.x];
        }
        __syncthreads();
      }

      if (row < rows && column < columns) {
        output[row * columns + column] = accumulator;
      }
    }
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

Shape output_shape(const Tensor& lhs, const Tensor& rhs, std::string_view operation) {
  HELIX_CHECK(lhs.rank() == 2 && rhs.rank() == 2, ErrorCode::kInvalidArgument,
              tensor_error(operation, "inputs", "must both have rank two"));
  HELIX_CHECK(lhs.shape()[1] == rhs.shape()[0], ErrorCode::kInvalidArgument,
              tensor_error(operation, "inputs", "must have equal inner dimensions"));
  return Shape{lhs.shape()[0], rhs.shape()[1]};
}

void validate(const Tensor& lhs, const Tensor& rhs, const Tensor& output, const Stream& stream,
              std::string_view operation) {
  require_cuda_float_tensor(lhs, "lhs", operation);
  require_cuda_float_tensor(rhs, "rhs", operation);
  require_cuda_float_tensor(output, "output", operation);
  HELIX_CHECK(lhs.device() == rhs.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "inputs", "must be on the same device"));
  HELIX_CHECK(output.device() == lhs.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must be on the input device"));
  HELIX_CHECK(output.shape() == output_shape(lhs, rhs, operation), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "has an invalid matrix-product shape"));
  HELIX_CHECK(stream.device_index() == lhs.device().index(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must belong to the input device"));
  HELIX_CHECK(lhs.nbytes() == 0 || lhs.data() != output.data(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must not alias lhs"));
  HELIX_CHECK(rhs.nbytes() == 0 || rhs.data() != output.data(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "output", "must not alias rhs"));
}

std::size_t tile_count(std::size_t elements) {
  HELIX_ASSERT(elements != 0, "cannot compute a positive tile count for zero elements");
  return 1 + (elements - 1) / kTile;
}

unsigned int grid_extent(std::size_t tiles) {
  HELIX_ASSERT(tiles != 0, "cannot select a CUDA grid for zero tiles");
  const std::size_t extent = std::min(tiles, kMaximumGridExtent);
  HELIX_ASSERT(extent <= std::numeric_limits<unsigned int>::max(),
               "CUDA grid dimension does not fit unsigned int");
  return static_cast<unsigned int>(extent);
}

void check_launch(std::string_view kernel_name) {
  std::ostringstream context;
  context << "launching " << kernel_name;
  HELIX_CUDA_CHECK_CONTEXT(cudaGetLastError(), context.str());
}

} // namespace

void matmul_out(const Tensor& lhs, const Tensor& rhs, Tensor& output, const Stream& stream,
                MatmulAlgorithm algorithm) {
  constexpr std::string_view kOperation = "cuda::kernels::matmul_out";
  validate(lhs, rhs, output, stream, kOperation);
  HELIX_CHECK(algorithm == MatmulAlgorithm::kNaive || algorithm == MatmulAlgorithm::kTiled,
              ErrorCode::kInvalidArgument, "cuda::kernels::matmul_out: unknown algorithm");

  const std::size_t rows = static_cast<std::size_t>(lhs.shape()[0]);
  const std::size_t inner = static_cast<std::size_t>(lhs.shape()[1]);
  const std::size_t columns = static_cast<std::size_t>(rhs.shape()[1]);
  if (rows == 0 || columns == 0) {
    return;
  }
  const std::size_t row_tiles = tile_count(rows);
  const std::size_t column_tiles = tile_count(columns);
  const dim3 block(kTile, kTile);
  const dim3 grid(grid_extent(column_tiles), grid_extent(row_tiles));
  DeviceGuard guard(stream.device_index());

  switch (algorithm) {
  case MatmulAlgorithm::kNaive:
    matmul_naive_kernel<<<grid, block, 0, stream.native_handle()>>>(
        lhs.data_as<float>(), rhs.data_as<float>(), output.data_as<float>(), rows, inner, columns);
    check_launch("matmul_naive_kernel");
    return;
  case MatmulAlgorithm::kTiled:
    matmul_tiled_kernel<<<grid, block, 0, stream.native_handle()>>>(
        lhs.data_as<float>(), rhs.data_as<float>(), output.data_as<float>(), rows, inner, columns,
        row_tiles, column_tiles);
    check_launch("matmul_tiled_kernel");
    return;
  }
  HELIX_ASSERT(false, "validated matmul algorithm was not dispatched");
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs, const Stream& stream,
              MatmulAlgorithm algorithm) {
  const Shape shape = output_shape(lhs, rhs, "cuda::kernels::matmul");
  Tensor output = Tensor::empty(shape, DType::kFloat32, lhs.device());
  matmul_out(lhs, rhs, output, stream, algorithm);
  return output;
}

} // namespace helix::cuda::kernels
