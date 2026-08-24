#include "helix/core/error.hpp"
#include "helix/kernels/cuda/embedding.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace helix::cuda::kernels {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr std::size_t kMaximumBlocks = 65'535;

__global__ void embedding_lookup_kernel(const std::uint32_t* token_ids, const float* table,
                                        float* output, std::size_t tokens, std::size_t vocabulary,
                                        std::size_t width) {
  for (std::size_t token = static_cast<std::size_t>(blockIdx.x); token < tokens;
       token += static_cast<std::size_t>(gridDim.x)) {
    const std::size_t id = static_cast<std::size_t>(token_ids[token]);
    if (id >= vocabulary) {
      for (std::size_t coordinate = static_cast<std::size_t>(threadIdx.x); coordinate < width;
           coordinate += static_cast<std::size_t>(blockDim.x)) {
        output[token * width + coordinate] = nanf("");
      }
      continue;
    }
    for (std::size_t coordinate = static_cast<std::size_t>(threadIdx.x); coordinate < width;
         coordinate += static_cast<std::size_t>(blockDim.x)) {
      output[token * width + coordinate] = table[id * width + coordinate];
    }
  }
}

std::string tensor_error(std::string_view operation, std::string_view tensor,
                         std::string_view requirement) {
  std::ostringstream message;
  message << operation << ": " << tensor << ' ' << requirement;
  return message.str();
}

void require_cuda_tensor(const Tensor& tensor, DType dtype, std::string_view name,
                         std::string_view operation) {
  HELIX_CHECK(tensor.device().is_cuda(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on CUDA"));
  HELIX_CHECK(tensor.dtype() == dtype, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "has an unsupported dtype"));
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be contiguous"));
}

void require_same_device(const Tensor& tensor, const Tensor& reference, std::string_view name,
                         std::string_view operation) {
  HELIX_CHECK(tensor.device() == reference.device(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on the token ID device"));
}

void require_stream(const Tensor& token_ids, const Stream& stream, std::string_view operation) {
  HELIX_CHECK(stream.native_handle() != nullptr, ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must not be moved-from"));
  HELIX_CHECK(stream.device_index() == token_ids.device().index(), ErrorCode::kInvalidArgument,
              tensor_error(operation, "stream", "must belong to the token ID device"));
}

void require_no_alias(const Tensor& input, const Tensor& output, std::string_view input_name,
                      std::string_view operation) {
  if (input.nbytes() == 0 || output.nbytes() == 0 || input.device() != output.device()) {
    return;
  }
  const auto input_begin = reinterpret_cast<std::uintptr_t>(input.data());
  const auto output_begin = reinterpret_cast<std::uintptr_t>(output.data());
  const auto maximum = std::numeric_limits<std::uintptr_t>::max();
  const bool invalid_range =
      input_begin > maximum - input.nbytes() || output_begin > maximum - output.nbytes();
  const bool overlaps = invalid_range || (input_begin < output_begin + output.nbytes() &&
                                          output_begin < input_begin + input.nbytes());
  HELIX_CHECK(
      !overlaps, ErrorCode::kInvalidArgument,
      tensor_error(operation, "output", std::string("must not alias ") + std::string(input_name)));
}

unsigned int block_count(std::size_t tokens) {
  HELIX_ASSERT(tokens != 0, "cannot select an embedding grid for zero tokens");
  const std::size_t blocks = std::min(tokens, kMaximumBlocks);
  HELIX_ASSERT(blocks <= std::numeric_limits<unsigned int>::max(),
               "embedding grid dimension does not fit unsigned int");
  return static_cast<unsigned int>(blocks);
}

void check_launch() {
  HELIX_CUDA_CHECK_CONTEXT(cudaGetLastError(), "launching embedding_lookup_kernel");
}

} // namespace

void embedding_lookup_out(const Tensor& token_ids, const Tensor& table, Tensor& output,
                          const Stream& stream) {
  constexpr std::string_view kOperation = "cuda::kernels::embedding_lookup_out";
  require_cuda_tensor(token_ids, DType::kUInt32, "token_ids", kOperation);
  require_cuda_tensor(table, DType::kFloat32, "table", kOperation);
  require_cuda_tensor(output, DType::kFloat32, "output", kOperation);
  HELIX_CHECK(token_ids.rank() == 1, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "token_ids", "must have rank one"));
  HELIX_CHECK(table.rank() == 2, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "table", "must have rank two"));
  HELIX_CHECK(table.shape()[1] != 0, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "table width", "must be non-zero"));
  const Shape expected_output{token_ids.shape()[0], table.shape()[1]};
  HELIX_CHECK(output.shape() == expected_output, ErrorCode::kInvalidArgument,
              tensor_error(kOperation, "output", "must have shape [tokens, width]"));
  require_same_device(table, token_ids, "table", kOperation);
  require_same_device(output, token_ids, "output", kOperation);
  require_stream(token_ids, stream, kOperation);
  require_no_alias(token_ids, output, "token_ids", kOperation);
  require_no_alias(table, output, "table", kOperation);

  if (token_ids.numel() == 0) {
    return;
  }
  const auto vocabulary = static_cast<std::size_t>(table.shape()[0]);
  const auto width = static_cast<std::size_t>(table.shape()[1]);
  DeviceGuard guard(stream.device_index());
  embedding_lookup_kernel<<<block_count(token_ids.numel()), kThreadsPerBlock, 0,
                            stream.native_handle()>>>(
      token_ids.data_as<std::uint32_t>(), table.data_as<float>(), output.data_as<float>(),
      token_ids.numel(), vocabulary, width);
  check_launch();
}

Tensor embedding_lookup(const Tensor& token_ids, const Tensor& table, const Stream& stream) {
  HELIX_CHECK(token_ids.rank() == 1, ErrorCode::kInvalidArgument,
              "cuda::kernels::embedding_lookup: token_ids must have rank one");
  HELIX_CHECK(table.rank() == 2, ErrorCode::kInvalidArgument,
              "cuda::kernels::embedding_lookup: table must have rank two");
  Tensor output =
      Tensor::empty({token_ids.shape()[0], table.shape()[1]}, DType::kFloat32, token_ids.device());
  embedding_lookup_out(token_ids, table, output, stream);
  return output;
}

} // namespace helix::cuda::kernels
