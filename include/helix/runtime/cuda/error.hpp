#pragma once

#include <cuda_runtime_api.h>
#include <source_location>
#include <string_view>

namespace helix::cuda {

void check(cudaError_t result, std::string_view expression, std::string_view context = {},
           std::source_location location = std::source_location::current());
void report_cleanup_error(cudaError_t result, std::string_view operation,
                          std::source_location location = std::source_location::current()) noexcept;

} // namespace helix::cuda

#define HELIX_CUDA_CHECK(expression)                                                               \
  ::helix::cuda::check((expression), #expression, {}, std::source_location::current())

#define HELIX_CUDA_CHECK_CONTEXT(expression, context)                                              \
  ::helix::cuda::check((expression), #expression, (context), std::source_location::current())
