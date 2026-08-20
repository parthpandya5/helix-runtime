#include "helix/runtime/cuda/error.hpp"

#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <sstream>
#include <string>

namespace helix::cuda {

void check(cudaError_t result, std::string_view expression, std::string_view context,
           std::source_location location) {
  if (result == cudaSuccess) {
    return;
  }

  std::ostringstream message;
  message << expression << " failed with " << cudaGetErrorName(result) << ": "
          << cudaGetErrorString(result);
  if (!context.empty()) {
    message << " (" << context << ')';
  }
  const ErrorCode code =
      result == cudaErrorMemoryAllocation ? ErrorCode::kResourceExhausted : ErrorCode::kRuntime;
  throw_error(code, message.str(), location);
}

void report_cleanup_error(cudaError_t result, std::string_view operation,
                          std::source_location location) noexcept {
  if (result == cudaSuccess) {
    return;
  }
  try {
    std::ostringstream message;
    message << operation << " failed during non-throwing cleanup with " << cudaGetErrorName(result)
            << ": " << cudaGetErrorString(result);
    log(LogLevel::kError, message.str(), location);
  } catch (...) {
    // Cleanup diagnostics are best-effort; destructors must not throw.
  }
}

} // namespace helix::cuda
