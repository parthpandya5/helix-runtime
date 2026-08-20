#include "helix/core/error.hpp"

#include <sstream>
#include <utility>

namespace helix {
namespace {

std::string format_error(ErrorCode code, std::string_view message,
                         const std::source_location& location) {
  std::ostringstream stream;
  stream << '[' << error_code_name(code) << "] " << message << " (" << location.file_name() << ':'
         << location.line() << " in " << location.function_name() << ')';
  return stream.str();
}

} // namespace

std::string_view error_code_name(ErrorCode code) noexcept {
  switch (code) {
  case ErrorCode::kInvalidArgument:
    return "invalid_argument";
  case ErrorCode::kOutOfRange:
    return "out_of_range";
  case ErrorCode::kUnsupported:
    return "unsupported";
  case ErrorCode::kResourceExhausted:
    return "resource_exhausted";
  case ErrorCode::kRuntime:
    return "runtime";
  case ErrorCode::kInternal:
    return "internal";
  }
  return "unknown";
}

Error::Error(ErrorCode code, std::string message, std::source_location location)
    : std::runtime_error(format_error(code, message, location)), code_(code),
      message_(std::move(message)), location_(location) {}

[[noreturn]] void throw_error(ErrorCode code, std::string message, std::source_location location) {
  throw Error(code, std::move(message), location);
}

[[noreturn]] void throw_invariant(std::string_view expression, std::string message,
                                  std::source_location location) {
  std::ostringstream stream;
  stream << "invariant '" << expression << "' failed";
  if (!message.empty()) {
    stream << ": " << message;
  }
  throw Error(ErrorCode::kInternal, stream.str(), location);
}

} // namespace helix
