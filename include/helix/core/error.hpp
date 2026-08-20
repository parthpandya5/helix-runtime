#pragma once

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace helix {

enum class ErrorCode {
  kInvalidArgument,
  kOutOfRange,
  kUnsupported,
  kResourceExhausted,
  kRuntime,
  kInternal,
};

[[nodiscard]] std::string_view error_code_name(ErrorCode code) noexcept;

class Error final : public std::runtime_error {
public:
  Error(ErrorCode code, std::string message,
        std::source_location location = std::source_location::current());

  [[nodiscard]] ErrorCode code() const noexcept {
    return code_;
  }
  [[nodiscard]] const std::string& message() const noexcept {
    return message_;
  }
  [[nodiscard]] const std::source_location& location() const noexcept {
    return location_;
  }

private:
  ErrorCode code_;
  std::string message_;
  std::source_location location_;
};

[[noreturn]] void throw_error(ErrorCode code, std::string message,
                              std::source_location location = std::source_location::current());

[[noreturn]] void throw_invariant(std::string_view expression, std::string message,
                                  std::source_location location = std::source_location::current());

} // namespace helix

#define HELIX_CHECK(condition, code, message)                                                      \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      ::helix::throw_error((code), (message), std::source_location::current());                    \
    }                                                                                              \
  } while (false)

#define HELIX_ASSERT(condition, message)                                                           \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      ::helix::throw_invariant(#condition, (message), std::source_location::current());            \
    }                                                                                              \
  } while (false)
