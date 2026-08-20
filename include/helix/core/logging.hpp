#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <source_location>
#include <string>
#include <string_view>
#include <thread>

namespace helix {

enum class LogLevel : std::uint8_t {
  kDebug,
  kInfo,
  kWarning,
  kError,
  kOff,
};

struct LogRecord {
  LogLevel level;
  std::string message;
  std::source_location location;
  std::chrono::system_clock::time_point timestamp;
  std::thread::id thread_id;
};

using LogSink = std::function<void(const LogRecord&)>;

[[nodiscard]] std::string_view log_level_name(LogLevel level) noexcept;
void set_log_level(LogLevel level) noexcept;
[[nodiscard]] LogLevel log_level() noexcept;
void set_log_sink(LogSink sink);
void reset_log_sink();
void log(LogLevel level, std::string message,
         std::source_location location = std::source_location::current()) noexcept;

} // namespace helix

#define HELIX_LOG_DEBUG(message)                                                                   \
  ::helix::log(::helix::LogLevel::kDebug, (message), std::source_location::current())
#define HELIX_LOG_INFO(message)                                                                    \
  ::helix::log(::helix::LogLevel::kInfo, (message), std::source_location::current())
#define HELIX_LOG_WARNING(message)                                                                 \
  ::helix::log(::helix::LogLevel::kWarning, (message), std::source_location::current())
#define HELIX_LOG_ERROR(message)                                                                   \
  ::helix::log(::helix::LogLevel::kError, (message), std::source_location::current())
