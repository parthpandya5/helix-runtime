#include "helix/core/logging.hpp"

#include "helix/core/error.hpp"

#include <atomic>
#include <iostream>
#include <mutex>
#include <utility>

namespace helix {
namespace {

void default_sink(const LogRecord& record) {
  std::clog << '[' << log_level_name(record.level) << "] " << record.location.file_name() << ':'
            << record.location.line() << " " << record.message << '\n';
}

struct LoggerState {
  std::atomic<LogLevel> minimum_level{LogLevel::kInfo};
  std::mutex sink_mutex;
  LogSink sink{default_sink};
};

LoggerState& logger_state() {
  static LoggerState state;
  return state;
}

} // namespace

std::string_view log_level_name(LogLevel level) noexcept {
  switch (level) {
  case LogLevel::kDebug:
    return "DEBUG";
  case LogLevel::kInfo:
    return "INFO";
  case LogLevel::kWarning:
    return "WARNING";
  case LogLevel::kError:
    return "ERROR";
  case LogLevel::kOff:
    return "OFF";
  }
  return "UNKNOWN";
}

void set_log_level(LogLevel level) noexcept {
  logger_state().minimum_level.store(level, std::memory_order_relaxed);
}

LogLevel log_level() noexcept {
  return logger_state().minimum_level.load(std::memory_order_relaxed);
}

void set_log_sink(LogSink sink) {
  HELIX_CHECK(static_cast<bool>(sink), ErrorCode::kInvalidArgument, "log sink must be callable");
  auto& state = logger_state();
  std::scoped_lock lock(state.sink_mutex);
  state.sink = std::move(sink);
}

void reset_log_sink() {
  auto& state = logger_state();
  std::scoped_lock lock(state.sink_mutex);
  state.sink = default_sink;
}

void log(LogLevel level, std::string message, std::source_location location) noexcept {
  auto& state = logger_state();
  const auto minimum = state.minimum_level.load(std::memory_order_relaxed);
  if (minimum == LogLevel::kOff ||
      static_cast<std::uint8_t>(level) < static_cast<std::uint8_t>(minimum)) {
    return;
  }

  const LogRecord record{level, std::move(message), location, std::chrono::system_clock::now(),
                         std::this_thread::get_id()};
  LogSink sink;
  {
    std::scoped_lock lock(state.sink_mutex);
    sink = state.sink;
  }

  try {
    sink(record);
  } catch (...) {
    try {
      default_sink(LogRecord{LogLevel::kError, "configured log sink threw an exception", location,
                             std::chrono::system_clock::now(), std::this_thread::get_id()});
    } catch (...) {
      // Logging is best-effort and must not terminate an inference process.
    }
  }
}

} // namespace helix
