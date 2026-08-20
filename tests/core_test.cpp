#include "helix/core/build_info.hpp"
#include "helix/core/device.hpp"
#include "helix/core/dtype.hpp"
#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <atomic>
#include <cstdint>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace helix {
namespace {

TEST(DeviceTest, RepresentsCpuAndCudaDevices) {
  const Device cpu = Device::cpu();
  const Device cuda = Device::cuda(3);

  EXPECT_TRUE(cpu.is_cpu());
  EXPECT_FALSE(cpu.is_cuda());
  EXPECT_EQ(cpu.index(), 0);
  EXPECT_EQ(cpu.to_string(), "cpu");

  EXPECT_TRUE(cuda.is_cuda());
  EXPECT_EQ(cuda.index(), 3);
  EXPECT_EQ(cuda.to_string(), "cuda:3");
  EXPECT_NE(cpu, cuda);
}

TEST(DeviceTest, ValidatesDeviceKindsAndIndices) {
  EXPECT_THROW(static_cast<void>(Device(DeviceType::kCpu, 1)), Error);
  EXPECT_THROW(static_cast<void>(Device::cuda(-1)), Error);
  EXPECT_THROW(static_cast<void>(Device(static_cast<DeviceType>(255), 0)), Error);
}

TEST(DeviceTest, StreamsReadableRepresentation) {
  std::ostringstream stream;
  stream << Device::cuda(2);
  EXPECT_EQ(stream.str(), "cuda:2");
}

TEST(DTypeTest, ReportsStableMetadata) {
  EXPECT_EQ(dtype_name(DType::kFloat32), "float32");
  EXPECT_EQ(dtype_size(DType::kFloat32), sizeof(float));
  EXPECT_TRUE(dtype_is_floating_point(DType::kFloat32));

  EXPECT_EQ(dtype_name(DType::kFloat16), "float16");
  EXPECT_EQ(dtype_size(DType::kFloat16), 2U);
  EXPECT_TRUE(dtype_is_floating_point(DType::kFloat16));
}

TEST(DTypeTest, RejectsUnknownValues) {
  const auto invalid = static_cast<DType>(255);
  EXPECT_THROW(static_cast<void>(dtype_name(invalid)), Error);
  EXPECT_THROW(static_cast<void>(dtype_size(invalid)), Error);
  EXPECT_THROW(static_cast<void>(dtype_is_floating_point(invalid)), Error);
}

TEST(ErrorTest, PreservesCodeMessageAndSourceContext) {
  try {
    throw_error(ErrorCode::kResourceExhausted, "KV block pool is empty");
    FAIL() << "throw_error returned";
  } catch (const Error& error) {
    EXPECT_EQ(error.code(), ErrorCode::kResourceExhausted);
    EXPECT_EQ(error.message(), "KV block pool is empty");
    EXPECT_NE(std::string(error.what()).find("resource_exhausted"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("KV block pool is empty"), std::string::npos);
    EXPECT_GT(error.location().line(), 0U);
  }
}

TEST(ErrorTest, ChecksDoNotEvaluateFailureMessageOnSuccess) {
  bool evaluated = false;
  const auto make_message = [&evaluated]() {
    evaluated = true;
    return std::string("should not be built");
  };

  HELIX_CHECK(true, ErrorCode::kInternal, make_message());
  EXPECT_FALSE(evaluated);
}

TEST(ErrorTest, InvariantIncludesFailedExpression) {
  try {
    HELIX_ASSERT(2 + 2 == 5, "arithmetic contract");
    FAIL() << "HELIX_ASSERT returned";
  } catch (const Error& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInternal);
    EXPECT_NE(error.message().find("2 + 2 == 5"), std::string::npos);
    EXPECT_NE(error.message().find("arithmetic contract"), std::string::npos);
  }
}

class LoggingTest : public ::testing::Test {
protected:
  void TearDown() override {
    reset_log_sink();
    set_log_level(LogLevel::kInfo);
  }
};

TEST_F(LoggingTest, FiltersBelowConfiguredSeverity) {
  std::vector<LogRecord> records;
  set_log_sink([&records](const LogRecord& record) { records.push_back(record); });
  set_log_level(LogLevel::kWarning);

  log(LogLevel::kInfo, "hidden");
  log(LogLevel::kWarning, "visible");

  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records.front().level, LogLevel::kWarning);
  EXPECT_EQ(records.front().message, "visible");
}

TEST_F(LoggingTest, SupportsConcurrentProducers) {
  constexpr std::size_t kThreadCount = 4;
  constexpr std::size_t kRecordsPerThread = 128;
  std::atomic<std::size_t> observed{0};
  set_log_sink([&observed](const LogRecord&) { observed.fetch_add(1, std::memory_order_relaxed); });
  set_log_level(LogLevel::kDebug);

  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (std::size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([] {
      for (std::size_t record_index = 0; record_index < kRecordsPerThread; ++record_index) {
        log(LogLevel::kDebug, "concurrent record");
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(observed.load(std::memory_order_relaxed), kThreadCount * kRecordsPerThread);
}

TEST(BuildInfoTest, ExposesConfiguredVersionAndBackend) {
  const auto info = build_info();
  EXPECT_FALSE(info.version.empty());
#if HELIX_ENABLE_CUDA
  EXPECT_TRUE(info.cuda_enabled);
#else
  EXPECT_FALSE(info.cuda_enabled);
#endif
}

} // namespace
} // namespace helix
