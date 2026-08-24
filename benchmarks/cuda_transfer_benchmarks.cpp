#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/event.hpp"
#include "helix/runtime/cuda/memory.hpp"
#include "helix/runtime/cuda/stream.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace helix::cuda {
namespace {

constexpr int kWarmupCopies = 10;

enum class HostMemoryKind {
  kPageable,
  kPinned,
};

struct HostBufferFixture {
  explicit HostBufferFixture(std::size_t bytes, HostMemoryKind kind) {
    if (kind == HostMemoryKind::kPinned) {
      pinned.emplace(bytes);
      data = pinned->data();
    } else {
      pageable.resize(bytes);
      data = pageable.data();
    }
  }

  std::vector<std::byte> pageable;
  std::optional<PinnedHostBuffer> pinned;
  void* data = nullptr;
};

void record_transfer_metadata(benchmark::State& state, HostMemoryKind kind) {
  state.counters["pinned_host_memory"] = kind == HostMemoryKind::kPinned ? 1.0 : 0.0;
  state.SetBytesProcessed(state.iterations() * state.range(0));
}

void BM_CudaHostToDeviceTransfer(benchmark::State& state, HostMemoryKind kind) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  HostBufferFixture host(bytes, kind);
  std::fill_n(static_cast<std::byte*>(host.data), bytes, std::byte{0x5A});
  Buffer device(bytes);
  Stream stream;
  for (int warmup = 0; warmup < kWarmupCopies; ++warmup) {
    copy_host_to_device_async(device.data(), host.data, bytes, stream);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    copy_host_to_device_async(device.data(), host.data, bytes, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  record_transfer_metadata(state, kind);
}

void BM_CudaDeviceToHostTransfer(benchmark::State& state, HostMemoryKind kind) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  HostBufferFixture host(bytes, kind);
  Buffer device(bytes);
  set_device_bytes(device.data(), 0xA5, bytes);
  Stream stream;
  for (int warmup = 0; warmup < kWarmupCopies; ++warmup) {
    copy_device_to_host_async(host.data, device.data(), bytes, stream);
  }
  stream.synchronize();

  Event start;
  Event stop;
  for ([[maybe_unused]] auto iteration : state) {
    start.record(stream);
    copy_device_to_host_async(host.data, device.data(), bytes, stream);
    stop.record(stream);
    stop.synchronize();
    state.SetIterationTime(static_cast<double>(elapsed_time_ms(start, stop)) / 1000.0);
  }
  record_transfer_metadata(state, kind);
}

BENCHMARK_CAPTURE(BM_CudaHostToDeviceTransfer, Pageable, HostMemoryKind::kPageable)
    ->Arg(4'096)
    ->Arg(1 << 20)
    ->Arg(16 << 20)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_CudaHostToDeviceTransfer, Pinned, HostMemoryKind::kPinned)
    ->Arg(4'096)
    ->Arg(1 << 20)
    ->Arg(16 << 20)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_CudaDeviceToHostTransfer, Pageable, HostMemoryKind::kPageable)
    ->Arg(4'096)
    ->Arg(1 << 20)
    ->Arg(16 << 20)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_CudaDeviceToHostTransfer, Pinned, HostMemoryKind::kPinned)
    ->Arg(4'096)
    ->Arg(1 << 20)
    ->Arg(16 << 20)
    ->UseManualTime();

} // namespace
} // namespace helix::cuda
