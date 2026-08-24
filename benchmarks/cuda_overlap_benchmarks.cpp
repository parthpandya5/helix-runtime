#include "helix/core/error.hpp"
#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/elementwise.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"
#include "helix/runtime/cuda/event.hpp"
#include "helix/runtime/cuda/memory.hpp"
#include "helix/runtime/cuda/stream.hpp"

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace helix::cuda::kernels {
namespace {

constexpr int kWarmupPipelines = 10;
constexpr std::size_t kPipelineSlots = 2;

enum class PipelineMode {
  kSequential,
  kDoubleBuffered,
};

std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* context) {
  HELIX_CHECK(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
              ErrorCode::kOutOfRange, context);
  return lhs * rhs;
}

std::size_t checked_chunk_count(std::size_t value) {
  HELIX_CHECK(value >= kPipelineSlots, ErrorCode::kInvalidArgument,
              "overlap benchmark requires at least two chunks");
  return value;
}

Dimension checked_positive_dimension(std::size_t value, const char* context) {
  HELIX_CHECK(value != 0 &&
                  value <= static_cast<std::size_t>(std::numeric_limits<Dimension>::max()),
              ErrorCode::kInvalidArgument, context);
  return static_cast<Dimension>(value);
}

int device_attribute(cudaDeviceAttr attribute, const char* context) {
  int value = 0;
  HELIX_CUDA_CHECK_CONTEXT(cudaDeviceGetAttribute(&value, attribute, 0), context);
  return value;
}

class SiluTransferPipeline final {
public:
  SiluTransferPipeline(std::size_t chunk_count, std::size_t elements_per_chunk)
      : chunk_count_(checked_chunk_count(chunk_count)), elements_per_chunk_(elements_per_chunk),
        elements_dimension_(checked_positive_dimension(
            elements_per_chunk_, "overlap benchmark chunk elements must fit a positive Dimension")),
        total_elements_(checked_product(chunk_count_, elements_per_chunk_,
                                        "overlap benchmark element-count overflow")),
        total_dimension_(checked_positive_dimension(
            total_elements_, "overlap benchmark total elements must fit a positive Dimension")),
        chunk_bytes_(checked_product(elements_per_chunk_, sizeof(float),
                                     "overlap benchmark chunk-byte overflow")),
        total_bytes_(checked_product(total_elements_, sizeof(float),
                                     "overlap benchmark total-byte overflow")),
        transfer_bytes_(
            checked_product(total_bytes_, 2, "overlap benchmark transfer-byte overflow")),
        host_input_(total_bytes_), host_output_(total_bytes_),
        device_input_{Tensor::empty({elements_dimension_}, DType::kFloat32, Device::cuda(0)),
                      Tensor::empty({elements_dimension_}, DType::kFloat32, Device::cuda(0))},
        device_output_{Tensor::empty({elements_dimension_}, DType::kFloat32, Device::cuda(0)),
                       Tensor::empty({elements_dimension_}, DType::kFloat32, Device::cuda(0))},
        input_ready_{Event(cudaEventDisableTiming), Event(cudaEventDisableTiming)},
        compute_ready_{Event(cudaEventDisableTiming), Event(cudaEventDisableTiming)},
        slot_done_{Event(cudaEventDisableTiming), Event(cudaEventDisableTiming)} {
    auto* values = static_cast<float*>(host_input_.data());
    for (std::size_t index = 0; index < total_elements_; ++index) {
      values[index] = static_cast<float>(static_cast<int>(index % 257U) - 128) / 16.0F;
    }
    expected_ = cpu::silu(
        Tensor::from_values({total_dimension_}, std::span<const float>(values, total_elements_)));
  }

  float execute(PipelineMode mode) {
    return mode == PipelineMode::kSequential ? execute_sequential() : execute_double_buffered();
  }

  void validate_output() const {
    const auto* actual = static_cast<const float*>(host_output_.data());
    const auto* expected = expected_.data_as<float>();
    for (std::size_t index = 0; index < total_elements_; ++index) {
      HELIX_CHECK(std::abs(actual[index] - expected[index]) <= 2.0e-6F, ErrorCode::kInternal,
                  "overlap benchmark output disagrees with the CPU SiLU oracle");
    }
  }

  [[nodiscard]] std::size_t chunk_count() const noexcept {
    return chunk_count_;
  }
  [[nodiscard]] std::size_t elements_per_chunk() const noexcept {
    return elements_per_chunk_;
  }
  [[nodiscard]] std::size_t total_elements() const noexcept {
    return total_elements_;
  }
  [[nodiscard]] std::size_t transfer_bytes() const noexcept {
    return transfer_bytes_;
  }

private:
  float execute_sequential() {
    start_.record(sequential_stream_);
    for (std::size_t chunk = 0; chunk < chunk_count_; ++chunk) {
      const std::size_t slot = chunk % kPipelineSlots;
      const std::size_t byte_offset = chunk * chunk_bytes_;
      copy_host_to_device_async(device_input_[slot].data(), input_at(byte_offset), chunk_bytes_,
                                sequential_stream_);
      silu_out(device_input_[slot], device_output_[slot], sequential_stream_);
      copy_device_to_host_async(output_at(byte_offset), device_output_[slot].data(), chunk_bytes_,
                                sequential_stream_);
    }
    stop_.record(sequential_stream_);
    stop_.synchronize();
    return elapsed_time_ms(start_, stop_);
  }

  float execute_double_buffered() {
    start_.record(completion_stream_);
    for (Stream& stream : transfer_streams_) {
      stream.wait(start_);
    }
    for (std::size_t chunk = 0; chunk < chunk_count_; ++chunk) {
      const std::size_t slot = chunk % kPipelineSlots;
      const std::size_t byte_offset = chunk * chunk_bytes_;
      copy_host_to_device_async(device_input_[slot].data(), input_at(byte_offset), chunk_bytes_,
                                transfer_streams_[slot]);
      input_ready_[slot].record(transfer_streams_[slot]);
      compute_stream_.wait(input_ready_[slot]);
      silu_out(device_input_[slot], device_output_[slot], compute_stream_);
      compute_ready_[slot].record(compute_stream_);
      transfer_streams_[slot].wait(compute_ready_[slot]);
      copy_device_to_host_async(output_at(byte_offset), device_output_[slot].data(), chunk_bytes_,
                                transfer_streams_[slot]);
      slot_done_[slot].record(transfer_streams_[slot]);
    }
    for (const Event& event : slot_done_) {
      completion_stream_.wait(event);
    }
    stop_.record(completion_stream_);
    stop_.synchronize();
    return elapsed_time_ms(start_, stop_);
  }

  [[nodiscard]] const void* input_at(std::size_t byte_offset) const noexcept {
    return static_cast<const void*>(static_cast<const std::byte*>(host_input_.data()) +
                                    byte_offset);
  }

  [[nodiscard]] void* output_at(std::size_t byte_offset) noexcept {
    return static_cast<void*>(static_cast<std::byte*>(host_output_.data()) + byte_offset);
  }

  std::size_t chunk_count_;
  std::size_t elements_per_chunk_;
  Dimension elements_dimension_;
  std::size_t total_elements_;
  Dimension total_dimension_;
  std::size_t chunk_bytes_;
  std::size_t total_bytes_;
  std::size_t transfer_bytes_;
  PinnedHostBuffer host_input_;
  PinnedHostBuffer host_output_;
  std::array<Tensor, kPipelineSlots> device_input_;
  std::array<Tensor, kPipelineSlots> device_output_;
  Tensor expected_ = Tensor::empty({0});
  Stream sequential_stream_;
  std::array<Stream, kPipelineSlots> transfer_streams_;
  Stream compute_stream_;
  Stream completion_stream_;
  Event start_;
  Event stop_;
  std::array<Event, kPipelineSlots> input_ready_;
  std::array<Event, kPipelineSlots> compute_ready_;
  std::array<Event, kPipelineSlots> slot_done_;
};

void BM_CudaSiluTransferPipeline(benchmark::State& state, PipelineMode mode) {
  if (!runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  set_device(0);
  HELIX_CHECK(state.range(0) > 0 && state.range(1) > 0, ErrorCode::kInvalidArgument,
              "overlap benchmark dimensions must be positive");
  SiluTransferPipeline pipeline(static_cast<std::size_t>(state.range(0)),
                                static_cast<std::size_t>(state.range(1)));
  for (int warmup = 0; warmup < kWarmupPipelines; ++warmup) {
    static_cast<void>(pipeline.execute(mode));
  }
  pipeline.validate_output();

  for ([[maybe_unused]] auto iteration : state) {
    const float elapsed_ms = pipeline.execute(mode);
    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
  }
  pipeline.validate_output();
  state.counters["chunks"] = static_cast<double>(pipeline.chunk_count());
  state.counters["chunk_elements"] = static_cast<double>(pipeline.elements_per_chunk());
  state.counters["async_copy_engines"] = static_cast<double>(device_attribute(
      cudaDevAttrAsyncEngineCount, "querying CUDA asynchronous copy-engine count"));
  state.counters["copy_compute_overlap"] = static_cast<double>(
      device_attribute(cudaDevAttrGpuOverlap, "querying CUDA copy/compute overlap support"));
  state.counters["transfer_bytes"] = static_cast<double>(pipeline.transfer_bytes());
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(pipeline.total_elements()));
  state.SetBytesProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(pipeline.transfer_bytes()));
}

BENCHMARK_CAPTURE(BM_CudaSiluTransferPipeline, Sequential, PipelineMode::kSequential)
    ->Args({8, 1 << 14})
    ->Args({8, 1 << 18})
    ->Args({8, 1 << 20})
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_CudaSiluTransferPipeline, DoubleBuffered, PipelineMode::kDoubleBuffered)
    ->Args({8, 1 << 14})
    ->Args({8, 1 << 18})
    ->Args({8, 1 << 20})
    ->UseManualTime();

} // namespace
} // namespace helix::cuda::kernels
