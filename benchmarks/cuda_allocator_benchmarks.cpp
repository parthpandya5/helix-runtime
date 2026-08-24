#include "allocator_traces.hpp"
#include "helix/memory/cuda_memory_resource.hpp"
#include "helix/memory/helix_allocator.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace helix::memory {
namespace {

constexpr std::array<std::size_t, 16> kCudaTraceSizes{
    256,    4'096,  1'024,   65'536, 512,   2'048,   8'192,  131'072,
    16'384, 32'768, 262'144, 4'096,  1'536, 524'288, 65'536, 1'048'576};

void BM_CudaMallocFree(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  for ([[maybe_unused]] auto iteration : state) {
    void* pointer = nullptr;
    HELIX_CUDA_CHECK_CONTEXT(cudaMalloc(&pointer, bytes), "allocator benchmark cudaMalloc");
    benchmark::DoNotOptimize(pointer);
    HELIX_CUDA_CHECK_CONTEXT(cudaFree(pointer), "allocator benchmark cudaFree");
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_CudaHelixAllocatorFixed(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  HelixAllocator allocator(std::make_shared<CudaMemoryResource>(0));
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  for ([[maybe_unused]] auto iteration : state) {
    const Allocation allocation = allocator.allocate(bytes);
    benchmark::DoNotOptimize(allocation.pointer());
    allocator.deallocate(allocation);
  }
  const AllocatorStats stats = allocator.stats();
  state.counters["upstream_allocations"] = static_cast<double>(stats.underlying_allocations);
  state.counters["reuse_count"] = static_cast<double>(stats.reuse_count);
  state.counters["reserved_bytes"] = static_cast<double>(stats.bytes_reserved);
  state.SetItemsProcessed(state.iterations());
}

void BM_CudaMallocMixedTrace(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  std::vector<void*> pointers(kCudaTraceSizes.size());
  for ([[maybe_unused]] auto iteration : state) {
    for (std::size_t index = 0; index < kCudaTraceSizes.size(); ++index) {
      HELIX_CUDA_CHECK_CONTEXT(cudaMalloc(&pointers[index], kCudaTraceSizes[index]),
                               "mixed allocator trace cudaMalloc");
      benchmark::DoNotOptimize(pointers[index]);
    }
    for (std::size_t index = pointers.size(); index != 0; --index) {
      HELIX_CUDA_CHECK_CONTEXT(cudaFree(pointers[index - 1]), "mixed allocator trace cudaFree");
    }
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(kCudaTraceSizes.size()));
}

void BM_CudaHelixAllocatorMixedTrace(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  HelixAllocator allocator(std::make_shared<CudaMemoryResource>(0));
  std::vector<Allocation> allocations(kCudaTraceSizes.size());
  for ([[maybe_unused]] auto iteration : state) {
    for (std::size_t index = 0; index < kCudaTraceSizes.size(); ++index) {
      allocations[index] = allocator.allocate(kCudaTraceSizes[index]);
      benchmark::DoNotOptimize(allocations[index].pointer());
    }
    for (std::size_t index = allocations.size(); index != 0; --index) {
      allocator.deallocate(allocations[index - 1]);
    }
  }
  const AllocatorStats stats = allocator.stats();
  state.counters["upstream_allocations"] = static_cast<double>(stats.underlying_allocations);
  state.counters["reuse_count"] = static_cast<double>(stats.reuse_count);
  state.counters["reserved_bytes"] = static_cast<double>(stats.bytes_reserved);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(kCudaTraceSizes.size()));
}

void replay_cuda_malloc_trace(const bench::AllocationTrace& trace, std::vector<void*>& pointers) {
  for (const bench::TraceOperation& operation : trace.operations) {
    if (operation.kind == bench::TraceOperationKind::kAllocate) {
      HELIX_CUDA_CHECK_CONTEXT(cudaMalloc(&pointers[operation.slot], operation.bytes),
                               "allocator workload trace cudaMalloc");
      benchmark::DoNotOptimize(pointers[operation.slot]);
    } else {
      HELIX_CUDA_CHECK_CONTEXT(cudaFree(pointers[operation.slot]),
                               "allocator workload trace cudaFree");
      pointers[operation.slot] = nullptr;
    }
  }
}

void replay_cuda_helix_trace(HelixAllocator& allocator, const bench::AllocationTrace& trace,
                             std::vector<Allocation>& allocations) {
  for (const bench::TraceOperation& operation : trace.operations) {
    if (operation.kind == bench::TraceOperationKind::kAllocate) {
      allocations[operation.slot] = allocator.allocate(operation.bytes);
      benchmark::DoNotOptimize(allocations[operation.slot].pointer());
    } else {
      allocator.deallocate(allocations[operation.slot]);
      allocations[operation.slot] = Allocation{};
    }
  }
}

struct CudaFragmentationProfile {
  std::size_t bytes_reserved = 0;
  std::size_t peak_bytes_allocated = 0;
  std::size_t segment_count = 0;
  std::size_t underlying_allocations = 0;
  double maximum_external_fragmentation = 0.0;
  double maximum_internal_fragmentation = 0.0;
};

CudaFragmentationProfile profile_cuda_helix_trace(const bench::AllocationTrace& trace) {
  HelixAllocator allocator(std::make_shared<CudaMemoryResource>(0));
  std::vector<Allocation> allocations(trace.slot_count);
  CudaFragmentationProfile profile;
  for (const bench::TraceOperation& operation : trace.operations) {
    if (operation.kind == bench::TraceOperationKind::kAllocate) {
      allocations[operation.slot] = allocator.allocate(operation.bytes);
    } else {
      allocator.deallocate(allocations[operation.slot]);
      allocations[operation.slot] = Allocation{};
    }
    const AllocatorStats stats = allocator.stats();
    profile.maximum_external_fragmentation =
        std::max(profile.maximum_external_fragmentation, stats.external_fragmentation);
    profile.maximum_internal_fragmentation =
        std::max(profile.maximum_internal_fragmentation, stats.internal_fragmentation);
  }
  const AllocatorStats stats = allocator.stats();
  HELIX_CHECK(stats.active_allocation_count == 0 && stats.bytes_allocated == 0,
              ErrorCode::kInternal, "CUDA allocator benchmark trace leaked a profiled allocation");
  profile.bytes_reserved = stats.bytes_reserved;
  profile.peak_bytes_allocated = stats.peak_bytes_allocated;
  profile.segment_count = stats.segment_count;
  profile.underlying_allocations = stats.underlying_allocations;
  return profile;
}

void record_cuda_trace_counters(benchmark::State& state, const bench::AllocationTrace& trace) {
  state.counters["allocations_per_trace"] = static_cast<double>(trace.allocation_count);
  state.counters["operations_per_trace"] = static_cast<double>(trace.operations.size());
  state.counters["peak_live_requested"] = static_cast<double>(trace.peak_live_requested_bytes);
}

void BM_CudaMallocWorkloadTrace(benchmark::State& state, bench::AllocationTraceKind kind) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  const bench::AllocationTrace& trace = bench::allocation_trace(kind);
  std::vector<void*> pointers(trace.slot_count, nullptr);
  for ([[maybe_unused]] auto iteration : state) {
    replay_cuda_malloc_trace(trace, pointers);
  }
  record_cuda_trace_counters(state, trace);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(trace.allocation_count));
}

void BM_CudaHelixAllocatorWorkloadTrace(benchmark::State& state, bench::AllocationTraceKind kind) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  const bench::AllocationTrace& trace = bench::allocation_trace(kind);
  HelixAllocator allocator(std::make_shared<CudaMemoryResource>(0));
  std::vector<Allocation> allocations(trace.slot_count);
  for ([[maybe_unused]] auto iteration : state) {
    replay_cuda_helix_trace(allocator, trace, allocations);
  }
  const AllocatorStats timed_stats = allocator.stats();
  const CudaFragmentationProfile profile = profile_cuda_helix_trace(trace);
  HELIX_CHECK(timed_stats.active_allocation_count == 0 && timed_stats.bytes_allocated == 0,
              ErrorCode::kInternal, "CUDA allocator benchmark trace leaked a timed allocation");
  record_cuda_trace_counters(state, trace);
  state.counters["maximum_external_fragmentation"] = profile.maximum_external_fragmentation;
  state.counters["maximum_internal_fragmentation"] = profile.maximum_internal_fragmentation;
  state.counters["pool_peak_allocated"] = static_cast<double>(profile.peak_bytes_allocated);
  state.counters["pool_reserved"] = static_cast<double>(profile.bytes_reserved);
  state.counters["pool_segments"] = static_cast<double>(profile.segment_count);
  state.counters["reuse_fraction"] = timed_stats.allocation_count == 0
                                         ? 0.0
                                         : static_cast<double>(timed_stats.reuse_count) /
                                               static_cast<double>(timed_stats.allocation_count);
  state.counters["upstream_allocations"] = static_cast<double>(profile.underlying_allocations);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(trace.allocation_count));
}

BENCHMARK(BM_CudaMallocFree)->Arg(256)->Arg(4'096)->Arg(1 << 20)->UseRealTime();
BENCHMARK(BM_CudaHelixAllocatorFixed)->Arg(256)->Arg(4'096)->Arg(1 << 20)->UseRealTime();
BENCHMARK(BM_CudaMallocMixedTrace)->UseRealTime();
BENCHMARK(BM_CudaHelixAllocatorMixedTrace)->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaMallocWorkloadTrace, RandomSize, bench::AllocationTraceKind::kRandomSize)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaHelixAllocatorWorkloadTrace, RandomSize,
                  bench::AllocationTraceKind::kRandomSize)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaMallocWorkloadTrace, SmallHeavy, bench::AllocationTraceKind::kSmallHeavy)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaHelixAllocatorWorkloadTrace, SmallHeavy,
                  bench::AllocationTraceKind::kSmallHeavy)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaMallocWorkloadTrace, LargeHeavy, bench::AllocationTraceKind::kLargeHeavy)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaHelixAllocatorWorkloadTrace, LargeHeavy,
                  bench::AllocationTraceKind::kLargeHeavy)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaMallocWorkloadTrace, Bursty, bench::AllocationTraceKind::kBursty)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaHelixAllocatorWorkloadTrace, Bursty, bench::AllocationTraceKind::kBursty)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaMallocWorkloadTrace, LlmLike, bench::AllocationTraceKind::kLlmLike)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_CudaHelixAllocatorWorkloadTrace, LlmLike, bench::AllocationTraceKind::kLlmLike)
    ->UseRealTime();

} // namespace
} // namespace helix::memory
