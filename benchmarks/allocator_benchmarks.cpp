#include "allocator_traces.hpp"
#include "helix/memory/helix_allocator.hpp"
#include "helix/memory/system_memory_resource.hpp"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace helix::memory {
namespace {

constexpr std::size_t kAlignment = 256;

void BM_SystemResourceFixedAllocateFree(benchmark::State& state) {
  SystemMemoryResource resource;
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  for ([[maybe_unused]] auto iteration : state) {
    void* pointer = resource.allocate(bytes, kAlignment);
    benchmark::DoNotOptimize(pointer);
    resource.deallocate(pointer, bytes, kAlignment);
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_HelixAllocatorFixedAllocateFree(benchmark::State& state) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>());
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  for ([[maybe_unused]] auto iteration : state) {
    const Allocation allocation = allocator.allocate(bytes);
    benchmark::DoNotOptimize(allocation.pointer());
    allocator.deallocate(allocation);
  }
  const AllocatorStats stats = allocator.stats();
  state.counters["upstream_allocations"] = static_cast<double>(stats.underlying_allocations);
  state.counters["reuse_count"] = static_cast<double>(stats.reuse_count);
  state.SetItemsProcessed(state.iterations());
}

constexpr std::array<std::size_t, 16> kTraceSizes{64,    4'096, 128,   65'536, 512, 2'048,
                                                  256,   8'192, 1'024, 32'768, 192, 16'384,
                                                  4'096, 320,   1'536, 262'144};

void BM_SystemResourceMixedTrace(benchmark::State& state) {
  SystemMemoryResource resource;
  std::vector<void*> pointers(kTraceSizes.size());
  for ([[maybe_unused]] auto iteration : state) {
    for (std::size_t index = 0; index < kTraceSizes.size(); ++index) {
      pointers[index] = resource.allocate(kTraceSizes[index], kAlignment);
      benchmark::DoNotOptimize(pointers[index]);
    }
    for (std::size_t index = kTraceSizes.size(); index != 0; --index) {
      resource.deallocate(pointers[index - 1], kTraceSizes[index - 1], kAlignment);
    }
  }
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(kTraceSizes.size()));
}

void BM_HelixAllocatorMixedTrace(benchmark::State& state) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>());
  std::vector<Allocation> allocations(kTraceSizes.size());
  for ([[maybe_unused]] auto iteration : state) {
    for (std::size_t index = 0; index < kTraceSizes.size(); ++index) {
      allocations[index] = allocator.allocate(kTraceSizes[index]);
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
                          static_cast<benchmark::IterationCount>(kTraceSizes.size()));
}

void replay_system_trace(SystemMemoryResource& resource, const bench::AllocationTrace& trace,
                         std::vector<void*>& pointers, std::vector<std::size_t>& sizes) {
  for (const bench::TraceOperation& operation : trace.operations) {
    if (operation.kind == bench::TraceOperationKind::kAllocate) {
      pointers[operation.slot] = resource.allocate(operation.bytes, kAlignment);
      sizes[operation.slot] = operation.bytes;
      benchmark::DoNotOptimize(pointers[operation.slot]);
    } else {
      resource.deallocate(pointers[operation.slot], sizes[operation.slot], kAlignment);
      pointers[operation.slot] = nullptr;
      sizes[operation.slot] = 0;
    }
  }
}

void replay_helix_trace(HelixAllocator& allocator, const bench::AllocationTrace& trace,
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

struct FragmentationProfile {
  std::size_t bytes_reserved = 0;
  std::size_t peak_bytes_allocated = 0;
  std::size_t segment_count = 0;
  std::size_t underlying_allocations = 0;
  double maximum_external_fragmentation = 0.0;
  double maximum_internal_fragmentation = 0.0;
};

FragmentationProfile profile_helix_trace(const bench::AllocationTrace& trace) {
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>());
  std::vector<Allocation> allocations(trace.slot_count);
  FragmentationProfile profile;
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
              ErrorCode::kInternal, "allocator benchmark trace leaked a profiled allocation");
  profile.bytes_reserved = stats.bytes_reserved;
  profile.peak_bytes_allocated = stats.peak_bytes_allocated;
  profile.segment_count = stats.segment_count;
  profile.underlying_allocations = stats.underlying_allocations;
  return profile;
}

void record_trace_counters(benchmark::State& state, const bench::AllocationTrace& trace) {
  state.counters["allocations_per_trace"] = static_cast<double>(trace.allocation_count);
  state.counters["operations_per_trace"] = static_cast<double>(trace.operations.size());
  state.counters["peak_live_requested"] = static_cast<double>(trace.peak_live_requested_bytes);
}

void BM_SystemResourceWorkloadTrace(benchmark::State& state, bench::AllocationTraceKind kind) {
  const bench::AllocationTrace& trace = bench::allocation_trace(kind);
  SystemMemoryResource resource;
  std::vector<void*> pointers(trace.slot_count, nullptr);
  std::vector<std::size_t> sizes(trace.slot_count, 0);
  for ([[maybe_unused]] auto iteration : state) {
    replay_system_trace(resource, trace, pointers, sizes);
  }
  record_trace_counters(state, trace);
  state.SetItemsProcessed(state.iterations() *
                          static_cast<benchmark::IterationCount>(trace.allocation_count));
}

void BM_HelixAllocatorWorkloadTrace(benchmark::State& state, bench::AllocationTraceKind kind) {
  const bench::AllocationTrace& trace = bench::allocation_trace(kind);
  HelixAllocator allocator(std::make_shared<SystemMemoryResource>());
  std::vector<Allocation> allocations(trace.slot_count);
  for ([[maybe_unused]] auto iteration : state) {
    replay_helix_trace(allocator, trace, allocations);
  }
  const AllocatorStats timed_stats = allocator.stats();
  const FragmentationProfile profile = profile_helix_trace(trace);
  HELIX_CHECK(timed_stats.active_allocation_count == 0 && timed_stats.bytes_allocated == 0,
              ErrorCode::kInternal, "allocator benchmark trace leaked a timed allocation");
  record_trace_counters(state, trace);
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

BENCHMARK(BM_SystemResourceFixedAllocateFree)->Arg(256)->Arg(4'096)->Arg(1 << 20);
BENCHMARK(BM_HelixAllocatorFixedAllocateFree)->Arg(256)->Arg(4'096)->Arg(1 << 20);
BENCHMARK(BM_SystemResourceMixedTrace);
BENCHMARK(BM_HelixAllocatorMixedTrace);
BENCHMARK_CAPTURE(BM_SystemResourceWorkloadTrace, RandomSize,
                  bench::AllocationTraceKind::kRandomSize);
BENCHMARK_CAPTURE(BM_HelixAllocatorWorkloadTrace, RandomSize,
                  bench::AllocationTraceKind::kRandomSize);
BENCHMARK_CAPTURE(BM_SystemResourceWorkloadTrace, SmallHeavy,
                  bench::AllocationTraceKind::kSmallHeavy);
BENCHMARK_CAPTURE(BM_HelixAllocatorWorkloadTrace, SmallHeavy,
                  bench::AllocationTraceKind::kSmallHeavy);
BENCHMARK_CAPTURE(BM_SystemResourceWorkloadTrace, LargeHeavy,
                  bench::AllocationTraceKind::kLargeHeavy);
BENCHMARK_CAPTURE(BM_HelixAllocatorWorkloadTrace, LargeHeavy,
                  bench::AllocationTraceKind::kLargeHeavy);
BENCHMARK_CAPTURE(BM_SystemResourceWorkloadTrace, Bursty, bench::AllocationTraceKind::kBursty);
BENCHMARK_CAPTURE(BM_HelixAllocatorWorkloadTrace, Bursty, bench::AllocationTraceKind::kBursty);
BENCHMARK_CAPTURE(BM_SystemResourceWorkloadTrace, LlmLike, bench::AllocationTraceKind::kLlmLike);
BENCHMARK_CAPTURE(BM_HelixAllocatorWorkloadTrace, LlmLike, bench::AllocationTraceKind::kLlmLike);

} // namespace
} // namespace helix::memory
