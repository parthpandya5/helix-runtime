#include "helix/memory/helix_allocator.hpp"
#include "helix/memory/system_memory_resource.hpp"

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

BENCHMARK(BM_SystemResourceFixedAllocateFree)->Arg(256)->Arg(4'096)->Arg(1 << 20);
BENCHMARK(BM_HelixAllocatorFixedAllocateFree)->Arg(256)->Arg(4'096)->Arg(1 << 20);
BENCHMARK(BM_SystemResourceMixedTrace);
BENCHMARK(BM_HelixAllocatorMixedTrace);

} // namespace
} // namespace helix::memory
