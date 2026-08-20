#include "helix/memory/cuda_memory_resource.hpp"
#include "helix/memory/helix_allocator.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

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

BENCHMARK(BM_CudaMallocFree)->Arg(256)->Arg(4'096)->Arg(1 << 20)->UseRealTime();
BENCHMARK(BM_CudaHelixAllocatorFixed)->Arg(256)->Arg(4'096)->Arg(1 << 20)->UseRealTime();
BENCHMARK(BM_CudaMallocMixedTrace)->UseRealTime();
BENCHMARK(BM_CudaHelixAllocatorMixedTrace)->UseRealTime();

} // namespace
} // namespace helix::memory
