#include "allocator_traces.hpp"
#include "helix/memory/cuda_memory_resource.hpp"
#include "helix/memory/cuda_tensor_storage.hpp"
#include "helix/memory/helix_allocator.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/error.hpp"

#include <algorithm>
#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace helix::memory {
namespace {

constexpr std::array<std::size_t, 16> kCudaTraceSizes{
    256,    4'096,  1'024,   65'536, 512,   2'048,   8'192,  131'072,
    16'384, 32'768, 262'144, 4'096,  1'536, 524'288, 65'536, 1'048'576};
constexpr int kCudaAllocatorWarmups = 10;

#if CUDART_VERSION >= 11020
bool cuda_async_allocator_supported(int device_index) {
  int supported = 0;
  HELIX_CUDA_CHECK_CONTEXT(
      cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device_index),
      "querying CUDA stream-ordered allocator support");
  return supported != 0;
}

cudaMemPool_t prepare_default_cuda_pool(int device_index) {
  cudaMemPool_t pool = nullptr;
  HELIX_CUDA_CHECK_CONTEXT(cudaDeviceGetDefaultMemPool(&pool, device_index),
                           "querying the default CUDA memory pool");
  HELIX_CUDA_CHECK_CONTEXT(cudaMemPoolTrimTo(pool, 0),
                           "trimming the default CUDA memory pool before benchmarking");
  std::uint64_t zero = 0;
  HELIX_CUDA_CHECK_CONTEXT(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &zero),
                           "resetting the default CUDA memory-pool reserved high watermark");
  HELIX_CUDA_CHECK_CONTEXT(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero),
                           "resetting the default CUDA memory-pool used high watermark");
  return pool;
}

std::uint64_t cuda_pool_attribute(cudaMemPool_t pool, cudaMemPoolAttr attribute,
                                  const char* context) {
  std::uint64_t value = 0;
  HELIX_CUDA_CHECK_CONTEXT(cudaMemPoolGetAttribute(pool, attribute, &value), context);
  return value;
}

void cuda_malloc_async_free_synchronized(std::size_t bytes, const cuda::Stream& stream) {
  void* pointer = nullptr;
  HELIX_CUDA_CHECK_CONTEXT(cudaMallocAsync(&pointer, bytes, stream.native_handle()),
                           "allocator benchmark cudaMallocAsync");
  benchmark::DoNotOptimize(pointer);
  const cudaError_t free_result = cudaFreeAsync(pointer, stream.native_handle());
  if (free_result != cudaSuccess) {
    cuda::report_cleanup_error(cudaStreamSynchronize(stream.native_handle()),
                               "waiting after allocator benchmark cudaFreeAsync failure");
    cuda::report_cleanup_error(cudaFree(pointer),
                               "cleaning up after allocator benchmark cudaFreeAsync failure");
    HELIX_CUDA_CHECK_CONTEXT(free_result, "allocator benchmark cudaFreeAsync");
  }
  stream.synchronize();
}

void BM_CudaMallocAsyncFreeSynchronizedDefaultPool(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  constexpr int kDeviceIndex = 0;
  cuda::set_device(kDeviceIndex);
  if (!cuda_async_allocator_supported(kDeviceIndex)) {
    state.SkipWithError("CUDA device does not support stream-ordered memory pools");
    return;
  }

  cuda::Stream stream;
  const cudaMemPool_t pool = prepare_default_cuda_pool(kDeviceIndex);
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  for (int warmup = 0; warmup < kCudaAllocatorWarmups; ++warmup) {
    cuda_malloc_async_free_synchronized(bytes, stream);
  }
  for ([[maybe_unused]] auto iteration : state) {
    cuda_malloc_async_free_synchronized(bytes, stream);
  }

  const std::uint64_t used_current = cuda_pool_attribute(
      pool, cudaMemPoolAttrUsedMemCurrent, "querying the default CUDA memory-pool used bytes");
  HELIX_CHECK(used_current == 0, ErrorCode::kInternal,
              "cudaMallocAsync benchmark left live pool bytes");
  state.counters["pool_release_threshold_bytes"] = static_cast<double>(
      cuda_pool_attribute(pool, cudaMemPoolAttrReleaseThreshold,
                          "querying the default CUDA memory-pool release threshold"));
  state.counters["pool_reserved_bytes"] = static_cast<double>(
      cuda_pool_attribute(pool, cudaMemPoolAttrReservedMemCurrent,
                          "querying the default CUDA memory-pool reserved bytes"));
  state.counters["pool_reserved_high_bytes"] = static_cast<double>(
      cuda_pool_attribute(pool, cudaMemPoolAttrReservedMemHigh,
                          "querying the default CUDA memory-pool reserved high watermark"));
  state.SetItemsProcessed(state.iterations());
  HELIX_CUDA_CHECK_CONTEXT(cudaMemPoolTrimTo(pool, 0),
                           "trimming the default CUDA memory pool after benchmarking");
}
#endif

void BM_CudaTensorDirectLifecycle(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  const auto elements = static_cast<Dimension>(state.range(0));
  for (int warmup = 0; warmup < kCudaAllocatorWarmups; ++warmup) {
    const Tensor tensor = Tensor::empty({elements}, DType::kFloat32, Device::cuda(0));
    benchmark::DoNotOptimize(tensor.data());
  }
  for ([[maybe_unused]] auto iteration : state) {
    const Tensor tensor = Tensor::empty({elements}, DType::kFloat32, Device::cuda(0));
    benchmark::DoNotOptimize(tensor.data());
  }
  state.counters["logical_bytes"] =
      static_cast<double>(elements) * static_cast<double>(sizeof(float));
  state.SetItemsProcessed(state.iterations());
}

void BM_CudaTensorHelixStreamBoundLifecycle(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  const auto elements = static_cast<Dimension>(state.range(0));
  auto allocator = std::make_shared<StreamOrderedAllocator>(0);
  auto stream = std::make_shared<cuda::Stream>();
  for (int warmup = 0; warmup < kCudaAllocatorWarmups; ++warmup) {
    {
      const Tensor tensor =
          allocate_cuda_tensor_on_stream({elements}, DType::kFloat32, allocator, stream);
      benchmark::DoNotOptimize(tensor.data());
    }
    static_cast<void>(allocator->wait_and_reclaim_all());
  }
  for ([[maybe_unused]] auto iteration : state) {
    {
      const Tensor tensor =
          allocate_cuda_tensor_on_stream({elements}, DType::kFloat32, allocator, stream);
      benchmark::DoNotOptimize(tensor.data());
    }
    static_cast<void>(allocator->wait_and_reclaim_all());
  }
  const DeferredAllocatorStats stats = allocator->stats();
  HELIX_CHECK(stats.client_allocation_count == 0 && stats.pending_allocation_count == 0,
              ErrorCode::kInternal, "stream-bound CUDA Tensor benchmark leaked an allocation");
  state.counters["logical_bytes"] =
      static_cast<double>(elements) * static_cast<double>(sizeof(float));
  state.counters["reserved_bytes"] = static_cast<double>(stats.allocator.bytes_reserved);
  state.counters["reuse_count"] = static_cast<double>(stats.allocator.reuse_count);
  state.counters["upstream_allocations"] =
      static_cast<double>(stats.allocator.underlying_allocations);
  state.SetItemsProcessed(state.iterations());
}

void BM_CudaMallocFree(benchmark::State& state) {
  if (!cuda::runtime_available()) {
    state.SkipWithError("requires a visible NVIDIA CUDA device");
    return;
  }
  cuda::set_device(0);
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  for (int warmup = 0; warmup < kCudaAllocatorWarmups; ++warmup) {
    void* pointer = nullptr;
    HELIX_CUDA_CHECK_CONTEXT(cudaMalloc(&pointer, bytes), "allocator benchmark warmup cudaMalloc");
    HELIX_CUDA_CHECK_CONTEXT(cudaFree(pointer), "allocator benchmark warmup cudaFree");
  }
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
  for (int warmup = 0; warmup < kCudaAllocatorWarmups; ++warmup) {
    const Allocation allocation = allocator.allocate(bytes);
    allocator.deallocate(allocation);
  }
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
#if CUDART_VERSION >= 11020
BENCHMARK(BM_CudaMallocAsyncFreeSynchronizedDefaultPool)
    ->Arg(256)
    ->Arg(4'096)
    ->Arg(1 << 20)
    ->UseRealTime();
#endif
BENCHMARK(BM_CudaTensorDirectLifecycle)->Arg(1)->Arg(1'024)->Arg(1 << 20)->UseRealTime();
BENCHMARK(BM_CudaTensorHelixStreamBoundLifecycle)->Arg(1)->Arg(1'024)->Arg(1 << 20)->UseRealTime();
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
