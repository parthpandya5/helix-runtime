#include "helix/memory/system_memory_resource.hpp"
#include "helix/memory/tensor_storage.hpp"
#include "helix/tensor/tensor.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace helix {
namespace {

void BM_TensorZeros(benchmark::State& state) {
  const auto elements = static_cast<Dimension>(state.range(0));
  for ([[maybe_unused]] auto iteration : state) {
    Tensor tensor = Tensor::zeros({elements});
    benchmark::DoNotOptimize(tensor.data());
  }
  state.SetItemsProcessed(state.iterations() * state.range(0));
  state.SetBytesProcessed(state.iterations() * state.range(0) *
                          static_cast<std::int64_t>(sizeof(float)));
}

void BM_TensorDirectAllocationSteadyState(benchmark::State& state) {
  const auto elements = static_cast<Dimension>(state.range(0));
  {
    Tensor warmup = Tensor::empty({elements});
    benchmark::DoNotOptimize(warmup.data());
  }
  for ([[maybe_unused]] auto iteration : state) {
    Tensor tensor = Tensor::empty({elements});
    benchmark::DoNotOptimize(tensor.data());
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_TensorHelixAllocatorSteadyState(benchmark::State& state) {
  const auto elements = static_cast<Dimension>(state.range(0));
  auto allocator =
      std::make_shared<memory::HelixAllocator>(std::make_shared<memory::SystemMemoryResource>());
  {
    Tensor warmup = memory::allocate_cpu_tensor({elements}, DType::kFloat32, allocator);
    benchmark::DoNotOptimize(warmup.data());
  }
  for ([[maybe_unused]] auto iteration : state) {
    Tensor tensor = memory::allocate_cpu_tensor({elements}, DType::kFloat32, allocator);
    benchmark::DoNotOptimize(tensor.data());
    benchmark::ClobberMemory();
  }
  const memory::AllocatorStats stats = allocator->stats();
  state.counters["reserved_bytes"] = static_cast<double>(stats.bytes_reserved);
  state.counters["reuse_count"] = static_cast<double>(stats.reuse_count);
  state.counters["upstream_allocations"] = static_cast<double>(stats.underlying_allocations);
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_TensorZeros)->Arg(1)->Arg(1024)->Arg(1 << 20);
BENCHMARK(BM_TensorDirectAllocationSteadyState)
    ->ArgName("elements")
    ->Arg(1)
    ->Arg(1024)
    ->Arg(1 << 20);
BENCHMARK(BM_TensorHelixAllocatorSteadyState)->ArgName("elements")->Arg(1)->Arg(1024)->Arg(1 << 20);

} // namespace
} // namespace helix
