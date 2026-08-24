#include "helix/core/dtype.hpp"

#include <benchmark/benchmark.h>

namespace helix {
namespace {

void BM_DTypeMetadata(benchmark::State& state) {
  const auto dtype = static_cast<DType>(state.range(0));
  for ([[maybe_unused]] auto iteration : state) {
    benchmark::DoNotOptimize(dtype_size(dtype));
    benchmark::DoNotOptimize(dtype_name(dtype));
  }
}

BENCHMARK(BM_DTypeMetadata)
    ->Arg(static_cast<int>(DType::kFloat32))
    ->Arg(static_cast<int>(DType::kFloat16))
    ->Arg(static_cast<int>(DType::kUInt32));

} // namespace
} // namespace helix
