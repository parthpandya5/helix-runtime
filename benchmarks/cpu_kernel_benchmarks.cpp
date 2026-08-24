#include "helix/kernels/cpu/reference.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace helix::cpu {
namespace {

void finish_elementwise_iteration(benchmark::State& state, std::int64_t elements,
                                  std::int64_t tensors_touched) {
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() * elements * tensors_touched *
                          static_cast<std::int64_t>(sizeof(float)));
}

void BM_CpuAdd(benchmark::State& state) {
  const auto elements = static_cast<Dimension>(state.range(0));
  const Tensor lhs = Tensor::random_uniform({elements}, 101);
  const Tensor rhs = Tensor::random_uniform({elements}, 202);
  Tensor output = Tensor::empty({elements});
  for ([[maybe_unused]] auto iteration : state) {
    add_out(lhs, rhs, output);
    benchmark::ClobberMemory();
  }
  finish_elementwise_iteration(state, state.range(0), 3);
}

void BM_CpuSilu(benchmark::State& state) {
  const auto elements = static_cast<Dimension>(state.range(0));
  const Tensor input = Tensor::random_uniform({elements}, 303, -8.0F, 8.0F);
  Tensor output = Tensor::empty({elements});
  for ([[maybe_unused]] auto iteration : state) {
    silu_out(input, output);
    benchmark::ClobberMemory();
  }
  finish_elementwise_iteration(state, state.range(0), 2);
}

void BM_CpuSoftmax(benchmark::State& state) {
  const auto rows = static_cast<Dimension>(state.range(0));
  const auto width = static_cast<Dimension>(state.range(1));
  const Tensor input = Tensor::random_uniform({rows, width}, 404, -5.0F, 5.0F);
  Tensor output = Tensor::empty({rows, width});
  for ([[maybe_unused]] auto iteration : state) {
    softmax_out(input, output);
    benchmark::ClobberMemory();
  }
  finish_elementwise_iteration(state, state.range(0) * state.range(1), 2);
}

void BM_CpuMatmul(benchmark::State& state) {
  const auto rows = static_cast<Dimension>(state.range(0));
  const auto inner = static_cast<Dimension>(state.range(1));
  const auto columns = static_cast<Dimension>(state.range(2));
  const Tensor lhs = Tensor::random_uniform({rows, inner}, 505);
  const Tensor rhs = Tensor::random_uniform({inner, columns}, 606);
  Tensor output = Tensor::empty({rows, columns});
  for ([[maybe_unused]] auto iteration : state) {
    matmul_out(lhs, rhs, output);
    benchmark::ClobberMemory();
  }
  const auto operations = 2 * state.range(0) * state.range(1) * state.range(2);
  state.SetItemsProcessed(state.iterations() * operations);
  state.counters["FLOP/s"] = benchmark::Counter(static_cast<double>(operations),
                                                benchmark::Counter::kIsIterationInvariantRate);
}

void BM_CpuRmsNorm(benchmark::State& state) {
  const auto rows = static_cast<Dimension>(state.range(0));
  const auto width = static_cast<Dimension>(state.range(1));
  const Tensor input = Tensor::random_uniform({rows, width}, 707);
  const Tensor weight = Tensor::random_uniform({width}, 808, 0.5F, 1.5F);
  Tensor output = Tensor::empty({rows, width});
  for ([[maybe_unused]] auto iteration : state) {
    rms_norm_out(input, weight, 1.0e-5F, output);
    benchmark::ClobberMemory();
  }
  finish_elementwise_iteration(state, state.range(0) * state.range(1), 3);
}

void BM_CpuSwiGlu(benchmark::State& state) {
  const auto elements = static_cast<Dimension>(state.range(0));
  const Tensor gate = Tensor::random_uniform({elements}, 909, -8.0F, 8.0F);
  const Tensor up = Tensor::random_uniform({elements}, 1'010, -2.0F, 2.0F);
  Tensor output = Tensor::empty({elements});
  for ([[maybe_unused]] auto iteration : state) {
    swiglu_out(gate, up, output);
    benchmark::ClobberMemory();
  }
  finish_elementwise_iteration(state, state.range(0), 3);
}

void BM_CpuRope(benchmark::State& state) {
  const auto tokens = static_cast<Dimension>(state.range(0));
  const auto heads = static_cast<Dimension>(state.range(1));
  const auto head_dimension = static_cast<Dimension>(state.range(2));
  const Tensor input = Tensor::random_uniform({tokens, heads, head_dimension}, 1'111);
  const auto cache_elements =
      static_cast<std::size_t>(tokens) * static_cast<std::size_t>(head_dimension / 2);
  std::vector<float> cosine_values(cache_elements);
  std::vector<float> sine_values(cache_elements);
  for (std::size_t index = 0; index < cache_elements; ++index) {
    const float angle = static_cast<float>(index) * 0.001F;
    cosine_values[index] = std::cos(angle);
    sine_values[index] = std::sin(angle);
  }
  const Tensor cosine = Tensor::from_values({tokens, head_dimension / 2}, cosine_values);
  const Tensor sine = Tensor::from_values({tokens, head_dimension / 2}, sine_values);
  Tensor output = Tensor::empty(input.shape());
  for ([[maybe_unused]] auto iteration : state) {
    rope_out(input, cosine, sine, output);
    benchmark::ClobberMemory();
  }
  finish_elementwise_iteration(state, state.range(0) * state.range(1) * state.range(2), 3);
}

void BM_CpuEmbeddingLookup(benchmark::State& state) {
  const auto tokens = static_cast<Dimension>(state.range(0));
  const auto vocabulary = static_cast<Dimension>(state.range(1));
  const auto width = static_cast<Dimension>(state.range(2));
  std::vector<std::uint32_t> ids(static_cast<std::size_t>(tokens));
  for (std::size_t token = 0; token < ids.size(); ++token) {
    ids[token] =
        static_cast<std::uint32_t>((token * 7'919U + 17U) % static_cast<std::size_t>(vocabulary));
  }
  const Tensor token_ids = Tensor::from_values({tokens}, ids);
  const Tensor table = Tensor::random_uniform({vocabulary, width}, 1'212);
  Tensor output = Tensor::empty({tokens, width});
  for (int warmup = 0; warmup < 10; ++warmup) {
    embedding_lookup_out(token_ids, table, output);
  }
  for ([[maybe_unused]] auto iteration : state) {
    embedding_lookup_out(token_ids, table, output);
    benchmark::ClobberMemory();
  }
  const auto elements = state.range(0) * state.range(2);
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() *
                          (2 * elements * static_cast<std::int64_t>(sizeof(float)) +
                           state.range(0) * static_cast<std::int64_t>(sizeof(std::uint32_t))));
  state.counters["tokens"] = static_cast<double>(state.range(0));
  state.counters["vocabulary"] = static_cast<double>(state.range(1));
  state.counters["width"] = static_cast<double>(state.range(2));
}

template <bool TokenToHead> void run_cpu_head_layout(benchmark::State& state) {
  const auto tokens = static_cast<Dimension>(state.range(0));
  const auto heads = static_cast<Dimension>(state.range(1));
  const auto head_dimension = static_cast<Dimension>(state.range(2));
  const Tensor token_input =
      Tensor::random_uniform({tokens, heads, head_dimension}, 1'313, -2.0F, 2.0F);
  const Tensor head_input = token_heads_to_head_major(token_input);
  const Tensor input = TokenToHead ? token_input : head_input;
  Tensor output = Tensor::empty(TokenToHead ? Shape{heads, tokens, head_dimension}
                                            : Shape{tokens, heads, head_dimension});
  const auto execute = [&input, &output]() {
    if constexpr (TokenToHead) {
      token_heads_to_head_major_out(input, output);
    } else {
      head_major_to_token_heads_out(input, output);
    }
  };
  for (int warmup = 0; warmup < 10; ++warmup) {
    execute();
  }
  for ([[maybe_unused]] auto iteration : state) {
    execute();
    benchmark::ClobberMemory();
  }
  finish_elementwise_iteration(state, state.range(0) * state.range(1) * state.range(2), 2);
  state.counters["tokens"] = static_cast<double>(state.range(0));
  state.counters["heads"] = static_cast<double>(state.range(1));
  state.counters["head_dimension"] = static_cast<double>(state.range(2));
}

void BM_CpuTokenHeadsToHeadMajor(benchmark::State& state) {
  run_cpu_head_layout<true>(state);
}

void BM_CpuHeadMajorToTokenHeads(benchmark::State& state) {
  run_cpu_head_layout<false>(state);
}

BENCHMARK(BM_CpuAdd)->Arg(1)->Arg(1024)->Arg(1 << 20);
BENCHMARK(BM_CpuSilu)->Arg(1)->Arg(1024)->Arg(1 << 20);
BENCHMARK(BM_CpuSoftmax)->Args({1, 128})->Args({32, 128})->Args({16, 1023});
BENCHMARK(BM_CpuMatmul)->Args({31, 29, 37})->Args({64, 64, 64})->Args({128, 128, 128});
BENCHMARK(BM_CpuRmsNorm)->Args({1, 4096})->Args({32, 4096});
BENCHMARK(BM_CpuSwiGlu)->Arg(4096)->Arg(32 * 4096);
BENCHMARK(BM_CpuRope)->Args({1, 32, 128})->Args({128, 32, 128});
BENCHMARK(BM_CpuEmbeddingLookup)
    ->Args({1, 128, 32})
    ->Args({7, 128, 32})
    ->Args({1, 4096, 256})
    ->Args({128, 4096, 256})
    ->Args({1, 32000, 1024});
BENCHMARK(BM_CpuTokenHeadsToHeadMajor)
    ->Args({1, 32, 128})
    ->Args({128, 32, 128})
    ->Args({7, 3, 37})
    ->Args({257, 13, 65});
BENCHMARK(BM_CpuHeadMajorToTokenHeads)
    ->Args({1, 32, 128})
    ->Args({128, 32, 128})
    ->Args({7, 3, 37})
    ->Args({257, 13, 65});

} // namespace
} // namespace helix::cpu
