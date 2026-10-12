#include "helix/core/build_info.hpp"
#include "helix/kernels/cpu/reference.hpp"

#if HELIX_CONSUMER_EXPECT_CUDA
#include "helix/kv_cache/cuda_paged.hpp"
#include "helix/model/cuda_generation.hpp"
#include "helix/model/cuda_llama.hpp"
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

#if HELIX_CONSUMER_EXPECT_CUDA
static_assert(!std::is_copy_constructible_v<helix::kv_cache::CudaPagedKvCache>);
static_assert(!std::is_copy_constructible_v<helix::kv_cache::CudaPagedKvCache::DecodeView>);
static_assert(!std::is_move_constructible_v<helix::model::CudaLlamaGenerator>);
static_assert(!std::is_copy_constructible_v<helix::model::CudaLlamaModel>);
#endif

int main() {
  const bool expected_cuda = HELIX_CONSUMER_EXPECT_CUDA != 0;
  if (helix::build_info().cuda_enabled != expected_cuda) {
    return 1;
  }

  const helix::Tensor lhs = helix::Tensor::from_values({3}, std::array{1.0F, -2.0F, 3.5F});
  const helix::Tensor rhs = helix::Tensor::from_values({3}, std::array{4.0F, 5.0F, -0.5F});
  const helix::Tensor sum = helix::cpu::add(lhs, rhs);
  constexpr std::array expected{5.0F, 3.0F, 3.0F};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (std::abs(sum.data_as<float>()[index] - expected[index]) > 1.0e-6F) {
      return 2;
    }
  }
  return 0;
}
