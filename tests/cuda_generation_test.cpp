#include "helix/core/error.hpp"
#include "helix/model/cuda_generation.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace helix::model {
namespace {

class CudaGenerationTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!helix::cuda::runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    helix::cuda::set_device(0);
  }
};

LlamaConfig generation_config(std::size_t maximum_length = 8) {
  return LlamaConfig{7, 4, 5, 2, 2, maximum_length, 1.0e-5F, 10'000.0F};
}

Tensor make_tensor(Shape shape, std::size_t seed, bool normalization = false) {
  Tensor tensor = Tensor::empty(std::move(shape));
  float* values = tensor.data_as<float>();
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    values[index] =
        normalization
            ? 0.8F + static_cast<float>((index + seed) % 9) * 0.025F
            : static_cast<float>(static_cast<int>((index * 11 + seed * 7) % 31) - 15) * 0.0175F;
  }
  return tensor;
}

LlamaWeights make_weights(const LlamaConfig& config, bool zero_lm_head = false) {
  const auto vocabulary = static_cast<Dimension>(config.vocabulary_size);
  const auto hidden = static_cast<Dimension>(config.hidden_size);
  const auto intermediate = static_cast<Dimension>(config.intermediate_size);
  std::vector<LlamaLayerWeights> layers;
  layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::size_t seed = 20 + layer * 13;
    layers.emplace_back(
        make_tensor({hidden}, seed, true), make_tensor({hidden, hidden}, seed + 1),
        make_tensor({hidden, hidden}, seed + 2), make_tensor({hidden, hidden}, seed + 3),
        make_tensor({hidden, hidden}, seed + 4), make_tensor({hidden}, seed + 5, true),
        make_tensor({hidden, intermediate}, seed + 6),
        make_tensor({hidden, intermediate}, seed + 7),
        make_tensor({intermediate, hidden}, seed + 8));
  }
  Tensor lm_head =
      zero_lm_head ? Tensor::zeros({hidden, vocabulary}) : make_tensor({hidden, vocabulary}, 102);
  return LlamaWeights(make_tensor({vocabulary, hidden}, 1), std::move(layers),
                      make_tensor({hidden}, 101, true), std::move(lm_head));
}

void expect_device_tensor_near(const Tensor& actual_device, const Tensor& expected,
                               float tolerance = 2.0e-4F) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected.shape());
  ASSERT_EQ(actual.numel(), expected.numel());
  for (std::size_t index = 0; index < actual.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected.data_as<float>()[index], tolerance)
        << "at element " << index;
  }
}

void expect_decode_matches_last_row(const Tensor& actual_device, const Tensor& expected,
                                    float tolerance = 2.0e-4F) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape()[0], 1);
  ASSERT_EQ(actual.shape()[1], expected.shape()[1]);
  const std::size_t vocabulary = static_cast<std::size_t>(expected.shape()[1]);
  const std::size_t last_row = static_cast<std::size_t>(expected.shape()[0] - 1);
  for (std::size_t token = 0; token < vocabulary; ++token) {
    EXPECT_NEAR(actual.data_as<float>()[token],
                expected.data_as<float>()[last_row * vocabulary + token], tolerance)
        << "at vocabulary index " << token;
  }
}

std::vector<std::uint32_t> full_recompute_greedy(CpuLlamaModel& model,
                                                 std::vector<std::uint32_t> context,
                                                 std::size_t max_new_tokens) {
  std::vector<std::uint32_t> generated;
  generated.reserve(max_new_tokens);
  for (std::size_t step = 0; step < max_new_tokens; ++step) {
    const std::uint32_t token = select_greedy_token(model.forward(context));
    generated.push_back(token);
    if (step + 1 < max_new_tokens) {
      context.push_back(token);
    }
  }
  return generated;
}

TEST_F(CudaGenerationTest, SequentialPrefillAndDecodeMatchCpuAndPreserveValidationState) {
  const LlamaConfig config = generation_config();
  const LlamaWeights weights = make_weights(config);
  CpuLlamaModel reference(config, make_weights(config));
  CudaLlamaGenerator generator(CudaLlamaModel(config, weights), 2);
  const std::vector<std::uint32_t> prompt{2, 5, 1};

  const Tensor prefill_logits = generator.prefill(prompt);
  EXPECT_EQ(prefill_logits.device(), Device::cuda(0));
  expect_device_tensor_near(prefill_logits, reference.forward(prompt));
  EXPECT_TRUE(generator.has_prefilled());
  EXPECT_EQ(generator.cached_token_count(), prompt.size());
  EXPECT_EQ(generator.block_tokens(), 2U);
  EXPECT_EQ(generator.stream().device_index(), 0);

  const kv_cache::PagedKvCacheStats before = generator.cache_stats();
  EXPECT_THROW(static_cast<void>(generator.prefill(prompt)), Error);
  EXPECT_THROW(
      static_cast<void>(generator.decode(static_cast<std::uint32_t>(config.vocabulary_size))),
      Error);
  EXPECT_EQ(generator.cached_token_count(), prompt.size());
  EXPECT_EQ(generator.cache_stats().append_count, before.append_count);

  std::vector<std::uint32_t> context = prompt;
  context.push_back(4);
  const Tensor decode_logits = generator.decode(4);
  expect_decode_matches_last_row(decode_logits, reference.forward(context));
  EXPECT_EQ(generator.cached_token_count(), context.size());
  const kv_cache::PagedKvCacheStats stats = generator.cache_stats();
  EXPECT_EQ(stats.active_sequence_count, 1U);
  EXPECT_EQ(stats.active_tokens, context.size());
  EXPECT_EQ(stats.used_blocks, 2U);
  EXPECT_EQ(stats.free_blocks, 2U);
  EXPECT_EQ(stats.sequence_create_count, 2U);
  EXPECT_EQ(stats.release_count, 1U);
}

TEST_F(CudaGenerationTest, GreedyGenerationMatchesFullRecomputationAndEos) {
  const LlamaConfig config = generation_config();
  const LlamaWeights weights = make_weights(config, true);
  const std::vector<std::uint32_t> prompt{6, 1, 4};
  CpuLlamaModel reference(config, make_weights(config, true));
  const std::vector<std::uint32_t> expected = full_recompute_greedy(reference, prompt, 4);
  ASSERT_EQ(expected, (std::vector<std::uint32_t>{0, 0, 0, 0}));

  CudaLlamaGenerator generator(CudaLlamaModel(config, weights), 2);
  const GenerationResult result = generator.generate(prompt, GenerationOptions{4, std::nullopt});
  EXPECT_EQ(result.generated_tokens, expected);
  EXPECT_EQ(result.stop_reason, GenerationStopReason::kMaxNewTokens);
  EXPECT_EQ(result.cached_token_count, prompt.size() + expected.size() - 1);
  EXPECT_EQ(generator.cache_stats().active_tokens, result.cached_token_count);

  CudaLlamaGenerator eos_generator(CudaLlamaModel(config, weights), 2);
  const GenerationResult eos = eos_generator.generate(prompt, GenerationOptions{5, 0});
  EXPECT_EQ(eos.generated_tokens, (std::vector<std::uint32_t>{0}));
  EXPECT_EQ(eos.stop_reason, GenerationStopReason::kEosToken);
  EXPECT_EQ(eos.cached_token_count, prompt.size());
}

TEST_F(CudaGenerationTest, RejectsInvalidSetupAndStopsAtContextWithoutLeakingPages) {
  const LlamaConfig config = generation_config(4);
  const LlamaWeights weights = make_weights(config, true);
  EXPECT_THROW(
      {
        CudaLlamaGenerator invalid(CudaLlamaModel(config, weights), 0);
        static_cast<void>(invalid);
      },
      Error);

  CudaLlamaGenerator generator(CudaLlamaModel(config, weights), 3);
  EXPECT_THROW(static_cast<void>(generator.decode(1)), Error);
  EXPECT_THROW(static_cast<void>(
                   generator.generate(std::vector<std::uint32_t>{1, 2}, GenerationOptions{2, 7})),
               Error);
  EXPECT_THROW(static_cast<void>(generator.prefill(std::span<const std::uint32_t>{})), Error);
  EXPECT_THROW(static_cast<void>(generator.prefill(std::vector<std::uint32_t>{1, 7})), Error);
  EXPECT_FALSE(generator.has_prefilled());
  EXPECT_EQ(generator.cached_token_count(), 0U);
  EXPECT_EQ(generator.cache_stats().sequence_create_count, 1U);

  const std::vector<std::uint32_t> prompt{2, 3, 1};
  const GenerationResult limited = generator.generate(prompt, GenerationOptions{5, std::nullopt});
  EXPECT_EQ(limited.generated_tokens, (std::vector<std::uint32_t>{0, 0}));
  EXPECT_EQ(limited.stop_reason, GenerationStopReason::kContextLimit);
  EXPECT_EQ(limited.cached_token_count, config.max_sequence_length);
  EXPECT_EQ(generator.cache_stats().used_blocks, 2U);
  EXPECT_EQ(generator.cache_stats().free_blocks, 0U);
  EXPECT_THROW(static_cast<void>(generator.decode(0)), Error);
  EXPECT_EQ(generator.cached_token_count(), config.max_sequence_length);

  CudaLlamaGenerator zero_generator(CudaLlamaModel(config, weights), 2);
  const GenerationResult zero = zero_generator.generate(prompt, GenerationOptions{0, std::nullopt});
  EXPECT_TRUE(zero.generated_tokens.empty());
  EXPECT_EQ(zero.stop_reason, GenerationStopReason::kMaxNewTokens);
  EXPECT_EQ(zero.cached_token_count, prompt.size());
}

static_assert(!std::is_copy_constructible_v<CudaLlamaGenerator>);
static_assert(!std::is_move_constructible_v<CudaLlamaGenerator>);

} // namespace
} // namespace helix::model
