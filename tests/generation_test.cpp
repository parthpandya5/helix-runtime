#include "helix/core/error.hpp"
#include "helix/model/generation.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace helix::model {
namespace {

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

LlamaWeights make_weights(const LlamaConfig& config) {
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
  return LlamaWeights(make_tensor({vocabulary, hidden}, 1), std::move(layers),
                      make_tensor({hidden}, 101, true), make_tensor({hidden, vocabulary}, 102));
}

CpuLlamaModel make_model(const LlamaConfig& config) {
  return CpuLlamaModel(config, make_weights(config));
}

void expect_tensor_near(const Tensor& actual, const Tensor& expected, float tolerance) {
  ASSERT_EQ(actual.shape(), expected.shape());
  ASSERT_EQ(actual.numel(), expected.numel());
  for (std::size_t index = 0; index < actual.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected.data_as<float>()[index], tolerance)
        << "at element " << index;
  }
}

void expect_decode_matches_last_row(const Tensor& decode_logits, const Tensor& full_logits,
                                    float tolerance) {
  ASSERT_EQ(decode_logits.shape()[0], 1);
  ASSERT_EQ(decode_logits.shape()[1], full_logits.shape()[1]);
  const std::size_t vocabulary = static_cast<std::size_t>(full_logits.shape()[1]);
  const std::size_t last_row = static_cast<std::size_t>(full_logits.shape()[0] - 1);
  for (std::size_t token = 0; token < vocabulary; ++token) {
    EXPECT_NEAR(decode_logits.data_as<float>()[token],
                full_logits.data_as<float>()[last_row * vocabulary + token], tolerance)
        << "at vocabulary index " << token;
  }
}

std::vector<std::uint32_t> full_recompute_greedy(CpuLlamaModel& model,
                                                 std::vector<std::uint32_t> context,
                                                 std::size_t max_new_tokens) {
  std::vector<std::uint32_t> generated;
  generated.reserve(max_new_tokens);
  for (std::size_t step = 0; step < max_new_tokens; ++step) {
    const Tensor logits = model.forward(context);
    const std::uint32_t token = select_greedy_token(logits);
    generated.push_back(token);
    if (step + 1 < max_new_tokens) {
      context.push_back(token);
    }
  }
  return generated;
}

TEST(GreedySelectionTest, UsesLastRowLowestTieAndRejectsInvalidLogits) {
  const Tensor logits =
      Tensor::from_values({2, 4}, std::array{9.0F, 8.0F, 7.0F, 6.0F, 3.0F, 5.0F, 5.0F, 1.0F});
  EXPECT_EQ(select_greedy_token(logits), 1U);

  EXPECT_THROW(static_cast<void>(select_greedy_token(Tensor::zeros({4}))), Error);
  EXPECT_THROW(static_cast<void>(select_greedy_token(Tensor::zeros({0, 4}))), Error);
  EXPECT_THROW(static_cast<void>(select_greedy_token(Tensor::zeros({1, 4}, DType::kFloat16))),
               Error);
  Tensor non_finite = Tensor::zeros({1, 3});
  non_finite.data_as<float>()[1] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(static_cast<void>(select_greedy_token(non_finite)), Error);
  EXPECT_THROW(
      static_cast<void>(generation_stop_reason_name(static_cast<GenerationStopReason>(999))),
      Error);
}

TEST(CpuLlamaGeneratorTest, CachedDecodeMatchesFullPrefixRecomputation) {
  const LlamaConfig config = generation_config();
  CpuLlamaModel reference = make_model(config);
  CpuLlamaGenerator generator(make_model(config));
  const std::vector<std::uint32_t> prompt{2, 5, 1};

  const Tensor prefill_logits = generator.prefill(prompt);
  expect_tensor_near(prefill_logits, reference.forward(prompt), 5.0e-5F);
  EXPECT_TRUE(generator.has_prefilled());
  EXPECT_EQ(generator.cached_token_count(), prompt.size());
  EXPECT_EQ(generator.cache_stats().active_tokens, prompt.size());
  EXPECT_EQ(generator.cache_stats().append_count, prompt.size());

  std::vector<std::uint32_t> context = prompt;
  context.push_back(4);
  const Tensor first_decode = generator.decode(4);
  expect_decode_matches_last_row(first_decode, reference.forward(context), 5.0e-5F);
  context.push_back(0);
  const Tensor second_decode = generator.decode(0);
  expect_decode_matches_last_row(second_decode, reference.forward(context), 5.0e-5F);
  EXPECT_EQ(generator.cached_token_count(), context.size());
  EXPECT_EQ(generator.cache_stats().active_tokens, context.size());
}

TEST(CpuLlamaModelTest, PagedSequentialDecodeMatchesEveryFullPrefixPosition) {
  const LlamaConfig config = generation_config();
  CpuLlamaModel model = make_model(config);
  kv_cache::PagedKvCache cache(kv_cache::PagedKvCacheConfig{
      config.layer_count, config.attention_head_count, config.head_dimension(), 2, 4});
  const kv_cache::PagedSequenceHandle sequence = cache.create_sequence();
  std::vector<std::uint32_t> context;

  for (const std::uint32_t token : std::array<std::uint32_t, 4>{2, 5, 1, 4}) {
    const CpuLlamaDecodeResult decoded = model.decode_one(token, context.size(), cache, sequence);
    context.push_back(token);
    expect_decode_matches_last_row(decoded.logits, model.forward(context), 5.0e-5F);
    cache.append(sequence, decoded.token_keys, decoded.token_values);
    EXPECT_EQ(cache.length(sequence), context.size());
  }
  EXPECT_EQ(cache.block_table(sequence).size(), 2U);
}

TEST(CpuLlamaGeneratorTest, GreedyGenerationMatchesRepeatedFullRecomputation) {
  const LlamaConfig config = generation_config();
  const std::vector<std::uint32_t> prompt{6, 1, 4};
  CpuLlamaModel reference = make_model(config);
  const std::vector<std::uint32_t> expected = full_recompute_greedy(reference, prompt, 4);
  CpuLlamaGenerator generator(make_model(config));

  const GenerationResult result = generator.generate(prompt, GenerationOptions{4, std::nullopt});
  EXPECT_EQ(result.generated_tokens, expected);
  EXPECT_EQ(result.stop_reason, GenerationStopReason::kMaxNewTokens);
  EXPECT_EQ(result.cached_token_count, prompt.size() + expected.size() - 1);
  EXPECT_EQ(generator.cache_stats().active_tokens, result.cached_token_count);
}

TEST(CpuLlamaGeneratorTest, AppliesEosZeroBudgetAndContextLimitStopConditions) {
  const std::vector<std::uint32_t> prompt{2, 3, 1};
  const LlamaConfig normal_config = generation_config();
  CpuLlamaModel reference = make_model(normal_config);
  const std::uint32_t first_token = select_greedy_token(reference.forward(prompt));

  CpuLlamaGenerator eos_generator(make_model(normal_config));
  const GenerationResult eos_result =
      eos_generator.generate(prompt, GenerationOptions{5, first_token});
  EXPECT_EQ(eos_result.generated_tokens, (std::vector<std::uint32_t>{first_token}));
  EXPECT_EQ(eos_result.stop_reason, GenerationStopReason::kEosToken);
  EXPECT_EQ(eos_result.cached_token_count, prompt.size());

  CpuLlamaGenerator zero_generator(make_model(normal_config));
  const GenerationResult zero_result =
      zero_generator.generate(prompt, GenerationOptions{0, std::nullopt});
  EXPECT_TRUE(zero_result.generated_tokens.empty());
  EXPECT_EQ(zero_result.stop_reason, GenerationStopReason::kMaxNewTokens);
  EXPECT_EQ(zero_result.cached_token_count, prompt.size());

  const LlamaConfig short_config = generation_config(4);
  CpuLlamaGenerator limited_generator(make_model(short_config));
  const GenerationResult limited_result =
      limited_generator.generate(prompt, GenerationOptions{5, std::nullopt});
  EXPECT_EQ(limited_result.generated_tokens.size(), 2U);
  EXPECT_EQ(limited_result.stop_reason, GenerationStopReason::kContextLimit);
  EXPECT_EQ(limited_result.cached_token_count, short_config.max_sequence_length);
}

TEST(CpuLlamaGeneratorTest, StateAndInputFailuresDoNotAdvanceCache) {
  const LlamaConfig config = generation_config(4);
  CpuLlamaGenerator generator(make_model(config));
  EXPECT_THROW(static_cast<void>(generator.decode(1)), Error);
  EXPECT_EQ(generator.cached_token_count(), 0U);
  EXPECT_THROW(static_cast<void>(generator.prefill(std::span<const std::uint32_t>{})), Error);
  EXPECT_FALSE(generator.has_prefilled());
  EXPECT_EQ(generator.cached_token_count(), 0U);
  EXPECT_THROW(static_cast<void>(generator.prefill(std::vector<std::uint32_t>{1, 7})), Error);
  EXPECT_FALSE(generator.has_prefilled());
  EXPECT_EQ(generator.cached_token_count(), 0U);

  const std::vector<std::uint32_t> prompt{1, 2, 3};
  static_cast<void>(generator.prefill(prompt));
  EXPECT_THROW(static_cast<void>(generator.prefill(prompt)), Error);
  EXPECT_THROW(static_cast<void>(generator.decode(7)), Error);
  EXPECT_EQ(generator.cached_token_count(), prompt.size());
  static_cast<void>(generator.decode(4));
  EXPECT_THROW(static_cast<void>(generator.decode(1)), Error);
  EXPECT_EQ(generator.cached_token_count(), config.max_sequence_length);
}

TEST(CpuLlamaGeneratorTest, InvalidEosAndCacheGeometryAreRejectedBeforeMutation) {
  const LlamaConfig config = generation_config();
  CpuLlamaGenerator generator(make_model(config));
  EXPECT_THROW(static_cast<void>(
                   generator.generate(std::vector<std::uint32_t>{1, 2}, GenerationOptions{2, 7})),
               Error);
  EXPECT_FALSE(generator.has_prefilled());
  EXPECT_EQ(generator.cached_token_count(), 0U);

  CpuLlamaModel model = make_model(config);
  kv_cache::ContiguousKvCache wrong_cache(
      kv_cache::ContiguousKvCacheConfig{config.layer_count, 1, config.head_dimension()});
  const kv_cache::SequenceHandle wrong_sequence = wrong_cache.create_sequence(2);
  EXPECT_THROW(static_cast<void>(model.decode_one(1, 0, wrong_cache, wrong_sequence)), Error);

  kv_cache::ContiguousKvCache cache(kv_cache::ContiguousKvCacheConfig{
      config.layer_count, config.attention_head_count, config.head_dimension()});
  const kv_cache::SequenceHandle sequence = cache.create_sequence(2);
  EXPECT_THROW(static_cast<void>(model.decode_one(1, 1, cache, sequence)), Error);
  EXPECT_EQ(cache.length(sequence), 0U);

  const CpuLlamaDecodeResult first = model.decode_one(1, 0, cache, sequence);
  EXPECT_EQ(first.token_keys.shape(), (Shape{2, 2, 2}));
  EXPECT_EQ(first.token_values.shape(), (Shape{2, 2, 2}));
  expect_decode_matches_last_row(first.logits, model.forward(std::vector<std::uint32_t>{1}),
                                 5.0e-5F);
  EXPECT_EQ(cache.length(sequence), 0U);
  cache.append(sequence, first.token_keys, first.token_values);
  EXPECT_EQ(cache.length(sequence), 1U);
}

static_assert(!std::is_move_constructible_v<CpuLlamaGenerator>);

} // namespace
} // namespace helix::model
