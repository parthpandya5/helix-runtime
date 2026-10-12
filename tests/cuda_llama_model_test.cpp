#include "helix/core/error.hpp"
#include "helix/kv_cache/cuda_paged.hpp"
#include "helix/model/cuda_llama.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace helix::model {
namespace {

constexpr float kAbsoluteTolerance = 2.0e-4F;
constexpr float kRelativeTolerance = 2.0e-3F;

LlamaConfig tiny_config() {
  return LlamaConfig{11, 8, 12, 2, 2, 8, 1.0e-5F, 10'000.0F};
}

LlamaWeights make_weights(const LlamaConfig& config) {
  const Dimension vocabulary = static_cast<Dimension>(config.vocabulary_size);
  const Dimension hidden = static_cast<Dimension>(config.hidden_size);
  const Dimension intermediate = static_cast<Dimension>(config.intermediate_size);
  std::vector<LlamaLayerWeights> layers;
  layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::uint64_t seed = 100 + layer * 20;
    layers.emplace_back(Tensor::random_uniform({hidden}, seed, 0.8F, 1.2F),
                        Tensor::random_uniform({hidden, hidden}, seed + 1, -0.2F, 0.2F),
                        Tensor::random_uniform({hidden, hidden}, seed + 2, -0.2F, 0.2F),
                        Tensor::random_uniform({hidden, hidden}, seed + 3, -0.2F, 0.2F),
                        Tensor::random_uniform({hidden, hidden}, seed + 4, -0.2F, 0.2F),
                        Tensor::random_uniform({hidden}, seed + 5, 0.8F, 1.2F),
                        Tensor::random_uniform({hidden, intermediate}, seed + 6, -0.2F, 0.2F),
                        Tensor::random_uniform({hidden, intermediate}, seed + 7, -0.2F, 0.2F),
                        Tensor::random_uniform({intermediate, hidden}, seed + 8, -0.2F, 0.2F));
  }
  return LlamaWeights(Tensor::random_uniform({vocabulary, hidden}, 1, -0.25F, 0.25F),
                      std::move(layers), Tensor::random_uniform({hidden}, 2, 0.8F, 1.2F),
                      Tensor::random_uniform({hidden, vocabulary}, 3, -0.2F, 0.2F));
}

std::pair<Tensor, Tensor> make_rope_cache(const LlamaConfig& config, std::size_t tokens,
                                          std::size_t position_offset = 0) {
  const std::size_t head_dimension = config.head_dimension();
  const std::size_t pairs = head_dimension / 2;
  std::vector<float> cosine(tokens * pairs);
  std::vector<float> sine(tokens * pairs);
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const double exponent =
          -2.0 * static_cast<double>(pair) / static_cast<double>(head_dimension);
      const double angle = static_cast<double>(position_offset + token) *
                           std::pow(static_cast<double>(config.rope_theta), exponent);
      cosine[token * pairs + pair] = static_cast<float>(std::cos(angle));
      sine[token * pairs + pair] = static_cast<float>(std::sin(angle));
    }
  }
  const Shape shape{static_cast<Dimension>(tokens), static_cast<Dimension>(pairs)};
  return {Tensor::from_values(shape, cosine, Device::cuda(0)),
          Tensor::from_values(shape, sine, Device::cuda(0))};
}

Tensor pack_logical_blocks(const Tensor& logical, std::span<const std::uint32_t> block_table,
                           Dimension physical_blocks, Dimension block_tokens, float padding_value) {
  const auto heads = static_cast<std::size_t>(logical.shape()[0]);
  const auto tokens = static_cast<std::size_t>(logical.shape()[1]);
  const auto width = static_cast<std::size_t>(logical.shape()[2]);
  const auto physical_count = static_cast<std::size_t>(physical_blocks);
  const auto tokens_per_block = static_cast<std::size_t>(block_tokens);
  std::vector<float> packed(physical_count * heads * tokens_per_block * width, padding_value);
  const float* source = logical.data_as<float>();
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t token = 0; token < tokens; ++token) {
      const std::size_t logical_block = token / tokens_per_block;
      const std::size_t token_in_block = token % tokens_per_block;
      const std::size_t physical_block = static_cast<std::size_t>(block_table[logical_block]);
      for (std::size_t dimension = 0; dimension < width; ++dimension) {
        const std::size_t source_index = (head * tokens + token) * width + dimension;
        const std::size_t destination_index =
            (((physical_block * heads + head) * tokens_per_block + token_in_block) * width) +
            dimension;
        packed[destination_index] = source[source_index];
      }
    }
  }
  return Tensor::from_values(
      {physical_blocks, logical.shape()[0], block_tokens, logical.shape()[2]}, packed,
      Device::cuda(0));
}

std::vector<std::uint32_t> reversed_block_table(std::size_t logical_blocks,
                                                std::size_t physical_blocks) {
  std::vector<std::uint32_t> result(logical_blocks);
  for (std::size_t logical = 0; logical < logical_blocks; ++logical) {
    result[logical] = static_cast<std::uint32_t>(physical_blocks - 1 - logical);
  }
  return result;
}

void expect_close(const Tensor& actual_device, const Tensor& expected_host) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected_host.shape());
  ASSERT_EQ(actual.dtype(), DType::kFloat32);
  for (std::size_t index = 0; index < actual.numel(); ++index) {
    const float actual_value = actual.data_as<float>()[index];
    const float expected_value = expected_host.data_as<float>()[index];
    const float allowed_error = kAbsoluteTolerance + kRelativeTolerance * std::abs(expected_value);
    EXPECT_LE(std::abs(actual_value - expected_value), allowed_error) << "at index " << index;
  }
}

class CudaLlamaModelTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!helix::cuda::runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    helix::cuda::set_device(0);
  }
};

TEST_F(CudaLlamaModelTest, FullPrefillMatchesCpuForSingleAndOddTokenCounts) {
  const LlamaConfig config = tiny_config();
  CpuLlamaModel cpu_model(config, make_weights(config));
  CudaLlamaModel cuda_model = [&config] {
    const LlamaWeights source = make_weights(config);
    return CudaLlamaModel(config, source);
  }();
  helix::cuda::Stream stream;

  cuda_model.weights().validate(config, Device::cuda(0));
  EXPECT_EQ(cuda_model.device_index(), 0);
  for (const std::vector<std::uint32_t>& prompt :
       {std::vector<std::uint32_t>{3}, std::vector<std::uint32_t>{1, 5, 3, 7, 2}}) {
    const Tensor expected = cpu_model.forward(prompt);
    const Tensor actual = cuda_model.forward(prompt, stream);
    EXPECT_EQ(actual.device(), Device::cuda(0));
    expect_close(actual, expected);
  }
}

TEST_F(CudaLlamaModelTest, RejectsInvalidHostAndDeviceMetadataBeforeExecution) {
  const LlamaConfig config = tiny_config();
  const LlamaWeights source = make_weights(config);
  CudaLlamaModel model(config, source);
  helix::cuda::Stream stream;

  EXPECT_THROW(model.forward({}, stream), Error);
  const std::vector<std::uint32_t> overlength(config.max_sequence_length + 1, 0);
  EXPECT_THROW(model.forward(overlength, stream), Error);
  const std::array<std::uint32_t, 1> invalid_id{static_cast<std::uint32_t>(config.vocabulary_size)};
  EXPECT_THROW(model.forward(invalid_id, stream), Error);

  const std::array<std::uint32_t, 1> valid_id{2};
  const Tensor device_ids = Tensor::from_values({1}, valid_id, Device::cuda(0));
  const auto [cosine, sine] = make_rope_cache(config, 1);
  EXPECT_THROW(model.forward_device(Tensor::zeros({1}, DType::kUInt32), cosine, sine, stream),
               Error);
  EXPECT_THROW(model.forward_device(Tensor::zeros({1}, DType::kFloat32, Device::cuda(0)), cosine,
                                    sine, stream),
               Error);
  EXPECT_THROW(model.forward_device(Tensor::zeros({1, 1}, DType::kUInt32, Device::cuda(0)), cosine,
                                    sine, stream),
               Error);
  EXPECT_THROW(model.forward_device(Tensor::zeros({0}, DType::kUInt32, Device::cuda(0)), cosine,
                                    sine, stream),
               Error);
  EXPECT_THROW(
      model.forward_device(Tensor::zeros({static_cast<Dimension>(config.max_sequence_length + 1)},
                                         DType::kUInt32, Device::cuda(0)),
                           cosine, sine, stream),
      Error);
  EXPECT_THROW(model.forward_device(device_ids, Tensor::zeros(cosine.shape()), sine, stream),
               Error);
  EXPECT_THROW(model.forward_device(device_ids,
                                    Tensor::zeros(cosine.shape(), DType::kUInt32, Device::cuda(0)),
                                    sine, stream),
               Error);
  EXPECT_THROW(model.forward_device(device_ids, cosine,
                                    Tensor::zeros({1, 1}, DType::kFloat32, Device::cuda(0)),
                                    stream),
               Error);

  helix::cuda::Stream active_stream = std::move(stream);
  EXPECT_THROW(model.forward(valid_id, stream), Error);
  const Tensor actual = model.forward_device(device_ids, cosine, sine, active_stream);
  EXPECT_EQ(actual.shape(), (Shape{1, static_cast<Dimension>(config.vocabulary_size)}));
}

TEST_F(CudaLlamaModelTest, DeviceOutOfVocabularyTokenProducesOnlyNanLogits) {
  const LlamaConfig config = tiny_config();
  const LlamaWeights source = make_weights(config);
  CudaLlamaModel model(config, source);
  helix::cuda::Stream stream;
  const std::array<std::uint32_t, 1> invalid_id{static_cast<std::uint32_t>(config.vocabulary_size)};
  const Tensor device_ids = Tensor::from_values({1}, invalid_id, Device::cuda(0));
  const auto [cosine, sine] = make_rope_cache(config, 1);

  const Tensor logits = model.forward_device(device_ids, cosine, sine, stream).to(Device::cpu());
  for (std::size_t index = 0; index < logits.numel(); ++index) {
    EXPECT_TRUE(std::isnan(logits.data_as<float>()[index])) << "at index " << index;
  }
}

TEST_F(CudaLlamaModelTest, PagedDecodeMatchesCpuAtPositionZeroAndAcrossOddPages) {
  const LlamaConfig config = tiny_config();
  CpuLlamaModel cpu_model(config, make_weights(config));
  const LlamaWeights source = make_weights(config);
  CudaLlamaModel cuda_model(config, source);
  helix::cuda::Stream stream;
  constexpr Dimension kBlockTokens = 2;

  for (const std::size_t position : {std::size_t{0}, std::size_t{5}}) {
    std::vector<std::uint32_t> prefix(position);
    for (std::size_t index = 0; index < position; ++index) {
      prefix[index] = static_cast<std::uint32_t>((index * 3 + 1) % config.vocabulary_size);
    }

    std::vector<Tensor> logical_keys;
    std::vector<Tensor> logical_values;
    logical_keys.reserve(config.layer_count);
    logical_values.reserve(config.layer_count);
    if (position == 0) {
      for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        logical_keys.push_back(Tensor::zeros({static_cast<Dimension>(config.attention_head_count),
                                              0, static_cast<Dimension>(config.head_dimension())}));
        logical_values.push_back(Tensor::zeros(logical_keys.back().shape()));
      }
    } else {
      CpuLlamaPrefillResult prefill = cpu_model.prefill_with_kv(prefix);
      logical_keys = std::move(prefill.layer_keys);
      logical_values = std::move(prefill.layer_values);
    }

    const std::size_t logical_blocks =
        position == 0 ? 0 : 1 + (position - 1) / static_cast<std::size_t>(kBlockTokens);
    const std::size_t physical_blocks = logical_blocks + 2;
    const std::vector<std::uint32_t> table = reversed_block_table(logical_blocks, physical_blocks);
    std::vector<Tensor> key_blocks;
    std::vector<Tensor> value_blocks;
    key_blocks.reserve(config.layer_count);
    value_blocks.reserve(config.layer_count);
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
      key_blocks.push_back(
          pack_logical_blocks(logical_keys[layer], table, static_cast<Dimension>(physical_blocks),
                              kBlockTokens, std::numeric_limits<float>::quiet_NaN()));
      value_blocks.push_back(
          pack_logical_blocks(logical_values[layer], table, static_cast<Dimension>(physical_blocks),
                              kBlockTokens, std::numeric_limits<float>::quiet_NaN()));
    }
    attention::cuda::PagedBlockTable device_table(table, physical_blocks, 0);

    kv_cache::PagedKvCache cpu_cache(kv_cache::PagedKvCacheConfig{
        config.layer_count, config.attention_head_count, config.head_dimension(),
        static_cast<std::size_t>(kBlockTokens), physical_blocks});
    const kv_cache::PagedSequenceHandle sequence = cpu_cache.create_sequence();
    if (position != 0) {
      cpu_cache.append_layer_major(sequence, logical_keys, logical_values);
    }
    const std::uint32_t token_id =
        static_cast<std::uint32_t>((position * 5 + 3) % config.vocabulary_size);
    const CpuLlamaDecodeResult expected =
        cpu_model.decode_one(token_id, position, cpu_cache, sequence);

    const CudaLlamaDecodeResult host_result = cuda_model.decode_one_paged(
        token_id, position, key_blocks, value_blocks, device_table, stream);
    EXPECT_EQ(host_result.token_keys.device(), Device::cuda(0));
    EXPECT_EQ(host_result.token_keys.shape(), expected.token_keys.shape());
    EXPECT_EQ(host_result.token_values.shape(), expected.token_values.shape());
    expect_close(host_result.token_keys, expected.token_keys);
    expect_close(host_result.token_values, expected.token_values);
    expect_close(host_result.logits, expected.logits);

    const std::array<std::uint32_t, 1> token_ids{token_id};
    const Tensor device_id = Tensor::from_values({1}, token_ids, Device::cuda(0));
    const auto [cosine, sine] = make_rope_cache(config, 1, position);
    const CudaLlamaDecodeResult device_result = cuda_model.decode_one_paged_device(
        device_id, cosine, sine, key_blocks, value_blocks, device_table, position, stream);
    expect_close(device_result.token_keys, expected.token_keys);
    expect_close(device_result.token_values, expected.token_values);
    expect_close(device_result.logits, expected.logits);
  }
}

TEST_F(CudaLlamaModelTest, GpuCacheDecodeCommitMatchesFullPrefixRecomputation) {
  const LlamaConfig config = tiny_config();
  CpuLlamaModel cpu_model(config, make_weights(config));
  const LlamaWeights source = make_weights(config);
  CudaLlamaModel cuda_model(config, source);
  kv_cache::CudaPagedKvCache cache(kv_cache::PagedKvCacheConfig{
      config.layer_count, config.attention_head_count, config.head_dimension(), 2, 4});
  const kv_cache::CudaPagedSequenceHandle sequence = cache.create_sequence();
  helix::cuda::Stream stream;
  const std::array<std::uint32_t, 5> tokens{1, 5, 3, 7, 2};
  std::vector<std::uint32_t> prefix;

  for (std::size_t position = 0; position < tokens.size(); ++position) {
    prefix.push_back(tokens[position]);
    const Tensor expected_logits = cpu_model.forward(prefix);
    CudaLlamaDecodeResult result = [&] {
      auto view = cache.read_sequence(sequence);
      EXPECT_EQ(view.length(), position);
      return cuda_model.decode_one_paged(tokens[position], position, view.layer_key_blocks(),
                                         view.layer_value_blocks(), view.block_table(), stream);
    }();

    expect_close(result.logits, expected_logits);
    cache.append(sequence, result.token_keys, result.token_values, stream);
    EXPECT_EQ(cache.length(sequence), position + 1);
  }

  {
    auto view = cache.read_sequence(sequence);
    ASSERT_EQ(view.physical_blocks().size(), 3U);
    EXPECT_EQ(view.physical_blocks()[0], 0U);
    EXPECT_EQ(view.physical_blocks()[1], 1U);
    EXPECT_EQ(view.physical_blocks()[2], 2U);
    EXPECT_THROW(cuda_model.decode_one_paged(static_cast<std::uint32_t>(config.vocabulary_size),
                                             view.length(), view.layer_key_blocks(),
                                             view.layer_value_blocks(), view.block_table(), stream),
                 Error);
  }
  EXPECT_EQ(cache.length(sequence), tokens.size());
  EXPECT_EQ(cache.stats().append_count, tokens.size());
  EXPECT_EQ(cache.stats().used_blocks, 3U);
}

TEST_F(CudaLlamaModelTest, PagedDecodeRejectsInvalidContractsBeforeExecution) {
  const LlamaConfig config = tiny_config();
  const LlamaWeights source = make_weights(config);
  CudaLlamaModel model(config, source);
  const Dimension heads = static_cast<Dimension>(config.attention_head_count);
  const Dimension width = static_cast<Dimension>(config.head_dimension());
  std::vector<Tensor> key_blocks;
  std::vector<Tensor> value_blocks;
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    key_blocks.push_back(Tensor::zeros({1, heads, 2, width}, DType::kFloat32, Device::cuda(0)));
    value_blocks.push_back(Tensor::zeros({1, heads, 2, width}, DType::kFloat32, Device::cuda(0)));
  }
  attention::cuda::PagedBlockTable empty_table(std::span<const std::uint32_t>{}, 1, 0);
  helix::cuda::Stream stream;

  EXPECT_THROW(model.decode_one_paged(static_cast<std::uint32_t>(config.vocabulary_size), 0,
                                      key_blocks, value_blocks, empty_table, stream),
               Error);
  EXPECT_THROW(model.decode_one_paged(1, config.max_sequence_length, key_blocks, value_blocks,
                                      empty_table, stream),
               Error);
  EXPECT_THROW(model.decode_one_paged(1, 0, std::span<const Tensor>(key_blocks).first(1),
                                      value_blocks, empty_table, stream),
               Error);

  std::vector<Tensor> wrong_values = value_blocks;
  wrong_values[1] = Tensor::zeros({1, heads, 3, width}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(model.decode_one_paged(1, 0, key_blocks, wrong_values, empty_table, stream), Error);
  std::vector<Tensor> host_keys = key_blocks;
  host_keys[0] = Tensor::zeros({1, heads, 2, width});
  EXPECT_THROW(model.decode_one_paged(1, 0, host_keys, value_blocks, empty_table, stream), Error);
  std::vector<Tensor> integer_keys = key_blocks;
  integer_keys[0] = Tensor::zeros({1, heads, 2, width}, DType::kUInt32, Device::cuda(0));
  EXPECT_THROW(model.decode_one_paged(1, 0, integer_keys, value_blocks, empty_table, stream),
               Error);

  constexpr std::array<std::uint32_t, 1> kOneBlock{0};
  attention::cuda::PagedBlockTable nonempty_table(kOneBlock, 1, 0);
  EXPECT_THROW(model.decode_one_paged(1, 0, key_blocks, value_blocks, nonempty_table, stream),
               Error);
  attention::cuda::PagedBlockTable wrong_capacity(std::span<const std::uint32_t>{}, 2, 0);
  EXPECT_THROW(model.decode_one_paged(1, 0, key_blocks, value_blocks, wrong_capacity, stream),
               Error);

  const std::array<std::uint32_t, 1> token_value{1};
  const Tensor token_id = Tensor::from_values({1}, token_value, Device::cuda(0));
  const auto [cosine, sine] = make_rope_cache(config, 1);
  EXPECT_THROW(model.decode_one_paged_device(Tensor::zeros({1}, DType::kFloat32, Device::cuda(0)),
                                             cosine, sine, key_blocks, value_blocks, empty_table, 0,
                                             stream),
               Error);
  EXPECT_THROW(model.decode_one_paged_device(Tensor::zeros({2}, DType::kUInt32, Device::cuda(0)),
                                             cosine, sine, key_blocks, value_blocks, empty_table, 0,
                                             stream),
               Error);
  EXPECT_THROW(model.decode_one_paged_device(token_id, Tensor::zeros({1, 1}), sine, key_blocks,
                                             value_blocks, empty_table, 0, stream),
               Error);
  EXPECT_THROW(model.decode_one_paged_device(token_id, cosine, sine, key_blocks, value_blocks,
                                             empty_table, 1, stream),
               Error);

  helix::cuda::Stream active_stream = std::move(stream);
  EXPECT_THROW(model.decode_one_paged(1, 0, key_blocks, value_blocks, empty_table, stream), Error);
  EXPECT_NO_THROW(static_cast<void>(
      model.decode_one_paged(1, 0, key_blocks, value_blocks, empty_table, active_stream)));
}

} // namespace
} // namespace helix::model
