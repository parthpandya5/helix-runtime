#include "helix/core/error.hpp"
#include "helix/model/cuda_llama.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
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

std::pair<Tensor, Tensor> make_rope_cache(const LlamaConfig& config, std::size_t tokens) {
  const std::size_t head_dimension = config.head_dimension();
  const std::size_t pairs = head_dimension / 2;
  std::vector<float> cosine(tokens * pairs);
  std::vector<float> sine(tokens * pairs);
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const double exponent =
          -2.0 * static_cast<double>(pair) / static_cast<double>(head_dimension);
      const double angle =
          static_cast<double>(token) * std::pow(static_cast<double>(config.rope_theta), exponent);
      cosine[token * pairs + pair] = static_cast<float>(std::cos(angle));
      sine[token * pairs + pair] = static_cast<float>(std::sin(angle));
    }
  }
  const Shape shape{static_cast<Dimension>(tokens), static_cast<Dimension>(pairs)};
  return {Tensor::from_values(shape, cosine, Device::cuda(0)),
          Tensor::from_values(shape, sine, Device::cuda(0))};
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

} // namespace
} // namespace helix::model
