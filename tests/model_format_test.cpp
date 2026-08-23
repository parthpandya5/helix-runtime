#include "helix/core/error.hpp"
#include "helix/model/format.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace helix::model {
namespace {

constexpr std::size_t kHeaderBytes = 128;
constexpr std::size_t kDirectoryEntryBytes = 80;
constexpr std::size_t kMetadataChecksumOffset = 120;
constexpr std::uint64_t kFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;

LlamaConfig test_config() {
  return LlamaConfig{7, 4, 5, 2, 2, 9, 1.0e-5F, 10'000.0F};
}

Tensor make_tensor(Shape shape, std::size_t seed, bool norm = false) {
  Tensor tensor = Tensor::empty(std::move(shape));
  float* values = tensor.data_as<float>();
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    values[index] =
        norm ? 0.8F + static_cast<float>((index + seed) % 9) * 0.025F
             : static_cast<float>(static_cast<int>((index * 11 + seed * 7) % 31) - 15) * 0.0175F;
  }
  return tensor;
}

LlamaWeights make_weights(const LlamaConfig& config, bool non_finite_embedding = false) {
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
  Tensor embedding = make_tensor({vocabulary, hidden}, 1);
  if (non_finite_embedding) {
    embedding.data_as<float>()[0] = std::numeric_limits<float>::quiet_NaN();
  }
  return LlamaWeights(std::move(embedding), std::move(layers), make_tensor({hidden}, 101, true),
                      make_tensor({hidden, vocabulary}, 102));
}

void expect_tensor_equal(const Tensor& actual, const Tensor& expected) {
  ASSERT_EQ(actual.shape(), expected.shape());
  ASSERT_EQ(actual.dtype(), expected.dtype());
  ASSERT_EQ(actual.device(), expected.device());
  ASSERT_EQ(actual.numel(), expected.numel());
  for (std::size_t index = 0; index < actual.numel(); ++index) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.data_as<float>()[index]),
              std::bit_cast<std::uint32_t>(expected.data_as<float>()[index]))
        << "at element " << index;
  }
}

void expect_weights_equal(const LlamaWeights& actual, const LlamaWeights& expected) {
  expect_tensor_equal(actual.token_embedding(), expected.token_embedding());
  ASSERT_EQ(actual.layers().size(), expected.layers().size());
  for (std::size_t index = 0; index < actual.layers().size(); ++index) {
    const LlamaLayerWeights& lhs = actual.layers()[index];
    const LlamaLayerWeights& rhs = expected.layers()[index];
    expect_tensor_equal(lhs.attention_norm_weight, rhs.attention_norm_weight);
    expect_tensor_equal(lhs.query_weight, rhs.query_weight);
    expect_tensor_equal(lhs.key_weight, rhs.key_weight);
    expect_tensor_equal(lhs.value_weight, rhs.value_weight);
    expect_tensor_equal(lhs.attention_output_weight, rhs.attention_output_weight);
    expect_tensor_equal(lhs.ffn_norm_weight, rhs.ffn_norm_weight);
    expect_tensor_equal(lhs.gate_weight, rhs.gate_weight);
    expect_tensor_equal(lhs.up_weight, rhs.up_weight);
    expect_tensor_equal(lhs.down_weight, rhs.down_weight);
  }
  expect_tensor_equal(actual.final_norm_weight(), expected.final_norm_weight());
  expect_tensor_equal(actual.lm_head_weight(), expected.lm_head_weight());
}

std::string serialize(const LlamaConfig& config, const LlamaWeights& weights) {
  std::ostringstream output(std::ios::out | std::ios::binary);
  write_llama_model_file(output, config, weights);
  return output.str();
}

LoadedLlamaModel deserialize(const std::string& bytes) {
  std::istringstream input(bytes, std::ios::in | std::ios::binary);
  return read_llama_model_file(input);
}

std::uint64_t read_u64(const std::string& bytes, std::size_t offset) {
  std::uint64_t value = 0;
  for (std::size_t shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[offset + shift / 8]))
             << shift;
  }
  return value;
}

void write_u64(std::string& bytes, std::size_t offset, std::uint64_t value) {
  for (std::size_t shift = 0; shift < 64; shift += 8) {
    bytes[offset + shift / 8] = static_cast<char>((value >> shift) & 0xffU);
  }
}

void refresh_metadata_checksum(std::string& bytes, std::size_t tensor_count) {
  write_u64(bytes, kMetadataChecksumOffset, 0);
  std::uint64_t hash = kFnvOffset;
  const std::size_t metadata_bytes = kHeaderBytes + tensor_count * kDirectoryEntryBytes;
  for (std::size_t index = 0; index < metadata_bytes; ++index) {
    hash ^= static_cast<unsigned char>(bytes[index]);
    hash *= kFnvPrime;
  }
  write_u64(bytes, kMetadataChecksumOffset, hash);
}

void expect_invalid_file(const std::string& bytes) {
  try {
    static_cast<void>(deserialize(bytes));
    FAIL() << "expected malformed model file to be rejected";
  } catch (const Error& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
    EXPECT_NE(error.message().find("model::format"), std::string::npos);
  }
}

TEST(ModelFormatTest, RoundTripsDeterministicallyWithExactFloatBits) {
  const LlamaConfig config = test_config();
  const LlamaWeights weights = make_weights(config);
  const std::string first = serialize(config, weights);
  const std::string second = serialize(config, weights);
  EXPECT_EQ(first, second);

  LoadedLlamaModel loaded = deserialize(first);
  EXPECT_EQ(loaded.config.vocabulary_size, config.vocabulary_size);
  EXPECT_EQ(loaded.config.hidden_size, config.hidden_size);
  EXPECT_EQ(loaded.config.intermediate_size, config.intermediate_size);
  EXPECT_EQ(loaded.config.layer_count, config.layer_count);
  EXPECT_EQ(loaded.config.attention_head_count, config.attention_head_count);
  EXPECT_EQ(loaded.config.max_sequence_length, config.max_sequence_length);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(loaded.config.rms_norm_epsilon),
            std::bit_cast<std::uint32_t>(config.rms_norm_epsilon));
  EXPECT_EQ(std::bit_cast<std::uint32_t>(loaded.config.rope_theta),
            std::bit_cast<std::uint32_t>(config.rope_theta));
  expect_weights_equal(loaded.weights, weights);

  ASSERT_GE(first.size(), kHeaderBytes + kDirectoryEntryBytes);
  EXPECT_EQ(first.substr(0, 8), "HELIXLM1");
  const std::uint64_t data_offset = read_u64(first, 56);
  const std::uint64_t first_tensor_offset = read_u64(first, kHeaderBytes + 48);
  EXPECT_EQ(data_offset % 64, 0U);
  EXPECT_EQ(first_tensor_offset, data_offset);
}

TEST(ModelFormatTest, LoadedWeightsExecuteIdentically) {
  const LlamaConfig config = test_config();
  const std::string bytes = serialize(config, make_weights(config));
  LoadedLlamaModel first = deserialize(bytes);
  LoadedLlamaModel second = deserialize(bytes);
  CpuLlamaModel first_model(first.config, std::move(first.weights));
  CpuLlamaModel second_model(second.config, std::move(second.weights));
  const std::vector<std::uint32_t> tokens{6, 1, 4, 0, 2};
  const Tensor first_logits = first_model.forward(tokens);
  const Tensor second_logits = second_model.forward(tokens);
  expect_tensor_equal(first_logits, second_logits);
}

TEST(ModelFormatTest, RejectsTruncationTrailingBytesAndBasicHeaderCorruption) {
  const LlamaConfig config = test_config();
  const std::string valid = serialize(config, make_weights(config));

  expect_invalid_file("");
  expect_invalid_file(valid.substr(0, kHeaderBytes - 1));
  expect_invalid_file(valid.substr(0, valid.size() - 1));
  expect_invalid_file(valid + std::string(1, '\0'));

  std::string bad_magic = valid;
  bad_magic[0] = 'X';
  expect_invalid_file(bad_magic);
  std::string bad_version = valid;
  bad_version[8] = 2;
  expect_invalid_file(bad_version);
  std::string bad_endian = valid;
  bad_endian[12] = 0;
  expect_invalid_file(bad_endian);
}

TEST(ModelFormatTest, RejectsMetadataPayloadAndPaddingCorruption) {
  const LlamaConfig config = test_config();
  const std::string valid = serialize(config, make_weights(config));
  const std::size_t tensor_count = 3 + config.layer_count * 9;
  const std::size_t metadata_end = kHeaderBytes + tensor_count * kDirectoryEntryBytes;
  const std::size_t data_offset = static_cast<std::size_t>(read_u64(valid, 56));
  ASSERT_LT(metadata_end, data_offset);

  std::string bad_metadata = valid;
  bad_metadata[kHeaderBytes] ^= 1;
  expect_invalid_file(bad_metadata);
  std::string bad_payload = valid;
  bad_payload.back() ^= 1;
  expect_invalid_file(bad_payload);
  std::string bad_padding = valid;
  bad_padding[metadata_end] = 1;
  expect_invalid_file(bad_padding);

  std::string bad_rank = valid;
  bad_rank[kHeaderBytes + 12] = 3;
  refresh_metadata_checksum(bad_rank, tensor_count);
  expect_invalid_file(bad_rank);
  std::string bad_offset = valid;
  write_u64(bad_offset, kHeaderBytes + 48, data_offset + 64);
  refresh_metadata_checksum(bad_offset, tensor_count);
  expect_invalid_file(bad_offset);
}

TEST(ModelFormatTest, RejectsInvalidWeightsBeforeWriting) {
  const LlamaConfig config = test_config();
  LlamaWeights wrong_shape(Tensor::zeros({7, 5}), {}, Tensor::zeros({4}), Tensor::zeros({4, 7}));
  std::ostringstream output(std::ios::out | std::ios::binary);
  EXPECT_THROW(write_llama_model_file(output, config, wrong_shape), Error);

  LlamaWeights non_finite = make_weights(config, true);
  std::ostringstream non_finite_output(std::ios::out | std::ios::binary);
  EXPECT_THROW(write_llama_model_file(non_finite_output, config, non_finite), Error);

  std::ostringstream nonzero_output(std::ios::out | std::ios::binary);
  nonzero_output.put('x');
  EXPECT_THROW(write_llama_model_file(nonzero_output, config, make_weights(config)), Error);
}

TEST(ModelFormatTest, FilesystemOverloadsRoundTripAndReportMissingFileContext) {
  const LlamaConfig config = test_config();
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("helix-model-format-" + std::to_string(nonce) + ".bin");
  struct Cleanup final {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  } cleanup{path};
  const LlamaWeights weights = make_weights(config);
  write_llama_model_file(path, config, weights);
  LoadedLlamaModel loaded = read_llama_model_file(path);
  expect_weights_equal(loaded.weights, weights);

  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "helix-model-format-file-that-does-not-exist.bin";
  try {
    static_cast<void>(read_llama_model_file(missing));
    FAIL() << "expected a missing model file error";
  } catch (const Error& error) {
    EXPECT_EQ(error.code(), ErrorCode::kRuntime);
    EXPECT_NE(error.message().find(missing.string()), std::string::npos);
  }
}

} // namespace
} // namespace helix::model
