#include "helix/core/error.hpp"
#include "helix/model/format.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
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

std::vector<const Tensor*> canonical_tensors(const LlamaWeights& weights) {
  std::vector<const Tensor*> tensors{&weights.token_embedding()};
  tensors.reserve(3 + weights.layers().size() * 9);
  for (const LlamaLayerWeights& layer : weights.layers()) {
    tensors.push_back(&layer.attention_norm_weight);
    tensors.push_back(&layer.query_weight);
    tensors.push_back(&layer.key_weight);
    tensors.push_back(&layer.value_weight);
    tensors.push_back(&layer.attention_output_weight);
    tensors.push_back(&layer.ffn_norm_weight);
    tensors.push_back(&layer.gate_weight);
    tensors.push_back(&layer.up_weight);
    tensors.push_back(&layer.down_weight);
  }
  tensors.push_back(&weights.final_norm_weight());
  tensors.push_back(&weights.lm_head_weight());
  return tensors;
}

void write_raw_tensor(const std::filesystem::path& path, const Tensor& tensor) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(output) << path;
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(tensor.data_as<float>()[index]);
    for (std::size_t shift = 0; shift < 32; shift += 8) {
      output.put(static_cast<char>((bits >> shift) & 0xffU));
    }
  }
  ASSERT_TRUE(output) << path;
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  EXPECT_TRUE(input) << path;
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(output) << path;
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(output) << path;
}

struct TemporaryDirectory final {
  explicit TemporaryDirectory(std::string_view prefix) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           (std::string(prefix) + "-" + std::to_string(nonce));
    std::filesystem::create_directory(path);
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }

  std::filesystem::path path;
};

std::vector<std::filesystem::path> write_raw_tensors(const std::filesystem::path& directory,
                                                     const LlamaWeights& weights) {
  const std::vector<const Tensor*> tensors = canonical_tensors(weights);
  std::vector<std::filesystem::path> paths;
  paths.reserve(tensors.size());
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const std::filesystem::path path = directory / (std::to_string(index) + ".f32");
    write_raw_tensor(path, *tensors[index]);
    paths.push_back(path);
  }
  return paths;
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

void expect_invalid_mapped_file(const std::filesystem::path& path, const std::string& bytes) {
  write_file(path, bytes);
  try {
    static_cast<void>(read_llama_model_file_mapped(path));
    FAIL() << "expected malformed mapped model file to be rejected";
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

TEST(ModelFormatTest, MappedLoaderIsBitExactReadOnlyAndOwnsLifetime) {
  if (!mapped_model_loading_supported()) {
    GTEST_SKIP() << "read-only model mapping is unavailable on this platform";
  }

  const LlamaConfig config = test_config();
  const LlamaWeights weights = make_weights(config);
  TemporaryDirectory temporary("helix-model-mapped");
  const std::filesystem::path path = temporary.path / "model.helix";
  write_llama_model_file(path, config, weights);

  LoadedLlamaModel copied = read_llama_model_file(path);
  LoadedLlamaModel mapped = read_llama_model_file_mapped(path);
  expect_weights_equal(mapped.weights, weights);
  expect_weights_equal(mapped.weights, copied.weights);
  for (const Tensor* tensor : canonical_tensors(mapped.weights)) {
    EXPECT_TRUE(tensor->is_read_only());
    EXPECT_EQ(tensor->allocated_bytes(), tensor->nbytes());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(tensor->data()) % 64, 0U);
  }

  Tensor mapped_alias = mapped.weights.token_embedding();
  EXPECT_THROW(static_cast<void>(mapped_alias.data()), Error);
  EXPECT_THROW(static_cast<void>(mapped_alias.zero_()), Error);
  Tensor clone = mapped_alias.clone();
  EXPECT_FALSE(clone.is_read_only());
  clone.data_as<float>()[0] = -7.0F;
  EXPECT_NE(clone.data_as<float>()[0], std::as_const(mapped_alias).data_as<float>()[0]);

  CpuLlamaModel copied_model(copied.config, std::move(copied.weights));
  CpuLlamaModel mapped_model(mapped.config, std::move(mapped.weights));
  const std::vector<std::uint32_t> tokens{6, 1, 4, 0, 2};
  const Tensor copied_logits = copied_model.forward(tokens);
  const Tensor first_mapped_logits = mapped_model.forward(tokens);
  expect_tensor_equal(first_mapped_logits, copied_logits);

  ASSERT_TRUE(std::filesystem::remove(path));
  const Tensor after_unlink_logits = mapped_model.forward(tokens);
  expect_tensor_equal(after_unlink_logits, copied_logits);
}

TEST(ModelFormatTest, MappedLoaderRejectsCorruptionAndReportsPathErrors) {
  if (!mapped_model_loading_supported()) {
    GTEST_SKIP() << "read-only model mapping is unavailable on this platform";
  }

  const LlamaConfig config = test_config();
  const std::string valid = serialize(config, make_weights(config));
  const std::size_t tensor_count = 3 + config.layer_count * 9;
  const std::size_t metadata_end = kHeaderBytes + tensor_count * kDirectoryEntryBytes;
  const std::size_t first_payload = static_cast<std::size_t>(read_u64(valid, kHeaderBytes + 48));
  const std::size_t first_payload_bytes =
      static_cast<std::size_t>(read_u64(valid, kHeaderBytes + 56));
  TemporaryDirectory temporary("helix-model-mapped-invalid");
  const std::filesystem::path path = temporary.path / "model.helix";

  expect_invalid_mapped_file(path, valid.substr(0, valid.size() - 1));
  expect_invalid_mapped_file(path, valid + std::string(1, '\0'));

  std::string bad_metadata = valid;
  bad_metadata[kHeaderBytes] ^= 1;
  expect_invalid_mapped_file(path, bad_metadata);

  std::string bad_padding = valid;
  ASSERT_LT(metadata_end, first_payload);
  bad_padding[metadata_end] = 1;
  expect_invalid_mapped_file(path, bad_padding);

  std::string non_finite = valid;
  const std::uint32_t infinity =
      std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity());
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    non_finite[first_payload + shift / 8] = static_cast<char>((infinity >> shift) & 0xffU);
  }
  std::uint64_t payload_checksum = kFnvOffset;
  for (std::size_t index = 0; index < first_payload_bytes; ++index) {
    payload_checksum ^= static_cast<unsigned char>(non_finite[first_payload + index]);
    payload_checksum *= kFnvPrime;
  }
  write_u64(non_finite, kHeaderBytes + 72, payload_checksum);
  refresh_metadata_checksum(non_finite, tensor_count);
  expect_invalid_mapped_file(path, non_finite);

  const std::filesystem::path missing = temporary.path / "missing.helix";
  try {
    static_cast<void>(read_llama_model_file_mapped(missing));
    FAIL() << "expected a missing mapped model file error";
  } catch (const Error& error) {
    EXPECT_EQ(error.code(), ErrorCode::kRuntime);
    EXPECT_NE(error.message().find(missing.string()), std::string::npos);
  }
}

TEST(ModelFormatTest, RawFloat32FilesProduceCanonicalContainer) {
  const LlamaConfig config = test_config();
  const LlamaWeights weights = make_weights(config);
  TemporaryDirectory temporary("helix-model-raw");
  const std::vector<std::filesystem::path> paths = write_raw_tensors(temporary.path, weights);
  const std::filesystem::path output_path = temporary.path / "converted.helix";

  write_llama_model_file_from_raw_float32_files(output_path, config, paths);
  EXPECT_EQ(read_file(output_path), serialize(config, weights));
  LoadedLlamaModel loaded = read_llama_model_file(output_path);
  expect_weights_equal(loaded.weights, weights);
}

TEST(ModelFormatTest, RawFloat32FilesRejectInvalidInputsBeforeOpeningOutput) {
  const LlamaConfig config = test_config();
  const LlamaWeights weights = make_weights(config);
  TemporaryDirectory temporary("helix-model-raw-invalid");
  std::vector<std::filesystem::path> paths = write_raw_tensors(temporary.path, weights);
  const std::filesystem::path output_path = temporary.path / "converted.helix";

  const std::span<const std::filesystem::path> too_few(paths.data(), paths.size() - 1);
  EXPECT_THROW(write_llama_model_file_from_raw_float32_files(output_path, config, too_few), Error);
  EXPECT_FALSE(std::filesystem::exists(output_path));

  const std::string first_bytes = read_file(paths.front());
  ASSERT_GT(first_bytes.size(), sizeof(float));
  std::filesystem::resize_file(paths.front(), first_bytes.size() - sizeof(float));
  EXPECT_THROW(write_llama_model_file_from_raw_float32_files(output_path, config, paths), Error);
  EXPECT_FALSE(std::filesystem::exists(output_path));

  {
    std::ofstream restore(paths.front(), std::ios::binary | std::ios::trunc);
    restore.write(first_bytes.data(), static_cast<std::streamsize>(first_bytes.size()));
  }
  std::string non_finite = first_bytes;
  const std::uint32_t infinity =
      std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity());
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    non_finite[shift / 8] = static_cast<char>((infinity >> shift) & 0xffU);
  }
  {
    std::ofstream corrupt(paths.front(), std::ios::binary | std::ios::trunc);
    corrupt.write(non_finite.data(), static_cast<std::streamsize>(non_finite.size()));
  }
  EXPECT_THROW(write_llama_model_file_from_raw_float32_files(output_path, config, paths), Error);
  EXPECT_FALSE(std::filesystem::exists(output_path));

  write_raw_tensor(paths.front(), weights.token_embedding());
  EXPECT_THROW(write_llama_model_file_from_raw_float32_files(paths.front(), config, paths), Error);
  EXPECT_EQ(read_file(paths.front()), first_bytes);

  const std::filesystem::path hard_link = temporary.path / "payload-hard-link.helix";
  std::filesystem::create_hard_link(paths.front(), hard_link);
  EXPECT_THROW(write_llama_model_file_from_raw_float32_files(hard_link, config, paths), Error);
  EXPECT_EQ(read_file(paths.front()), first_bytes);
}

} // namespace
} // namespace helix::model
