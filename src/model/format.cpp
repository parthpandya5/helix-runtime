#include "helix/model/format.hpp"

#include "helix/core/error.hpp"
#include "helix/tensor/shape.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#define HELIX_MODEL_HAS_POSIX_MMAP 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define HELIX_MODEL_HAS_POSIX_MMAP 0
#endif

namespace helix::model {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'H', 'E', 'L', 'I', 'X', 'L', 'M', '1'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kHeaderBytes = 128;
constexpr std::uint32_t kModelKindLlama = 1;
constexpr std::uint64_t kDirectoryOffset = kHeaderBytes;
constexpr std::uint64_t kDirectoryEntryBytes = 80;
constexpr std::uint64_t kPayloadAlignment = 64;
constexpr std::uint32_t kFloat32Code = 1;
constexpr std::uint32_t kGlobalLayer = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint64_t kFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;
constexpr std::size_t kIoChunkBytes = 64 * 1024;
static_assert(sizeof(float) == sizeof(std::uint32_t),
              "HELIXLM1 requires a 32-bit host Float32 representation");

enum class TensorRole : std::uint32_t {
  kTokenEmbedding = 1,
  kAttentionNorm = 2,
  kQuery = 3,
  kKey = 4,
  kValue = 5,
  kAttentionOutput = 6,
  kFfnNorm = 7,
  kGate = 8,
  kUp = 9,
  kDown = 10,
  kFinalNorm = 11,
  kLmHead = 12,
};

struct TensorSpec {
  TensorRole role;
  std::uint32_t layer;
  Shape shape;
  const Tensor* tensor = nullptr;
};

struct DirectoryRecord {
  TensorRole role;
  std::uint32_t layer;
  std::uint32_t dtype;
  std::uint32_t rank;
  std::array<std::uint64_t, 4> dimensions{};
  std::uint64_t offset;
  std::uint64_t byte_length;
  std::uint64_t alignment;
  std::uint64_t checksum;
};

struct DecodedHeader {
  LlamaConfig config;
  std::uint64_t metadata_checksum;
  std::size_t tensor_count;
  std::size_t directory_bytes;
  std::size_t metadata_end;
  std::size_t data_offset;
};

struct ParsedModelMetadata {
  DecodedHeader header;
  std::vector<TensorSpec> specs;
  std::vector<DirectoryRecord> records;
};

[[noreturn]] void invalid_format(std::string message) {
  throw_error(ErrorCode::kInvalidArgument, "model::format: " + std::move(message));
}

[[noreturn]] void io_error(std::string message) {
  throw_error(ErrorCode::kRuntime, "model::format: " + std::move(message));
}

#if HELIX_MODEL_HAS_POSIX_MMAP
std::string posix_error_message(int error_number) {
  return std::error_code(error_number, std::generic_category()).message();
}

class FileDescriptor final {
public:
  explicit FileDescriptor(int value) noexcept : value_(value) {}

  ~FileDescriptor() {
    if (value_ >= 0) {
      static_cast<void>(::close(value_));
    }
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&&) = delete;
  FileDescriptor& operator=(FileDescriptor&&) = delete;

  [[nodiscard]] int get() const noexcept {
    return value_;
  }

private:
  int value_;
};

class MappedFile final {
public:
  explicit MappedFile(const std::filesystem::path& path) {
    const std::string native_path = path.string();
    if (native_path.find('\0') != std::string::npos) {
      io_error("model path contains an embedded null byte");
    }

    const int descriptor_value = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor_value < 0) {
      const int error_number = errno;
      io_error("could not open '" + path.string() +
               "' for memory mapping: " + posix_error_message(error_number));
    }
    const FileDescriptor descriptor(descriptor_value);

    struct stat status{};
    if (::fstat(descriptor.get(), &status) != 0) {
      const int error_number = errno;
      io_error("could not inspect '" + path.string() +
               "' for memory mapping: " + posix_error_message(error_number));
    }
    if (!S_ISREG(status.st_mode)) {
      io_error("memory-mapped model path '" + path.string() + "' is not a regular file");
    }
    if (status.st_size < 0) {
      io_error("memory-mapped model file '" + path.string() + "' has a negative size");
    }
    const auto unsigned_size = static_cast<std::uintmax_t>(status.st_size);
    if (unsigned_size > std::numeric_limits<std::size_t>::max()) {
      throw_error(ErrorCode::kOutOfRange,
                  "model::format: memory-mapped file exceeds this host's addressable range");
    }
    size_bytes_ = static_cast<std::size_t>(unsigned_size);
    if (size_bytes_ == 0) {
      return;
    }

    void* const mapping = ::mmap(nullptr, size_bytes_, PROT_READ, MAP_PRIVATE, descriptor.get(), 0);
    if (mapping == MAP_FAILED) {
      const int error_number = errno;
      io_error("could not memory-map '" + path.string() +
               "': " + posix_error_message(error_number));
    }
    data_ = static_cast<const std::uint8_t*>(mapping);
  }

  ~MappedFile() {
    if (data_ != nullptr) {
      static_cast<void>(::munmap(const_cast<std::uint8_t*>(data_), size_bytes_));
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return {data_, size_bytes_};
  }

private:
  const std::uint8_t* data_ = nullptr;
  std::size_t size_bytes_ = 0;
};
#endif

std::size_t checked_add(std::size_t lhs, std::size_t rhs, std::string_view context) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw_error(ErrorCode::kOutOfRange,
                "model::format: " + std::string(context) + " size overflow");
  }
  return lhs + rhs;
}

std::size_t aligned_size(std::size_t value, std::size_t alignment, std::string_view context) {
  HELIX_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0,
               "model file alignment must be a power of two");
  const std::size_t padded = checked_add(value, alignment - 1, context);
  return padded & ~(alignment - 1);
}

std::uint64_t to_u64(std::size_t value) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(value);
}

std::streamoff to_streamoff(std::size_t value, std::string_view field) {
  if (value > static_cast<std::uintmax_t>(std::numeric_limits<std::streamoff>::max())) {
    throw_error(ErrorCode::kOutOfRange,
                "model::format: " + std::string(field) + " exceeds the stream offset range");
  }
  return static_cast<std::streamoff>(value);
}

std::size_t to_size(std::uint64_t value, std::string_view field) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    invalid_format(std::string(field) + " exceeds this host's size range");
  }
  return static_cast<std::size_t>(value);
}

Dimension model_dimension(std::size_t value) {
  HELIX_ASSERT(value <= static_cast<std::size_t>(std::numeric_limits<Dimension>::max()),
               "validated model dimension must fit Dimension");
  return static_cast<Dimension>(value);
}

std::string_view role_name(TensorRole role) {
  switch (role) {
  case TensorRole::kTokenEmbedding:
    return "token_embedding";
  case TensorRole::kAttentionNorm:
    return "attention_norm";
  case TensorRole::kQuery:
    return "query";
  case TensorRole::kKey:
    return "key";
  case TensorRole::kValue:
    return "value";
  case TensorRole::kAttentionOutput:
    return "attention_output";
  case TensorRole::kFfnNorm:
    return "ffn_norm";
  case TensorRole::kGate:
    return "gate";
  case TensorRole::kUp:
    return "up";
  case TensorRole::kDown:
    return "down";
  case TensorRole::kFinalNorm:
    return "final_norm";
  case TensorRole::kLmHead:
    return "lm_head";
  }
  return "unknown";
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
  for (std::size_t shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void append_float(std::vector<std::uint8_t>& bytes, float value) {
  append_u32(bytes, std::bit_cast<std::uint32_t>(value));
}

void set_u64(std::span<std::uint8_t> bytes, std::size_t offset, std::uint64_t value) {
  HELIX_ASSERT(offset <= bytes.size() && bytes.size() - offset >= sizeof(value),
               "model metadata checksum offset must be in bounds");
  for (std::size_t shift = 0; shift < 64; shift += 8) {
    bytes[offset + shift / 8] = static_cast<std::uint8_t>((value >> shift) & 0xffU);
  }
}

class ByteReader final {
public:
  explicit ByteReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  [[nodiscard]] std::uint32_t u32(std::string_view field) {
    require(sizeof(std::uint32_t), field);
    std::uint32_t value = 0;
    for (std::size_t shift = 0; shift < 32; shift += 8) {
      value |= static_cast<std::uint32_t>(bytes_[offset_++]) << shift;
    }
    return value;
  }

  [[nodiscard]] std::uint64_t u64(std::string_view field) {
    require(sizeof(std::uint64_t), field);
    std::uint64_t value = 0;
    for (std::size_t shift = 0; shift < 64; shift += 8) {
      value |= static_cast<std::uint64_t>(bytes_[offset_++]) << shift;
    }
    return value;
  }

  [[nodiscard]] std::size_t offset() const noexcept {
    return offset_;
  }

private:
  void require(std::size_t count, std::string_view field) const {
    if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
      invalid_format(std::string("truncated metadata while reading ") + std::string(field));
    }
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

std::uint64_t fnv_bytes(std::span<const std::uint8_t> bytes, std::uint64_t hash = kFnvOffset) {
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= kFnvPrime;
  }
  return hash;
}

void fnv_u32(std::uint64_t& hash, std::uint32_t value) {
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    hash ^= static_cast<std::uint8_t>((value >> shift) & 0xffU);
    hash *= kFnvPrime;
  }
}

std::uint64_t tensor_checksum(const Tensor& tensor, TensorRole role) {
  std::uint64_t hash = kFnvOffset;
  const float* values = tensor.data_as<float>();
  for (std::size_t index = 0; index < tensor.numel(); ++index) {
    if (!std::isfinite(values[index])) {
      throw_error(ErrorCode::kInvalidArgument, "model::format: " + std::string(role_name(role)) +
                                                   " contains a non-finite value");
    }
    fnv_u32(hash, std::bit_cast<std::uint32_t>(values[index]));
  }
  return hash;
}

void read_bytes(std::istream& input, std::span<std::uint8_t> bytes, std::string_view context);

std::size_t file_size(const std::filesystem::path& path, std::string_view context) {
  std::error_code error;
  const std::uintmax_t bytes = std::filesystem::file_size(path, error);
  if (error) {
    io_error(std::string(context) + " could not inspect '" + path.string() +
             "': " + error.message());
  }
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    throw_error(ErrorCode::kOutOfRange,
                "model::format: " + std::string(context) + " exceeds this host's size range");
  }
  return static_cast<std::size_t>(bytes);
}

std::uint64_t raw_tensor_checksum(const std::filesystem::path& path, const TensorSpec& spec) {
  const std::size_t expected_bytes = checked_nbytes(spec.shape, DType::kFloat32);
  const std::size_t actual_bytes = file_size(path, "raw tensor payload");
  if (actual_bytes != expected_bytes) {
    invalid_format("raw " + std::string(role_name(spec.role)) + " payload '" + path.string() +
                   "' has " + std::to_string(actual_bytes) + " bytes; expected " +
                   std::to_string(expected_bytes));
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    io_error("could not open raw tensor payload '" + path.string() + "' for reading");
  }
  std::vector<std::uint8_t> bytes(kIoChunkBytes);
  std::size_t remaining = expected_bytes;
  std::uint64_t checksum = kFnvOffset;
  while (remaining != 0) {
    const std::size_t byte_count = std::min(remaining, bytes.size());
    HELIX_ASSERT(byte_count % sizeof(float) == 0,
                 "raw Float32 checksum chunk must contain complete values");
    read_bytes(input, std::span<std::uint8_t>(bytes.data(), byte_count), "raw tensor payload");
    checksum = fnv_bytes(std::span<const std::uint8_t>(bytes.data(), byte_count), checksum);
    for (std::size_t offset = 0; offset < byte_count; offset += sizeof(float)) {
      std::uint32_t bits = 0;
      for (std::size_t shift = 0; shift < 32; shift += 8) {
        bits |= static_cast<std::uint32_t>(bytes[offset + shift / 8]) << shift;
      }
      if (!std::isfinite(std::bit_cast<float>(bits))) {
        invalid_format("raw " + std::string(role_name(spec.role)) + " payload '" + path.string() +
                       "' contains a non-finite value");
      }
    }
    remaining -= byte_count;
  }
  return checksum;
}

std::vector<TensorSpec> make_specs(const LlamaConfig& config, const LlamaWeights* weights) {
  config.validate();
  if (config.layer_count > kGlobalLayer) {
    throw_error(ErrorCode::kOutOfRange,
                "model::format: layer count exceeds the v1 file-format range");
  }

  const std::size_t layer_tensor_count =
      checked_multiply(config.layer_count, 9, "model file layer tensor count");
  const std::size_t tensor_count = checked_add(layer_tensor_count, 3, "model file tensor count");
  std::vector<TensorSpec> specs;
  specs.reserve(tensor_count);

  const Dimension vocabulary = model_dimension(config.vocabulary_size);
  const Dimension hidden = model_dimension(config.hidden_size);
  const Dimension intermediate = model_dimension(config.intermediate_size);
  specs.push_back(TensorSpec{TensorRole::kTokenEmbedding, kGlobalLayer, Shape{vocabulary, hidden},
                             weights == nullptr ? nullptr : &weights->token_embedding()});

  for (std::size_t layer_index = 0; layer_index < config.layer_count; ++layer_index) {
    const auto serialized_layer = static_cast<std::uint32_t>(layer_index);
    const LlamaLayerWeights* layer = weights == nullptr ? nullptr : &weights->layers()[layer_index];
    specs.push_back(TensorSpec{TensorRole::kAttentionNorm, serialized_layer, Shape{hidden},
                               layer == nullptr ? nullptr : &layer->attention_norm_weight});
    specs.push_back(TensorSpec{TensorRole::kQuery, serialized_layer, Shape{hidden, hidden},
                               layer == nullptr ? nullptr : &layer->query_weight});
    specs.push_back(TensorSpec{TensorRole::kKey, serialized_layer, Shape{hidden, hidden},
                               layer == nullptr ? nullptr : &layer->key_weight});
    specs.push_back(TensorSpec{TensorRole::kValue, serialized_layer, Shape{hidden, hidden},
                               layer == nullptr ? nullptr : &layer->value_weight});
    specs.push_back(TensorSpec{TensorRole::kAttentionOutput, serialized_layer,
                               Shape{hidden, hidden},
                               layer == nullptr ? nullptr : &layer->attention_output_weight});
    specs.push_back(TensorSpec{TensorRole::kFfnNorm, serialized_layer, Shape{hidden},
                               layer == nullptr ? nullptr : &layer->ffn_norm_weight});
    specs.push_back(TensorSpec{TensorRole::kGate, serialized_layer, Shape{hidden, intermediate},
                               layer == nullptr ? nullptr : &layer->gate_weight});
    specs.push_back(TensorSpec{TensorRole::kUp, serialized_layer, Shape{hidden, intermediate},
                               layer == nullptr ? nullptr : &layer->up_weight});
    specs.push_back(TensorSpec{TensorRole::kDown, serialized_layer, Shape{intermediate, hidden},
                               layer == nullptr ? nullptr : &layer->down_weight});
  }

  specs.push_back(TensorSpec{TensorRole::kFinalNorm, kGlobalLayer, Shape{hidden},
                             weights == nullptr ? nullptr : &weights->final_norm_weight()});
  specs.push_back(TensorSpec{TensorRole::kLmHead, kGlobalLayer, Shape{hidden, vocabulary},
                             weights == nullptr ? nullptr : &weights->lm_head_weight()});
  HELIX_ASSERT(specs.size() == tensor_count, "model tensor specification count must match");
  return specs;
}

std::vector<DirectoryRecord> make_records(std::span<const TensorSpec> specs,
                                          std::span<const std::uint64_t> checksums,
                                          std::size_t data_offset, std::size_t& file_size) {
  HELIX_ASSERT(specs.size() == checksums.size(),
               "model tensor specs and checksums must have equal sizes");
  std::vector<DirectoryRecord> records;
  records.reserve(specs.size());
  std::size_t cursor = data_offset;
  for (std::size_t index = 0; index < specs.size(); ++index) {
    const TensorSpec& spec = specs[index];
    cursor = aligned_size(cursor, kPayloadAlignment, "tensor payload alignment");
    const std::size_t payload_bytes = checked_nbytes(spec.shape, DType::kFloat32);
    DirectoryRecord record{spec.role,
                           spec.layer,
                           kFloat32Code,
                           static_cast<std::uint32_t>(spec.shape.size()),
                           {},
                           to_u64(cursor),
                           to_u64(payload_bytes),
                           kPayloadAlignment,
                           checksums[index]};
    for (std::size_t dimension_index = 0; dimension_index < spec.shape.size(); ++dimension_index) {
      record.dimensions[dimension_index] = static_cast<std::uint64_t>(spec.shape[dimension_index]);
    }
    records.push_back(record);
    cursor = checked_add(cursor, payload_bytes, "model file payload");
  }
  file_size = cursor;
  return records;
}

std::vector<std::uint8_t> encode_directory(std::span<const DirectoryRecord> records) {
  const std::size_t byte_count = checked_multiply(
      records.size(), static_cast<std::size_t>(kDirectoryEntryBytes), "model directory");
  std::vector<std::uint8_t> bytes;
  bytes.reserve(byte_count);
  for (const DirectoryRecord& record : records) {
    append_u32(bytes, static_cast<std::uint32_t>(record.role));
    append_u32(bytes, record.layer);
    append_u32(bytes, record.dtype);
    append_u32(bytes, record.rank);
    for (const std::uint64_t dimension : record.dimensions) {
      append_u64(bytes, dimension);
    }
    append_u64(bytes, record.offset);
    append_u64(bytes, record.byte_length);
    append_u64(bytes, record.alignment);
    append_u64(bytes, record.checksum);
  }
  HELIX_ASSERT(bytes.size() == byte_count, "encoded model directory size must match");
  return bytes;
}

std::vector<std::uint8_t> encode_header(const LlamaConfig& config, std::size_t tensor_count,
                                        std::size_t data_offset, std::size_t file_size) {
  std::vector<std::uint8_t> bytes(kMagic.begin(), kMagic.end());
  bytes.reserve(kHeaderBytes);
  append_u32(bytes, kFormatVersion);
  append_u32(bytes, kEndianMarker);
  append_u32(bytes, kHeaderBytes);
  append_u32(bytes, kModelKindLlama);
  append_u64(bytes, to_u64(file_size));
  append_u64(bytes, kDirectoryOffset);
  append_u64(bytes, to_u64(tensor_count));
  append_u64(bytes, kDirectoryEntryBytes);
  append_u64(bytes, to_u64(data_offset));
  append_u64(bytes, to_u64(config.vocabulary_size));
  append_u64(bytes, to_u64(config.hidden_size));
  append_u64(bytes, to_u64(config.intermediate_size));
  append_u64(bytes, to_u64(config.layer_count));
  append_u64(bytes, to_u64(config.attention_head_count));
  append_u64(bytes, to_u64(config.max_sequence_length));
  append_float(bytes, config.rms_norm_epsilon);
  append_float(bytes, config.rope_theta);
  append_u64(bytes, 0);
  HELIX_ASSERT(bytes.size() == kHeaderBytes, "encoded model header size must match");
  return bytes;
}

void write_bytes(std::ostream& output, std::span<const std::uint8_t> bytes,
                 std::string_view context) {
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    io_error(std::string(context) + " exceeds stream write range");
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    io_error("failed while writing " + std::string(context));
  }
}

void write_padding(std::ostream& output, std::size_t count) {
  const std::array<std::uint8_t, 64> zeros{};
  while (count != 0) {
    const std::size_t chunk = std::min(count, zeros.size());
    write_bytes(output, std::span<const std::uint8_t>(zeros.data(), chunk), "alignment padding");
    count -= chunk;
  }
}

void write_tensor_payload(std::ostream& output, const Tensor& tensor) {
  std::vector<std::uint8_t> bytes(kIoChunkBytes);
  const float* values = tensor.data_as<float>();
  std::size_t value_offset = 0;
  while (value_offset < tensor.numel()) {
    const std::size_t value_count =
        std::min(tensor.numel() - value_offset, bytes.size() / sizeof(float));
    for (std::size_t index = 0; index < value_count; ++index) {
      const std::uint32_t bits = std::bit_cast<std::uint32_t>(values[value_offset + index]);
      const std::size_t byte_offset = index * sizeof(float);
      for (std::size_t shift = 0; shift < 32; shift += 8) {
        bytes[byte_offset + shift / 8] = static_cast<std::uint8_t>((bits >> shift) & 0xffU);
      }
    }
    write_bytes(output, std::span<const std::uint8_t>(bytes.data(), value_count * sizeof(float)),
                "tensor payload");
    value_offset += value_count;
  }
}

void write_raw_tensor_payload(std::ostream& output, const std::filesystem::path& path,
                              std::size_t byte_count, std::uint64_t expected_checksum) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    io_error("could not reopen raw tensor payload '" + path.string() + "' for reading");
  }
  std::vector<std::uint8_t> bytes(kIoChunkBytes);
  std::size_t remaining = byte_count;
  std::uint64_t checksum = kFnvOffset;
  while (remaining != 0) {
    const std::size_t chunk = std::min(remaining, bytes.size());
    read_bytes(input, std::span<std::uint8_t>(bytes.data(), chunk), "raw tensor payload");
    checksum = fnv_bytes(std::span<const std::uint8_t>(bytes.data(), chunk), checksum);
    write_bytes(output, std::span<const std::uint8_t>(bytes.data(), chunk), "raw tensor payload");
    remaining -= chunk;
  }
  if (checksum != expected_checksum || input.peek() != std::char_traits<char>::eof()) {
    invalid_format("raw tensor payload '" + path.string() +
                   "' changed while the model was being written");
  }
}

template <typename PayloadWriter>
void write_llama_model_stream(std::ostream& output, const LlamaConfig& config,
                              std::span<const TensorSpec> specs,
                              std::span<const std::uint64_t> checksums,
                              PayloadWriter&& write_payload) {
  if (output.tellp() != std::streampos(0)) {
    invalid_format("output stream must be seekable and positioned at offset zero");
  }
  const std::size_t directory_bytes = checked_multiply(
      specs.size(), static_cast<std::size_t>(kDirectoryEntryBytes), "model directory");
  const std::size_t metadata_end = checked_add(kHeaderBytes, directory_bytes, "model metadata");
  const std::size_t data_offset =
      aligned_size(metadata_end, kPayloadAlignment, "model payload start");
  std::size_t model_file_size = 0;
  const std::vector<DirectoryRecord> records =
      make_records(specs, checksums, data_offset, model_file_size);
  const std::streamoff expected_stream_size = to_streamoff(model_file_size, "model file size");
  const std::vector<std::uint8_t> directory = encode_directory(records);
  std::vector<std::uint8_t> header =
      encode_header(config, specs.size(), data_offset, model_file_size);
  std::uint64_t metadata_checksum = fnv_bytes(header);
  metadata_checksum = fnv_bytes(directory, metadata_checksum);
  set_u64(header, kHeaderBytes - sizeof(std::uint64_t), metadata_checksum);

  write_bytes(output, header, "model header");
  write_bytes(output, directory, "tensor directory");
  std::size_t cursor = metadata_end;
  for (std::size_t index = 0; index < specs.size(); ++index) {
    const std::size_t offset = to_size(records[index].offset, "tensor offset");
    write_padding(output, offset - cursor);
    write_payload(output, index, to_size(records[index].byte_length, "tensor byte length"));
    cursor = checked_add(offset, to_size(records[index].byte_length, "tensor byte length"),
                         "model payload cursor");
  }
  HELIX_ASSERT(cursor == model_file_size, "written model byte count must match header");
  if (output.tellp() != std::streampos(expected_stream_size)) {
    io_error("written stream length does not match the model header");
  }
}

std::size_t stream_size(std::istream& input) {
  input.clear();
  input.seekg(0, std::ios::end);
  if (!input) {
    invalid_format("input stream must be seekable");
  }
  const std::streampos end = input.tellg();
  if (end < std::streampos(0)) {
    invalid_format("could not determine input size");
  }
  const auto unsigned_end = static_cast<std::uintmax_t>(end);
  if (unsigned_end > std::numeric_limits<std::size_t>::max()) {
    invalid_format("input size exceeds this host's addressable range");
  }
  input.seekg(0, std::ios::beg);
  if (!input) {
    invalid_format("could not seek to model-file start");
  }
  return static_cast<std::size_t>(unsigned_end);
}

void read_bytes(std::istream& input, std::span<std::uint8_t> bytes, std::string_view context) {
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    invalid_format(std::string(context) + " exceeds stream read range");
  }
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    invalid_format("truncated " + std::string(context));
  }
}

void read_zero_padding(std::istream& input, std::size_t count) {
  std::array<std::uint8_t, 64> bytes{};
  while (count != 0) {
    const std::size_t chunk = std::min(count, bytes.size());
    read_bytes(input, std::span<std::uint8_t>(bytes.data(), chunk), "alignment padding");
    if (!std::all_of(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(chunk),
                     [](std::uint8_t value) { return value == 0; })) {
      invalid_format("alignment padding must contain only zero bytes");
    }
    count -= chunk;
  }
}

DirectoryRecord decode_record(ByteReader& reader) {
  DirectoryRecord record{};
  record.role = static_cast<TensorRole>(reader.u32("tensor role"));
  record.layer = reader.u32("tensor layer");
  record.dtype = reader.u32("tensor dtype");
  record.rank = reader.u32("tensor rank");
  for (std::uint64_t& dimension : record.dimensions) {
    dimension = reader.u64("tensor dimension");
  }
  record.offset = reader.u64("tensor offset");
  record.byte_length = reader.u64("tensor byte length");
  record.alignment = reader.u64("tensor alignment");
  record.checksum = reader.u64("tensor checksum");
  return record;
}

void validate_record(const DirectoryRecord& record, const TensorSpec& spec,
                     std::size_t expected_offset, std::size_t file_size) {
  const std::string tensor_name(role_name(spec.role));
  if (record.role != spec.role || record.layer != spec.layer) {
    invalid_format("tensor directory order or layer is invalid at " + tensor_name);
  }
  if (record.dtype != kFloat32Code) {
    invalid_format(tensor_name + " has an unsupported dtype code");
  }
  if (record.rank != spec.shape.size() || record.rank > record.dimensions.size()) {
    invalid_format(tensor_name + " has an invalid rank");
  }
  for (std::size_t dimension = 0; dimension < record.dimensions.size(); ++dimension) {
    const std::uint64_t expected =
        dimension < spec.shape.size() ? static_cast<std::uint64_t>(spec.shape[dimension]) : 0;
    if (record.dimensions[dimension] != expected) {
      invalid_format(tensor_name + " has an unexpected shape");
    }
  }
  const std::size_t expected_bytes = checked_nbytes(spec.shape, DType::kFloat32);
  if (record.byte_length != to_u64(expected_bytes)) {
    invalid_format(tensor_name + " has an invalid byte length");
  }
  if (record.alignment != kPayloadAlignment || record.offset != to_u64(expected_offset) ||
      record.offset % kPayloadAlignment != 0) {
    invalid_format(tensor_name + " has a non-canonical payload offset or alignment");
  }
  const std::size_t offset = to_size(record.offset, "tensor offset");
  const std::size_t byte_length = to_size(record.byte_length, "tensor byte length");
  if (offset > file_size || byte_length > file_size - offset) {
    invalid_format(tensor_name + " payload exceeds the file bounds");
  }
}

DecodedHeader decode_header(std::span<const std::uint8_t> header, std::size_t actual_size) {
  if (actual_size < kHeaderBytes || header.size() != kHeaderBytes) {
    invalid_format("file is smaller than the v1 header");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
    invalid_format("magic does not identify a Helix Llama model");
  }

  ByteReader reader(header.subspan(kMagic.size()));
  const std::uint32_t version = reader.u32("format version");
  const std::uint32_t endian_marker = reader.u32("endianness marker");
  const std::uint32_t header_bytes = reader.u32("header size");
  const std::uint32_t model_kind = reader.u32("model kind");
  const std::uint64_t declared_file_size = reader.u64("file size");
  const std::uint64_t directory_offset = reader.u64("directory offset");
  const std::uint64_t tensor_count_value = reader.u64("tensor count");
  const std::uint64_t directory_entry_bytes = reader.u64("directory entry size");
  const std::uint64_t data_offset_value = reader.u64("data offset");
  LlamaConfig config;
  config.vocabulary_size = to_size(reader.u64("vocabulary size"), "vocabulary size");
  config.hidden_size = to_size(reader.u64("hidden size"), "hidden size");
  config.intermediate_size = to_size(reader.u64("intermediate size"), "intermediate size");
  config.layer_count = to_size(reader.u64("layer count"), "layer count");
  config.attention_head_count = to_size(reader.u64("attention head count"), "attention head count");
  config.max_sequence_length =
      to_size(reader.u64("maximum sequence length"), "maximum sequence length");
  config.rms_norm_epsilon = std::bit_cast<float>(reader.u32("RMSNorm epsilon"));
  config.rope_theta = std::bit_cast<float>(reader.u32("RoPE theta"));
  const std::uint64_t metadata_checksum = reader.u64("metadata checksum");
  HELIX_ASSERT(reader.offset() + kMagic.size() == kHeaderBytes,
               "decoded model header size must match");

  if (version != kFormatVersion) {
    invalid_format("unsupported format version " + std::to_string(version));
  }
  if (endian_marker != kEndianMarker) {
    invalid_format("invalid endianness marker");
  }
  if (header_bytes != kHeaderBytes || model_kind != kModelKindLlama ||
      directory_offset != kDirectoryOffset || directory_entry_bytes != kDirectoryEntryBytes) {
    invalid_format("header layout or model kind is not supported by v1");
  }
  if (declared_file_size != to_u64(actual_size)) {
    invalid_format("declared file size does not match the stream length");
  }
  config.validate();

  const std::size_t tensor_count = to_size(tensor_count_value, "tensor count");
  if (tensor_count > (actual_size - kHeaderBytes) / kDirectoryEntryBytes) {
    invalid_format("tensor directory exceeds the file bounds");
  }
  const std::size_t directory_bytes = checked_multiply(
      tensor_count, static_cast<std::size_t>(kDirectoryEntryBytes), "model directory");
  const std::size_t metadata_end = checked_add(kHeaderBytes, directory_bytes, "model metadata");
  const std::size_t expected_data_offset =
      aligned_size(metadata_end, kPayloadAlignment, "model payload start");
  const std::size_t data_offset = to_size(data_offset_value, "data offset");
  if (data_offset != expected_data_offset || data_offset > actual_size) {
    invalid_format("data offset is not canonical or exceeds the file bounds");
  }

  return DecodedHeader{std::move(config), metadata_checksum, tensor_count,
                       directory_bytes,   metadata_end,      data_offset};
}

ParsedModelMetadata parse_model_metadata(DecodedHeader header,
                                         std::span<const std::uint8_t> header_bytes,
                                         std::span<const std::uint8_t> directory,
                                         std::size_t actual_size) {
  if (directory.size() != header.directory_bytes) {
    invalid_format("truncated tensor directory");
  }

  std::vector<TensorSpec> specs = make_specs(header.config, nullptr);
  if (header.tensor_count != specs.size()) {
    invalid_format("tensor count does not match the model configuration");
  }

  std::vector<std::uint8_t> checksum_header(header_bytes.begin(), header_bytes.end());
  set_u64(checksum_header, kHeaderBytes - sizeof(std::uint64_t), 0);
  std::uint64_t computed_metadata_checksum = fnv_bytes(checksum_header);
  computed_metadata_checksum = fnv_bytes(directory, computed_metadata_checksum);
  if (computed_metadata_checksum != header.metadata_checksum) {
    invalid_format("metadata checksum mismatch");
  }

  ByteReader directory_reader(directory);
  std::vector<DirectoryRecord> records;
  records.reserve(header.tensor_count);
  std::size_t expected_offset = header.data_offset;
  for (std::size_t index = 0; index < header.tensor_count; ++index) {
    DirectoryRecord record = decode_record(directory_reader);
    expected_offset = aligned_size(expected_offset, kPayloadAlignment, "tensor payload offset");
    validate_record(record, specs[index], expected_offset, actual_size);
    expected_offset = checked_add(expected_offset, to_size(record.byte_length, "tensor bytes"),
                                  "model payload bounds");
    records.push_back(record);
  }
  HELIX_ASSERT(directory_reader.offset() == directory.size(),
               "decoded model directory size must match");
  if (expected_offset != actual_size) {
    invalid_format("payload extents do not exactly cover the declared file");
  }

  return ParsedModelMetadata{std::move(header), std::move(specs), std::move(records)};
}

void validate_zero_padding(std::span<const std::uint8_t> padding) {
  if (!std::all_of(padding.begin(), padding.end(), [](std::uint8_t value) { return value == 0; })) {
    invalid_format("alignment padding must contain only zero bytes");
  }
}

float decode_finite_float32(std::span<const std::uint8_t> bytes, std::size_t offset,
                            TensorRole role) {
  HELIX_ASSERT(offset <= bytes.size() && bytes.size() - offset >= sizeof(float),
               "Float32 payload decode must remain in bounds");
  std::uint32_t bits = 0;
  for (std::size_t shift = 0; shift < 32; shift += 8) {
    bits |= static_cast<std::uint32_t>(bytes[offset + shift / 8]) << shift;
  }
  const float value = std::bit_cast<float>(bits);
  if (!std::isfinite(value)) {
    invalid_format(std::string(role_name(role)) + " contains a non-finite value");
  }
  return value;
}

void validate_finite_float32_payload(std::span<const std::uint8_t> bytes, TensorRole role) {
  HELIX_ASSERT(bytes.size() % sizeof(float) == 0, "Float32 payload must contain complete values");
  for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
    static_cast<void>(decode_finite_float32(bytes, offset, role));
  }
}

Tensor read_tensor_payload(std::istream& input, const TensorSpec& spec,
                           const DirectoryRecord& record) {
  Tensor tensor = Tensor::empty(spec.shape);
  float* values = tensor.data_as<float>();
  std::vector<std::uint8_t> bytes(kIoChunkBytes);
  std::size_t value_offset = 0;
  std::uint64_t checksum = kFnvOffset;
  while (value_offset < tensor.numel()) {
    const std::size_t value_count =
        std::min(tensor.numel() - value_offset, bytes.size() / sizeof(float));
    const std::size_t byte_count = value_count * sizeof(float);
    read_bytes(input, std::span<std::uint8_t>(bytes.data(), byte_count), "tensor payload");
    const std::span<const std::uint8_t> chunk(bytes.data(), byte_count);
    checksum = fnv_bytes(chunk, checksum);
    for (std::size_t index = 0; index < value_count; ++index) {
      const std::size_t byte_offset = index * sizeof(float);
      values[value_offset + index] = decode_finite_float32(chunk, byte_offset, spec.role);
    }
    value_offset += value_count;
  }
  if (checksum != record.checksum) {
    invalid_format(std::string(role_name(spec.role)) + " payload checksum mismatch");
  }
  return tensor;
}

Tensor map_tensor_payload(std::span<const std::uint8_t> file_bytes, const TensorSpec& spec,
                          const DirectoryRecord& record,
                          const std::shared_ptr<const void>& mapping_owner) {
  const std::size_t offset = to_size(record.offset, "tensor offset");
  const std::size_t byte_length = to_size(record.byte_length, "tensor byte length");
  const std::span<const std::uint8_t> payload = file_bytes.subspan(offset, byte_length);
  if (fnv_bytes(payload) != record.checksum) {
    invalid_format(std::string(role_name(spec.role)) + " payload checksum mismatch");
  }
  validate_finite_float32_payload(payload, spec.role);
  if (reinterpret_cast<std::uintptr_t>(payload.data()) % kPayloadAlignment != 0) {
    io_error("mapped tensor payload does not satisfy its declared alignment");
  }
  return Tensor::from_read_only_cpu_buffer(spec.shape, DType::kFloat32, payload.data(),
                                           mapping_owner, kPayloadAlignment);
}

LlamaWeights assemble_weights(const LlamaConfig& config, std::vector<Tensor> tensors) {
  HELIX_ASSERT(tensors.size() == 3 + config.layer_count * 9,
               "loaded tensor count must match model configuration");
  Tensor token_embedding = std::move(tensors[0]);
  std::vector<LlamaLayerWeights> layers;
  layers.reserve(config.layer_count);
  for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
    const std::size_t base = 1 + layer * 9;
    layers.emplace_back(
        std::move(tensors[base]), std::move(tensors[base + 1]), std::move(tensors[base + 2]),
        std::move(tensors[base + 3]), std::move(tensors[base + 4]), std::move(tensors[base + 5]),
        std::move(tensors[base + 6]), std::move(tensors[base + 7]), std::move(tensors[base + 8]));
  }
  const std::size_t final_index = 1 + config.layer_count * 9;
  LlamaWeights weights(std::move(token_embedding), std::move(layers),
                       std::move(tensors[final_index]), std::move(tensors[final_index + 1]));
  weights.validate(config);
  return weights;
}

} // namespace

void write_llama_model_file(std::ostream& output, const LlamaConfig& config,
                            const LlamaWeights& weights) {
  config.validate();
  weights.validate(config);
  const std::vector<TensorSpec> specs = make_specs(config, &weights);
  std::vector<std::uint64_t> checksums;
  checksums.reserve(specs.size());
  for (const TensorSpec& spec : specs) {
    HELIX_ASSERT(spec.tensor != nullptr, "write tensor specification must contain a tensor");
    checksums.push_back(tensor_checksum(*spec.tensor, spec.role));
  }
  write_llama_model_stream(
      output, config, specs, checksums,
      [&specs](std::ostream& payload_output, std::size_t index, std::size_t byte_length) {
        HELIX_ASSERT(specs[index].tensor->nbytes() == byte_length,
                     "tensor payload byte length must match its record");
        write_tensor_payload(payload_output, *specs[index].tensor);
      });
}

void write_llama_model_file(const std::filesystem::path& path, const LlamaConfig& config,
                            const LlamaWeights& weights) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    io_error("could not open '" + path.string() + "' for writing");
  }
  write_llama_model_file(output, config, weights);
  output.flush();
  if (!output) {
    io_error("could not finish writing '" + path.string() + "'");
  }
}

void write_llama_model_file_from_raw_float32_files(
    const std::filesystem::path& path, const LlamaConfig& config,
    std::span<const std::filesystem::path> tensor_paths) {
  config.validate();
  const std::vector<TensorSpec> specs = make_specs(config, nullptr);
  if (tensor_paths.size() != specs.size()) {
    invalid_format("raw tensor file count " + std::to_string(tensor_paths.size()) +
                   " does not match expected count " + std::to_string(specs.size()));
  }

  std::error_code path_error;
  const bool output_exists = std::filesystem::exists(path, path_error);
  if (path_error) {
    io_error("could not inspect output path '" + path.string() + "': " + path_error.message());
  }
  std::filesystem::path normalized_output;
  if (output_exists) {
    normalized_output = std::filesystem::weakly_canonical(path, path_error);
  } else {
    const std::filesystem::path output_parent = std::filesystem::weakly_canonical(
        path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path(), path_error);
    normalized_output = output_parent / path.filename();
  }
  if (path_error) {
    io_error("could not resolve output path '" + path.string() + "': " + path_error.message());
  }

  std::vector<std::uint64_t> checksums;
  checksums.reserve(specs.size());
  for (std::size_t index = 0; index < specs.size(); ++index) {
    path_error.clear();
    const std::filesystem::path normalized_input =
        std::filesystem::weakly_canonical(tensor_paths[index], path_error);
    if (path_error) {
      io_error("could not resolve raw tensor payload '" + tensor_paths[index].string() +
               "': " + path_error.message());
    }
    bool equivalent = false;
    if (output_exists) {
      equivalent = std::filesystem::equivalent(path, tensor_paths[index], path_error);
      if (path_error) {
        io_error("could not compare output path with raw tensor payload '" +
                 tensor_paths[index].string() + "': " + path_error.message());
      }
    }
    if (normalized_input == normalized_output || equivalent) {
      invalid_format("output path aliases raw tensor payload '" + tensor_paths[index].string() +
                     "'");
    }
    checksums.push_back(raw_tensor_checksum(tensor_paths[index], specs[index]));
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    io_error("could not open '" + path.string() + "' for writing");
  }
  write_llama_model_stream(output, config, specs, checksums,
                           [&tensor_paths, &checksums](std::ostream& payload_output,
                                                       std::size_t index, std::size_t byte_length) {
                             write_raw_tensor_payload(payload_output, tensor_paths[index],
                                                      byte_length, checksums[index]);
                           });
  output.flush();
  if (!output) {
    io_error("could not finish writing '" + path.string() + "'");
  }
}

LoadedLlamaModel read_llama_model_file(std::istream& input) {
  const std::size_t actual_size = stream_size(input);
  if (actual_size < kHeaderBytes) {
    invalid_format("file is smaller than the v1 header");
  }

  std::vector<std::uint8_t> header(kHeaderBytes);
  read_bytes(input, header, "model header");
  DecodedHeader decoded_header = decode_header(header, actual_size);
  std::vector<std::uint8_t> directory(decoded_header.directory_bytes);
  read_bytes(input, directory, "tensor directory");
  ParsedModelMetadata metadata =
      parse_model_metadata(std::move(decoded_header), header, directory, actual_size);

  std::size_t cursor = metadata.header.metadata_end;
  std::vector<Tensor> tensors;
  tensors.reserve(metadata.header.tensor_count);
  for (std::size_t index = 0; index < metadata.header.tensor_count; ++index) {
    const std::size_t offset = to_size(metadata.records[index].offset, "tensor offset");
    read_zero_padding(input, offset - cursor);
    tensors.push_back(read_tensor_payload(input, metadata.specs[index], metadata.records[index]));
    cursor = checked_add(offset, to_size(metadata.records[index].byte_length, "tensor bytes"),
                         "model payload cursor");
  }
  HELIX_ASSERT(cursor == actual_size, "read model byte count must match stream size");

  LlamaConfig config = metadata.header.config;
  LlamaWeights weights = assemble_weights(config, std::move(tensors));
  return LoadedLlamaModel{std::move(config), std::move(weights)};
}

LoadedLlamaModel read_llama_model_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    io_error("could not open '" + path.string() + "' for reading");
  }
  return read_llama_model_file(input);
}

bool mapped_model_loading_supported() noexcept {
#if HELIX_MODEL_HAS_POSIX_MMAP
  return std::endian::native == std::endian::little && std::numeric_limits<float>::is_iec559;
#else
  return false;
#endif
}

LoadedLlamaModel read_llama_model_file_mapped(const std::filesystem::path& path) {
#if HELIX_MODEL_HAS_POSIX_MMAP
  HELIX_CHECK(std::endian::native == std::endian::little && std::numeric_limits<float>::is_iec559,
              ErrorCode::kUnsupported,
              "model::format: memory-mapped model loading requires a little-endian IEEE-754 "
              "Float32 host");
  auto mapping = std::make_shared<MappedFile>(path);
  const std::span<const std::uint8_t> file_bytes = mapping->bytes();
  if (file_bytes.size() < kHeaderBytes) {
    invalid_format("file is smaller than the v1 header");
  }
  if (reinterpret_cast<std::uintptr_t>(file_bytes.data()) % kPayloadAlignment != 0) {
    io_error("model-file mapping does not satisfy payload alignment");
  }

  const std::span<const std::uint8_t> header = file_bytes.first(kHeaderBytes);
  DecodedHeader decoded_header = decode_header(header, file_bytes.size());
  const std::span<const std::uint8_t> directory =
      file_bytes.subspan(kHeaderBytes, decoded_header.directory_bytes);
  ParsedModelMetadata metadata =
      parse_model_metadata(std::move(decoded_header), header, directory, file_bytes.size());
  const std::shared_ptr<const void> mapping_owner = mapping;

  std::size_t cursor = metadata.header.metadata_end;
  std::vector<Tensor> tensors;
  tensors.reserve(metadata.header.tensor_count);
  for (std::size_t index = 0; index < metadata.header.tensor_count; ++index) {
    const std::size_t offset = to_size(metadata.records[index].offset, "tensor offset");
    validate_zero_padding(file_bytes.subspan(cursor, offset - cursor));
    tensors.push_back(map_tensor_payload(file_bytes, metadata.specs[index], metadata.records[index],
                                         mapping_owner));
    cursor = checked_add(offset, to_size(metadata.records[index].byte_length, "tensor bytes"),
                         "model payload cursor");
  }
  HELIX_ASSERT(cursor == file_bytes.size(), "mapped model byte count must match file size");

  LlamaConfig config = metadata.header.config;
  LlamaWeights weights = assemble_weights(config, std::move(tensors));
  return LoadedLlamaModel{std::move(config), std::move(weights)};
#else
  static_cast<void>(path);
  throw_error(ErrorCode::kUnsupported,
              "model::format: memory-mapped model loading requires a POSIX host");
#endif
}

} // namespace helix::model
