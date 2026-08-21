#include "helix/kv_cache/contiguous.hpp"

#include "helix/core/error.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace helix::kv_cache {
namespace {

std::uint64_t next_cache_id() {
  static std::atomic<std::uint64_t> next{1};
  std::uint64_t current = next.load(std::memory_order_relaxed);
  while (true) {
    HELIX_CHECK(current != std::numeric_limits<std::uint64_t>::max(), ErrorCode::kResourceExhausted,
                "contiguous KV cache identity space exhausted");
    if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return current;
    }
  }
}

Dimension checked_dimension(std::size_t value, std::string_view name) {
  HELIX_CHECK(value <= static_cast<std::size_t>(std::numeric_limits<Dimension>::max()),
              ErrorCode::kOutOfRange, std::string(name) + " does not fit Tensor dimension type");
  return static_cast<Dimension>(value);
}

std::string tensor_error(std::string_view operation, std::string_view name,
                         std::string_view requirement) {
  std::ostringstream message;
  message << operation << ": " << name << ' ' << requirement;
  return message.str();
}

void require_cpu_float_tensor(const Tensor& tensor, std::string_view name,
                              std::string_view operation) {
  HELIX_CHECK(tensor.device().is_cpu(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on CPU"));
  HELIX_CHECK(tensor.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must have dtype float32"));
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be contiguous"));
}

void require_shape(const Tensor& tensor, const Shape& expected, std::string_view name,
                   std::string_view operation) {
  if (tensor.shape() != expected) {
    std::ostringstream message;
    message << operation << ": " << name << " shape " << shape_string(tensor.shape())
            << " does not match expected shape " << shape_string(expected);
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
}

} // namespace

struct ContiguousKvCache::Impl {
  struct SequenceState {
    SequenceState(const ContiguousKvCacheConfig& config, std::size_t token_capacity)
        : capacity(token_capacity),
          keys(Tensor::empty({checked_dimension(config.layers, "KV layer count"),
                              checked_dimension(config.kv_heads, "KV head count"),
                              checked_dimension(token_capacity, "KV sequence capacity"),
                              checked_dimension(config.head_dimension, "KV head dimension")})),
          values(Tensor::empty({checked_dimension(config.layers, "KV layer count"),
                                checked_dimension(config.kv_heads, "KV head count"),
                                checked_dimension(token_capacity, "KV sequence capacity"),
                                checked_dimension(config.head_dimension, "KV head dimension")})) {}

    std::size_t length = 0;
    std::size_t capacity;
    Tensor keys;
    Tensor values;
  };

  explicit Impl(ContiguousKvCacheConfig cache_config)
      : config(cache_config), cache_id(next_cache_id()) {
    HELIX_CHECK(config.layers != 0, ErrorCode::kInvalidArgument,
                "contiguous KV cache layer count must be positive");
    HELIX_CHECK(config.kv_heads != 0, ErrorCode::kInvalidArgument,
                "contiguous KV cache head count must be positive");
    HELIX_CHECK(config.head_dimension != 0, ErrorCode::kInvalidArgument,
                "contiguous KV cache head dimension must be positive");
    HELIX_CHECK(config.maximum_bytes != 0, ErrorCode::kInvalidArgument,
                "contiguous KV cache byte limit must be positive");
    static_cast<void>(checked_dimension(config.layers, "KV layer count"));
    static_cast<void>(checked_dimension(config.kv_heads, "KV head count"));
    static_cast<void>(checked_dimension(config.head_dimension, "KV head dimension"));

    std::size_t elements = checked_multiply(config.layers, 2, "KV bytes-per-token elements");
    elements = checked_multiply(elements, config.kv_heads, "KV bytes-per-token elements");
    elements = checked_multiply(elements, config.head_dimension, "KV bytes-per-token elements");
    per_token_bytes = checked_multiply(elements, sizeof(float), "KV bytes per token");
  }

  [[nodiscard]] SequenceState& require_sequence(const SequenceHandle& handle,
                                                std::string_view operation) {
    HELIX_CHECK(handle && handle.cache_id() == cache_id, ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence handle belongs to another cache or is empty");
    const auto iterator = sequences.find(handle.id());
    HELIX_CHECK(iterator != sequences.end(), ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence is unknown or was already released");
    return iterator->second;
  }

  [[nodiscard]] const SequenceState& require_sequence(const SequenceHandle& handle,
                                                      std::string_view operation) const {
    HELIX_CHECK(handle && handle.cache_id() == cache_id, ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence handle belongs to another cache or is empty");
    const auto iterator = sequences.find(handle.id());
    HELIX_CHECK(iterator != sequences.end(), ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence is unknown or was already released");
    return iterator->second;
  }

  [[nodiscard]] std::size_t bytes_for_capacity(std::size_t capacity) const {
    HELIX_CHECK(capacity != 0, ErrorCode::kInvalidArgument,
                "contiguous KV sequence capacity must be positive");
    static_cast<void>(checked_dimension(capacity, "KV sequence capacity"));
    return checked_multiply(capacity, per_token_bytes, "contiguous KV sequence bytes");
  }

  void copy_layer(const SequenceState& sequence, std::size_t layer, Tensor& keys_output,
                  Tensor& values_output, std::string_view operation) const {
    HELIX_CHECK(layer < config.layers, ErrorCode::kOutOfRange,
                std::string(operation) + ": layer index is out of range");
    require_cpu_float_tensor(keys_output, "keys output", operation);
    require_cpu_float_tensor(values_output, "values output", operation);
    const Shape output_shape{checked_dimension(config.kv_heads, "KV head count"),
                             checked_dimension(sequence.length, "KV sequence length"),
                             checked_dimension(config.head_dimension, "KV head dimension")};
    require_shape(keys_output, output_shape, "keys output", operation);
    require_shape(values_output, output_shape, "values output", operation);
    HELIX_CHECK(keys_output.nbytes() == 0 || keys_output.data() != values_output.data(),
                ErrorCode::kInvalidArgument,
                std::string(operation) + ": key and value outputs must not alias");
    if (sequence.length == 0) {
      return;
    }

    const auto* keys_source = sequence.keys.data_as<float>();
    const auto* values_source = sequence.values.data_as<float>();
    auto* keys_destination = keys_output.data_as<float>();
    auto* values_destination = values_output.data_as<float>();
    const std::size_t head_copy_elements =
        checked_multiply(sequence.length, config.head_dimension, "KV layer copy elements");
    const std::size_t head_copy_bytes =
        checked_multiply(head_copy_elements, sizeof(float), "KV layer copy bytes");
    for (std::size_t head = 0; head < config.kv_heads; ++head) {
      const std::size_t source_offset =
          ((layer * config.kv_heads + head) * sequence.capacity) * config.head_dimension;
      const std::size_t destination_offset = head * head_copy_elements;
      std::memcpy(keys_destination + destination_offset, keys_source + source_offset,
                  head_copy_bytes);
      std::memcpy(values_destination + destination_offset, values_source + source_offset,
                  head_copy_bytes);
    }
  }

  ContiguousKvCacheConfig config;
  const std::uint64_t cache_id;
  std::size_t per_token_bytes = 0;
  mutable std::mutex mutex;
  std::unordered_map<std::uint64_t, SequenceState> sequences;
  std::uint64_t next_sequence_id = 1;
  ContiguousKvCacheStats stats;
};

ContiguousKvCache::ContiguousKvCache(ContiguousKvCacheConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

ContiguousKvCache::~ContiguousKvCache() = default;

SequenceHandle ContiguousKvCache::create_sequence(std::size_t capacity_tokens) {
  std::scoped_lock lock(impl_->mutex);
  const std::size_t bytes = impl_->bytes_for_capacity(capacity_tokens);
  HELIX_CHECK(bytes <= impl_->config.maximum_bytes -
                           std::min(impl_->stats.bytes_allocated, impl_->config.maximum_bytes),
              ErrorCode::kResourceExhausted, "contiguous KV cache byte limit would be exceeded");
  HELIX_CHECK(impl_->next_sequence_id != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted, "contiguous KV sequence id space exhausted");

  const std::uint64_t sequence_id = impl_->next_sequence_id;
  Impl::SequenceState state(impl_->config, capacity_tokens);
  const auto [iterator, inserted] = impl_->sequences.emplace(sequence_id, std::move(state));
  static_cast<void>(iterator);
  HELIX_ASSERT(inserted, "contiguous KV sequence id unexpectedly collided");

  ++impl_->next_sequence_id;
  impl_->stats.bytes_allocated += bytes;
  impl_->stats.peak_bytes_allocated =
      std::max(impl_->stats.peak_bytes_allocated, impl_->stats.bytes_allocated);
  ++impl_->stats.active_sequence_count;
  impl_->stats.active_capacity_tokens += capacity_tokens;
  ++impl_->stats.sequence_create_count;
  return SequenceHandle(sequence_id, impl_->cache_id);
}

void ContiguousKvCache::append(const SequenceHandle& sequence, const Tensor& token_keys,
                               const Tensor& token_values) {
  constexpr std::string_view kOperation = "ContiguousKvCache::append";
  std::scoped_lock lock(impl_->mutex);
  Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  require_cpu_float_tensor(token_keys, "token keys", kOperation);
  require_cpu_float_tensor(token_values, "token values", kOperation);
  const Shape token_shape{checked_dimension(impl_->config.layers, "KV layer count"),
                          checked_dimension(impl_->config.kv_heads, "KV head count"),
                          checked_dimension(impl_->config.head_dimension, "KV head dimension")};
  require_shape(token_keys, token_shape, "token keys", kOperation);
  require_shape(token_values, token_shape, "token values", kOperation);
  HELIX_CHECK(state.length < state.capacity, ErrorCode::kResourceExhausted,
              "ContiguousKvCache::append sequence capacity is exhausted");

  const auto* keys_source = token_keys.data_as<float>();
  const auto* values_source = token_values.data_as<float>();
  auto* keys_destination = state.keys.data_as<float>();
  auto* values_destination = state.values.data_as<float>();
  const std::size_t vector_bytes =
      checked_multiply(impl_->config.head_dimension, sizeof(float), "KV append vector bytes");
  for (std::size_t layer = 0; layer < impl_->config.layers; ++layer) {
    for (std::size_t head = 0; head < impl_->config.kv_heads; ++head) {
      const std::size_t source_offset =
          (layer * impl_->config.kv_heads + head) * impl_->config.head_dimension;
      const std::size_t destination_offset =
          ((layer * impl_->config.kv_heads + head) * state.capacity + state.length) *
          impl_->config.head_dimension;
      std::memcpy(keys_destination + destination_offset, keys_source + source_offset, vector_bytes);
      std::memcpy(values_destination + destination_offset, values_source + source_offset,
                  vector_bytes);
    }
  }
  ++state.length;
  ++impl_->stats.active_tokens;
  ++impl_->stats.append_count;
}

KvLayer ContiguousKvCache::retrieve_layer(const SequenceHandle& sequence, std::size_t layer) const {
  constexpr std::string_view kOperation = "ContiguousKvCache::retrieve_layer";
  std::scoped_lock lock(impl_->mutex);
  const Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  const Shape output_shape{checked_dimension(impl_->config.kv_heads, "KV head count"),
                           checked_dimension(state.length, "KV sequence length"),
                           checked_dimension(impl_->config.head_dimension, "KV head dimension")};
  KvLayer result{Tensor::empty(output_shape), Tensor::empty(output_shape)};
  impl_->copy_layer(state, layer, result.keys, result.values, kOperation);
  return result;
}

void ContiguousKvCache::copy_layer_out(const SequenceHandle& sequence, std::size_t layer,
                                       Tensor& keys, Tensor& values) const {
  constexpr std::string_view kOperation = "ContiguousKvCache::copy_layer_out";
  std::scoped_lock lock(impl_->mutex);
  const Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  impl_->copy_layer(state, layer, keys, values, kOperation);
}

void ContiguousKvCache::release_sequence(const SequenceHandle& sequence) {
  constexpr std::string_view kOperation = "ContiguousKvCache::release_sequence";
  std::scoped_lock lock(impl_->mutex);
  const Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  const std::size_t capacity_tokens = state.capacity;
  const std::size_t active_tokens = state.length;
  const std::size_t bytes = impl_->bytes_for_capacity(capacity_tokens);
  impl_->sequences.erase(sequence.id());
  impl_->stats.bytes_allocated -= bytes;
  --impl_->stats.active_sequence_count;
  impl_->stats.active_capacity_tokens -= capacity_tokens;
  impl_->stats.active_tokens -= active_tokens;
  ++impl_->stats.release_count;
}

bool ContiguousKvCache::contains(const SequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  return sequence && sequence.cache_id() == impl_->cache_id &&
         impl_->sequences.contains(sequence.id());
}

std::size_t ContiguousKvCache::length(const SequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->require_sequence(sequence, "ContiguousKvCache::length").length;
}

std::size_t ContiguousKvCache::capacity(const SequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->require_sequence(sequence, "ContiguousKvCache::capacity").capacity;
}

std::size_t ContiguousKvCache::bytes_per_token() const noexcept {
  return impl_->per_token_bytes;
}

std::size_t ContiguousKvCache::sequence_bytes(std::size_t capacity_tokens) const {
  return impl_->bytes_for_capacity(capacity_tokens);
}

ContiguousKvCacheStats ContiguousKvCache::stats() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->stats;
}

const ContiguousKvCacheConfig& ContiguousKvCache::config() const noexcept {
  return impl_->config;
}

} // namespace helix::kv_cache
