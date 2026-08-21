#include "helix/kv_cache/paged.hpp"

#include "helix/core/error.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace helix::kv_cache {
namespace {

struct PagedLayout {
  PagedKvCacheConfig config;
  Shape pool_shape;
  std::size_t bytes_per_token = 0;
  std::size_t block_bytes = 0;
  std::size_t pool_bytes = 0;
  std::size_t block_elements = 0;
  std::size_t layer_elements = 0;
  std::size_t head_elements = 0;
};

std::uint64_t next_paged_cache_id() {
  static std::atomic<std::uint64_t> next{1};
  std::uint64_t current = next.load(std::memory_order_relaxed);
  while (true) {
    HELIX_CHECK(current != std::numeric_limits<std::uint64_t>::max(), ErrorCode::kResourceExhausted,
                "paged KV cache identity space exhausted");
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

PagedLayout make_layout(PagedKvCacheConfig config) {
  HELIX_CHECK(config.layers != 0, ErrorCode::kInvalidArgument,
              "paged KV cache layer count must be positive");
  HELIX_CHECK(config.kv_heads != 0, ErrorCode::kInvalidArgument,
              "paged KV cache head count must be positive");
  HELIX_CHECK(config.head_dimension != 0, ErrorCode::kInvalidArgument,
              "paged KV cache head dimension must be positive");
  HELIX_CHECK(config.block_tokens != 0, ErrorCode::kInvalidArgument,
              "paged KV cache block token count must be positive");
  HELIX_CHECK(config.physical_blocks != 0, ErrorCode::kInvalidArgument,
              "paged KV cache physical block count must be positive");
  HELIX_CHECK(config.physical_blocks <= std::numeric_limits<std::uint32_t>::max(),
              ErrorCode::kOutOfRange,
              "paged KV cache physical block count exceeds BlockId capacity");

  static_cast<void>(checked_dimension(config.layers, "KV layer count"));
  static_cast<void>(checked_dimension(config.kv_heads, "KV head count"));
  static_cast<void>(checked_dimension(config.head_dimension, "KV head dimension"));
  static_cast<void>(checked_dimension(config.block_tokens, "KV block token count"));
  static_cast<void>(checked_dimension(config.physical_blocks, "physical KV block count"));

  PagedLayout layout;
  layout.config = config;
  layout.head_elements =
      checked_multiply(config.block_tokens, config.head_dimension, "paged KV head-block elements");
  layout.layer_elements =
      checked_multiply(config.kv_heads, layout.head_elements, "paged KV layer-block elements");
  layout.block_elements =
      checked_multiply(config.layers, layout.layer_elements, "paged KV physical-block elements");
  const std::size_t one_pool_block_bytes =
      checked_multiply(layout.block_elements, sizeof(float), "paged KV key block bytes");
  layout.block_bytes = checked_multiply(one_pool_block_bytes, 2, "paged KV key/value block bytes");
  layout.pool_bytes =
      checked_multiply(config.physical_blocks, layout.block_bytes, "paged KV pool bytes");

  std::size_t token_elements =
      checked_multiply(config.layers, 2, "paged KV bytes-per-token elements");
  token_elements =
      checked_multiply(token_elements, config.kv_heads, "paged KV bytes-per-token elements");
  token_elements =
      checked_multiply(token_elements, config.head_dimension, "paged KV bytes-per-token elements");
  layout.bytes_per_token =
      checked_multiply(token_elements, sizeof(float), "paged KV bytes per token");
  HELIX_ASSERT(layout.block_bytes == checked_multiply(layout.bytes_per_token, config.block_tokens,
                                                      "paged KV block-byte consistency"),
               "paged KV block byte formulas disagree");

  layout.pool_shape = {checked_dimension(config.physical_blocks, "physical KV block count"),
                       checked_dimension(config.layers, "KV layer count"),
                       checked_dimension(config.kv_heads, "KV head count"),
                       checked_dimension(config.block_tokens, "KV block token count"),
                       checked_dimension(config.head_dimension, "KV head dimension")};
  return layout;
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

#ifndef NDEBUG
std::size_t blocks_for_tokens(std::size_t tokens, std::size_t block_tokens) noexcept {
  return tokens / block_tokens + static_cast<std::size_t>(tokens % block_tokens != 0);
}
#endif

class FreeBlockQueue final {
public:
  explicit FreeBlockQueue(std::size_t block_count) : slots_(block_count), count_(block_count) {
    for (std::size_t index = 0; index < block_count; ++index) {
      slots_[index] = PhysicalBlockId(static_cast<std::uint32_t>(index));
    }
  }

  [[nodiscard]] bool empty() const noexcept {
    return count_ == 0;
  }

  [[nodiscard]] std::size_t size() const noexcept {
    return count_;
  }

  [[nodiscard]] PhysicalBlockId front() const {
    HELIX_ASSERT(!empty(), "front requested from empty KV free-block queue");
    return slots_[head_];
  }

  void pop() {
    HELIX_ASSERT(!empty(), "pop requested from empty KV free-block queue");
    head_ = increment(head_);
    --count_;
  }

  void push(PhysicalBlockId block) {
    HELIX_ASSERT(count_ < slots_.size(), "KV free-block queue overflow");
    slots_[tail_] = block;
    tail_ = increment(tail_);
    ++count_;
  }

  [[nodiscard]] PhysicalBlockId at(std::size_t logical_index) const {
    HELIX_ASSERT(logical_index < count_, "KV free-block queue index is out of range");
    return slots_[(head_ + logical_index) % slots_.size()];
  }

private:
  [[nodiscard]] std::size_t increment(std::size_t index) const noexcept {
    ++index;
    return index == slots_.size() ? 0 : index;
  }

  std::vector<PhysicalBlockId> slots_;
  std::size_t head_ = 0;
  std::size_t tail_ = 0;
  std::size_t count_ = 0;
};

} // namespace

struct PagedKvCache::Impl {
  struct SequenceState {
    std::size_t length = 0;
    std::vector<PhysicalBlockId> blocks;
  };

  explicit Impl(PagedLayout layout)
      : config(layout.config), cache_id(next_paged_cache_id()),
        per_token_bytes(layout.bytes_per_token), per_block_bytes(layout.block_bytes),
        block_elements(layout.block_elements), layer_elements(layout.layer_elements),
        head_elements(layout.head_elements), keys(Tensor::empty(layout.pool_shape)),
        values(Tensor::empty(layout.pool_shape)), free_blocks(config.physical_blocks),
        block_owners(config.physical_blocks, 0), block_was_allocated(config.physical_blocks, 0) {
    HELIX_ASSERT(keys.nbytes() + values.nbytes() == layout.pool_bytes,
                 "paged KV pool tensors do not match computed reserved bytes");
    stats.bytes_reserved = layout.pool_bytes;
    stats.bytes_free = layout.pool_bytes;
    stats.total_blocks = config.physical_blocks;
    stats.free_blocks = config.physical_blocks;
    validate_invariants();
  }

  [[nodiscard]] SequenceState& require_sequence(const PagedSequenceHandle& handle,
                                                std::string_view operation) {
    HELIX_CHECK(handle && handle.cache_id() == cache_id, ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence handle belongs to another cache or is empty");
    const auto iterator = sequences.find(handle.id());
    HELIX_CHECK(iterator != sequences.end(), ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence is unknown or was already released");
    return iterator->second;
  }

  [[nodiscard]] const SequenceState& require_sequence(const PagedSequenceHandle& handle,
                                                      std::string_view operation) const {
    HELIX_CHECK(handle && handle.cache_id() == cache_id, ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence handle belongs to another cache or is empty");
    const auto iterator = sequences.find(handle.id());
    HELIX_CHECK(iterator != sequences.end(), ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence is unknown or was already released");
    return iterator->second;
  }

  [[nodiscard]] std::size_t pool_offset(PhysicalBlockId block, std::size_t layer, std::size_t head,
                                        std::size_t token_offset) const noexcept {
    return static_cast<std::size_t>(block.value()) * block_elements + layer * layer_elements +
           head * head_elements + token_offset * config.head_dimension;
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

    const auto* keys_source = keys.data_as<float>();
    const auto* values_source = values.data_as<float>();
    auto* keys_destination = keys_output.data_as<float>();
    auto* values_destination = values_output.data_as<float>();
    for (std::size_t head = 0; head < config.kv_heads; ++head) {
      for (std::size_t logical_block = 0; logical_block < sequence.blocks.size(); ++logical_block) {
        const std::size_t logical_start = logical_block * config.block_tokens;
        const std::size_t tokens_in_block =
            std::min(config.block_tokens, sequence.length - logical_start);
        const std::size_t copy_elements = tokens_in_block * config.head_dimension;
        const std::size_t copy_bytes = copy_elements * sizeof(float);
        const std::size_t source_offset =
            pool_offset(sequence.blocks[logical_block], layer, head, 0);
        const std::size_t destination_offset =
            (head * sequence.length + logical_start) * config.head_dimension;
        std::memcpy(keys_destination + destination_offset, keys_source + source_offset, copy_bytes);
        std::memcpy(values_destination + destination_offset, values_source + source_offset,
                    copy_bytes);
      }
    }
  }

  void validate_invariants() const {
#ifndef NDEBUG
    std::size_t mapped_blocks = 0;
    std::size_t active_tokens = 0;
    for (const auto& [sequence_id, sequence] : sequences) {
      HELIX_ASSERT(sequence_id != 0, "paged KV sequence uses reserved zero identity");
      HELIX_ASSERT(sequence.blocks.size() ==
                       blocks_for_tokens(sequence.length, config.block_tokens),
                   "paged KV block-table length disagrees with token length");
      active_tokens += sequence.length;
      mapped_blocks += sequence.blocks.size();
      for (const PhysicalBlockId block : sequence.blocks) {
        HELIX_ASSERT(block.valid() && block.value() < config.physical_blocks,
                     "paged KV block table contains an invalid physical block");
        HELIX_ASSERT(block_owners[block.value()] == sequence_id,
                     "paged KV block owner disagrees with sequence table");
      }
    }

    std::size_t owned_blocks = 0;
    std::size_t free_members = 0;
    for (std::size_t block = 0; block < config.physical_blocks; ++block) {
      if (block_owners[block] == 0) {
        ++free_members;
        HELIX_ASSERT(block_is_free[block] != 0,
                     "free paged KV block is absent from free membership");
      } else {
        ++owned_blocks;
        HELIX_ASSERT(block_is_free[block] == 0, "owned paged KV block is marked as free");
        HELIX_ASSERT(block_was_allocated[block] != 0,
                     "owned paged KV block lacks allocation history");
        HELIX_ASSERT(sequences.contains(block_owners[block]),
                     "paged KV block is owned by an unknown sequence");
      }
    }
    for (std::size_t offset = 0; offset < free_blocks.size(); ++offset) {
      const PhysicalBlockId block = free_blocks.at(offset);
      HELIX_ASSERT(block.valid() && block.value() < config.physical_blocks,
                   "paged KV free queue contains an invalid block");
      HELIX_ASSERT(block_is_free[block.value()] != 0 && block_owners[block.value()] == 0,
                   "paged KV free queue contains an owned block");
    }

    HELIX_ASSERT(mapped_blocks == owned_blocks, "paged KV mapped and owned block counts disagree");
    HELIX_ASSERT(free_members == free_blocks.size(),
                 "paged KV free membership and queue counts disagree");
    HELIX_ASSERT(owned_blocks + free_blocks.size() == config.physical_blocks,
                 "paged KV block accounting does not cover the pool");
    HELIX_ASSERT(stats.total_blocks == config.physical_blocks,
                 "paged KV total-block statistic is inconsistent");
    HELIX_ASSERT(stats.used_blocks == owned_blocks && stats.free_blocks == free_blocks.size(),
                 "paged KV used/free block statistics are inconsistent");
    HELIX_ASSERT(stats.active_sequence_count == sequences.size(),
                 "paged KV sequence statistic is inconsistent");
    HELIX_ASSERT(stats.active_tokens == active_tokens,
                 "paged KV active-token statistic is inconsistent");
    HELIX_ASSERT(stats.allocated_token_slots == owned_blocks * config.block_tokens,
                 "paged KV token-slot statistic is inconsistent");
    HELIX_ASSERT(stats.bytes_used == owned_blocks * per_block_bytes,
                 "paged KV used-byte statistic is inconsistent");
    HELIX_ASSERT(stats.bytes_free == free_blocks.size() * per_block_bytes,
                 "paged KV free-byte statistic is inconsistent");
    HELIX_ASSERT(stats.bytes_used + stats.bytes_free == stats.bytes_reserved,
                 "paged KV byte statistics do not cover the pool");
#endif
  }

  PagedKvCacheConfig config;
  const std::uint64_t cache_id;
  std::size_t per_token_bytes = 0;
  std::size_t per_block_bytes = 0;
  std::size_t block_elements = 0;
  std::size_t layer_elements = 0;
  std::size_t head_elements = 0;
  Tensor keys;
  Tensor values;
  mutable std::mutex mutex;
  FreeBlockQueue free_blocks;
  std::vector<std::uint64_t> block_owners;
  std::vector<std::uint8_t> block_was_allocated;
  std::vector<std::uint8_t> block_is_free = std::vector<std::uint8_t>(config.physical_blocks, 1);
  std::unordered_map<std::uint64_t, SequenceState> sequences;
  std::uint64_t next_sequence_id = 1;
  PagedKvCacheStats stats;
};

PagedKvCache::PagedKvCache(PagedKvCacheConfig config) try
    : impl_(std::make_unique<Impl>(make_layout(config))) {
} catch (const std::bad_alloc&) {
  throw_error(ErrorCode::kResourceExhausted,
              "PagedKvCache construction could not allocate pool metadata");
}

PagedKvCache::~PagedKvCache() = default;

PagedSequenceHandle PagedKvCache::create_sequence() {
  std::scoped_lock lock(impl_->mutex);
  HELIX_CHECK(impl_->next_sequence_id != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted, "paged KV sequence id space exhausted");
  const std::uint64_t sequence_id = impl_->next_sequence_id;
  try {
    const auto [iterator, inserted] = impl_->sequences.try_emplace(sequence_id);
    static_cast<void>(iterator);
    HELIX_ASSERT(inserted, "paged KV sequence id unexpectedly collided");
  } catch (const std::bad_alloc&) {
    throw_error(ErrorCode::kResourceExhausted,
                "PagedKvCache::create_sequence could not allocate sequence metadata");
  }
  ++impl_->next_sequence_id;
  ++impl_->stats.active_sequence_count;
  ++impl_->stats.sequence_create_count;
  impl_->validate_invariants();
  return PagedSequenceHandle(sequence_id, impl_->cache_id);
}

void PagedKvCache::append(const PagedSequenceHandle& sequence, const Tensor& token_keys,
                          const Tensor& token_values) {
  constexpr std::string_view kOperation = "PagedKvCache::append";
  std::scoped_lock lock(impl_->mutex);
  Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  require_cpu_float_tensor(token_keys, "token keys", kOperation);
  require_cpu_float_tensor(token_values, "token values", kOperation);
  const Shape token_shape{checked_dimension(impl_->config.layers, "KV layer count"),
                          checked_dimension(impl_->config.kv_heads, "KV head count"),
                          checked_dimension(impl_->config.head_dimension, "KV head dimension")};
  require_shape(token_keys, token_shape, "token keys", kOperation);
  require_shape(token_values, token_shape, "token values", kOperation);

  const bool needs_block = state.length % impl_->config.block_tokens == 0;
  PhysicalBlockId block;
  if (needs_block) {
    HELIX_CHECK(!impl_->free_blocks.empty(), ErrorCode::kResourceExhausted,
                "PagedKvCache::append has no free physical KV block");
    block = impl_->free_blocks.front();
    try {
      state.blocks.push_back(block);
    } catch (const std::bad_alloc&) {
      throw_error(ErrorCode::kResourceExhausted,
                  "PagedKvCache::append could not grow the sequence block table");
    }
    impl_->free_blocks.pop();
    HELIX_ASSERT(impl_->block_is_free[block.value()] != 0 &&
                     impl_->block_owners[block.value()] == 0,
                 "paged KV free block has conflicting ownership");
    impl_->block_is_free[block.value()] = 0;
    impl_->block_owners[block.value()] = sequence.id();
    if (impl_->block_was_allocated[block.value()] != 0) {
      ++impl_->stats.block_reuse_count;
    }
    impl_->block_was_allocated[block.value()] = 1;
    ++impl_->stats.used_blocks;
    --impl_->stats.free_blocks;
    impl_->stats.bytes_used += impl_->per_block_bytes;
    impl_->stats.bytes_free -= impl_->per_block_bytes;
    impl_->stats.peak_bytes_used = std::max(impl_->stats.peak_bytes_used, impl_->stats.bytes_used);
    impl_->stats.allocated_token_slots += impl_->config.block_tokens;
    ++impl_->stats.block_allocation_count;
  } else {
    HELIX_ASSERT(!state.blocks.empty(), "nonempty paged KV sequence has no physical block");
    block = state.blocks.back();
  }

  const std::size_t token_offset = state.length % impl_->config.block_tokens;
  const auto* keys_source = token_keys.data_as<float>();
  const auto* values_source = token_values.data_as<float>();
  auto* keys_destination = impl_->keys.data_as<float>();
  auto* values_destination = impl_->values.data_as<float>();
  const std::size_t vector_bytes = impl_->config.head_dimension * sizeof(float);
  for (std::size_t layer = 0; layer < impl_->config.layers; ++layer) {
    for (std::size_t head = 0; head < impl_->config.kv_heads; ++head) {
      const std::size_t source_offset =
          (layer * impl_->config.kv_heads + head) * impl_->config.head_dimension;
      const std::size_t destination_offset = impl_->pool_offset(block, layer, head, token_offset);
      std::memcpy(keys_destination + destination_offset, keys_source + source_offset, vector_bytes);
      std::memcpy(values_destination + destination_offset, values_source + source_offset,
                  vector_bytes);
    }
  }

  ++state.length;
  ++impl_->stats.active_tokens;
  ++impl_->stats.append_count;
  impl_->validate_invariants();
}

KvLayer PagedKvCache::retrieve_layer(const PagedSequenceHandle& sequence, std::size_t layer) const {
  constexpr std::string_view kOperation = "PagedKvCache::retrieve_layer";
  std::scoped_lock lock(impl_->mutex);
  const Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  HELIX_CHECK(layer < impl_->config.layers, ErrorCode::kOutOfRange,
              "PagedKvCache::retrieve_layer layer index is out of range");
  const Shape output_shape{checked_dimension(impl_->config.kv_heads, "KV head count"),
                           checked_dimension(state.length, "KV sequence length"),
                           checked_dimension(impl_->config.head_dimension, "KV head dimension")};
  KvLayer result{Tensor::empty(output_shape), Tensor::empty(output_shape)};
  impl_->copy_layer(state, layer, result.keys, result.values, kOperation);
  return result;
}

void PagedKvCache::copy_layer_out(const PagedSequenceHandle& sequence, std::size_t layer,
                                  Tensor& keys, Tensor& values) const {
  constexpr std::string_view kOperation = "PagedKvCache::copy_layer_out";
  std::scoped_lock lock(impl_->mutex);
  const Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  impl_->copy_layer(state, layer, keys, values, kOperation);
}

void PagedKvCache::release_sequence(const PagedSequenceHandle& sequence) {
  constexpr std::string_view kOperation = "PagedKvCache::release_sequence";
  std::scoped_lock lock(impl_->mutex);
  const Impl::SequenceState& state = impl_->require_sequence(sequence, kOperation);
  HELIX_ASSERT(state.blocks.size() <= impl_->config.physical_blocks - impl_->free_blocks.size(),
               "paged KV release would overflow the free-block queue");
  for (const PhysicalBlockId block : state.blocks) {
    HELIX_ASSERT(impl_->block_owners[block.value()] == sequence.id() &&
                     impl_->block_is_free[block.value()] == 0,
                 "paged KV release found inconsistent block ownership");
  }

  const std::size_t released_blocks = state.blocks.size();
  const std::size_t released_tokens = state.length;
  for (const PhysicalBlockId block : state.blocks) {
    impl_->block_owners[block.value()] = 0;
    impl_->block_is_free[block.value()] = 1;
    impl_->free_blocks.push(block);
  }
  impl_->sequences.erase(sequence.id());
  impl_->stats.used_blocks -= released_blocks;
  impl_->stats.free_blocks += released_blocks;
  const std::size_t released_bytes = released_blocks * impl_->per_block_bytes;
  impl_->stats.bytes_used -= released_bytes;
  impl_->stats.bytes_free += released_bytes;
  --impl_->stats.active_sequence_count;
  impl_->stats.active_tokens -= released_tokens;
  impl_->stats.allocated_token_slots -= released_blocks * impl_->config.block_tokens;
  ++impl_->stats.release_count;
  impl_->validate_invariants();
}

bool PagedKvCache::contains(const PagedSequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  return sequence && sequence.cache_id() == impl_->cache_id &&
         impl_->sequences.contains(sequence.id());
}

std::size_t PagedKvCache::length(const PagedSequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->require_sequence(sequence, "PagedKvCache::length").length;
}

std::vector<PhysicalBlockId> PagedKvCache::block_table(const PagedSequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  const auto& blocks = impl_->require_sequence(sequence, "PagedKvCache::block_table").blocks;
  try {
    return blocks;
  } catch (const std::bad_alloc&) {
    throw_error(ErrorCode::kResourceExhausted,
                "PagedKvCache::block_table could not allocate the table snapshot");
  }
}

std::size_t PagedKvCache::bytes_per_token() const noexcept {
  return impl_->per_token_bytes;
}

std::size_t PagedKvCache::block_bytes() const noexcept {
  return impl_->per_block_bytes;
}

std::size_t PagedKvCache::token_capacity() const noexcept {
  return impl_->config.physical_blocks * impl_->config.block_tokens;
}

PagedKvCacheStats PagedKvCache::stats() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->stats;
}

const PagedKvCacheConfig& PagedKvCache::config() const noexcept {
  return impl_->config;
}

} // namespace helix::kv_cache
