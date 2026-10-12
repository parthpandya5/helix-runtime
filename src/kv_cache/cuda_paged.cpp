#include "helix/kv_cache/cuda_paged.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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

struct CudaPagedLayout final {
  PagedKvCacheConfig config;
  Shape layer_pool_shape;
  std::size_t bytes_per_token = 0;
  std::size_t block_bytes = 0;
  std::size_t pool_bytes = 0;
  std::size_t token_capacity = 0;
  std::size_t vector_bytes = 0;
  std::size_t layer_block_elements = 0;
  std::size_t head_block_elements = 0;
};

std::uint64_t next_cuda_paged_cache_id() {
  static std::atomic<std::uint64_t> next{1};
  std::uint64_t current = next.load(std::memory_order_relaxed);
  while (true) {
    HELIX_CHECK(current != std::numeric_limits<std::uint64_t>::max(), ErrorCode::kResourceExhausted,
                "CUDA paged KV cache identity space exhausted");
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

CudaPagedLayout make_cuda_paged_layout(PagedKvCacheConfig config, int device_index) {
  HELIX_CHECK(config.layers != 0, ErrorCode::kInvalidArgument,
              "CUDA paged KV cache layer count must be positive");
  HELIX_CHECK(config.kv_heads != 0, ErrorCode::kInvalidArgument,
              "CUDA paged KV cache head count must be positive");
  HELIX_CHECK(config.head_dimension != 0, ErrorCode::kInvalidArgument,
              "CUDA paged KV cache head dimension must be positive");
  HELIX_CHECK(config.block_tokens != 0, ErrorCode::kInvalidArgument,
              "CUDA paged KV cache block token count must be positive");
  HELIX_CHECK(config.physical_blocks != 0, ErrorCode::kInvalidArgument,
              "CUDA paged KV cache physical block count must be positive");
  HELIX_CHECK(config.physical_blocks <= std::numeric_limits<std::uint32_t>::max(),
              ErrorCode::kOutOfRange,
              "CUDA paged KV cache physical block count exceeds uint32 block-ID capacity");
  HELIX_CHECK(device_index >= 0, ErrorCode::kInvalidArgument,
              "CUDA paged KV cache device index must be non-negative");

  static_cast<void>(checked_dimension(config.layers, "CUDA KV layer count"));
  static_cast<void>(checked_dimension(config.kv_heads, "CUDA KV head count"));
  static_cast<void>(checked_dimension(config.head_dimension, "CUDA KV head dimension"));
  static_cast<void>(checked_dimension(config.block_tokens, "CUDA KV block token count"));
  static_cast<void>(checked_dimension(config.physical_blocks, "CUDA physical KV block count"));

  CudaPagedLayout layout;
  layout.config = config;
  layout.head_block_elements = checked_multiply(config.block_tokens, config.head_dimension,
                                                "CUDA paged KV head-block elements");
  layout.layer_block_elements = checked_multiply(config.kv_heads, layout.head_block_elements,
                                                 "CUDA paged KV layer-block elements");
  const std::size_t layer_pool_elements = checked_multiply(
      config.physical_blocks, layout.layer_block_elements, "CUDA paged KV layer-pool elements");
  const std::size_t layer_pool_bytes =
      checked_multiply(layer_pool_elements, sizeof(float), "CUDA paged KV layer-pool bytes");
  const std::size_t both_pools_per_layer =
      checked_multiply(layer_pool_bytes, 2, "CUDA paged KV layer key/value bytes");
  layout.pool_bytes =
      checked_multiply(config.layers, both_pools_per_layer, "CUDA paged KV pool bytes");

  std::size_t token_elements =
      checked_multiply(config.layers, 2, "CUDA paged KV bytes-per-token elements");
  token_elements =
      checked_multiply(token_elements, config.kv_heads, "CUDA paged KV bytes-per-token elements");
  token_elements = checked_multiply(token_elements, config.head_dimension,
                                    "CUDA paged KV bytes-per-token elements");
  layout.bytes_per_token =
      checked_multiply(token_elements, sizeof(float), "CUDA paged KV bytes per token");
  layout.block_bytes = checked_multiply(layout.bytes_per_token, config.block_tokens,
                                        "CUDA paged KV bytes per physical block");
  layout.token_capacity =
      checked_multiply(config.physical_blocks, config.block_tokens, "CUDA paged KV token capacity");
  layout.vector_bytes =
      checked_multiply(config.head_dimension, sizeof(float), "CUDA paged KV head-vector bytes");
  HELIX_ASSERT(layout.pool_bytes == checked_multiply(config.physical_blocks, layout.block_bytes,
                                                     "CUDA paged KV pool-byte consistency"),
               "CUDA paged KV pool byte formulas disagree");

  layout.layer_pool_shape = {
      checked_dimension(config.physical_blocks, "CUDA physical KV block count"),
      checked_dimension(config.kv_heads, "CUDA KV head count"),
      checked_dimension(config.block_tokens, "CUDA KV block token count"),
      checked_dimension(config.head_dimension, "CUDA KV head dimension")};
  return layout;
}

#ifndef NDEBUG
std::size_t blocks_for_tokens(std::size_t tokens, std::size_t block_tokens) noexcept {
  return tokens / block_tokens + static_cast<std::size_t>(tokens % block_tokens != 0);
}
#endif

std::string tensor_error(std::string_view operation, std::string_view name,
                         std::string_view requirement) {
  std::ostringstream message;
  message << operation << ": " << name << ' ' << requirement;
  return message.str();
}

void require_input_tensor(const Tensor& tensor, const Shape& expected_shape, Device device,
                          std::string_view name, std::string_view operation) {
  HELIX_CHECK(tensor.device() == device, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be on the cache device"));
  HELIX_CHECK(tensor.dtype() == DType::kFloat32, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must have dtype float32"));
  HELIX_CHECK(tensor.is_contiguous(), ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "must be contiguous"));
  HELIX_CHECK(tensor.shape() == expected_shape, ErrorCode::kInvalidArgument,
              tensor_error(operation, name, "shape does not match [layers, heads, dimension]"));
}

class FreeBlockQueue final {
public:
  explicit FreeBlockQueue(std::size_t block_count) : slots_(block_count), count_(block_count) {
    for (std::size_t index = 0; index < block_count; ++index) {
      slots_[index] = static_cast<std::uint32_t>(index);
    }
  }

  [[nodiscard]] bool empty() const noexcept {
    return count_ == 0;
  }
  [[nodiscard]] std::size_t size() const noexcept {
    return count_;
  }
  [[nodiscard]] std::uint32_t front() const {
    HELIX_ASSERT(!empty(), "front requested from empty CUDA KV free-block queue");
    return slots_[head_];
  }
  [[nodiscard]] std::uint32_t at(std::size_t logical_index) const {
    HELIX_ASSERT(logical_index < count_, "CUDA KV free-block queue index is out of range");
    return slots_[(head_ + logical_index) % slots_.size()];
  }
  void pop() {
    HELIX_ASSERT(!empty(), "pop requested from empty CUDA KV free-block queue");
    head_ = increment(head_);
    --count_;
  }
  void push(std::uint32_t block) {
    HELIX_ASSERT(count_ < slots_.size(), "CUDA KV free-block queue overflow");
    slots_[tail_] = block;
    tail_ = increment(tail_);
    ++count_;
  }

private:
  [[nodiscard]] std::size_t increment(std::size_t index) const noexcept {
    ++index;
    return index == slots_.size() ? 0 : index;
  }

  std::vector<std::uint32_t> slots_;
  std::size_t head_ = 0;
  std::size_t tail_ = 0;
  std::size_t count_ = 0;
};

bool synchronize_noexcept(const helix::cuda::Stream& stream) noexcept {
  try {
    stream.synchronize();
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace

struct CudaPagedKvCache::SequenceState final {
  std::size_t length = 0;
  std::vector<std::uint32_t> blocks;
};

struct CudaPagedKvCache::Impl final {
  Impl(CudaPagedLayout layout, int target_device)
      : config(layout.config), cache_id(next_cuda_paged_cache_id()),
        device(Device::cuda(target_device)), per_token_bytes(layout.bytes_per_token),
        per_block_bytes(layout.block_bytes), total_token_capacity(layout.token_capacity),
        vector_bytes(layout.vector_bytes), layer_block_elements(layout.layer_block_elements),
        head_block_elements(layout.head_block_elements), free_blocks(config.physical_blocks),
        block_token_counts(config.physical_blocks, 0),
        block_was_allocated(config.physical_blocks, 0), block_is_free(config.physical_blocks, 1) {
    layer_keys.reserve(config.layers);
    layer_values.reserve(config.layers);
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
      layer_keys.push_back(Tensor::empty(layout.layer_pool_shape, DType::kFloat32, device));
      layer_values.push_back(Tensor::empty(layout.layer_pool_shape, DType::kFloat32, device));
    }

    std::size_t actual_bytes = 0;
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
      actual_bytes += layer_keys[layer].nbytes();
      actual_bytes += layer_values[layer].nbytes();
    }
    HELIX_ASSERT(actual_bytes == layout.pool_bytes,
                 "CUDA paged KV tensors do not match computed reserved bytes");
    stats.bytes_reserved = layout.pool_bytes;
    stats.bytes_free = layout.pool_bytes;
    stats.total_blocks = config.physical_blocks;
    stats.free_blocks = config.physical_blocks;
    validate_invariants();
  }

  void require_healthy(std::string_view operation) const {
    HELIX_CHECK(!stream_state_uncertain, ErrorCode::kRuntime,
                std::string(operation) +
                    ": cache is unusable after a CUDA stream could not be drained");
  }

  [[nodiscard]] SequenceState& require_sequence(const CudaPagedSequenceHandle& handle,
                                                std::string_view operation) {
    HELIX_CHECK(handle && handle.cache_id() == cache_id, ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence belongs to another cache or is empty");
    const auto iterator = sequences.find(handle.id());
    HELIX_CHECK(iterator != sequences.end(), ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence is unknown or was already released");
    return iterator->second;
  }

  [[nodiscard]] const SequenceState& require_sequence(const CudaPagedSequenceHandle& handle,
                                                      std::string_view operation) const {
    HELIX_CHECK(handle && handle.cache_id() == cache_id, ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence belongs to another cache or is empty");
    const auto iterator = sequences.find(handle.id());
    HELIX_CHECK(iterator != sequences.end(), ErrorCode::kInvalidArgument,
                std::string(operation) + ": sequence is unknown or was already released");
    return iterator->second;
  }

  [[nodiscard]] std::size_t destination_element_offset(std::uint32_t physical_block,
                                                       std::size_t head,
                                                       std::size_t token_offset) const noexcept {
    return static_cast<std::size_t>(physical_block) * layer_block_elements +
           head * head_block_elements + token_offset * config.head_dimension;
  }

  void validate_invariants() const {
#ifndef NDEBUG
    std::size_t derived_tokens = 0;
    std::size_t derived_used_blocks = 0;
    for (const auto& [sequence_id, sequence] : sequences) {
      HELIX_ASSERT(sequence_id != 0, "CUDA paged KV sequence uses reserved zero identity");
      HELIX_ASSERT(sequence.blocks.size() ==
                       blocks_for_tokens(sequence.length, config.block_tokens),
                   "CUDA paged KV table length disagrees with sequence length");
      HELIX_ASSERT(sequence.length <= std::numeric_limits<std::size_t>::max() - derived_tokens,
                   "CUDA paged KV active-token invariant would overflow");
      derived_tokens += sequence.length;
      derived_used_blocks += sequence.blocks.size();
      for (std::size_t logical = 0; logical < sequence.blocks.size(); ++logical) {
        const std::uint32_t block = sequence.blocks[logical];
        HELIX_ASSERT(static_cast<std::size_t>(block) < config.physical_blocks,
                     "CUDA paged KV table contains an invalid physical block");
        HELIX_ASSERT(block_is_free[block] == 0,
                     "CUDA paged KV table references a free physical block");
        const std::size_t logical_start = logical * config.block_tokens;
        const std::size_t visible = std::min(config.block_tokens, sequence.length - logical_start);
        HELIX_ASSERT(block_token_counts[block] == visible,
                     "CUDA paged KV payload length disagrees with sequence mapping");
        for (const auto& [other_id, other] : sequences) {
          for (std::size_t other_logical = 0; other_logical < other.blocks.size();
               ++other_logical) {
            HELIX_ASSERT((other_id == sequence_id && other_logical == logical) ||
                             other.blocks[other_logical] != block,
                         "CUDA paged KV physical block has multiple owners");
          }
        }
      }
    }

    std::size_t free_members = 0;
    for (std::size_t block = 0; block < config.physical_blocks; ++block) {
      if (block_is_free[block] != 0) {
        ++free_members;
        HELIX_ASSERT(block_token_counts[block] == 0,
                     "free CUDA paged KV block retains written-token metadata");
      } else {
        HELIX_ASSERT(block_was_allocated[block] != 0,
                     "owned CUDA paged KV block lacks allocation history");
        HELIX_ASSERT(block_token_counts[block] != 0 &&
                         block_token_counts[block] <= config.block_tokens,
                     "owned CUDA paged KV block has invalid written-token metadata");
      }
    }
    for (std::size_t offset = 0; offset < free_blocks.size(); ++offset) {
      const std::uint32_t block = free_blocks.at(offset);
      HELIX_ASSERT(static_cast<std::size_t>(block) < config.physical_blocks,
                   "CUDA paged KV free queue contains an invalid block");
      HELIX_ASSERT(block_is_free[block] != 0, "CUDA paged KV free queue contains an owned block");
      for (std::size_t prior = 0; prior < offset; ++prior) {
        HELIX_ASSERT(free_blocks.at(prior) != block,
                     "CUDA paged KV free queue contains a duplicate block");
      }
    }

    HELIX_ASSERT(free_members == free_blocks.size(),
                 "CUDA paged KV free membership disagrees with free queue");
    HELIX_ASSERT(derived_used_blocks + free_members == config.physical_blocks,
                 "CUDA paged KV ownership does not cover the physical pool");
    HELIX_ASSERT(stats.active_sequence_count == sequences.size(),
                 "CUDA paged KV sequence statistic is inconsistent");
    HELIX_ASSERT(stats.active_tokens == derived_tokens && stats.resident_tokens == derived_tokens,
                 "CUDA paged KV token statistics are inconsistent");
    HELIX_ASSERT(stats.used_blocks == derived_used_blocks &&
                     stats.free_blocks == free_blocks.size(),
                 "CUDA paged KV block statistics are inconsistent");
    HELIX_ASSERT(stats.block_reference_count == derived_used_blocks && stats.shared_blocks == 0,
                 "CUDA paged KV reference statistics are inconsistent");
    HELIX_ASSERT(stats.allocated_token_slots == derived_used_blocks * config.block_tokens,
                 "CUDA paged KV allocated-slot statistic is inconsistent");
    HELIX_ASSERT(stats.bytes_used == derived_used_blocks * per_block_bytes &&
                     stats.bytes_free == free_blocks.size() * per_block_bytes,
                 "CUDA paged KV byte statistics are inconsistent");
    HELIX_ASSERT(stats.bytes_used + stats.bytes_free == stats.bytes_reserved,
                 "CUDA paged KV byte statistics do not cover the pool");
#endif
  }

  PagedKvCacheConfig config;
  const std::uint64_t cache_id;
  Device device;
  std::size_t per_token_bytes = 0;
  std::size_t per_block_bytes = 0;
  std::size_t total_token_capacity = 0;
  std::size_t vector_bytes = 0;
  std::size_t layer_block_elements = 0;
  std::size_t head_block_elements = 0;
  std::vector<Tensor> layer_keys;
  std::vector<Tensor> layer_values;
  mutable std::mutex mutex;
  FreeBlockQueue free_blocks;
  std::vector<std::size_t> block_token_counts;
  std::vector<std::uint8_t> block_was_allocated;
  std::vector<std::uint8_t> block_is_free;
  std::unordered_map<std::uint64_t, SequenceState> sequences;
  std::uint64_t next_sequence_id = 1;
  PagedKvCacheStats stats;
  bool stream_state_uncertain = false;
  // Destroyed before layer_keys/layer_values. Retaining the table here prevents
  // its address from being reused after a consumer reports uncertain stream
  // completion; the entire cache is poisoned at the same time.
  std::optional<cuda::PagedBlockTable> quarantined_table;
};

CudaPagedKvCache::DecodeView::DecodeView(std::shared_ptr<Impl> cache,
                                         std::unique_lock<std::mutex> lock,
                                         const SequenceState* sequence,
                                         cuda::PagedBlockTable block_table) noexcept
    : cache_(std::move(cache)), lock_(std::move(lock)), sequence_(sequence),
      block_table_(std::move(block_table)) {}

CudaPagedKvCache::DecodeView::~DecodeView() = default;
CudaPagedKvCache::DecodeView::DecodeView(DecodeView&&) noexcept = default;
CudaPagedKvCache::DecodeView& CudaPagedKvCache::DecodeView::operator=(DecodeView&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  // The lock refers to a mutex owned by cache_. Tear down the borrowed table,
  // unlock, and only then release the cache owner. Default member-wise move
  // assignment would reset cache_ first and could destroy a facade-less cache
  // while this view still held its mutex.
  block_table_.reset();
  sequence_ = nullptr;
  if (lock_.owns_lock()) {
    lock_.unlock();
  }
  cache_.reset();

  cache_ = std::move(other.cache_);
  lock_ = std::move(other.lock_);
  sequence_ = std::exchange(other.sequence_, nullptr);
  block_table_ = std::move(other.block_table_);
  return *this;
}

void CudaPagedKvCache::DecodeView::require_valid(std::string_view operation) const {
  HELIX_CHECK(cache_ != nullptr && lock_.owns_lock() && sequence_ != nullptr &&
                  block_table_.has_value() && block_table_->valid(),
              ErrorCode::kInvalidArgument,
              std::string(operation) + ": CUDA paged KV decode view is empty or moved from");
}

std::size_t CudaPagedKvCache::DecodeView::length() const {
  require_valid("CudaPagedKvCache::DecodeView::length");
  return sequence_->length;
}

std::span<const std::uint32_t> CudaPagedKvCache::DecodeView::physical_blocks() const {
  require_valid("CudaPagedKvCache::DecodeView::physical_blocks");
  return sequence_->blocks;
}

std::span<const Tensor> CudaPagedKvCache::DecodeView::layer_key_blocks() const {
  require_valid("CudaPagedKvCache::DecodeView::layer_key_blocks");
  return cache_->layer_keys;
}

std::span<const Tensor> CudaPagedKvCache::DecodeView::layer_value_blocks() const {
  require_valid("CudaPagedKvCache::DecodeView::layer_value_blocks");
  return cache_->layer_values;
}

const cuda::PagedBlockTable& CudaPagedKvCache::DecodeView::block_table() const {
  require_valid("CudaPagedKvCache::DecodeView::block_table");
  return *block_table_;
}

void CudaPagedKvCache::DecodeView::poison_after_consumer_failure() noexcept {
  if (cache_ == nullptr || !lock_.owns_lock() || !block_table_.has_value() ||
      !block_table_->valid()) {
    return;
  }
  cache_->stream_state_uncertain = true;
  // A second table cannot be lent after stream_state_uncertain is set because
  // read_sequence rejects the poisoned cache while this view still owns its
  // lock. PagedBlockTable's move assignment is nonthrowing.
  cache_->quarantined_table = std::move(block_table_);
  sequence_ = nullptr;
}

CudaPagedKvCache::CudaPagedKvCache(PagedKvCacheConfig config, int device_index) try
    : impl_(std::make_shared<Impl>(make_cuda_paged_layout(config, device_index), device_index)) {
} catch (const std::bad_alloc&) {
  throw_error(ErrorCode::kResourceExhausted,
              "CudaPagedKvCache construction could not allocate pool metadata");
}

CudaPagedKvCache::~CudaPagedKvCache() = default;

CudaPagedSequenceHandle CudaPagedKvCache::create_sequence() {
  constexpr std::string_view kOperation = "CudaPagedKvCache::create_sequence";
  std::scoped_lock lock(impl_->mutex);
  impl_->require_healthy(kOperation);
  HELIX_CHECK(impl_->next_sequence_id != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted, "CUDA paged KV sequence ID space exhausted");
  HELIX_CHECK(impl_->stats.active_sequence_count != std::numeric_limits<std::size_t>::max(),
              ErrorCode::kResourceExhausted,
              "CudaPagedKvCache::create_sequence active sequence count would overflow");
  HELIX_CHECK(impl_->stats.sequence_create_count != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted,
              "CudaPagedKvCache::create_sequence creation counter would overflow");
  const std::uint64_t sequence_id = impl_->next_sequence_id;
  try {
    const auto [iterator, inserted] = impl_->sequences.try_emplace(sequence_id);
    static_cast<void>(iterator);
    HELIX_ASSERT(inserted, "CUDA paged KV sequence identity unexpectedly collided");
  } catch (const std::bad_alloc&) {
    throw_error(ErrorCode::kResourceExhausted,
                "CudaPagedKvCache::create_sequence could not allocate sequence metadata");
  }
  ++impl_->next_sequence_id;
  ++impl_->stats.active_sequence_count;
  ++impl_->stats.sequence_create_count;
  impl_->validate_invariants();
  return CudaPagedSequenceHandle(sequence_id, impl_->cache_id);
}

void CudaPagedKvCache::append(const CudaPagedSequenceHandle& sequence, const Tensor& token_keys,
                              const Tensor& token_values, const helix::cuda::Stream& stream) {
  constexpr std::string_view kOperation = "CudaPagedKvCache::append";
  std::scoped_lock lock(impl_->mutex);
  impl_->require_healthy(kOperation);
  SequenceState& state = impl_->require_sequence(sequence, kOperation);
  HELIX_CHECK(stream.native_handle() != nullptr, ErrorCode::kInvalidArgument,
              "CudaPagedKvCache::append stream must not be moved-from");
  HELIX_CHECK(stream.device_index() == impl_->device.index(), ErrorCode::kInvalidArgument,
              "CudaPagedKvCache::append stream must belong to the cache device");
  const Shape expected_shape{
      checked_dimension(impl_->config.layers, "CUDA KV layer count"),
      checked_dimension(impl_->config.kv_heads, "CUDA KV head count"),
      checked_dimension(impl_->config.head_dimension, "CUDA KV head dimension")};
  require_input_tensor(token_keys, expected_shape, impl_->device, "token keys", kOperation);
  require_input_tensor(token_values, expected_shape, impl_->device, "token values", kOperation);
  for (std::size_t layer = 0; layer < impl_->config.layers; ++layer) {
    HELIX_CHECK(!token_keys.overlaps_logical_range(impl_->layer_keys[layer]) &&
                    !token_keys.overlaps_logical_range(impl_->layer_values[layer]) &&
                    !token_values.overlaps_logical_range(impl_->layer_keys[layer]) &&
                    !token_values.overlaps_logical_range(impl_->layer_values[layer]),
                ErrorCode::kInvalidArgument,
                "CudaPagedKvCache::append input must not alias cache-owned page storage");
  }
  HELIX_CHECK(state.length != std::numeric_limits<std::size_t>::max(),
              ErrorCode::kResourceExhausted,
              "CudaPagedKvCache::append sequence length would overflow");
  HELIX_CHECK(impl_->stats.active_tokens != std::numeric_limits<std::size_t>::max() &&
                  impl_->stats.resident_tokens != std::numeric_limits<std::size_t>::max(),
              ErrorCode::kResourceExhausted,
              "CudaPagedKvCache::append active token statistics would overflow");
  HELIX_CHECK(impl_->stats.append_count != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted,
              "CudaPagedKvCache::append append counter would overflow");

  const bool needs_block = state.length % impl_->config.block_tokens == 0;
  std::uint32_t block = 0;
  if (needs_block) {
    HELIX_CHECK(!impl_->free_blocks.empty(), ErrorCode::kResourceExhausted,
                "CudaPagedKvCache::append has no free physical KV block");
    block = impl_->free_blocks.front();
    HELIX_CHECK(impl_->stats.block_allocation_count != std::numeric_limits<std::uint64_t>::max(),
                ErrorCode::kResourceExhausted,
                "CudaPagedKvCache::append block allocation counter would overflow");
    if (impl_->block_was_allocated[block] != 0) {
      HELIX_CHECK(impl_->stats.block_reuse_count != std::numeric_limits<std::uint64_t>::max(),
                  ErrorCode::kResourceExhausted,
                  "CudaPagedKvCache::append block reuse counter would overflow");
    }
    HELIX_CHECK(state.blocks.size() != std::numeric_limits<std::size_t>::max(),
                ErrorCode::kResourceExhausted,
                "CudaPagedKvCache::append block-table size would overflow");
    try {
      state.blocks.reserve(state.blocks.size() + 1);
    } catch (const std::bad_alloc&) {
      throw_error(ErrorCode::kResourceExhausted,
                  "CudaPagedKvCache::append could not grow the sequence block table");
    }
  } else {
    HELIX_ASSERT(!state.blocks.empty(), "nonempty CUDA paged KV sequence has no physical block");
    block = state.blocks.back();
  }

  const std::size_t token_offset = state.length % impl_->config.block_tokens;
  HELIX_ASSERT(impl_->block_token_counts[block] == token_offset,
               "CUDA paged KV append offset disagrees with written-token metadata");
  const auto* key_source = static_cast<const std::byte*>(token_keys.data());
  const auto* value_source = static_cast<const std::byte*>(token_values.data());
  bool enqueued_work = false;
  try {
    for (std::size_t layer = 0; layer < impl_->config.layers; ++layer) {
      auto* key_destination = static_cast<std::byte*>(impl_->layer_keys[layer].data());
      auto* value_destination = static_cast<std::byte*>(impl_->layer_values[layer].data());
      for (std::size_t head = 0; head < impl_->config.kv_heads; ++head) {
        const std::size_t source_vector = layer * impl_->config.kv_heads + head;
        const std::size_t destination_element =
            impl_->destination_element_offset(block, head, token_offset);
        helix::cuda::copy_device_to_device_async(
            key_destination + destination_element * sizeof(float),
            key_source + source_vector * impl_->vector_bytes, impl_->vector_bytes, stream);
        enqueued_work = true;
        helix::cuda::copy_device_to_device_async(
            value_destination + destination_element * sizeof(float),
            value_source + source_vector * impl_->vector_bytes, impl_->vector_bytes, stream);
      }
    }
    stream.synchronize();
  } catch (...) {
    if (enqueued_work && !synchronize_noexcept(stream)) {
      impl_->stream_state_uncertain = true;
    }
    throw;
  }

  if (needs_block) {
    state.blocks.push_back(block);
    impl_->free_blocks.pop();
    HELIX_ASSERT(impl_->block_is_free[block] != 0 && impl_->block_token_counts[block] == 0,
                 "CUDA paged KV free block has conflicting ownership metadata");
    impl_->block_is_free[block] = 0;
    if (impl_->block_was_allocated[block] != 0) {
      ++impl_->stats.block_reuse_count;
    }
    impl_->block_was_allocated[block] = 1;
    ++impl_->stats.used_blocks;
    --impl_->stats.free_blocks;
    impl_->stats.bytes_used += impl_->per_block_bytes;
    impl_->stats.bytes_free -= impl_->per_block_bytes;
    impl_->stats.peak_bytes_used = std::max(impl_->stats.peak_bytes_used, impl_->stats.bytes_used);
    impl_->stats.allocated_token_slots += impl_->config.block_tokens;
    ++impl_->stats.block_reference_count;
    ++impl_->stats.block_allocation_count;
  }

  ++state.length;
  ++impl_->block_token_counts[block];
  ++impl_->stats.active_tokens;
  ++impl_->stats.resident_tokens;
  ++impl_->stats.append_count;
  impl_->validate_invariants();
}

void CudaPagedKvCache::release_sequence(const CudaPagedSequenceHandle& sequence) {
  constexpr std::string_view kOperation = "CudaPagedKvCache::release_sequence";
  std::scoped_lock lock(impl_->mutex);
  SequenceState& state = impl_->require_sequence(sequence, kOperation);
  HELIX_CHECK(impl_->stats.release_count != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted,
              "CudaPagedKvCache::release_sequence release counter would overflow");

  const std::size_t released_tokens = state.length;
  const std::size_t released_blocks = state.blocks.size();
  for (const std::uint32_t block : state.blocks) {
    HELIX_ASSERT(impl_->block_is_free[block] == 0 && impl_->block_token_counts[block] != 0,
                 "CUDA paged KV release encountered conflicting block ownership");
    impl_->block_token_counts[block] = 0;
    impl_->block_is_free[block] = 1;
    impl_->free_blocks.push(block);
  }
  impl_->sequences.erase(sequence.id());
  --impl_->stats.active_sequence_count;
  impl_->stats.active_tokens -= released_tokens;
  impl_->stats.resident_tokens -= released_tokens;
  impl_->stats.used_blocks -= released_blocks;
  impl_->stats.free_blocks += released_blocks;
  const std::size_t released_bytes = released_blocks * impl_->per_block_bytes;
  impl_->stats.bytes_used -= released_bytes;
  impl_->stats.bytes_free += released_bytes;
  impl_->stats.allocated_token_slots -= released_blocks * impl_->config.block_tokens;
  impl_->stats.block_reference_count -= released_blocks;
  ++impl_->stats.release_count;
  impl_->validate_invariants();
}

CudaPagedKvCache::DecodeView
CudaPagedKvCache::read_sequence(const CudaPagedSequenceHandle& sequence) const {
  constexpr std::string_view kOperation = "CudaPagedKvCache::read_sequence";
  std::shared_ptr<Impl> cache = impl_;
  std::unique_lock lock(cache->mutex);
  cache->require_healthy(kOperation);
  const SequenceState& state = cache->require_sequence(sequence, kOperation);
  cuda::PagedBlockTable table(state.blocks, cache->config.physical_blocks, cache->device.index());
  return DecodeView(std::move(cache), std::move(lock), &state, std::move(table));
}

bool CudaPagedKvCache::is_healthy() const {
  std::scoped_lock lock(impl_->mutex);
  return !impl_->stream_state_uncertain;
}

bool CudaPagedKvCache::contains(const CudaPagedSequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  if (!sequence || sequence.cache_id() != impl_->cache_id) {
    return false;
  }
  return impl_->sequences.contains(sequence.id());
}

std::size_t CudaPagedKvCache::length(const CudaPagedSequenceHandle& sequence) const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->require_sequence(sequence, "CudaPagedKvCache::length").length;
}

PagedKvCacheStats CudaPagedKvCache::stats() const {
  std::scoped_lock lock(impl_->mutex);
  return impl_->stats;
}

std::size_t CudaPagedKvCache::bytes_per_token() const noexcept {
  return impl_->per_token_bytes;
}

std::size_t CudaPagedKvCache::block_bytes() const noexcept {
  return impl_->per_block_bytes;
}

std::size_t CudaPagedKvCache::token_capacity() const noexcept {
  return impl_->total_token_capacity;
}

int CudaPagedKvCache::device_index() const noexcept {
  return impl_->device.index();
}

const PagedKvCacheConfig& CudaPagedKvCache::config() const noexcept {
  return impl_->config;
}

} // namespace helix::kv_cache
