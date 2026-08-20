#include "helix/memory/helix_allocator.hpp"

#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace helix::memory {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs, std::string_view context) {
  HELIX_CHECK(rhs <= std::numeric_limits<std::size_t>::max() - lhs, ErrorCode::kOutOfRange,
              std::string(context));
  return lhs + rhs;
}

std::size_t round_up(std::size_t value, std::size_t alignment, std::string_view context) {
  HELIX_ASSERT(is_power_of_two(alignment), "round_up alignment must be a power of two");
  const std::size_t mask = alignment - 1;
  HELIX_CHECK(value <= std::numeric_limits<std::size_t>::max() - mask, ErrorCode::kOutOfRange,
              std::string(context));
  return (value + mask) & ~mask;
}

std::uintptr_t checked_address_add(std::uintptr_t address, std::size_t offset) {
  HELIX_CHECK(offset <= std::numeric_limits<std::uintptr_t>::max() - address,
              ErrorCode::kOutOfRange, "allocator pointer offset overflow");
  return address + offset;
}

std::size_t alignment_padding(void* base, std::size_t offset, std::size_t alignment) {
  const std::uintptr_t address =
      checked_address_add(reinterpret_cast<std::uintptr_t>(base), offset);
  const std::uintptr_t mask = static_cast<std::uintptr_t>(alignment - 1);
  HELIX_CHECK(address <= std::numeric_limits<std::uintptr_t>::max() - mask, ErrorCode::kOutOfRange,
              "allocator pointer alignment overflow");
  const std::uintptr_t aligned = (address + mask) & ~mask;
  return static_cast<std::size_t>(aligned - address);
}

void* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<void*>(
      checked_address_add(reinterpret_cast<std::uintptr_t>(base), offset));
}

std::string cleanup_message(const std::exception& exception) {
  std::ostringstream message;
  message << "HelixAllocator failed to release an upstream segment during destruction: "
          << exception.what();
  return message.str();
}

std::uint64_t next_allocator_id() {
  static std::atomic<std::uint64_t> next{1};
  std::uint64_t current = next.load(std::memory_order_relaxed);
  while (true) {
    HELIX_CHECK(current != std::numeric_limits<std::uint64_t>::max(), ErrorCode::kResourceExhausted,
                "HelixAllocator identity space exhausted");
    if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed,
                                   std::memory_order_relaxed)) {
      return current;
    }
  }
}

} // namespace

struct HelixAllocator::Impl {
  struct Block {
    std::size_t offset = 0;
    std::size_t bytes = 0;
    std::size_t requested_bytes = 0;
    std::uint64_t allocation_id = 0;

    [[nodiscard]] bool allocated() const noexcept {
      return allocation_id != 0;
    }
  };

  struct Segment {
    std::uint64_t id = 0;
    void* base = nullptr;
    std::size_t bytes = 0;
    std::size_t alignment = 0;
    std::vector<Block> blocks;
  };

  struct ActiveRecord {
    std::uint64_t segment_id = 0;
    std::size_t offset = 0;
    void* pointer = nullptr;
    std::size_t requested_bytes = 0;
    std::size_t allocated_bytes = 0;
    std::size_t alignment = 0;
  };

  struct Candidate {
    std::size_t segment_index = 0;
    std::size_t block_index = 0;
    std::size_t padding = 0;
    std::size_t waste = 0;
  };

  Impl(std::shared_ptr<MemoryResource> upstream, AllocatorConfig allocator_config)
      : resource(std::move(upstream)), config(allocator_config), allocator_id(next_allocator_id()) {
    HELIX_CHECK(resource != nullptr, ErrorCode::kInvalidArgument,
                "HelixAllocator requires a non-null memory resource");
    HELIX_CHECK(config.segment_bytes != 0, ErrorCode::kInvalidArgument,
                "HelixAllocator segment size must be non-zero");
    HELIX_CHECK(is_power_of_two(config.minimum_alignment), ErrorCode::kInvalidArgument,
                "HelixAllocator minimum alignment must be a non-zero power of two");
    HELIX_CHECK(config.maximum_reserved_bytes != 0, ErrorCode::kInvalidArgument,
                "HelixAllocator maximum reserved bytes must be non-zero");
  }

  [[nodiscard]] std::optional<Candidate> find_best_fit(std::size_t allocated_bytes,
                                                       std::size_t alignment) const {
    std::optional<Candidate> best;
    for (std::size_t segment_index = 0; segment_index < segments.size(); ++segment_index) {
      const Segment& segment = segments[segment_index];
      for (std::size_t block_index = 0; block_index < segment.blocks.size(); ++block_index) {
        const Block& block = segment.blocks[block_index];
        if (block.allocated()) {
          continue;
        }
        const std::size_t padding = alignment_padding(segment.base, block.offset, alignment);
        if (padding > block.bytes || allocated_bytes > block.bytes - padding) {
          continue;
        }
        const std::size_t waste = block.bytes - padding - allocated_bytes;
        if (!best || waste < best->waste ||
            (waste == best->waste && segment.id < segments[best->segment_index].id) ||
            (waste == best->waste && segment.id == segments[best->segment_index].id &&
             block.offset < segments[best->segment_index].blocks[best->block_index].offset)) {
          best = Candidate{segment_index, block_index, padding, waste};
        }
      }
    }
    return best;
  }

  void acquire_segment(std::size_t allocated_bytes, std::size_t alignment) {
    const std::size_t segment_alignment = std::max(config.minimum_alignment, alignment);
    const std::size_t desired_bytes = std::max(config.segment_bytes, allocated_bytes);
    const std::size_t segment_bytes =
        round_up(desired_bytes, segment_alignment, "allocator segment-size overflow");
    HELIX_CHECK(segment_bytes <= config.maximum_reserved_bytes -
                                     std::min(bytes_reserved, config.maximum_reserved_bytes),
                ErrorCode::kResourceExhausted,
                "HelixAllocator reservation limit would be exceeded");

    Segment segment;
    segment.id = next_segment_id;
    HELIX_CHECK(next_segment_id != std::numeric_limits<std::uint64_t>::max(),
                ErrorCode::kResourceExhausted, "HelixAllocator segment id space exhausted");
    segment.bytes = segment_bytes;
    segment.alignment = segment_alignment;
    segment.blocks.reserve(1);
    segment.blocks.push_back(Block{0, segment_bytes, 0, 0});
    segment.base = resource->allocate(segment_bytes, segment_alignment);
    HELIX_CHECK(segment.base != nullptr, ErrorCode::kResourceExhausted,
                "memory resource returned a null segment");
    const auto address = reinterpret_cast<std::uintptr_t>(segment.base);
    if (address % segment_alignment != 0) {
      resource->deallocate(segment.base, segment.bytes, segment.alignment);
      throw_error(ErrorCode::kRuntime,
                  "memory resource returned a segment with insufficient alignment");
    }

    try {
      segments.push_back(std::move(segment));
    } catch (...) {
      try {
        resource->deallocate(segment.base, segment.bytes, segment.alignment);
      } catch (const std::exception& exception) {
        HELIX_LOG_ERROR(cleanup_message(exception));
      }
      throw;
    }
    ++next_segment_id;
    bytes_reserved =
        checked_add(bytes_reserved, segment_bytes, "HelixAllocator reserved-byte counter overflow");
    ++underlying_allocations;
    assert_invariants();
  }

  Allocation commit_allocation(const Candidate& candidate, std::size_t requested_bytes,
                               std::size_t allocated_bytes, std::size_t alignment) {
    HELIX_CHECK(next_allocation_id != std::numeric_limits<std::uint64_t>::max(),
                ErrorCode::kResourceExhausted, "HelixAllocator allocation id space exhausted");
    Segment& segment = segments[candidate.segment_index];
    const Block& free_block = segment.blocks[candidate.block_index];
    const std::size_t allocation_offset = free_block.offset + candidate.padding;
    const std::size_t suffix_offset = allocation_offset + allocated_bytes;
    const std::size_t suffix_bytes = free_block.bytes - candidate.padding - allocated_bytes;
    const std::uint64_t allocation_id = next_allocation_id;
    void* pointer = offset_pointer(segment.base, allocation_offset);

    std::vector<Block> replacement;
    replacement.reserve(segment.blocks.size() + 2);
    for (std::size_t index = 0; index < segment.blocks.size(); ++index) {
      if (index != candidate.block_index) {
        replacement.push_back(segment.blocks[index]);
        continue;
      }
      if (candidate.padding != 0) {
        replacement.push_back(Block{free_block.offset, candidate.padding, 0, 0});
      }
      replacement.push_back(
          Block{allocation_offset, allocated_bytes, requested_bytes, allocation_id});
      if (suffix_bytes != 0) {
        replacement.push_back(Block{suffix_offset, suffix_bytes, 0, 0});
      }
    }

    const ActiveRecord record{segment.id,      allocation_offset, pointer,
                              requested_bytes, allocated_bytes,   alignment};
    const auto [active_iterator, inserted] = active.emplace(allocation_id, record);
    static_cast<void>(active_iterator);
    HELIX_ASSERT(inserted, "allocation id unexpectedly collided");
    segment.blocks.swap(replacement);

    ++next_allocation_id;
    bytes_allocated = checked_add(bytes_allocated, allocated_bytes,
                                  "HelixAllocator allocated-byte counter overflow");
    bytes_requested = checked_add(bytes_requested, requested_bytes,
                                  "HelixAllocator requested-byte counter overflow");
    peak_bytes_allocated = std::max(peak_bytes_allocated, bytes_allocated);
    ++allocation_count;
    assert_invariants();
    return Allocation(pointer, requested_bytes, allocated_bytes, alignment, allocation_id,
                      allocator_id, resource->device());
  }

  void assert_invariants() const {
#ifndef NDEBUG
    std::size_t scanned_reserved = 0;
    std::size_t scanned_allocated = 0;
    std::size_t scanned_requested = 0;
    std::size_t scanned_active = 0;
    for (const Segment& segment : segments) {
      HELIX_ASSERT(segment.base != nullptr, "allocator segment base is null");
      HELIX_ASSERT(segment.bytes != 0, "allocator segment is empty");
      HELIX_ASSERT(!segment.blocks.empty(), "allocator segment has no blocks");
      HELIX_ASSERT(segment.blocks.front().offset == 0,
                   "allocator segment does not begin at offset zero");
      std::size_t expected_offset = 0;
      bool previous_was_free = false;
      for (const Block& block : segment.blocks) {
        HELIX_ASSERT(block.bytes != 0, "allocator contains a zero-byte block");
        HELIX_ASSERT(block.offset == expected_offset, "allocator blocks are not contiguous");
        expected_offset += block.bytes;
        if (block.allocated()) {
          HELIX_ASSERT(block.requested_bytes != 0 && block.requested_bytes <= block.bytes,
                       "allocated block has invalid requested bytes");
          const auto active_iterator = active.find(block.allocation_id);
          HELIX_ASSERT(active_iterator != active.end(),
                       "allocated block has no active allocation record");
          const ActiveRecord& record = active_iterator->second;
          HELIX_ASSERT(record.segment_id == segment.id && record.offset == block.offset &&
                           record.requested_bytes == block.requested_bytes &&
                           record.allocated_bytes == block.bytes &&
                           record.pointer == offset_pointer(segment.base, block.offset),
                       "active allocation record disagrees with its block");
          HELIX_ASSERT(reinterpret_cast<std::uintptr_t>(record.pointer) % record.alignment == 0,
                       "active allocation pointer violates its alignment");
          scanned_allocated += block.bytes;
          scanned_requested += block.requested_bytes;
          ++scanned_active;
        } else {
          HELIX_ASSERT(block.requested_bytes == 0, "free block retains non-zero requested bytes");
          HELIX_ASSERT(!previous_was_free, "adjacent free blocks were not coalesced");
        }
        previous_was_free = !block.allocated();
      }
      HELIX_ASSERT(expected_offset == segment.bytes,
                   "allocator blocks do not cover their complete segment");
      scanned_reserved += segment.bytes;
    }
    HELIX_ASSERT(scanned_reserved == bytes_reserved,
                 "allocator reserved-byte counter disagrees with segments");
    HELIX_ASSERT(scanned_allocated == bytes_allocated,
                 "allocator allocated-byte counter disagrees with blocks");
    HELIX_ASSERT(scanned_requested == bytes_requested,
                 "allocator requested-byte counter disagrees with blocks");
    HELIX_ASSERT(scanned_active == active.size(),
                 "allocator active map disagrees with allocated blocks");
#endif
  }

  void release_all_noexcept() noexcept {
    for (const Segment& segment : segments) {
      try {
        resource->deallocate(segment.base, segment.bytes, segment.alignment);
      } catch (const std::exception& exception) {
        HELIX_LOG_ERROR(cleanup_message(exception));
      } catch (...) {
        HELIX_LOG_ERROR("HelixAllocator encountered an unknown upstream cleanup failure");
      }
    }
    segments.clear();
    active.clear();
  }

  std::shared_ptr<MemoryResource> resource;
  AllocatorConfig config;
  const std::uint64_t allocator_id;
  mutable std::mutex mutex;
  std::vector<Segment> segments;
  std::unordered_map<std::uint64_t, ActiveRecord> active;
  std::uint64_t next_segment_id = 1;
  std::uint64_t next_allocation_id = 1;
  std::size_t bytes_reserved = 0;
  std::size_t bytes_allocated = 0;
  std::size_t bytes_requested = 0;
  std::size_t peak_bytes_allocated = 0;
  std::uint64_t allocation_count = 0;
  std::uint64_t reuse_count = 0;
  std::uint64_t underlying_allocations = 0;
  std::uint64_t underlying_deallocations = 0;
};

Allocation::Allocation(void* pointer, std::size_t requested_bytes, std::size_t allocated_bytes,
                       std::size_t alignment, std::uint64_t id, std::uint64_t allocator_id,
                       Device device) noexcept
    : pointer_(pointer), requested_bytes_(requested_bytes), allocated_bytes_(allocated_bytes),
      alignment_(alignment), id_(id), allocator_id_(allocator_id), device_(device) {}

HelixAllocator::HelixAllocator(std::shared_ptr<MemoryResource> resource, AllocatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(resource), config)) {}

HelixAllocator::~HelixAllocator() {
  std::scoped_lock lock(impl_->mutex);
  impl_->release_all_noexcept();
}

Allocation HelixAllocator::allocate(std::size_t bytes, std::size_t alignment) {
  std::scoped_lock lock(impl_->mutex);
  HELIX_CHECK(bytes != 0, ErrorCode::kInvalidArgument, "HelixAllocator cannot allocate zero bytes");
  if (alignment == 0) {
    alignment = impl_->config.minimum_alignment;
  }
  HELIX_CHECK(is_power_of_two(alignment), ErrorCode::kInvalidArgument,
              "HelixAllocator alignment must be a non-zero power of two");
  const std::size_t effective_alignment = std::max(alignment, impl_->config.minimum_alignment);
  const std::size_t allocated_bytes =
      round_up(bytes, effective_alignment, "HelixAllocator allocation-size overflow");

  std::optional<Impl::Candidate> candidate =
      impl_->find_best_fit(allocated_bytes, effective_alignment);
  const bool reused = candidate.has_value();
  if (!candidate) {
    impl_->acquire_segment(allocated_bytes, effective_alignment);
    candidate = impl_->find_best_fit(allocated_bytes, effective_alignment);
    HELIX_ASSERT(candidate.has_value(), "new allocator segment cannot satisfy its request");
  }
  Allocation allocation =
      impl_->commit_allocation(*candidate, bytes, allocated_bytes, effective_alignment);
  if (reused) {
    ++impl_->reuse_count;
  }
  return allocation;
}

void HelixAllocator::deallocate(const Allocation& allocation) {
  std::scoped_lock lock(impl_->mutex);
  HELIX_CHECK(allocation.id() != 0 && allocation.pointer() != nullptr, ErrorCode::kInvalidArgument,
              "HelixAllocator cannot deallocate an empty handle");
  HELIX_CHECK(allocation.allocator_id_ == impl_->allocator_id, ErrorCode::kInvalidArgument,
              "HelixAllocator allocation belongs to another allocator instance");
  const auto active_iterator = impl_->active.find(allocation.id());
  HELIX_CHECK(active_iterator != impl_->active.end(), ErrorCode::kInvalidArgument,
              "HelixAllocator allocation is unknown or was already freed");
  const Impl::ActiveRecord& record = active_iterator->second;
  HELIX_CHECK(record.pointer == allocation.pointer() &&
                  record.requested_bytes == allocation.requested_bytes() &&
                  record.allocated_bytes == allocation.allocated_bytes() &&
                  record.alignment == allocation.alignment() &&
                  allocation.device() == impl_->resource->device(),
              ErrorCode::kInvalidArgument,
              "HelixAllocator allocation handle metadata does not match the active allocation");

  const auto segment_iterator =
      std::find_if(impl_->segments.begin(), impl_->segments.end(),
                   [&](const Impl::Segment& item) { return item.id == record.segment_id; });
  HELIX_ASSERT(segment_iterator != impl_->segments.end(),
               "active allocation references a missing segment");
  Impl::Segment& segment = *segment_iterator;
  bool found_block = false;
  std::vector<Impl::Block> replacement;
  replacement.reserve(segment.blocks.size());
  for (const Impl::Block& existing : segment.blocks) {
    Impl::Block block = existing;
    if (existing.offset == record.offset && existing.allocation_id == allocation.id()) {
      block.requested_bytes = 0;
      block.allocation_id = 0;
      found_block = true;
    }
    if (!replacement.empty() && !replacement.back().allocated() && !block.allocated()) {
      HELIX_ASSERT(replacement.back().offset + replacement.back().bytes == block.offset,
                   "allocator free blocks are not adjacent in sorted order");
      replacement.back().bytes += block.bytes;
    } else {
      replacement.push_back(block);
    }
  }
  HELIX_ASSERT(found_block, "active allocation has no matching block");
  segment.blocks.swap(replacement);
  impl_->bytes_allocated -= record.allocated_bytes;
  impl_->bytes_requested -= record.requested_bytes;
  impl_->active.erase(active_iterator);
  impl_->assert_invariants();
}

bool HelixAllocator::owns(const Allocation& allocation) const {
  std::scoped_lock lock(impl_->mutex);
  if (allocation.allocator_id_ != impl_->allocator_id) {
    return false;
  }
  const auto iterator = impl_->active.find(allocation.id());
  return iterator != impl_->active.end() && iterator->second.pointer == allocation.pointer();
}

AllocatorStats HelixAllocator::stats() const {
  std::scoped_lock lock(impl_->mutex);
  AllocatorStats result;
  result.bytes_reserved = impl_->bytes_reserved;
  result.bytes_allocated = impl_->bytes_allocated;
  result.bytes_requested = impl_->bytes_requested;
  result.bytes_free = impl_->bytes_reserved - impl_->bytes_allocated;
  result.peak_bytes_allocated = impl_->peak_bytes_allocated;
  result.segment_count = impl_->segments.size();
  result.active_allocation_count = impl_->active.size();
  result.allocation_count = impl_->allocation_count;
  result.reuse_count = impl_->reuse_count;
  result.underlying_allocations = impl_->underlying_allocations;
  result.underlying_deallocations = impl_->underlying_deallocations;
  for (const Impl::Segment& segment : impl_->segments) {
    for (const Impl::Block& block : segment.blocks) {
      if (!block.allocated()) {
        result.largest_free_block = std::max(result.largest_free_block, block.bytes);
      }
    }
  }
  if (result.bytes_free != 0) {
    result.external_fragmentation = 1.0 - static_cast<double>(result.largest_free_block) /
                                              static_cast<double>(result.bytes_free);
  }
  if (result.bytes_allocated != 0) {
    result.internal_fragmentation =
        static_cast<double>(result.bytes_allocated - result.bytes_requested) /
        static_cast<double>(result.bytes_allocated);
  }
  return result;
}

std::size_t HelixAllocator::release_cached() {
  std::scoped_lock lock(impl_->mutex);
  std::size_t released = 0;
  auto iterator = impl_->segments.begin();
  while (iterator != impl_->segments.end()) {
    const bool entirely_free =
        iterator->blocks.size() == 1 && !iterator->blocks.front().allocated();
    if (!entirely_free) {
      ++iterator;
      continue;
    }
    impl_->resource->deallocate(iterator->base, iterator->bytes, iterator->alignment);
    released =
        checked_add(released, iterator->bytes, "HelixAllocator released-byte counter overflow");
    impl_->bytes_reserved -= iterator->bytes;
    ++impl_->underlying_deallocations;
    iterator = impl_->segments.erase(iterator);
  }
  impl_->assert_invariants();
  return released;
}

Device HelixAllocator::device() const noexcept {
  return impl_->resource->device();
}

const AllocatorConfig& HelixAllocator::config() const noexcept {
  return impl_->config;
}

} // namespace helix::memory
