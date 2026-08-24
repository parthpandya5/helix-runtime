#include "helix/memory/helix_allocator.hpp"

#include "helix/core/error.hpp"
#include "helix/core/logging.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
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
  static constexpr std::size_t kNoActiveSlot = std::numeric_limits<std::size_t>::max();

  struct Block {
    std::size_t offset = 0;
    std::size_t bytes = 0;
    std::size_t requested_bytes = 0;
    std::uint64_t allocation_id = 0;
    std::size_t active_slot = kNoActiveSlot;

    [[nodiscard]] bool allocated() const noexcept {
      return allocation_id != 0;
    }
  };

  static_assert(std::is_nothrow_move_constructible_v<Block> &&
                    std::is_nothrow_move_assignable_v<Block>,
                "in-place block-table updates rely on non-throwing Block moves");

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

  struct ActiveSlot {
    ActiveRecord record;
    std::uint64_t allocation_id = 0;
    std::size_t next_free = kNoActiveSlot;
    bool occupied = false;
  };

  static_assert(std::is_nothrow_move_constructible_v<ActiveSlot> &&
                    std::is_nothrow_move_assignable_v<ActiveSlot>,
                "active-slot growth relies on non-throwing ActiveSlot moves");

  struct Candidate {
    std::size_t segment_index = 0;
    std::size_t block_index = 0;
    std::size_t padding = 0;
    std::size_t waste = 0;
  };

  [[nodiscard]] std::size_t acquire_active_slot(const ActiveRecord& record,
                                                std::uint64_t allocation_id) {
    std::size_t slot_index = kNoActiveSlot;
    if (free_active_slot != kNoActiveSlot) {
      slot_index = free_active_slot;
      HELIX_ASSERT(slot_index < active_slots.size(),
                   "active-slot free list contains an out-of-range slot");
      ActiveSlot& slot = active_slots[slot_index];
      HELIX_ASSERT(!slot.occupied, "active-slot free list contains an occupied slot");
      free_active_slot = slot.next_free;
      slot.record = record;
      slot.allocation_id = allocation_id;
      slot.next_free = kNoActiveSlot;
      slot.occupied = true;
    } else {
      slot_index = active_slots.size();
      active_slots.push_back(ActiveSlot{record, allocation_id, kNoActiveSlot, true});
    }
    ++active_count;
    return slot_index;
  }

  void release_active_slot(std::size_t slot_index) {
    HELIX_ASSERT(slot_index < active_slots.size(), "released active slot is out of range");
    ActiveSlot& slot = active_slots[slot_index];
    HELIX_ASSERT(slot.occupied, "released active slot is already free");
    HELIX_ASSERT(active_count != 0, "active-slot count underflow");
    slot.record = ActiveRecord{};
    slot.allocation_id = 0;
    slot.next_free = free_active_slot;
    slot.occupied = false;
    free_active_slot = slot_index;
    --active_count;
  }

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
    segment.blocks.push_back(Block{0, segment_bytes, 0, 0, kNoActiveSlot});
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

    const std::size_t added_blocks = static_cast<std::size_t>(candidate.padding != 0) +
                                     static_cast<std::size_t>(suffix_bytes != 0);
    const std::size_t required_blocks =
        checked_add(segment.blocks.size(), added_blocks, "allocator block-count overflow");
    if (required_blocks > segment.blocks.capacity()) {
      std::size_t new_capacity = std::max<std::size_t>(segment.blocks.capacity(), 8);
      while (new_capacity < required_blocks) {
        if (new_capacity > segment.blocks.max_size() / 2) {
          new_capacity = required_blocks;
          break;
        }
        new_capacity *= 2;
      }
      segment.blocks.reserve(new_capacity);
    }

    // Complete every potentially throwing metadata allocation before publishing
    // the active record. With reserved capacity and non-throwing Block moves, the
    // split below cannot strand a partially committed allocation.

    const std::size_t new_bytes_allocated = checked_add(
        bytes_allocated, allocated_bytes, "HelixAllocator allocated-byte counter overflow");
    const std::size_t new_bytes_requested = checked_add(
        bytes_requested, requested_bytes, "HelixAllocator requested-byte counter overflow");

    const ActiveRecord record{segment.id,      allocation_offset, pointer,
                              requested_bytes, allocated_bytes,   alignment};
    const std::size_t active_slot = acquire_active_slot(record, allocation_id);

    const Block allocated_block{allocation_offset, allocated_bytes, requested_bytes, allocation_id,
                                active_slot};
    if (candidate.padding == 0) {
      segment.blocks[candidate.block_index] = allocated_block;
      if (suffix_bytes != 0) {
        segment.blocks.insert(segment.blocks.begin() +
                                  static_cast<std::ptrdiff_t>(candidate.block_index + 1),
                              Block{suffix_offset, suffix_bytes, 0, 0, kNoActiveSlot});
      }
    } else {
      segment.blocks[candidate.block_index] =
          Block{free_block.offset, candidate.padding, 0, 0, kNoActiveSlot};
      segment.blocks.insert(segment.blocks.begin() +
                                static_cast<std::ptrdiff_t>(candidate.block_index + 1),
                            allocated_block);
      if (suffix_bytes != 0) {
        segment.blocks.insert(segment.blocks.begin() +
                                  static_cast<std::ptrdiff_t>(candidate.block_index + 2),
                              Block{suffix_offset, suffix_bytes, 0, 0, kNoActiveSlot});
      }
    }

    ++next_allocation_id;
    bytes_allocated = new_bytes_allocated;
    bytes_requested = new_bytes_requested;
    peak_bytes_allocated = std::max(peak_bytes_allocated, bytes_allocated);
    ++allocation_count;
    assert_invariants();
    return Allocation(pointer, requested_bytes, allocated_bytes, alignment, allocation_id,
                      allocator_id, active_slot, resource->device());
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
          HELIX_ASSERT(block.active_slot < active_slots.size(),
                       "allocated block has an invalid active slot");
          const ActiveSlot& slot = active_slots[block.active_slot];
          HELIX_ASSERT(slot.occupied && slot.allocation_id == block.allocation_id,
                       "allocated block has no matching active slot");
          const ActiveRecord& record = slot.record;
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
          HELIX_ASSERT(block.active_slot == kNoActiveSlot,
                       "free block retains an active-slot index");
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
    std::size_t occupied_slots = 0;
    for (std::size_t index = 0; index < active_slots.size(); ++index) {
      const ActiveSlot& slot = active_slots[index];
      if (slot.occupied) {
        ++occupied_slots;
        std::size_t references = 0;
        for (const Segment& segment : segments) {
          references += static_cast<std::size_t>(
              std::count_if(segment.blocks.begin(), segment.blocks.end(), [&](const Block& block) {
                return block.allocated() && block.active_slot == index;
              }));
        }
        HELIX_ASSERT(references == 1, "occupied active slot must have exactly one block");
      }
    }
    std::size_t free_slots = 0;
    std::size_t free_cursor = free_active_slot;
    while (free_cursor != kNoActiveSlot) {
      HELIX_ASSERT(free_cursor < active_slots.size(), "active-slot free list is out of range");
      HELIX_ASSERT(!active_slots[free_cursor].occupied,
                   "active-slot free list contains an occupied slot");
      ++free_slots;
      HELIX_ASSERT(free_slots <= active_slots.size(), "active-slot free list contains a cycle");
      free_cursor = active_slots[free_cursor].next_free;
    }
    HELIX_ASSERT(occupied_slots == active_count && scanned_active == active_count,
                 "active-slot count disagrees with allocated blocks");
    HELIX_ASSERT(free_slots + occupied_slots == active_slots.size(),
                 "active-slot free list does not cover every unoccupied slot");
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
    active_slots.clear();
    free_active_slot = kNoActiveSlot;
    active_count = 0;
  }

  std::shared_ptr<MemoryResource> resource;
  AllocatorConfig config;
  const std::uint64_t allocator_id;
  mutable std::mutex mutex;
  std::vector<Segment> segments;
  std::vector<ActiveSlot> active_slots;
  std::size_t free_active_slot = kNoActiveSlot;
  std::size_t active_count = 0;
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
                       std::size_t active_slot, Device device) noexcept
    : pointer_(pointer), requested_bytes_(requested_bytes), allocated_bytes_(allocated_bytes),
      alignment_(alignment), id_(id), allocator_id_(allocator_id), active_slot_(active_slot),
      device_(device) {}

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
  HELIX_CHECK(allocation.active_slot_ < impl_->active_slots.size(), ErrorCode::kInvalidArgument,
              "HelixAllocator allocation is unknown or was already freed");
  const Impl::ActiveSlot& active_slot = impl_->active_slots[allocation.active_slot_];
  HELIX_CHECK(active_slot.occupied && active_slot.allocation_id == allocation.id(),
              ErrorCode::kInvalidArgument,
              "HelixAllocator allocation is unknown or was already freed");
  const Impl::ActiveRecord record = active_slot.record;
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
  HELIX_CHECK(segment_iterator != impl_->segments.end(), ErrorCode::kInternal,
              "active allocation references a missing segment");
  Impl::Segment& segment = *segment_iterator;
  const auto block_iterator =
      std::find_if(segment.blocks.begin(), segment.blocks.end(), [&](const Impl::Block& block) {
        return block.offset == record.offset && block.allocation_id == allocation.id() &&
               block.active_slot == allocation.active_slot_;
      });
  HELIX_CHECK(block_iterator != segment.blocks.end(), ErrorCode::kInternal,
              "active allocation has no matching block");
  std::size_t block_index =
      static_cast<std::size_t>(std::distance(segment.blocks.begin(), block_iterator));
  segment.blocks[block_index].requested_bytes = 0;
  segment.blocks[block_index].allocation_id = 0;
  segment.blocks[block_index].active_slot = Impl::kNoActiveSlot;

  if (block_index != 0 && !segment.blocks[block_index - 1].allocated()) {
    Impl::Block& previous = segment.blocks[block_index - 1];
    HELIX_ASSERT(previous.offset + previous.bytes == segment.blocks[block_index].offset,
                 "allocator free blocks are not adjacent in sorted order");
    previous.bytes += segment.blocks[block_index].bytes;
    segment.blocks.erase(segment.blocks.begin() + static_cast<std::ptrdiff_t>(block_index));
    --block_index;
  }
  if (block_index + 1 < segment.blocks.size() && !segment.blocks[block_index + 1].allocated()) {
    Impl::Block& current = segment.blocks[block_index];
    HELIX_ASSERT(current.offset + current.bytes == segment.blocks[block_index + 1].offset,
                 "allocator free blocks are not adjacent in sorted order");
    current.bytes += segment.blocks[block_index + 1].bytes;
    segment.blocks.erase(segment.blocks.begin() + static_cast<std::ptrdiff_t>(block_index + 1));
  }
  impl_->bytes_allocated -= record.allocated_bytes;
  impl_->bytes_requested -= record.requested_bytes;
  impl_->release_active_slot(allocation.active_slot_);
  impl_->assert_invariants();
}

bool HelixAllocator::owns(const Allocation& allocation) const {
  std::scoped_lock lock(impl_->mutex);
  if (allocation.allocator_id_ != impl_->allocator_id) {
    return false;
  }
  if (allocation.active_slot_ >= impl_->active_slots.size()) {
    return false;
  }
  const Impl::ActiveSlot& slot = impl_->active_slots[allocation.active_slot_];
  return slot.occupied && slot.allocation_id == allocation.id() &&
         slot.record.pointer == allocation.pointer();
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
  result.active_allocation_count = impl_->active_count;
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
