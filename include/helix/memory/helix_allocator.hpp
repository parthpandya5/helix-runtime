#pragma once

#include "helix/memory/memory_resource.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace helix::memory {

struct AllocatorConfig {
  std::size_t segment_bytes = 64U * 1024U * 1024U;
  std::size_t minimum_alignment = 256;
  std::size_t maximum_reserved_bytes = std::numeric_limits<std::size_t>::max();
};

class Allocation final {
public:
  Allocation() = default;

  [[nodiscard]] explicit operator bool() const noexcept {
    return pointer_ != nullptr;
  }
  [[nodiscard]] void* pointer() const noexcept {
    return pointer_;
  }
  [[nodiscard]] std::size_t requested_bytes() const noexcept {
    return requested_bytes_;
  }
  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return allocated_bytes_;
  }
  [[nodiscard]] std::size_t alignment() const noexcept {
    return alignment_;
  }
  [[nodiscard]] std::uint64_t id() const noexcept {
    return id_;
  }
  [[nodiscard]] Device device() const noexcept {
    return device_;
  }

private:
  friend class HelixAllocator;

  Allocation(void* pointer, std::size_t requested_bytes, std::size_t allocated_bytes,
             std::size_t alignment, std::uint64_t id, std::uint64_t allocator_id,
             Device device) noexcept;

  void* pointer_ = nullptr;
  std::size_t requested_bytes_ = 0;
  std::size_t allocated_bytes_ = 0;
  std::size_t alignment_ = 0;
  std::uint64_t id_ = 0;
  std::uint64_t allocator_id_ = 0;
  Device device_ = Device::cpu();
};

struct AllocatorStats {
  std::size_t bytes_reserved = 0;
  std::size_t bytes_allocated = 0;
  std::size_t bytes_requested = 0;
  std::size_t bytes_free = 0;
  std::size_t peak_bytes_allocated = 0;
  std::size_t largest_free_block = 0;
  std::size_t segment_count = 0;
  std::size_t active_allocation_count = 0;
  std::uint64_t allocation_count = 0;
  std::uint64_t reuse_count = 0;
  std::uint64_t underlying_allocations = 0;
  std::uint64_t underlying_deallocations = 0;
  double external_fragmentation = 0.0;
  double internal_fragmentation = 0.0;
};

class HelixAllocator final {
public:
  explicit HelixAllocator(std::shared_ptr<MemoryResource> resource, AllocatorConfig config = {});
  ~HelixAllocator();

  HelixAllocator(const HelixAllocator&) = delete;
  HelixAllocator& operator=(const HelixAllocator&) = delete;
  HelixAllocator(HelixAllocator&&) = delete;
  HelixAllocator& operator=(HelixAllocator&&) = delete;

  [[nodiscard]] Allocation allocate(std::size_t bytes, std::size_t alignment = 0);
  void deallocate(const Allocation& allocation);
  [[nodiscard]] bool owns(const Allocation& allocation) const;
  [[nodiscard]] AllocatorStats stats() const;
  std::size_t release_cached();

  [[nodiscard]] Device device() const noexcept;
  [[nodiscard]] const AllocatorConfig& config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace helix::memory
