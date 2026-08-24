#pragma once

#include "helix/core/error.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace helix::memory::bench {

enum class TraceOperationKind : std::uint8_t {
  kAllocate,
  kDeallocate,
};

struct TraceOperation {
  TraceOperationKind kind;
  std::size_t slot;
  std::size_t bytes;
};

struct AllocationTrace {
  std::string_view name;
  std::vector<TraceOperation> operations;
  std::size_t slot_count = 0;
  std::size_t allocation_count = 0;
  std::size_t peak_live_requested_bytes = 0;
};

enum class AllocationTraceKind : std::uint8_t {
  kRandomSize,
  kSmallHeavy,
  kLargeHeavy,
  kBursty,
  kLlmLike,
};

inline constexpr std::array<AllocationTraceKind, 5> kAllocationTraceKinds{
    AllocationTraceKind::kRandomSize, AllocationTraceKind::kSmallHeavy,
    AllocationTraceKind::kLargeHeavy, AllocationTraceKind::kBursty,
    AllocationTraceKind::kLlmLike};

class TraceBuilder final {
public:
  explicit TraceBuilder(std::size_t slot_count)
      : slot_bytes_(slot_count, 0), trace_{"", {}, slot_count, 0, 0} {
    HELIX_CHECK(slot_count != 0, ErrorCode::kInvalidArgument,
                "allocator benchmark trace requires at least one slot");
  }

  void allocate(std::size_t slot, std::size_t bytes) {
    validate_slot(slot);
    HELIX_CHECK(slot_bytes_[slot] == 0, ErrorCode::kInternal,
                "allocator benchmark trace allocates an active slot");
    HELIX_CHECK(bytes != 0, ErrorCode::kInternal,
                "allocator benchmark trace contains a zero-byte allocation");
    HELIX_CHECK(bytes <= std::numeric_limits<std::size_t>::max() - live_requested_bytes_,
                ErrorCode::kOutOfRange,
                "allocator benchmark trace live-byte accounting overflow");
    slot_bytes_[slot] = bytes;
    live_requested_bytes_ += bytes;
    trace_.peak_live_requested_bytes =
        std::max(trace_.peak_live_requested_bytes, live_requested_bytes_);
    ++trace_.allocation_count;
    trace_.operations.push_back(TraceOperation{TraceOperationKind::kAllocate, slot, bytes});
  }

  void deallocate(std::size_t slot) {
    validate_slot(slot);
    HELIX_CHECK(slot_bytes_[slot] != 0, ErrorCode::kInternal,
                "allocator benchmark trace frees an inactive slot");
    live_requested_bytes_ -= slot_bytes_[slot];
    slot_bytes_[slot] = 0;
    trace_.operations.push_back(TraceOperation{TraceOperationKind::kDeallocate, slot, 0});
  }

  [[nodiscard]] bool active(std::size_t slot) const {
    validate_slot(slot);
    return slot_bytes_[slot] != 0;
  }

  [[nodiscard]] AllocationTrace finish(std::string_view name) && {
    HELIX_CHECK(!name.empty(), ErrorCode::kInternal,
                "allocator benchmark trace requires a name");
    HELIX_CHECK(live_requested_bytes_ == 0, ErrorCode::kInternal,
                "allocator benchmark trace finishes with live bytes");
    for (const std::size_t bytes : slot_bytes_) {
      HELIX_CHECK(bytes == 0, ErrorCode::kInternal,
                  "allocator benchmark trace finishes with an active slot");
    }
    HELIX_CHECK(trace_.allocation_count != 0 && !trace_.operations.empty(),
                ErrorCode::kInternal, "allocator benchmark trace contains no work");
    trace_.name = name;
    return std::move(trace_);
  }

private:
  void validate_slot(std::size_t slot) const {
    HELIX_CHECK(slot < slot_bytes_.size(), ErrorCode::kInternal,
                "allocator benchmark trace slot is out of range");
  }

  std::vector<std::size_t> slot_bytes_;
  AllocationTrace trace_;
  std::size_t live_requested_bytes_ = 0;
};

inline std::uint64_t next_random(std::uint64_t& state) noexcept {
  state ^= state >> 12U;
  state ^= state << 25U;
  state ^= state >> 27U;
  return state * 0x2545F4914F6CDD1DULL;
}

inline std::size_t select_slot(const TraceBuilder& builder, std::size_t slot_count,
                               bool active, std::uint64_t random) {
  const std::size_t start = static_cast<std::size_t>(random % slot_count);
  for (std::size_t offset = 0; offset < slot_count; ++offset) {
    const std::size_t slot = (start + offset) % slot_count;
    if (builder.active(slot) == active) {
      return slot;
    }
  }
  throw_error(ErrorCode::kInternal, "allocator benchmark trace cannot select a matching slot");
}

template <typename SizeGenerator>
AllocationTrace make_churn_trace(std::string_view name, std::size_t slot_count,
                                 std::size_t target_allocations, std::uint64_t seed,
                                 SizeGenerator&& size_generator) {
  TraceBuilder builder(slot_count);
  std::size_t active_count = 0;
  std::size_t allocations = 0;
  while (allocations < target_allocations) {
    const std::uint64_t decision = next_random(seed);
    const bool should_allocate =
        active_count == 0 || (active_count != slot_count && decision % 100U < 61U);
    if (should_allocate) {
      const std::size_t slot = select_slot(builder, slot_count, false, next_random(seed));
      builder.allocate(slot, size_generator(seed));
      ++active_count;
      ++allocations;
    } else {
      const std::size_t slot = select_slot(builder, slot_count, true, next_random(seed));
      builder.deallocate(slot);
      --active_count;
    }
  }
  while (active_count != 0) {
    const std::size_t slot = select_slot(builder, slot_count, true, next_random(seed));
    builder.deallocate(slot);
    --active_count;
  }
  return std::move(builder).finish(name);
}

inline AllocationTrace make_random_size_trace() {
  return make_churn_trace("RandomSize", 64, 512, 0x9E3779B97F4A7C15ULL,
                          [](std::uint64_t& random) {
                            constexpr std::size_t kMaximumBytes = 1U << 20;
                            return 64U + static_cast<std::size_t>(next_random(random) %
                                                                  (kMaximumBytes - 63U));
                          });
}

inline AllocationTrace make_small_heavy_trace() {
  return make_churn_trace("SmallHeavy", 64, 768, 0xD1B54A32D192ED03ULL,
                          [](std::uint64_t& random) {
                            constexpr std::array<std::size_t, 7> kSmall{
                                64, 128, 256, 512, 1'024, 2'048, 4'096};
                            constexpr std::array<std::size_t, 4> kOccasional{
                                8'192, 16'384, 32'768, 65'536};
                            const bool small = next_random(random) % 100U < 85U;
                            if (small) {
                              return kSmall[static_cast<std::size_t>(next_random(random) %
                                                                      kSmall.size())];
                            }
                            return kOccasional[static_cast<std::size_t>(
                                next_random(random) % kOccasional.size())];
                          });
}

inline AllocationTrace make_large_heavy_trace() {
  return make_churn_trace("LargeHeavy", 16, 192, 0x94D049BB133111EBULL,
                          [](std::uint64_t& random) {
                            constexpr std::array<std::size_t, 4> kLarge{
                                512U << 10, 1U << 20, 2U << 20, 4U << 20};
                            constexpr std::array<std::size_t, 4> kOccasional{
                                64U << 10, 128U << 10, 256U << 10, 384U << 10};
                            const bool large = next_random(random) % 100U < 75U;
                            const auto& sizes = large ? kLarge : kOccasional;
                            return sizes[static_cast<std::size_t>(next_random(random) %
                                                                  sizes.size())];
                          });
}

inline AllocationTrace make_bursty_trace() {
  constexpr std::size_t kSlots = 64;
  constexpr std::size_t kBursts = 12;
  TraceBuilder builder(kSlots);
  for (std::size_t burst = 0; burst < kBursts; ++burst) {
    for (std::size_t slot = 0; slot < kSlots; ++slot) {
      const std::size_t bytes = 128U << ((slot + burst) % 9U);
      builder.allocate(slot, bytes);
    }
    for (std::size_t slot = 0; slot < kSlots; slot += 2) {
      builder.deallocate(slot);
    }
    for (std::size_t slot = 1; slot < kSlots; slot += 2) {
      builder.deallocate(slot);
    }
  }
  return std::move(builder).finish("Bursty");
}

inline AllocationTrace make_llm_like_trace() {
  constexpr std::size_t kWeightStart = 0;
  constexpr std::size_t kWeightCount = 4;
  constexpr std::size_t kKvStart = 4;
  constexpr std::size_t kKvGroups = 4;
  constexpr std::size_t kKvPerGroup = 4;
  constexpr std::size_t kWorkspaceStart = 20;
  constexpr std::array<std::size_t, 5> kWorkspaceBytes{
      1U << 20, 3U << 20, 4U << 20, 8U << 20, 16U << 20};
  constexpr std::size_t kRounds = 12;

  TraceBuilder builder(kWorkspaceStart + kWorkspaceBytes.size());
  for (std::size_t index = 0; index < kWeightCount; ++index) {
    builder.allocate(kWeightStart + index, 8U << 20);
  }
  for (std::size_t round = 0; round < kRounds; ++round) {
    const std::size_t group = round % kKvGroups;
    const std::size_t group_start = kKvStart + group * kKvPerGroup;
    if (round >= kKvGroups) {
      for (std::size_t index = 0; index < kKvPerGroup; ++index) {
        builder.deallocate(group_start + index);
      }
    }
    for (std::size_t index = 0; index < kKvPerGroup; ++index) {
      builder.allocate(group_start + index, 256U << 10);
    }
    for (std::size_t index = 0; index < kWorkspaceBytes.size(); ++index) {
      builder.allocate(kWorkspaceStart + index, kWorkspaceBytes[index]);
    }
    for (std::size_t index = kWorkspaceBytes.size(); index != 0; --index) {
      builder.deallocate(kWorkspaceStart + index - 1);
    }
  }
  for (std::size_t index = 0; index < kKvGroups * kKvPerGroup; ++index) {
    builder.deallocate(kKvStart + index);
  }
  for (std::size_t index = kWeightCount; index != 0; --index) {
    builder.deallocate(kWeightStart + index - 1);
  }
  return std::move(builder).finish("LlmLike");
}

inline const AllocationTrace& allocation_trace(AllocationTraceKind kind) {
  static const AllocationTrace random_size = make_random_size_trace();
  static const AllocationTrace small_heavy = make_small_heavy_trace();
  static const AllocationTrace large_heavy = make_large_heavy_trace();
  static const AllocationTrace bursty = make_bursty_trace();
  static const AllocationTrace llm_like = make_llm_like_trace();
  switch (kind) {
  case AllocationTraceKind::kRandomSize:
    return random_size;
  case AllocationTraceKind::kSmallHeavy:
    return small_heavy;
  case AllocationTraceKind::kLargeHeavy:
    return large_heavy;
  case AllocationTraceKind::kBursty:
    return bursty;
  case AllocationTraceKind::kLlmLike:
    return llm_like;
  }
  throw_error(ErrorCode::kInvalidArgument, "unknown allocator benchmark trace kind");
}

} // namespace helix::memory::bench
