#include "helix/kv_cache/cuda_block_table.hpp"

#include "helix/core/error.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>

namespace helix::kv_cache::cuda {
namespace {

std::string table_error(const char* requirement) {
  return std::string("kv_cache::cuda::PagedBlockTable: ") + requirement;
}

} // namespace

PagedBlockTable::PagedBlockTable(std::span<const std::uint32_t> physical_blocks,
                                 std::size_t physical_block_count, int device_index)
    : logical_block_count_(physical_blocks.size()), physical_block_count_(physical_block_count),
      device_index_(device_index) {
  HELIX_CHECK(physical_block_count != 0 &&
                  physical_block_count <= std::numeric_limits<std::uint32_t>::max(),
              ErrorCode::kInvalidArgument,
              table_error("physical block capacity must fit the uint32 block-ID representation"));
  HELIX_CHECK(device_index >= 0, ErrorCode::kInvalidArgument,
              table_error("device index must be non-negative"));
  HELIX_CHECK(
      physical_blocks.size() <= std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t),
      ErrorCode::kResourceExhausted, table_error("block-table byte count overflows size_t"));
  for (const std::uint32_t physical_block : physical_blocks) {
    HELIX_CHECK(static_cast<std::size_t>(physical_block) < physical_block_count,
                ErrorCode::kOutOfRange,
                table_error("physical block ID exceeds the packed block capacity"));
  }

  if (physical_blocks.empty()) {
    return;
  }
  device_blocks_.emplace(size_bytes(), device_index_);
  helix::cuda::DeviceGuard guard(device_index_);
  helix::cuda::copy_host_to_device(device_blocks_->data(), physical_blocks.data(), size_bytes());
}

PagedBlockTable::PagedBlockTable(PagedBlockTable&& other) noexcept
    : device_blocks_(std::move(other.device_blocks_)),
      logical_block_count_(std::exchange(other.logical_block_count_, 0)),
      physical_block_count_(std::exchange(other.physical_block_count_, 0)),
      device_index_(std::exchange(other.device_index_, -1)) {
  other.device_blocks_.reset();
}

PagedBlockTable& PagedBlockTable::operator=(PagedBlockTable&& other) noexcept {
  if (this != &other) {
    device_blocks_ = std::move(other.device_blocks_);
    logical_block_count_ = std::exchange(other.logical_block_count_, 0);
    physical_block_count_ = std::exchange(other.physical_block_count_, 0);
    device_index_ = std::exchange(other.device_index_, -1);
    other.device_blocks_.reset();
  }
  return *this;
}

bool PagedBlockTable::valid() const noexcept {
  if (physical_block_count_ == 0 || device_index_ < 0) {
    return false;
  }
  if (logical_block_count_ == 0) {
    return !device_blocks_.has_value();
  }
  return device_blocks_.has_value() && device_blocks_->data() != nullptr;
}

const std::uint32_t* PagedBlockTable::data() const noexcept {
  return device_blocks_.has_value() ? static_cast<const std::uint32_t*>(device_blocks_->data())
                                    : nullptr;
}

} // namespace helix::kv_cache::cuda
