#pragma once

#include "helix/core/device.hpp"

#include <cstddef>

namespace helix::memory {

class MemoryResource {
public:
  virtual ~MemoryResource() = default;

  MemoryResource(const MemoryResource&) = delete;
  MemoryResource& operator=(const MemoryResource&) = delete;
  MemoryResource(MemoryResource&&) = delete;
  MemoryResource& operator=(MemoryResource&&) = delete;

  [[nodiscard]] virtual void* allocate(std::size_t bytes, std::size_t alignment) = 0;
  virtual void deallocate(void* pointer, std::size_t bytes, std::size_t alignment) = 0;
  [[nodiscard]] virtual Device device() const noexcept = 0;

protected:
  MemoryResource() = default;
};

} // namespace helix::memory
