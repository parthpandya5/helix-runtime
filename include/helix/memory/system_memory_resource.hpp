#pragma once

#include "helix/memory/memory_resource.hpp"

namespace helix::memory {

class SystemMemoryResource final : public MemoryResource {
public:
  [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment) override;
  void deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
  [[nodiscard]] Device device() const noexcept override;
};

} // namespace helix::memory
