#pragma once

#include "helix/memory/memory_resource.hpp"

namespace helix::memory {

class CudaMemoryResource final : public MemoryResource {
public:
  explicit CudaMemoryResource(int device_index);

  [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment) override;
  void deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override;
  [[nodiscard]] Device device() const noexcept override;

private:
  int device_index_;
};

} // namespace helix::memory
