#pragma once

#include <cstddef>

namespace helix::cuda {

class Stream;

[[nodiscard]] void* allocate_device_bytes(std::size_t size_bytes, int device_index);
void free_device_bytes_noexcept(void* pointer, int device_index) noexcept;

void copy_host_to_device(void* destination, const void* source, std::size_t size_bytes);
void copy_device_to_host(void* destination, const void* source, std::size_t size_bytes);
void copy_device_to_device(void* destination, const void* source, std::size_t size_bytes);
void copy_device_to_device_peer(void* destination, int destination_device, const void* source,
                                int source_device, std::size_t size_bytes);
void set_device_bytes(void* destination, int byte_value, std::size_t size_bytes);

void copy_host_to_device_async(void* destination, const void* source, std::size_t size_bytes,
                               const Stream& stream);
void copy_device_to_host_async(void* destination, const void* source, std::size_t size_bytes,
                               const Stream& stream);
void copy_device_to_device_async(void* destination, const void* source, std::size_t size_bytes,
                                 const Stream& stream);
void set_device_bytes_async(void* destination, int byte_value, std::size_t size_bytes,
                            const Stream& stream);

class Buffer final {
public:
  explicit Buffer(std::size_t size_bytes, int device_index = -1);
  ~Buffer();

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  [[nodiscard]] void* data() noexcept {
    return pointer_;
  }
  [[nodiscard]] const void* data() const noexcept {
    return pointer_;
  }
  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return size_bytes_;
  }
  [[nodiscard]] int device_index() const noexcept {
    return device_index_;
  }
  [[nodiscard]] void* release() noexcept;

private:
  void destroy() noexcept;

  void* pointer_{nullptr};
  std::size_t size_bytes_{0};
  int device_index_{-1};
};

} // namespace helix::cuda
