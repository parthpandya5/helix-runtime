#pragma once

#include "helix/core/device.hpp"
#include "helix/core/dtype.hpp"
#include "helix/core/error.hpp"
#include "helix/tensor/shape.hpp"
#include "helix/tensor/storage.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>

namespace helix {

class Tensor final {
public:
  explicit Tensor(Shape shape, DType dtype = DType::kFloat32, Device device = Device::cpu());

  [[nodiscard]] static Tensor empty(Shape shape, DType dtype = DType::kFloat32,
                                    Device device = Device::cpu());
  [[nodiscard]] static Tensor zeros(Shape shape, DType dtype = DType::kFloat32,
                                    Device device = Device::cpu());
  [[nodiscard]] static Tensor random_uniform(Shape shape, std::uint64_t seed, float lower = -1.0F,
                                             float upper = 1.0F, DType dtype = DType::kFloat32,
                                             Device device = Device::cpu());
  [[nodiscard]] static Tensor from_values(Shape shape, std::span<const float> values,
                                          Device device = Device::cpu());
  [[nodiscard]] static Tensor from_values(Shape shape, std::span<const std::uint32_t> values,
                                          Device device = Device::cpu());
  [[nodiscard]] static Tensor
  from_read_only_cpu_buffer(Shape shape, DType dtype, const void* data,
                            std::shared_ptr<const void> owner,
                            std::size_t alignment = kDefaultStorageAlignment);
  [[nodiscard]] static Tensor
  from_mutable_cpu_buffer(Shape shape, DType dtype, void* data, std::size_t storage_bytes,
                          std::shared_ptr<void> owner,
                          std::size_t alignment = kDefaultStorageAlignment);
  // Low-level CUDA interop with the same pointer-provenance contract as
  // Storage::wrap_mutable_cuda. Prefer a subsystem-specific owning factory.
  [[nodiscard]] static Tensor
  from_mutable_cuda_buffer(Shape shape, DType dtype, void* data, std::size_t storage_bytes,
                           Device device, std::shared_ptr<void> owner,
                           std::size_t alignment = kDefaultStorageAlignment);

  [[nodiscard]] const Shape& shape() const noexcept {
    return shape_;
  }
  [[nodiscard]] const Strides& strides() const noexcept {
    return strides_;
  }
  [[nodiscard]] std::size_t rank() const noexcept {
    return shape_.size();
  }
  [[nodiscard]] DType dtype() const noexcept {
    return dtype_;
  }
  [[nodiscard]] Device device() const noexcept {
    return device_;
  }
  [[nodiscard]] std::size_t numel() const noexcept {
    return numel_;
  }
  [[nodiscard]] std::size_t nbytes() const noexcept {
    return nbytes_;
  }
  [[nodiscard]] std::size_t allocated_bytes() const noexcept {
    return storage_->size_bytes();
  }
  [[nodiscard]] std::size_t storage_offset_bytes() const noexcept {
    return storage_offset_bytes_;
  }
  [[nodiscard]] bool is_contiguous() const;
  [[nodiscard]] bool is_read_only() const noexcept {
    return storage_->is_read_only();
  }

  [[nodiscard]] void* data();
  [[nodiscard]] const void* data() const noexcept;

  template <typename T> [[nodiscard]] T* data_as() {
    using Value = std::remove_cv_t<T>;
    static_assert(std::is_same_v<Value, float> || std::is_same_v<Value, std::uint32_t>,
                  "typed Tensor access supports only float and uint32_t");
    constexpr DType expected = std::is_same_v<Value, float> ? DType::kFloat32 : DType::kUInt32;
    HELIX_CHECK(dtype_ == expected, ErrorCode::kInvalidArgument,
                "requested C++ element type does not match the Tensor dtype");
    return static_cast<T*>(data());
  }

  template <typename T> [[nodiscard]] const T* data_as() const {
    using Value = std::remove_cv_t<T>;
    static_assert(std::is_same_v<Value, float> || std::is_same_v<Value, std::uint32_t>,
                  "typed Tensor access supports only float and uint32_t");
    constexpr DType expected = std::is_same_v<Value, float> ? DType::kFloat32 : DType::kUInt32;
    HELIX_CHECK(dtype_ == expected, ErrorCode::kInvalidArgument,
                "requested C++ element type does not match the Tensor dtype");
    return static_cast<const T*>(data());
  }

  Tensor& zero_();
  Tensor& random_uniform_(std::uint64_t seed, float lower = -1.0F, float upper = 1.0F);
  Tensor& copy_from(const Tensor& source);

  [[nodiscard]] Tensor clone() const;
  [[nodiscard]] Tensor to(Device target) const;
  [[nodiscard]] Tensor reshape(Shape new_shape) const;
  [[nodiscard]] Tensor view(Shape new_shape) const;

private:
  Tensor(Shape shape, Strides strides, DType dtype, Device device, std::shared_ptr<Storage> storage,
         std::size_t storage_offset_bytes);

  void validate_storage_bounds() const;
  void require_cpu(std::string_view operation) const;
  void require_writable(std::string_view operation) const;

  Shape shape_;
  Strides strides_;
  DType dtype_;
  Device device_;
  std::size_t numel_;
  std::size_t nbytes_;
  std::shared_ptr<Storage> storage_;
  std::size_t storage_offset_bytes_;
};

} // namespace helix
