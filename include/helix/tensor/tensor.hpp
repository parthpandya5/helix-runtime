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

  [[nodiscard]] void* data() noexcept;
  [[nodiscard]] const void* data() const noexcept;

  template <typename T> [[nodiscard]] T* data_as() {
    using Value = std::remove_cv_t<T>;
    static_assert(std::is_same_v<Value, float>,
                  "Milestone 1 exposes typed access only for Float32");
    HELIX_CHECK(dtype_ == DType::kFloat32, ErrorCode::kInvalidArgument,
                "typed float access requires a float32 tensor");
    return static_cast<T*>(data());
  }

  template <typename T> [[nodiscard]] const T* data_as() const {
    using Value = std::remove_cv_t<T>;
    static_assert(std::is_same_v<Value, float>,
                  "Milestone 1 exposes typed access only for Float32");
    HELIX_CHECK(dtype_ == DType::kFloat32, ErrorCode::kInvalidArgument,
                "typed float access requires a float32 tensor");
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
