#include "helix/tensor/tensor.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <utility>

namespace helix {
namespace {

std::string operation_device_message(std::string_view operation, const Device& device) {
  std::ostringstream message;
  message << operation << " requires CPU storage, but tensor is on " << device;
  return message.str();
}

} // namespace

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)), dtype_(dtype),
      device_(device), numel_(checked_numel(shape_)), nbytes_(checked_nbytes(shape_, dtype_)),
      storage_(Storage::allocate(nbytes_, device_)), storage_offset_bytes_(0) {
  validate_storage_bounds();
}

Tensor::Tensor(Shape shape, Strides strides, DType dtype, Device device,
               std::shared_ptr<Storage> storage, std::size_t storage_offset_bytes)
    : shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype), device_(device),
      numel_(checked_numel(shape_)), nbytes_(checked_nbytes(shape_, dtype_)),
      storage_(std::move(storage)), storage_offset_bytes_(storage_offset_bytes) {
  HELIX_CHECK(storage_ != nullptr, ErrorCode::kInvalidArgument, "tensor storage must not be null");
  HELIX_CHECK(storage_->device() == device_, ErrorCode::kInvalidArgument,
              "tensor device does not match its storage device");
  HELIX_CHECK(strides_.size() == shape_.size(), ErrorCode::kInvalidArgument,
              "tensor shape and stride ranks differ");
  validate_storage_bounds();
}

Tensor Tensor::empty(Shape shape, DType dtype, Device device) {
  return Tensor(std::move(shape), dtype, device);
}

Tensor Tensor::zeros(Shape shape, DType dtype, Device device) {
  Tensor result(std::move(shape), dtype, device);
  result.zero_();
  return result;
}

Tensor Tensor::random_uniform(Shape shape, std::uint64_t seed, float lower, float upper,
                              DType dtype, Device device) {
  Tensor result(std::move(shape), dtype, device);
  result.random_uniform_(seed, lower, upper);
  return result;
}

Tensor Tensor::from_values(Shape shape, std::span<const float> values, Device device) {
  HELIX_CHECK(values.size() == checked_numel(shape), ErrorCode::kInvalidArgument,
              "value count does not match tensor element count");
  Tensor result(std::move(shape), DType::kFloat32, device);
  result.require_cpu("Tensor::from_values");
  if (!values.empty()) {
    std::memcpy(result.data(), values.data(), result.nbytes());
  }
  return result;
}

bool Tensor::is_contiguous() const {
  return has_contiguous_strides(shape_, strides_);
}

void* Tensor::data() noexcept {
  if (storage_->data() == nullptr) {
    return nullptr;
  }
  return static_cast<void*>(static_cast<std::byte*>(storage_->data()) + storage_offset_bytes_);
}

const void* Tensor::data() const noexcept {
  if (storage_->data() == nullptr) {
    return nullptr;
  }
  return static_cast<const void*>(static_cast<const std::byte*>(storage_->data()) +
                                  storage_offset_bytes_);
}

Tensor& Tensor::zero_() {
  require_cpu("Tensor::zero_");
  HELIX_CHECK(is_contiguous(), ErrorCode::kUnsupported,
              "Tensor::zero_ currently requires contiguous storage");
  if (nbytes_ != 0) {
    std::memset(data(), 0, nbytes_);
  }
  return *this;
}

Tensor& Tensor::random_uniform_(std::uint64_t seed, float lower, float upper) {
  require_cpu("Tensor::random_uniform_");
  HELIX_CHECK(dtype_ == DType::kFloat32, ErrorCode::kUnsupported,
              "random initialization currently supports float32 only");
  HELIX_CHECK(is_contiguous(), ErrorCode::kUnsupported,
              "random initialization currently requires contiguous storage");
  HELIX_CHECK(std::isfinite(lower) && std::isfinite(upper), ErrorCode::kInvalidArgument,
              "random bounds must be finite");
  HELIX_CHECK(lower <= upper, ErrorCode::kInvalidArgument,
              "random lower bound must not exceed upper bound");

  std::mt19937_64 generator(seed);
  auto* values = data_as<float>();
  constexpr double kUnitScale = 1.0 / 16777216.0;
  const double range = static_cast<double>(upper) - static_cast<double>(lower);
  for (std::size_t index = 0; index < numel_; ++index) {
    const auto random_bits = generator() >> 40U;
    const double unit = static_cast<double>(random_bits) * kUnitScale;
    values[index] = static_cast<float>(static_cast<double>(lower) + range * unit);
  }
  return *this;
}

Tensor& Tensor::copy_from(const Tensor& source) {
  HELIX_CHECK(shape_ == source.shape_, ErrorCode::kInvalidArgument,
              "tensor copy requires equal shapes");
  HELIX_CHECK(dtype_ == source.dtype_, ErrorCode::kInvalidArgument,
              "tensor copy requires equal dtypes");
  HELIX_CHECK(device_ == source.device_, ErrorCode::kInvalidArgument,
              "tensor copy requires equal devices");
  HELIX_CHECK(is_contiguous() && source.is_contiguous(), ErrorCode::kUnsupported,
              "tensor copy currently requires contiguous tensors");
  require_cpu("Tensor::copy_from");
  if (nbytes_ != 0 && data() != source.data()) {
    std::memmove(data(), source.data(), nbytes_);
  }
  return *this;
}

Tensor Tensor::clone() const {
  Tensor result(shape_, dtype_, device_);
  result.copy_from(*this);
  return result;
}

Tensor Tensor::to(Device target) const {
  if (target == device_) {
    return clone();
  }
  std::ostringstream message;
  message << "device transfer from " << device_ << " to " << target
          << " is unavailable until the CUDA runtime layer is implemented";
  throw_error(ErrorCode::kUnsupported, message.str());
}

Tensor Tensor::reshape(Shape new_shape) const {
  HELIX_CHECK(is_contiguous(), ErrorCode::kUnsupported,
              "reshape without copying requires a contiguous tensor");
  const auto new_numel = checked_numel(new_shape);
  if (new_numel != numel_) {
    std::ostringstream message;
    message << "cannot reshape tensor from " << shape_string(shape_) << " to "
            << shape_string(new_shape) << ": element counts differ (" << numel_ << " vs "
            << new_numel << ')';
    throw_error(ErrorCode::kInvalidArgument, message.str());
  }
  auto new_strides = contiguous_strides(new_shape);
  return Tensor(std::move(new_shape), std::move(new_strides), dtype_, device_, storage_,
                storage_offset_bytes_);
}

Tensor Tensor::view(Shape new_shape) const {
  return reshape(std::move(new_shape));
}

void Tensor::validate_storage_bounds() const {
  HELIX_CHECK(storage_offset_bytes_ <= storage_->size_bytes(), ErrorCode::kOutOfRange,
              "tensor storage offset exceeds allocation size");
  const auto available_bytes = storage_->size_bytes() - storage_offset_bytes_;
  HELIX_CHECK(nbytes_ <= available_bytes, ErrorCode::kOutOfRange,
              "tensor view exceeds its backing storage");
}

void Tensor::require_cpu(std::string_view operation) const {
  HELIX_CHECK(device_.is_cpu(), ErrorCode::kUnsupported,
              operation_device_message(operation, device_));
}

} // namespace helix
