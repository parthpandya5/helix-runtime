#include "helix/tensor/tensor.hpp"

#include "helix/core/config.hpp"

#if HELIX_ENABLE_CUDA
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/memory.hpp"
#endif

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
  const Device initialization_device = device.is_cpu() ? device : Device::cpu();
  Tensor result(std::move(shape), dtype, initialization_device);
  result.random_uniform_(seed, lower, upper);
  return device == initialization_device ? result : result.to(device);
}

Tensor Tensor::from_values(Shape shape, std::span<const float> values, Device device) {
  HELIX_CHECK(values.size() == checked_numel(shape), ErrorCode::kInvalidArgument,
              "value count does not match tensor element count");
  Tensor result(std::move(shape), DType::kFloat32, Device::cpu());
  if (!values.empty()) {
    std::memcpy(result.data(), values.data(), result.nbytes());
  }
  return device.is_cpu() ? result : result.to(device);
}

Tensor Tensor::from_values(Shape shape, std::span<const std::uint32_t> values, Device device) {
  HELIX_CHECK(values.size() == checked_numel(shape), ErrorCode::kInvalidArgument,
              "value count does not match tensor element count");
  Tensor result(std::move(shape), DType::kUInt32, Device::cpu());
  if (!values.empty()) {
    std::memcpy(result.data(), values.data(), result.nbytes());
  }
  return device.is_cpu() ? result : result.to(device);
}

Tensor Tensor::from_read_only_cpu_buffer(Shape shape, DType dtype, const void* data,
                                         std::shared_ptr<const void> owner, std::size_t alignment) {
  const std::size_t size_bytes = checked_nbytes(shape, dtype);
  auto storage = Storage::wrap_read_only_cpu(data, size_bytes, std::move(owner), alignment);
  auto strides = contiguous_strides(shape);
  return Tensor(std::move(shape), std::move(strides), dtype, Device::cpu(), std::move(storage), 0);
}

Tensor Tensor::from_mutable_cpu_buffer(Shape shape, DType dtype, void* data,
                                       std::size_t storage_bytes, std::shared_ptr<void> owner,
                                       std::size_t alignment) {
  const std::size_t logical_bytes = checked_nbytes(shape, dtype);
  HELIX_CHECK(storage_bytes >= logical_bytes, ErrorCode::kOutOfRange,
              "mutable CPU buffer is smaller than the Tensor logical byte count");
  auto storage = Storage::wrap_mutable_cpu(data, storage_bytes, std::move(owner), alignment);
  auto strides = contiguous_strides(shape);
  return Tensor(std::move(shape), std::move(strides), dtype, Device::cpu(), std::move(storage), 0);
}

Tensor Tensor::from_mutable_cuda_buffer(Shape shape, DType dtype, void* data,
                                        std::size_t storage_bytes, Device device,
                                        std::shared_ptr<void> owner, std::size_t alignment) {
  HELIX_CHECK(device.is_cuda(), ErrorCode::kInvalidArgument,
              "mutable CUDA Tensor buffer requires a CUDA device");
  const std::size_t logical_bytes = checked_nbytes(shape, dtype);
  HELIX_CHECK(storage_bytes >= logical_bytes, ErrorCode::kOutOfRange,
              "mutable CUDA buffer is smaller than the Tensor logical byte count");
  auto storage =
      Storage::wrap_mutable_cuda(data, storage_bytes, device, std::move(owner), alignment);
  auto strides = contiguous_strides(shape);
  return Tensor(std::move(shape), std::move(strides), dtype, device, std::move(storage), 0);
}

bool Tensor::is_contiguous() const {
  return has_contiguous_strides(shape_, strides_);
}

void* Tensor::data() {
  if (static_cast<const Storage&>(*storage_).data() == nullptr) {
    return nullptr;
  }
  return static_cast<void*>(static_cast<std::byte*>(storage_->mutable_data()) +
                            storage_offset_bytes_);
}

const void* Tensor::data() const noexcept {
  if (storage_->data() == nullptr) {
    return nullptr;
  }
  return static_cast<const void*>(static_cast<const std::byte*>(storage_->data()) +
                                  storage_offset_bytes_);
}

Tensor& Tensor::zero_() {
  require_writable("Tensor::zero_");
  HELIX_CHECK(is_contiguous(), ErrorCode::kUnsupported,
              "Tensor::zero_ currently requires contiguous storage");
  if (device_.is_cpu() && nbytes_ != 0) {
    std::memset(data(), 0, nbytes_);
  } else if (device_.is_cuda()) {
#if HELIX_ENABLE_CUDA
    cuda::DeviceGuard guard(device_.index());
    cuda::set_device_bytes(data(), 0, nbytes_);
#else
    throw_error(ErrorCode::kUnsupported, "Tensor::zero_ requires a CUDA-enabled build");
#endif
  }
  return *this;
}

Tensor& Tensor::random_uniform_(std::uint64_t seed, float lower, float upper) {
  require_writable("Tensor::random_uniform_");
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
  require_writable("Tensor::copy_from");
  HELIX_CHECK(shape_ == source.shape_, ErrorCode::kInvalidArgument,
              "tensor copy requires equal shapes");
  HELIX_CHECK(dtype_ == source.dtype_, ErrorCode::kInvalidArgument,
              "tensor copy requires equal dtypes");
  HELIX_CHECK(device_ == source.device_, ErrorCode::kInvalidArgument,
              "tensor copy requires equal devices; use Tensor::to for transfers");
  HELIX_CHECK(is_contiguous() && source.is_contiguous(), ErrorCode::kUnsupported,
              "tensor copy currently requires contiguous tensors");
  if (nbytes_ == 0 || data() == source.data()) {
    return *this;
  }
  if (device_.is_cpu()) {
    std::memmove(data(), source.data(), nbytes_);
  } else {
#if HELIX_ENABLE_CUDA
    cuda::DeviceGuard guard(device_.index());
    cuda::copy_device_to_device(data(), source.data(), nbytes_);
#else
    throw_error(ErrorCode::kUnsupported, "Tensor::copy_from requires a CUDA-enabled build");
#endif
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
#if HELIX_ENABLE_CUDA
  Tensor result(shape_, dtype_, target);
  if (nbytes_ == 0) {
    return result;
  }
  if (device_.is_cpu() && target.is_cuda()) {
    cuda::DeviceGuard guard(target.index());
    cuda::copy_host_to_device(result.data(), data(), nbytes_);
    return result;
  }
  if (device_.is_cuda() && target.is_cpu()) {
    cuda::DeviceGuard guard(device_.index());
    cuda::copy_device_to_host(result.data(), data(), nbytes_);
    return result;
  }
  if (device_.is_cuda() && target.is_cuda()) {
    cuda::copy_device_to_device_peer(result.data(), target.index(), data(), device_.index(),
                                     nbytes_);
    return result;
  }
#endif
  std::ostringstream message;
  message << "device transfer from " << device_ << " to " << target
          << " is unavailable in this build";
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

void Tensor::require_writable(std::string_view operation) const {
  HELIX_CHECK(!is_read_only(), ErrorCode::kInvalidArgument,
              std::string(operation) + " cannot mutate read-only storage");
}

} // namespace helix
