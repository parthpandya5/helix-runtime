#include "helix/core/config.hpp"
#include "helix/tensor/shape.hpp"
#include "helix/tensor/storage.hpp"
#include "helix/tensor/tensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace helix {
namespace {

std::vector<float> values_of(const Tensor& tensor) {
  const auto* begin = tensor.data_as<float>();
  return std::vector<float>(begin, begin + tensor.numel());
}

TEST(ShapeTest, ComputesScalarAndContiguousMetadata) {
  EXPECT_EQ(checked_numel(Shape{}), 1U);
  EXPECT_EQ(contiguous_strides(Shape{}), Strides{});

  const Shape shape{2, 3, 4};
  EXPECT_EQ(checked_numel(shape), 24U);
  EXPECT_EQ(checked_nbytes(shape, DType::kFloat32), 96U);
  EXPECT_EQ(contiguous_strides(shape), (Strides{12, 4, 1}));
  EXPECT_TRUE(has_contiguous_strides(shape, Strides{12, 4, 1}));
  EXPECT_FALSE(has_contiguous_strides(shape, Strides{12, 1, 3}));
  EXPECT_EQ(shape_string(shape), "[2, 3, 4]");
}

TEST(ShapeTest, DefinesZeroSizedTensorMetadata) {
  const Shape shape{2, 0, 3};
  EXPECT_EQ(checked_numel(shape), 0U);
  EXPECT_EQ(contiguous_strides(shape), (Strides{0, 3, 1}));
}

TEST(ShapeTest, RejectsNegativeDimensionsAndOverflow) {
  EXPECT_THROW(static_cast<void>(checked_numel(Shape{2, -1, 3})), Error);
  EXPECT_THROW(static_cast<void>(checked_numel(Shape{std::numeric_limits<Dimension>::max(), 3})),
               Error);
  EXPECT_THROW(static_cast<void>(
                   checked_nbytes(Shape{std::numeric_limits<Dimension>::max()}, DType::kFloat32)),
               Error);
}

TEST(StorageTest, AllocatesAlignedCpuMemory) {
  constexpr std::size_t kBytes = 257;
  constexpr std::size_t kAlignment = 128;
  const auto storage = Storage::allocate(kBytes, Device::cpu(), kAlignment);

  ASSERT_NE(storage->data(), nullptr);
  EXPECT_EQ(storage->size_bytes(), kBytes);
  EXPECT_EQ(storage->alignment(), kAlignment);
  EXPECT_EQ(storage->device(), Device::cpu());
  const auto address = reinterpret_cast<std::uintptr_t>(storage->data());
  EXPECT_EQ(address % kAlignment, 0U);
}

TEST(StorageTest, HandlesEmptyStorageWithoutSentinelAllocation) {
  const auto storage = Storage::allocate(0, Device::cpu());
  EXPECT_EQ(storage->data(), nullptr);
  EXPECT_EQ(storage->size_bytes(), 0U);
}

TEST(StorageTest, RejectsInvalidAlignmentAndUnavailableCuda) {
  constexpr std::size_t kTooSmallAlignment = alignof(std::max_align_t) / 2;
  EXPECT_THROW(static_cast<void>(Storage::allocate(16, Device::cpu(), 0)), Error);
  EXPECT_THROW(static_cast<void>(Storage::allocate(16, Device::cpu(), 24)), Error);
  EXPECT_THROW(static_cast<void>(Storage::allocate(16, Device::cpu(), kTooSmallAlignment)), Error);
#if !HELIX_ENABLE_CUDA
  EXPECT_THROW(static_cast<void>(Storage::allocate(16, Device::cuda())), Error);
  EXPECT_THROW(static_cast<void>(Storage::wrap_mutable_cuda(nullptr, 0, Device::cuda(), {}, 64)),
               Error);
#endif
}

TEST(StorageTest, ReadOnlyExternalStorageValidatesOwnershipAndAlignment) {
  struct alignas(64) Buffer final {
    std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  };
  const auto owner = std::make_shared<Buffer>();
  const auto storage =
      Storage::wrap_read_only_cpu(owner->values.data(), sizeof(owner->values), owner, 64);

  EXPECT_EQ(storage->data(), owner->values.data());
  EXPECT_TRUE(storage->is_read_only());
  EXPECT_THROW(static_cast<void>(storage->mutable_data()), Error);
  EXPECT_THROW(static_cast<void>(Storage::wrap_read_only_cpu(owner->values.data(),
                                                             sizeof(owner->values), {}, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Storage::wrap_read_only_cpu(
                   reinterpret_cast<const std::byte*>(owner->values.data()) + sizeof(float),
                   sizeof(owner->values) - sizeof(float), owner, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Storage::wrap_read_only_cpu(owner->values.data(), 0, {}, 64)),
               Error);
  EXPECT_EQ(Storage::wrap_read_only_cpu(nullptr, 0, {}, 64)->data(), nullptr);
}

TEST(StorageTest, MutableExternalStoragePreservesLifetimeAndValidatesContract) {
  struct alignas(64) Buffer final {
    std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  };
  auto owner = std::make_shared<Buffer>();
  std::weak_ptr<Buffer> lifetime = owner;
  auto storage = Storage::wrap_mutable_cpu(owner->values.data(), sizeof(owner->values), owner, 64);
  owner.reset();

  EXPECT_FALSE(lifetime.expired());
  EXPECT_EQ(storage->data(), lifetime.lock()->values.data());
  EXPECT_EQ(storage->mutable_data(), lifetime.lock()->values.data());
  EXPECT_FALSE(storage->is_read_only());
  static_cast<float*>(storage->mutable_data())[2] = 9.0F;
  EXPECT_FLOAT_EQ(static_cast<const float*>(storage->data())[2], 9.0F);

  const auto validation_owner = std::make_shared<Buffer>();
  EXPECT_THROW(static_cast<void>(Storage::wrap_mutable_cpu(
                   validation_owner->values.data(), sizeof(validation_owner->values), {}, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Storage::wrap_mutable_cpu(
                   reinterpret_cast<std::byte*>(validation_owner->values.data()) + sizeof(float),
                   sizeof(validation_owner->values) - sizeof(float), validation_owner, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Storage::wrap_mutable_cpu(validation_owner->values.data(), 0,
                                                           validation_owner, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Storage::wrap_mutable_cpu(
                   nullptr, sizeof(validation_owner->values), validation_owner, 64)),
               Error);
  EXPECT_EQ(Storage::wrap_mutable_cpu(nullptr, 0, {}, 64)->mutable_data(), nullptr);

  storage.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST(TensorTest, OwnsCanonicalContiguousMetadata) {
  const Tensor tensor({2, 3, 4}, DType::kFloat32, Device::cpu());

  EXPECT_EQ(tensor.shape(), (Shape{2, 3, 4}));
  EXPECT_EQ(tensor.strides(), (Strides{12, 4, 1}));
  EXPECT_EQ(tensor.rank(), 3U);
  EXPECT_EQ(tensor.numel(), 24U);
  EXPECT_EQ(tensor.nbytes(), 96U);
  EXPECT_EQ(tensor.allocated_bytes(), 96U);
  EXPECT_EQ(tensor.storage_offset_bytes(), 0U);
  EXPECT_EQ(tensor.dtype(), DType::kFloat32);
  EXPECT_EQ(tensor.device(), Device::cpu());
  EXPECT_TRUE(tensor.is_contiguous());
  EXPECT_NE(tensor.data(), nullptr);
}

TEST(TensorTest, SupportsScalarAndZeroSizedTensors) {
  const Tensor scalar = Tensor::zeros({});
  EXPECT_EQ(scalar.rank(), 0U);
  EXPECT_EQ(scalar.numel(), 1U);
  EXPECT_FLOAT_EQ(scalar.data_as<float>()[0], 0.0F);

  const Tensor empty = Tensor::zeros({4, 0, 7});
  EXPECT_EQ(empty.numel(), 0U);
  EXPECT_EQ(empty.nbytes(), 0U);
  EXPECT_EQ(empty.data(), nullptr);
}

TEST(TensorTest, ZeroInitializesSupportedStorageRepresentations) {
  const Tensor float_tensor = Tensor::zeros({7}, DType::kFloat32);
  EXPECT_TRUE(std::all_of(float_tensor.data_as<float>(),
                          float_tensor.data_as<float>() + float_tensor.numel(),
                          [](float value) { return value == 0.0F; }));

  const Tensor half_bytes = Tensor::zeros({5}, DType::kFloat16);
  const auto* bytes = static_cast<const std::byte*>(half_bytes.data());
  EXPECT_TRUE(std::all_of(bytes, bytes + half_bytes.nbytes(),
                          [](std::byte value) { return value == std::byte{0}; }));
  EXPECT_THROW(static_cast<void>(half_bytes.data_as<float>()), Error);
}

TEST(TensorTest, RandomInitializationIsDeterministicAndBounded) {
  const Tensor first = Tensor::random_uniform({257}, 1234, -0.25F, 0.75F);
  const Tensor second = Tensor::random_uniform({257}, 1234, -0.25F, 0.75F);
  const Tensor third = Tensor::random_uniform({257}, 4321, -0.25F, 0.75F);

  EXPECT_EQ(values_of(first), values_of(second));
  EXPECT_NE(values_of(first), values_of(third));
  for (const float value : values_of(first)) {
    EXPECT_GE(value, -0.25F);
    EXPECT_LT(value, 0.75F);
  }
}

TEST(TensorTest, RandomInitializationValidatesContract) {
  EXPECT_THROW(static_cast<void>(Tensor::random_uniform({2}, 0, 1.0F, -1.0F)), Error);
  EXPECT_THROW(static_cast<void>(
                   Tensor::random_uniform({2}, 0, 0.0F, std::numeric_limits<float>::infinity())),
               Error);
  EXPECT_THROW(
      static_cast<void>(Tensor::random_uniform({2}, 0, 0.0F, 1.0F, DType::kFloat16, Device::cpu())),
      Error);
}

TEST(TensorTest, ShallowCopiesAndViewsShareStorage) {
  Tensor base = Tensor::from_values({2, 3}, std::array{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
  Tensor alias = base;
  Tensor view = base.reshape({3, 2});

  EXPECT_EQ(alias.data(), base.data());
  EXPECT_EQ(view.data(), base.data());
  EXPECT_EQ(view.shape(), (Shape{3, 2}));
  EXPECT_EQ(view.strides(), (Strides{2, 1}));

  view.data_as<float>()[4] = 42.0F;
  EXPECT_FLOAT_EQ(base.data_as<float>()[4], 42.0F);
}

TEST(TensorTest, ViewKeepsStorageAlive) {
  const Tensor view = [] {
    Tensor temporary = Tensor::from_values({2, 2}, std::array{1.0F, 2.0F, 3.0F, 4.0F});
    return temporary.view({4});
  }();

  EXPECT_EQ(values_of(view), (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}));
}

TEST(TensorTest, ReadOnlyExternalBufferPreservesLifetimeAndSupportsDeepClone) {
  struct alignas(64) Buffer final {
    std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
  };

  auto owner = std::make_shared<Buffer>();
  std::weak_ptr<Buffer> lifetime = owner;
  Tensor tensor =
      Tensor::from_read_only_cpu_buffer({2, 2}, DType::kFloat32, owner->values.data(), owner, 64);
  owner.reset();

  EXPECT_FALSE(lifetime.expired());
  EXPECT_TRUE(tensor.is_read_only());
  EXPECT_EQ(values_of(tensor), (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}));
  EXPECT_THROW(static_cast<void>(tensor.data()), Error);
  EXPECT_THROW(static_cast<void>(tensor.data_as<float>()), Error);
  EXPECT_THROW(static_cast<void>(tensor.zero_()), Error);
  EXPECT_THROW(static_cast<void>(tensor.random_uniform_(7)), Error);
  EXPECT_THROW(static_cast<void>(tensor.copy_from(Tensor::zeros({2, 2}))), Error);

  Tensor view = tensor.view({4});
  EXPECT_TRUE(view.is_read_only());
  EXPECT_EQ(std::as_const(view).data(), std::as_const(tensor).data());

  Tensor clone = tensor.clone();
  EXPECT_FALSE(clone.is_read_only());
  clone.data_as<float>()[0] = -5.0F;
  EXPECT_FLOAT_EQ(std::as_const(tensor).data_as<float>()[0], 1.0F);

  tensor = Tensor::zeros({1});
  EXPECT_FALSE(lifetime.expired());
  view = Tensor::zeros({1});
  EXPECT_TRUE(lifetime.expired());
}

TEST(TensorTest, MutableExternalBufferExposesCapacityAndSharesOwnershipWithViews) {
  struct alignas(64) Buffer final {
    std::array<float, 8> values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  };

  auto owner = std::make_shared<Buffer>();
  std::weak_ptr<Buffer> lifetime = owner;
  Tensor tensor = Tensor::from_mutable_cpu_buffer({2, 3}, DType::kFloat32, owner->values.data(),
                                                  sizeof(owner->values), owner, 64);
  owner.reset();

  EXPECT_FALSE(lifetime.expired());
  EXPECT_FALSE(tensor.is_read_only());
  EXPECT_EQ(tensor.nbytes(), 6U * sizeof(float));
  EXPECT_EQ(tensor.allocated_bytes(), 8U * sizeof(float));
  tensor.data_as<float>()[4] = 42.0F;
  EXPECT_FLOAT_EQ(lifetime.lock()->values[4], 42.0F);

  Tensor view = tensor.view({3, 2});
  EXPECT_EQ(view.data(), tensor.data());
  tensor = Tensor::zeros({1});
  EXPECT_FALSE(lifetime.expired());

  Tensor clone = view.clone();
  EXPECT_NE(clone.data(), view.data());
  clone.data_as<float>()[0] = -1.0F;
  EXPECT_FLOAT_EQ(std::as_const(view).data_as<float>()[0], 1.0F);

  view = Tensor::zeros({1});
  EXPECT_TRUE(lifetime.expired());
}

TEST(TensorTest, MutableExternalBufferRejectsInvalidBoundsPointerOwnerAndAlignment) {
  struct alignas(64) Buffer final {
    std::array<float, 4> values{};
  };
  const auto owner = std::make_shared<Buffer>();

  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cpu_buffer(
                   {4}, DType::kFloat32, owner->values.data(), 3U * sizeof(float), owner, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cpu_buffer(
                   {4}, DType::kFloat32, owner->values.data(), sizeof(owner->values), {}, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cpu_buffer(
                   {0}, DType::kFloat32, owner->values.data(), 0, owner, 64)),
               Error);
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cpu_buffer({4}, DType::kFloat32, nullptr,
                                                                 sizeof(owner->values), owner, 64)),
               Error);
  EXPECT_THROW(
      static_cast<void>(Tensor::from_mutable_cpu_buffer(
          {4}, DType::kFloat32, reinterpret_cast<std::byte*>(owner->values.data()) + sizeof(float),
          sizeof(owner->values) - sizeof(float), owner, 64)),
      Error);
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cpu_buffer(
                   {4}, DType::kFloat32, owner->values.data(), sizeof(owner->values), owner, 24)),
               Error);

  const Tensor empty = Tensor::from_mutable_cpu_buffer({3, 0}, DType::kFloat32, nullptr, 0, {}, 64);
  EXPECT_EQ(empty.data(), nullptr);
  EXPECT_EQ(empty.allocated_bytes(), 0U);
#if !HELIX_ENABLE_CUDA
  EXPECT_THROW(static_cast<void>(Tensor::from_mutable_cuda_buffer({0}, DType::kFloat32, nullptr, 0,
                                                                  Device::cuda(), {}, 64)),
               Error);
#endif
}

TEST(TensorTest, ReshapeRejectsIncompatibleElementCounts) {
  const Tensor tensor = Tensor::zeros({2, 3});
  EXPECT_THROW(static_cast<void>(tensor.reshape({5})), Error);
  EXPECT_THROW(static_cast<void>(tensor.reshape({-1, 6})), Error);
}

TEST(TensorTest, CloneAndSameDeviceTransferDeepCopy) {
  Tensor source = Tensor::from_values({3}, std::array{1.0F, 2.0F, 3.0F});
  Tensor clone = source.clone();
  Tensor transferred = source.to(Device::cpu());

  EXPECT_NE(clone.data(), source.data());
  EXPECT_NE(transferred.data(), source.data());
  EXPECT_EQ(values_of(clone), values_of(source));
  EXPECT_EQ(values_of(transferred), values_of(source));

  clone.data_as<float>()[0] = -9.0F;
  EXPECT_FLOAT_EQ(source.data_as<float>()[0], 1.0F);
}

TEST(TensorTest, CopyValidatesShapeDTypeAndDeviceTransfer) {
  Tensor destination = Tensor::zeros({2, 2});
  const Tensor wrong_shape = Tensor::zeros({4});
  const Tensor wrong_dtype = Tensor::zeros({2, 2}, DType::kFloat16);

  EXPECT_THROW(static_cast<void>(destination.copy_from(wrong_shape)), Error);
  EXPECT_THROW(static_cast<void>(destination.copy_from(wrong_dtype)), Error);
#if !HELIX_ENABLE_CUDA
  EXPECT_THROW(static_cast<void>(destination.to(Device::cuda())), Error);
  EXPECT_THROW(static_cast<void>(Tensor({1}, DType::kFloat32, Device::cuda())), Error);
#endif
}

TEST(TensorTest, FromValuesValidatesElementCount) {
  EXPECT_THROW(static_cast<void>(Tensor::from_values({2, 2}, std::array{1.0F, 2.0F, 3.0F})), Error);
}

} // namespace
} // namespace helix
