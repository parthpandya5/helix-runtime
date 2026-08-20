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
#include <span>
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
#endif
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
