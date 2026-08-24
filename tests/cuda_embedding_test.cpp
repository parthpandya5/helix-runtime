#include "helix/kernels/cpu/reference.hpp"
#include "helix/kernels/cuda/embedding.hpp"
#include "helix/runtime/cuda/device.hpp"
#include "helix/runtime/cuda/memory.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace helix::cuda::kernels {
namespace {

class CudaEmbeddingTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    set_device(0);
  }
};

TEST_F(CudaEmbeddingTest, MatchesCpuForEmptyTinyOddAndNonBlockAlignedShapes) {
  struct Case final {
    Dimension tokens;
    Dimension vocabulary;
    Dimension width;
  };
  constexpr std::array kCases{Case{0, 4, 5}, Case{1, 1, 1}, Case{5, 11, 37}, Case{257, 1'021, 513}};
  Stream stream;
  for (const Case test_case : kCases) {
    std::vector<std::uint32_t> ids(static_cast<std::size_t>(test_case.tokens));
    for (std::size_t token = 0; token < ids.size(); ++token) {
      ids[token] = static_cast<std::uint32_t>((token * 17U + 3U) %
                                              static_cast<std::size_t>(test_case.vocabulary));
    }
    const Tensor ids_host = Tensor::from_values({test_case.tokens}, ids);
    const Tensor table_host =
        Tensor::random_uniform({test_case.vocabulary, test_case.width}, 6'001, -3.0F, 3.0F);
    const Tensor expected = cpu::embedding_lookup(ids_host, table_host);
    const Tensor ids_device = ids_host.to(Device::cuda(0));
    const Tensor table_device = table_host.to(Device::cuda(0));

    const Tensor actual_device = embedding_lookup(ids_device, table_device, stream);
    stream.synchronize();
    const Tensor actual = actual_device.to(Device::cpu());
    ASSERT_EQ(actual.shape(), expected.shape());
    for (std::size_t index = 0; index < expected.numel(); ++index) {
      EXPECT_FLOAT_EQ(actual.data_as<float>()[index], expected.data_as<float>()[index])
          << "at index " << index;
    }
  }
}

TEST_F(CudaEmbeddingTest, InvalidDeviceIdProducesNanRowWithoutOutOfBoundsRead) {
  const Tensor ids =
      Tensor::from_values({3}, std::array<std::uint32_t, 3>{0, 3, 1}, Device::cuda(0));
  const Tensor table_host =
      Tensor::from_values({3, 5}, std::array{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 10.0F, 11.0F, 12.0F,
                                             13.0F, 14.0F, 20.0F, 21.0F, 22.0F, 23.0F, 24.0F});
  const Tensor table = table_host.to(Device::cuda(0));
  Stream stream;

  const Tensor output_device = embedding_lookup(ids, table, stream);
  stream.synchronize();
  const Tensor output = output_device.to(Device::cpu());
  for (std::size_t coordinate = 0; coordinate < 5; ++coordinate) {
    EXPECT_FLOAT_EQ(output.data_as<float>()[coordinate], table_host.data_as<float>()[coordinate]);
    EXPECT_TRUE(std::isnan(output.data_as<float>()[5 + coordinate]));
    EXPECT_FLOAT_EQ(output.data_as<float>()[10 + coordinate],
                    table_host.data_as<float>()[5 + coordinate]);
  }
}

TEST_F(CudaEmbeddingTest, RejectsMetadataDeviceAliasingAndMovedStreamErrors) {
  Stream stream;
  const Tensor ids = Tensor::from_values({2}, std::array<std::uint32_t, 2>{0, 1}, Device::cuda(0));
  const Tensor table = Tensor::zeros({2, 3}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({2, 3}, DType::kFloat32, Device::cuda(0));

  EXPECT_THROW(embedding_lookup_out(Tensor::zeros({2}, DType::kFloat32, Device::cuda(0)), table,
                                    output, stream),
               Error);
  EXPECT_THROW(embedding_lookup_out(ids.view({1, 2}), table, output, stream), Error);
  EXPECT_THROW(embedding_lookup_out(ids, Tensor::zeros({6}, DType::kFloat32, Device::cuda(0)),
                                    output, stream),
               Error);
  EXPECT_THROW(embedding_lookup_out(ids, Tensor::zeros({2, 3}, DType::kFloat16, Device::cuda(0)),
                                    output, stream),
               Error);
  const Tensor zero_width_table = Tensor::zeros({2, 0}, DType::kFloat32, Device::cuda(0));
  Tensor zero_width_output = Tensor::zeros({2, 0}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(embedding_lookup_out(ids, zero_width_table, zero_width_output, stream), Error);

  Tensor wrong_shape = Tensor::zeros({2, 4}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(embedding_lookup_out(ids, table, wrong_shape, stream), Error);
  Tensor wrong_dtype = Tensor::zeros({2, 3}, DType::kUInt32, Device::cuda(0));
  EXPECT_THROW(embedding_lookup_out(ids, table, wrong_dtype, stream), Error);

  Tensor aliased_table = Tensor::zeros({2, 3}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(embedding_lookup_out(ids, aliased_table, aliased_table, stream), Error);

  const auto allocation = std::make_shared<Buffer>(768, 0);
  const Tensor partial_table = Tensor::from_mutable_cuda_buffer(
      {2, 64}, DType::kFloat32, allocation->data(), 512, Device::cuda(0), allocation, 256);
  Tensor partial_output = Tensor::from_mutable_cuda_buffer(
      {2, 64}, DType::kFloat32,
      static_cast<void*>(static_cast<std::byte*>(allocation->data()) + 256), 512, Device::cuda(0),
      allocation, 256);
  EXPECT_THROW(embedding_lookup_out(ids, partial_table, partial_output, stream), Error);
  EXPECT_THROW(embedding_lookup_out(Tensor::zeros({2}, DType::kUInt32, Device::cpu()), table,
                                    output, stream),
               Error);

  Stream moved_to = std::move(stream);
  EXPECT_THROW(embedding_lookup_out(ids, table, output, stream), Error);
  embedding_lookup_out(ids, table, output, moved_to);
  moved_to.synchronize();
}

} // namespace
} // namespace helix::cuda::kernels
