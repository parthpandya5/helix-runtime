#include "helix/attention/cuda.hpp"
#include "helix/attention/cuda_paged.hpp"
#include "helix/attention/reference.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace helix::attention::cuda {
namespace {

void expect_close(const Tensor& actual_device, const Tensor& expected_host, float tolerance) {
  const Tensor actual = actual_device.to(Device::cpu());
  ASSERT_EQ(actual.shape(), expected_host.shape());
  for (std::size_t index = 0; index < expected_host.numel(); ++index) {
    EXPECT_NEAR(actual.data_as<float>()[index], expected_host.data_as<float>()[index], tolerance)
        << "at index " << index;
  }
}

Tensor pack_logical_blocks(const Tensor& logical, std::span<const std::uint32_t> block_table,
                           Dimension physical_blocks, Dimension block_tokens, float padding_value) {
  const auto heads = static_cast<std::size_t>(logical.shape()[0]);
  const auto tokens = static_cast<std::size_t>(logical.shape()[1]);
  const auto width = static_cast<std::size_t>(logical.shape()[2]);
  const auto physical_count = static_cast<std::size_t>(physical_blocks);
  const auto tokens_per_block = static_cast<std::size_t>(block_tokens);
  std::vector<float> packed(physical_count * heads * tokens_per_block * width, padding_value);
  const auto* source = logical.data_as<float>();
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t token = 0; token < tokens; ++token) {
      const std::size_t logical_block = token / tokens_per_block;
      const std::size_t token_in_block = token % tokens_per_block;
      const std::size_t physical_block = static_cast<std::size_t>(block_table[logical_block]);
      for (std::size_t dimension = 0; dimension < width; ++dimension) {
        const std::size_t source_index = (head * tokens + token) * width + dimension;
        const std::size_t destination_index =
            (((physical_block * heads + head) * tokens_per_block + token_in_block) * width) +
            dimension;
        packed[destination_index] = source[source_index];
      }
    }
  }
  return Tensor::from_values(
      {physical_blocks, logical.shape()[0], block_tokens, logical.shape()[2]}, packed);
}

std::vector<std::uint32_t> reversed_block_table(std::size_t logical_blocks,
                                                std::size_t physical_blocks) {
  std::vector<std::uint32_t> result(logical_blocks);
  for (std::size_t logical_block = 0; logical_block < logical_blocks; ++logical_block) {
    result[logical_block] = static_cast<std::uint32_t>(physical_blocks - 1 - logical_block);
  }
  return result;
}

class CudaAttentionTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!helix::cuda::runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    helix::cuda::set_device(0);
  }
};

TEST_F(CudaAttentionTest, MatchesKnownCpuCausalResultAndProbabilities) {
  const Tensor query_host = Tensor::from_values({1, 2, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F});
  const Tensor key_host = query_host.clone();
  const Tensor value_host = Tensor::from_values({1, 2, 2}, std::array{10.0F, 0.0F, 0.0F, 20.0F});
  Tensor expected_output = Tensor::empty({1, 2, 2});
  Tensor expected_probabilities = Tensor::empty({1, 2, 2});
  cpu::scaled_dot_product_attention_out(query_host, key_host, value_host, 0, expected_output,
                                        expected_probabilities);
  const Tensor query = query_host.to(Device::cuda(0));
  const Tensor key = key_host.to(Device::cuda(0));
  const Tensor value = value_host.to(Device::cuda(0));
  Tensor output = Tensor::empty({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::empty({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;

  scaled_dot_product_attention_out(query, key, value, 0, output, workspace, stream);
  stream.synchronize();
  expect_close(output, expected_output, 2.0e-5F);
  expect_close(workspace, expected_probabilities, 2.0e-6F);
}

TEST_F(CudaAttentionTest, MatchesCpuForDecodeAndOddNonTileAlignedShapes) {
  struct Problem {
    Dimension heads;
    Dimension queries;
    Dimension keys;
    Dimension head_dimension;
    Dimension value_dimension;
  };
  constexpr std::array kProblems{Problem{1, 1, 3, 1, 2}, Problem{2, 3, 5, 5, 7},
                                 Problem{3, 7, 11, 37, 41}};
  helix::cuda::Stream stream;
  for (const Problem problem : kProblems) {
    const std::size_t position_offset = static_cast<std::size_t>(problem.keys - problem.queries);
    const Tensor query_host = Tensor::random_uniform(
        {problem.heads, problem.queries, problem.head_dimension}, 8'001, -1.0F, 1.0F);
    const Tensor key_host = Tensor::random_uniform(
        {problem.heads, problem.keys, problem.head_dimension}, 8'002, -1.0F, 1.0F);
    const Tensor value_host = Tensor::random_uniform(
        {problem.heads, problem.keys, problem.value_dimension}, 8'003, -1.0F, 1.0F);
    const Tensor expected =
        cpu::scaled_dot_product_attention(query_host, key_host, value_host, position_offset);
    const Tensor query = query_host.to(Device::cuda(0));
    const Tensor key = key_host.to(Device::cuda(0));
    const Tensor value = value_host.to(Device::cuda(0));
    const Tensor actual = scaled_dot_product_attention(query, key, value, position_offset, stream);

    stream.synchronize();
    const float tolerance = 2.0e-5F + static_cast<float>(problem.head_dimension) * 1.0e-6F;
    expect_close(actual, expected, tolerance);
  }
}

TEST_F(CudaAttentionTest, AcceptsEmptyQueryWithoutLaunching) {
  const Tensor query = Tensor::zeros({2, 0, 5}, DType::kFloat32, Device::cuda(0));
  const Tensor key = Tensor::zeros({2, 7, 5}, DType::kFloat32, Device::cuda(0));
  const Tensor value = Tensor::zeros({2, 7, 3}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;
  const Tensor output = scaled_dot_product_attention(query, key, value, 7, stream);
  EXPECT_EQ(output.shape(), (Shape{2, 0, 3}));
  EXPECT_EQ(output.numel(), 0U);
  EXPECT_EQ(score_workspace_bytes(query, key), 0U);
}

TEST_F(CudaAttentionTest, RejectsInvalidShapesOffsetsDevicesAndAliasing) {
  const Tensor query = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  const Tensor key = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  const Tensor value = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;

  EXPECT_THROW(static_cast<void>(scaled_dot_product_attention(
                   Tensor::zeros({2, 2}, DType::kFloat32, Device::cuda(0)), key, value, 0, stream)),
               Error);
  EXPECT_THROW(
      static_cast<void>(scaled_dot_product_attention(
          query, Tensor::zeros({1, 2, 3}, DType::kFloat32, Device::cuda(0)), value, 0, stream)),
      Error);
  EXPECT_THROW(static_cast<void>(scaled_dot_product_attention(query, key, value, 1, stream)),
               Error);
  EXPECT_THROW(static_cast<void>(
                   scaled_dot_product_attention(Tensor::zeros({1, 2, 2}), key, value, 0, stream)),
               Error);

  Tensor output_alias = query;
  EXPECT_THROW(
      scaled_dot_product_attention_out(query, key, value, 0, output_alias, workspace, stream),
      Error);
  Tensor workspace_alias = query;
  EXPECT_THROW(
      scaled_dot_product_attention_out(query, key, value, 0, output, workspace_alias, stream),
      Error);
  EXPECT_THROW(scaled_dot_product_attention_out(query, key, value, 0, workspace, workspace, stream),
               Error);
}

TEST_F(CudaAttentionTest, PagedPathMatchesKnownReorderedBlocksAndIgnoresTailPadding) {
  constexpr std::array<std::uint32_t, 2> kBlockTable{2, 0};
  constexpr Dimension kPhysicalBlocks = 3;
  constexpr Dimension kBlockTokens = 2;
  constexpr std::size_t kKvTokens = 3;
  const Tensor query_host = Tensor::from_values({1, 2, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F});
  const Tensor key_host =
      Tensor::from_values({1, 3, 2}, std::array{1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F});
  const Tensor value_host = Tensor::from_values(
      {1, 3, 3}, std::array{10.0F, 1.0F, -1.0F, 20.0F, 2.0F, -2.0F, 30.0F, 3.0F, -3.0F});
  Tensor expected_output = Tensor::empty({1, 2, 3});
  Tensor expected_probabilities = Tensor::empty({1, 2, 3});
  cpu::scaled_dot_product_attention_out(query_host, key_host, value_host, 1, expected_output,
                                        expected_probabilities);

  const Tensor query = query_host.to(Device::cuda(0));
  const Tensor key_blocks =
      pack_logical_blocks(key_host, kBlockTable, kPhysicalBlocks, kBlockTokens, 10'000.0F)
          .to(Device::cuda(0));
  const Tensor value_blocks =
      pack_logical_blocks(value_host, kBlockTable, kPhysicalBlocks, kBlockTokens, -10'000.0F)
          .to(Device::cuda(0));
  PagedBlockTable block_table(kBlockTable, static_cast<std::size_t>(kPhysicalBlocks), 0);
  Tensor output = Tensor::empty({1, 2, 3}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::empty({1, 2, 3}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;

  paged_scaled_dot_product_attention_out(query, key_blocks, value_blocks, block_table, kKvTokens, 1,
                                         output, workspace, stream);
  stream.synchronize();
  expect_close(output, expected_output, 2.0e-5F);
  expect_close(workspace, expected_probabilities, 2.0e-6F);
}

TEST_F(CudaAttentionTest, PagedPathMatchesCpuAcrossOddPagesHeadsAndWidths) {
  struct Problem {
    Dimension heads;
    Dimension queries;
    Dimension keys;
    Dimension key_dimension;
    Dimension value_dimension;
    Dimension block_tokens;
  };
  constexpr std::array kProblems{Problem{1, 1, 3, 1, 2, 2}, Problem{2, 3, 5, 5, 7, 2},
                                 Problem{3, 7, 11, 37, 41, 4}};
  helix::cuda::Stream stream;
  for (const Problem problem : kProblems) {
    const std::size_t kv_tokens = static_cast<std::size_t>(problem.keys);
    const std::size_t tokens_per_block = static_cast<std::size_t>(problem.block_tokens);
    const std::size_t logical_blocks = 1 + (kv_tokens - 1) / tokens_per_block;
    const std::size_t physical_blocks = logical_blocks + 2;
    const std::vector<std::uint32_t> table = reversed_block_table(logical_blocks, physical_blocks);
    const std::size_t position_offset = static_cast<std::size_t>(problem.keys - problem.queries);
    const Tensor query_host = Tensor::random_uniform(
        {problem.heads, problem.queries, problem.key_dimension}, 8'101, -1.0F, 1.0F);
    const Tensor key_host = Tensor::random_uniform(
        {problem.heads, problem.keys, problem.key_dimension}, 8'102, -1.0F, 1.0F);
    const Tensor value_host = Tensor::random_uniform(
        {problem.heads, problem.keys, problem.value_dimension}, 8'103, -1.0F, 1.0F);
    const Tensor expected =
        cpu::scaled_dot_product_attention(query_host, key_host, value_host, position_offset);
    const auto physical_blocks_dimension = static_cast<Dimension>(physical_blocks);
    const Tensor query = query_host.to(Device::cuda(0));
    const Tensor key_blocks =
        pack_logical_blocks(key_host, table, physical_blocks_dimension, problem.block_tokens, 77.0F)
            .to(Device::cuda(0));
    const Tensor value_blocks = pack_logical_blocks(value_host, table, physical_blocks_dimension,
                                                    problem.block_tokens, -77.0F)
                                    .to(Device::cuda(0));
    PagedBlockTable block_table(table, physical_blocks, 0);
    const Tensor actual = paged_scaled_dot_product_attention(
        query, key_blocks, value_blocks, block_table, kv_tokens, position_offset, stream);

    stream.synchronize();
    const float tolerance = 2.0e-5F + static_cast<float>(problem.key_dimension) * 1.0e-6F;
    expect_close(actual, expected, tolerance);
  }
}

TEST_F(CudaAttentionTest, PagedPathAcceptsEmptyQueriesWithoutLaunching) {
  constexpr std::array<std::uint32_t, 3> kBlockTable{3, 1, 0};
  const Tensor query = Tensor::zeros({2, 0, 5}, DType::kFloat32, Device::cuda(0));
  const Tensor key_blocks = Tensor::zeros({4, 2, 3, 5}, DType::kFloat32, Device::cuda(0));
  const Tensor value_blocks = Tensor::zeros({4, 2, 3, 7}, DType::kFloat32, Device::cuda(0));
  PagedBlockTable block_table(kBlockTable, 4, 0);
  helix::cuda::Stream stream;

  const Tensor output = paged_scaled_dot_product_attention(query, key_blocks, value_blocks,
                                                           block_table, 7, 7, stream);
  EXPECT_EQ(output.shape(), (Shape{2, 0, 7}));
  EXPECT_EQ(output.numel(), 0U);
  EXPECT_EQ(paged_score_workspace_bytes(query, key_blocks, block_table, 7), 0U);
}

TEST_F(CudaAttentionTest, PagedPathRejectsInvalidTablesGeometryOffsetsAndAliasing) {
  static_assert(!std::is_copy_constructible_v<PagedBlockTable>);
  static_assert(!std::is_copy_assignable_v<PagedBlockTable>);
  static_assert(std::is_nothrow_move_constructible_v<PagedBlockTable>);
  static_assert(std::is_nothrow_move_assignable_v<PagedBlockTable>);

  constexpr std::array<std::uint32_t, 2> kBlockIds{2, 0};
  constexpr std::array<std::uint32_t, 1> kShortBlockIds{0};
  constexpr std::array<std::uint32_t, 1> kOutOfRangeBlockId{3};
  const Tensor query = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  const Tensor key_blocks = Tensor::zeros({3, 1, 2, 2}, DType::kFloat32, Device::cuda(0));
  const Tensor value_blocks = Tensor::zeros({3, 1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor output = Tensor::zeros({1, 2, 2}, DType::kFloat32, Device::cuda(0));
  Tensor workspace = Tensor::zeros({1, 2, 3}, DType::kFloat32, Device::cuda(0));
  helix::cuda::Stream stream;

  EXPECT_THROW(static_cast<void>(PagedBlockTable(std::span<const std::uint32_t>{}, 3, 0)), Error);
  EXPECT_THROW(static_cast<void>(PagedBlockTable(kShortBlockIds, 0, 0)), Error);
  EXPECT_THROW(static_cast<void>(PagedBlockTable(kOutOfRangeBlockId, 3, 0)), Error);
  EXPECT_THROW(static_cast<void>(PagedBlockTable(kShortBlockIds, 3, -1)), Error);

  PagedBlockTable original(kBlockIds, 3, 0);
  PagedBlockTable block_table(std::move(original));
  EXPECT_FALSE(original.valid());
  EXPECT_TRUE(block_table.valid());
  EXPECT_EQ(block_table.logical_block_count(), 2U);
  EXPECT_EQ(block_table.physical_block_count(), 3U);
  EXPECT_EQ(block_table.size_bytes(), 2U * sizeof(std::uint32_t));
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(query, key_blocks, value_blocks,
                                                                    original, 3, 1, stream)),
               Error);

  PagedBlockTable short_table(kShortBlockIds, 3, 0);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(query, key_blocks, value_blocks,
                                                                    short_table, 3, 1, stream)),
               Error);
  PagedBlockTable wrong_capacity(kBlockIds, 4, 0);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(query, key_blocks, value_blocks,
                                                                    wrong_capacity, 3, 1, stream)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(query, key_blocks, value_blocks,
                                                                    block_table, 0, 0, stream)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(query, key_blocks, value_blocks,
                                                                    block_table, 3, 2, stream)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(
                   Tensor::zeros({1, 2, 2}), key_blocks, value_blocks, block_table, 3, 1, stream)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(
                   query, Tensor::zeros({3, 2, 2}, DType::kFloat32, Device::cuda(0)), value_blocks,
                   block_table, 3, 1, stream)),
               Error);
  EXPECT_THROW(static_cast<void>(paged_scaled_dot_product_attention(
                   query, key_blocks, Tensor::zeros({3, 1, 3, 2}, DType::kFloat32, Device::cuda(0)),
                   block_table, 3, 1, stream)),
               Error);

  Tensor output_alias = query;
  EXPECT_THROW(paged_scaled_dot_product_attention_out(query, key_blocks, value_blocks, block_table,
                                                      3, 1, output_alias, workspace, stream),
               Error);
  Tensor shared_output_workspace = Tensor::zeros({1, 2, 3}, DType::kFloat32, Device::cuda(0));
  const Tensor three_value_blocks = Tensor::zeros({3, 1, 2, 3}, DType::kFloat32, Device::cuda(0));
  EXPECT_THROW(paged_scaled_dot_product_attention_out(query, key_blocks, three_value_blocks,
                                                      block_table, 3, 1, shared_output_workspace,
                                                      shared_output_workspace, stream),
               Error);

  PagedBlockTable reassigned(kShortBlockIds, 3, 0);
  reassigned = std::move(block_table);
  EXPECT_FALSE(block_table.valid());
  EXPECT_TRUE(reassigned.valid());
  EXPECT_NO_THROW(paged_scaled_dot_product_attention_out(
      query, key_blocks, value_blocks, reassigned, 3, 1, output, workspace, stream));
  stream.synchronize();
}

} // namespace
} // namespace helix::attention::cuda
