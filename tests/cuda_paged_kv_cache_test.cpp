#include "helix/core/error.hpp"
#include "helix/kv_cache/cuda_paged.hpp"
#include "helix/runtime/cuda/device.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace helix::kv_cache {
namespace {

static_assert(!std::is_copy_constructible_v<CudaPagedKvCache::DecodeView>);
static_assert(!std::is_copy_assignable_v<CudaPagedKvCache::DecodeView>);
static_assert(std::is_nothrow_move_constructible_v<CudaPagedKvCache::DecodeView>);
static_assert(std::is_nothrow_move_assignable_v<CudaPagedKvCache::DecodeView>);
static_assert(
    noexcept(std::declval<CudaPagedKvCache::DecodeView&>().poison_after_consumer_failure()));

class CudaPagedKvCacheTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!helix::cuda::runtime_available()) {
      GTEST_SKIP() << "requires a visible NVIDIA CUDA device";
    }
    helix::cuda::set_device(0);
  }
};

std::pair<Tensor, Tensor> make_token(const PagedKvCacheConfig& config, std::size_t token) {
  const std::size_t elements = config.layers * config.kv_heads * config.head_dimension;
  std::vector<float> keys(elements);
  std::vector<float> values(elements);
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    for (std::size_t head = 0; head < config.kv_heads; ++head) {
      for (std::size_t coordinate = 0; coordinate < config.head_dimension; ++coordinate) {
        const std::size_t index =
            (layer * config.kv_heads + head) * config.head_dimension + coordinate;
        const float encoded =
            static_cast<float>(token * 1'000 + layer * 100 + head * 10 + coordinate);
        keys[index] = encoded + 0.25F;
        values[index] = -encoded - 0.5F;
      }
    }
  }
  const Shape shape{static_cast<Dimension>(config.layers), static_cast<Dimension>(config.kv_heads),
                    static_cast<Dimension>(config.head_dimension)};
  return {Tensor::from_values(shape, keys, Device::cuda(0)),
          Tensor::from_values(shape, values, Device::cuda(0))};
}

float expected_value(std::size_t token, std::size_t layer, std::size_t head, std::size_t coordinate,
                     bool keys) {
  const float encoded = static_cast<float>(token * 1'000 + layer * 100 + head * 10 + coordinate);
  return keys ? encoded + 0.25F : -encoded - 0.5F;
}

void expect_error_code(const std::function<void()>& operation, ErrorCode code) {
  try {
    operation();
    FAIL() << "expected helix::Error";
  } catch (const Error& error) {
    EXPECT_EQ(error.code(), code);
  }
}

TEST_F(CudaPagedKvCacheTest, MapsOddHistoryAcrossPagesAndReportsExactAccounting) {
  const PagedKvCacheConfig config{2, 3, 5, 2, 3};
  CudaPagedKvCache cache(config);
  EXPECT_TRUE(cache.is_healthy());
  helix::cuda::Stream stream;
  const CudaPagedSequenceHandle sequence = cache.create_sequence();

  {
    auto empty = cache.read_sequence(sequence);
    EXPECT_EQ(empty.length(), 0U);
    EXPECT_TRUE(empty.physical_blocks().empty());
    EXPECT_TRUE(empty.block_table().valid());
    EXPECT_EQ(empty.block_table().data(), nullptr);
    EXPECT_EQ(empty.block_table().logical_block_count(), 0U);
    ASSERT_EQ(empty.layer_key_blocks().size(), config.layers);
    EXPECT_EQ(empty.layer_key_blocks()[0].shape(), (Shape{3, 3, 2, 5}));
  }

  for (std::size_t token = 0; token < 5; ++token) {
    const auto [keys, values] = make_token(config, token);
    cache.append(sequence, keys, values, stream);
  }

  {
    auto view = cache.read_sequence(sequence);
    EXPECT_EQ(view.length(), 5U);
    constexpr std::array<std::uint32_t, 3> kExpectedBlocks{0, 1, 2};
    EXPECT_TRUE(std::equal(view.physical_blocks().begin(), view.physical_blocks().end(),
                           kExpectedBlocks.begin(), kExpectedBlocks.end()));
    EXPECT_EQ(view.block_table().logical_block_count(), 3U);
    EXPECT_EQ(view.block_table().physical_block_count(), 3U);
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
      const Tensor host_keys = view.layer_key_blocks()[layer].to(Device::cpu());
      const Tensor host_values = view.layer_value_blocks()[layer].to(Device::cpu());
      for (std::size_t token = 0; token < 5; ++token) {
        const std::size_t physical_block = token / config.block_tokens;
        const std::size_t token_offset = token % config.block_tokens;
        for (std::size_t head = 0; head < config.kv_heads; ++head) {
          for (std::size_t coordinate = 0; coordinate < config.head_dimension; ++coordinate) {
            const std::size_t index =
                (((physical_block * config.kv_heads + head) * config.block_tokens + token_offset) *
                 config.head_dimension) +
                coordinate;
            EXPECT_FLOAT_EQ(host_keys.data_as<float>()[index],
                            expected_value(token, layer, head, coordinate, true));
            EXPECT_FLOAT_EQ(host_values.data_as<float>()[index],
                            expected_value(token, layer, head, coordinate, false));
          }
        }
      }
    }
  }

  const PagedKvCacheStats stats = cache.stats();
  EXPECT_EQ(cache.bytes_per_token(), 2U * 2U * 3U * 5U * sizeof(float));
  EXPECT_EQ(cache.block_bytes(), cache.bytes_per_token() * config.block_tokens);
  EXPECT_EQ(cache.token_capacity(), 6U);
  EXPECT_EQ(stats.bytes_reserved, cache.block_bytes() * config.physical_blocks);
  EXPECT_EQ(stats.bytes_used, stats.bytes_reserved);
  EXPECT_EQ(stats.bytes_free, 0U);
  EXPECT_EQ(stats.peak_bytes_used, stats.bytes_reserved);
  EXPECT_EQ(stats.total_blocks, 3U);
  EXPECT_EQ(stats.used_blocks, 3U);
  EXPECT_EQ(stats.free_blocks, 0U);
  EXPECT_EQ(stats.active_sequence_count, 1U);
  EXPECT_EQ(stats.active_tokens, 5U);
  EXPECT_EQ(stats.resident_tokens, 5U);
  EXPECT_EQ(stats.allocated_token_slots, 6U);
  EXPECT_EQ(stats.unused_token_slots(), 1U);
  EXPECT_EQ(stats.block_reference_count, 3U);
  EXPECT_EQ(stats.shared_blocks, 0U);
  EXPECT_EQ(stats.sequence_create_count, 1U);
  EXPECT_EQ(stats.append_count, 5U);
  EXPECT_EQ(stats.block_allocation_count, 3U);
  EXPECT_EQ(stats.block_reuse_count, 0U);
}

TEST_F(CudaPagedKvCacheTest, ExhaustionIsTransactionalAndReleasedBlocksReuseInFifoOrder) {
  const PagedKvCacheConfig config{1, 2, 3, 2, 3};
  CudaPagedKvCache cache(config);
  helix::cuda::Stream stream;
  const CudaPagedSequenceHandle first = cache.create_sequence();
  const CudaPagedSequenceHandle second = cache.create_sequence();

  for (std::size_t token = 0; token < 4; ++token) {
    const auto [keys, values] = make_token(config, token);
    cache.append(first, keys, values, stream);
  }
  const auto [second_keys, second_values] = make_token(config, 10);
  cache.append(second, second_keys, second_values, stream);
  const auto [last_slack_keys, last_slack_values] = make_token(config, 11);
  cache.append(second, last_slack_keys, last_slack_values, stream);
  ASSERT_EQ(cache.stats().free_blocks, 0U);

  const PagedKvCacheStats before = cache.stats();
  const auto [rejected_keys, rejected_values] = make_token(config, 12);
  expect_error_code([&] { cache.append(second, rejected_keys, rejected_values, stream); },
                    ErrorCode::kResourceExhausted);
  EXPECT_EQ(cache.length(second), 2U);
  EXPECT_EQ(cache.stats().append_count, before.append_count);
  EXPECT_EQ(cache.stats().block_allocation_count, before.block_allocation_count);
  EXPECT_EQ(cache.stats().bytes_used, before.bytes_used);

  cache.release_sequence(first);
  EXPECT_FALSE(cache.contains(first));
  EXPECT_EQ(cache.stats().free_blocks, 2U);
  cache.append(second, rejected_keys, rejected_values, stream);
  {
    auto view = cache.read_sequence(second);
    ASSERT_EQ(view.physical_blocks().size(), 2U);
    EXPECT_EQ(view.physical_blocks()[0], 2U);
    EXPECT_EQ(view.physical_blocks()[1], 0U);
  }
  EXPECT_EQ(cache.stats().block_reuse_count, 1U);
  EXPECT_EQ(cache.stats().release_count, 1U);
}

TEST_F(CudaPagedKvCacheTest, RejectsInvalidConfigurationHandlesTensorsStreamsAndAliasing) {
  EXPECT_THROW(CudaPagedKvCache(PagedKvCacheConfig{0, 1, 1, 1, 1}), Error);
  EXPECT_THROW(CudaPagedKvCache(PagedKvCacheConfig{1, 0, 1, 1, 1}), Error);
  EXPECT_THROW(CudaPagedKvCache(PagedKvCacheConfig{1, 1, 0, 1, 1}), Error);
  EXPECT_THROW(CudaPagedKvCache(PagedKvCacheConfig{1, 1, 1, 0, 1}), Error);
  EXPECT_THROW(CudaPagedKvCache(PagedKvCacheConfig{1, 1, 1, 1, 0}), Error);
  EXPECT_THROW(CudaPagedKvCache(PagedKvCacheConfig{1, 1, 1, 1, 1}, -1), Error);

  const PagedKvCacheConfig config{1, 1, 2, 1, 1};
  CudaPagedKvCache first(config);
  CudaPagedKvCache second(config);
  const CudaPagedSequenceHandle sequence = first.create_sequence();
  const CudaPagedSequenceHandle foreign = second.create_sequence();
  helix::cuda::Stream stream;
  const auto [keys, values] = make_token(config, 0);

  EXPECT_THROW(first.append(foreign, keys, values, stream), Error);
  EXPECT_THROW(first.append(CudaPagedSequenceHandle{}, keys, values, stream), Error);
  EXPECT_THROW(first.append(sequence, Tensor::zeros({1, 1, 2}), values, stream), Error);
  EXPECT_THROW(first.append(sequence, Tensor::zeros({1, 1, 2}, DType::kUInt32, Device::cuda(0)),
                            values, stream),
               Error);
  EXPECT_THROW(first.append(sequence, Tensor::zeros({1, 2}, DType::kFloat32, Device::cuda(0)),
                            values, stream),
               Error);

  helix::cuda::Stream active_stream = std::move(stream);
  EXPECT_THROW(first.append(sequence, keys, values, stream), Error);

  std::optional<Tensor> aliased_keys;
  {
    auto view = first.read_sequence(sequence);
    aliased_keys.emplace(view.layer_key_blocks()[0].reshape({1, 1, 2}));
  }
  EXPECT_THROW(first.append(sequence, *aliased_keys, values, active_stream), Error);

  first.append(sequence, keys, values, active_stream);
  first.release_sequence(sequence);
  EXPECT_THROW(first.release_sequence(sequence), Error);
  EXPECT_THROW(static_cast<void>(first.read_sequence(sequence)), Error);
}

TEST_F(CudaPagedKvCacheTest, DecodeViewPinsMappingAndPoolLifetimeAndMovesSafely) {
  const PagedKvCacheConfig config{1, 1, 3, 2, 2};
  auto cache = std::make_unique<CudaPagedKvCache>(config);
  auto replacement_cache = std::make_unique<CudaPagedKvCache>(config);
  helix::cuda::Stream stream;
  const CudaPagedSequenceHandle sequence = cache->create_sequence();
  const CudaPagedSequenceHandle replacement_sequence = replacement_cache->create_sequence();
  const auto [keys, values] = make_token(config, 0);
  cache->append(sequence, keys, values, stream);
  replacement_cache->append(replacement_sequence, keys, values, stream);

  std::optional<CudaPagedKvCache::DecodeView> view;
  view.emplace(cache->read_sequence(sequence));
  CudaPagedKvCache::DecodeView moved = std::move(*view);
  EXPECT_THROW(static_cast<void>(view->length()), Error);
  view.reset();
  cache.reset();

  EXPECT_EQ(moved.length(), 1U);
  EXPECT_EQ(moved.physical_blocks().front(), 0U);
  EXPECT_EQ(moved.layer_key_blocks()[0].device(), Device::cuda(0));
  EXPECT_TRUE(moved.block_table().valid());

  // Exercise replacement while the destination view is the final owner of a
  // cache facade. Its mutex must be unlocked before that Impl can be destroyed.
  auto replacement = replacement_cache->read_sequence(replacement_sequence);
  replacement_cache.reset();
  moved = std::move(replacement);
  EXPECT_THROW(static_cast<void>(replacement.length()), Error);
  EXPECT_EQ(moved.length(), 1U);
  EXPECT_EQ(moved.physical_blocks().front(), 0U);
}

TEST_F(CudaPagedKvCacheTest, DecodeViewSerializesReleaseUntilBorrowedPointersExpire) {
  const PagedKvCacheConfig config{1, 1, 2, 2, 1};
  CudaPagedKvCache cache(config);
  helix::cuda::Stream stream;
  const CudaPagedSequenceHandle sequence = cache.create_sequence();
  const auto [keys, values] = make_token(config, 0);
  cache.append(sequence, keys, values, stream);

  std::optional<CudaPagedKvCache::DecodeView> view;
  view.emplace(cache.read_sequence(sequence));
  std::promise<void> started;
  std::future<void> started_future = started.get_future();
  std::future<void> release = std::async(std::launch::async, [&] {
    started.set_value();
    cache.release_sequence(sequence);
  });
  started_future.wait();
  EXPECT_EQ(release.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);

  view.reset();
  EXPECT_EQ(release.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  release.get();
  EXPECT_FALSE(cache.contains(sequence));
  EXPECT_EQ(cache.stats().free_blocks, 1U);
}

TEST_F(CudaPagedKvCacheTest, FailedConsumerQuarantinesTableAndPreventsPageReuse) {
  const PagedKvCacheConfig config{1, 1, 2, 2, 1};
  CudaPagedKvCache cache(config);
  helix::cuda::Stream stream;
  const CudaPagedSequenceHandle sequence = cache.create_sequence();
  const auto [keys, values] = make_token(config, 0);
  cache.append(sequence, keys, values, stream);

  {
    auto view = cache.read_sequence(sequence);
    ASSERT_NE(view.block_table().data(), nullptr);
    view.poison_after_consumer_failure();
    EXPECT_THROW(static_cast<void>(view.length()), Error);
  }

  EXPECT_FALSE(cache.is_healthy());
  EXPECT_TRUE(cache.contains(sequence));
  EXPECT_EQ(cache.length(sequence), 1U);
  expect_error_code([&] { static_cast<void>(cache.create_sequence()); }, ErrorCode::kRuntime);
  expect_error_code([&] { static_cast<void>(cache.read_sequence(sequence)); }, ErrorCode::kRuntime);
  expect_error_code([&] { cache.append(sequence, keys, values, stream); }, ErrorCode::kRuntime);

  cache.release_sequence(sequence);
  EXPECT_FALSE(cache.contains(sequence));
  EXPECT_EQ(cache.stats().used_blocks, 0U);
  EXPECT_EQ(cache.stats().free_blocks, 1U);
  expect_error_code([&] { static_cast<void>(cache.create_sequence()); }, ErrorCode::kRuntime);
}

} // namespace
} // namespace helix::kv_cache
