#include "helix/core/error.hpp"
#include "helix/scheduler/request.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace helix::scheduler {
namespace {

using namespace std::chrono_literals;

constexpr RequestTimePoint kCreated{};

std::vector<TokenId> prompt(std::size_t size) {
  std::vector<TokenId> tokens;
  tokens.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    tokens.emplace_back(static_cast<std::uint32_t>(100 + index));
  }
  return tokens;
}

Request make_request(std::uint64_t id = 1, std::size_t prompt_size = 3,
                     std::size_t max_new_tokens = 2) {
  return Request(RequestId{id}, prompt(prompt_size), max_new_tokens, kCreated);
}

TEST(RequestTest, StrongIdsAndInitialStateExposeImmutableRequestDefinition) {
  const RequestId first{7};
  const RequestId second{8};
  EXPECT_TRUE(first.valid());
  EXPECT_FALSE(RequestId{}.valid());
  EXPECT_LT(first, second);
  EXPECT_EQ(TokenId{42}.value(), 42U);

  Request request(first, prompt(3), 4, kCreated);
  EXPECT_EQ(request.id(), first);
  EXPECT_EQ(request.state(), RequestState::kWaiting);
  EXPECT_EQ(request.finish_reason(), FinishReason::kNone);
  EXPECT_FALSE(request.terminal());
  ASSERT_EQ(request.prompt_tokens().size(), 3U);
  EXPECT_EQ(request.prompt_tokens()[2], TokenId{102});
  EXPECT_TRUE(request.generated_tokens().empty());
  EXPECT_EQ(request.max_new_tokens(), 4U);
  EXPECT_EQ(request.prompt_tokens_processed(), 0U);
  EXPECT_EQ(request.remaining_prompt_tokens(), 3U);
  EXPECT_EQ(request.remaining_generation_tokens(), 4U);
  EXPECT_EQ(request.created_at(), kCreated);
  EXPECT_FALSE(request.first_scheduled_at().has_value());
  EXPECT_FALSE(request.completed_at().has_value());
  EXPECT_FALSE(request.waiting_time().has_value());
  EXPECT_FALSE(request.time_to_first_token().has_value());
}

TEST(RequestTest, ChunkedPrefillTransitionsToDecodeOnlyAfterWholePrompt) {
  Request request = make_request();
  request.begin_prefill(kCreated + 5ms);
  EXPECT_EQ(request.state(), RequestState::kPrefill);
  EXPECT_EQ(request.first_scheduled_at(), kCreated + 5ms);
  EXPECT_EQ(request.waiting_time(), 5ms);

  request.record_prefill_progress(1, kCreated + 7ms);
  EXPECT_EQ(request.state(), RequestState::kPrefill);
  EXPECT_EQ(request.prompt_tokens_processed(), 1U);
  EXPECT_EQ(request.remaining_prompt_tokens(), 2U);
  EXPECT_FALSE(request.prefill_completed_at().has_value());

  request.record_prefill_progress(2, kCreated + 9ms);
  EXPECT_EQ(request.state(), RequestState::kDecoding);
  EXPECT_EQ(request.prompt_tokens_processed(), 3U);
  EXPECT_EQ(request.prefill_completed_at(), kCreated + 9ms);
}

TEST(RequestTest, ZeroGenerationBudgetFinishesAtPrefillCompletion) {
  Request request = make_request(2, 2, 0);
  request.begin_prefill(kCreated + 1ms);
  request.record_prefill_progress(2, kCreated + 3ms);

  EXPECT_EQ(request.state(), RequestState::kFinished);
  EXPECT_EQ(request.finish_reason(), FinishReason::kLength);
  EXPECT_TRUE(request.terminal());
  EXPECT_EQ(request.completed_at(), kCreated + 3ms);
  EXPECT_EQ(request.remaining_generation_tokens(), 0U);
  EXPECT_TRUE(request.generated_tokens().empty());
}

TEST(RequestTest, GeneratedTokensRecordMetricsAndFinishAtLengthLimit) {
  Request request = make_request(3, 1, 2);
  request.begin_prefill(kCreated + 2ms);
  request.record_prefill_progress(1, kCreated + 4ms);
  request.append_generated_token(TokenId{501}, kCreated + 10ms);

  EXPECT_EQ(request.state(), RequestState::kDecoding);
  EXPECT_EQ(request.first_token_at(), kCreated + 10ms);
  EXPECT_EQ(request.last_token_at(), kCreated + 10ms);
  EXPECT_EQ(request.time_to_first_token(), 10ms);
  EXPECT_EQ(request.remaining_generation_tokens(), 1U);

  request.append_generated_token(TokenId{502}, kCreated + 13ms);
  EXPECT_EQ(request.state(), RequestState::kFinished);
  EXPECT_EQ(request.finish_reason(), FinishReason::kLength);
  EXPECT_EQ(request.completed_at(), kCreated + 13ms);
  EXPECT_EQ(request.last_token_at(), kCreated + 13ms);
  ASSERT_EQ(request.generated_tokens().size(), 2U);
  EXPECT_EQ(request.generated_tokens()[0], (GeneratedToken{TokenId{501}, kCreated + 10ms}));
  EXPECT_EQ(request.generated_tokens()[1], (GeneratedToken{TokenId{502}, kCreated + 13ms}));
}

TEST(RequestTest, EndOfSequenceCanFinishBeforeLengthLimit) {
  Request request = make_request(4, 1, 5);
  request.begin_prefill(kCreated);
  request.record_prefill_progress(1, kCreated);
  request.append_generated_token(TokenId{9}, kCreated + 1ms);
  request.finish_decoding(kCreated + 2ms);

  EXPECT_EQ(request.state(), RequestState::kFinished);
  EXPECT_EQ(request.finish_reason(), FinishReason::kEndOfSequence);
  EXPECT_EQ(request.generated_tokens().size(), 1U);
  EXPECT_EQ(request.completed_at(), kCreated + 2ms);
}

TEST(RequestTest, TransitionMatrixIsExplicitAndTerminalStatesHaveNoExits) {
  constexpr std::array states{RequestState::kWaiting,   RequestState::kPrefill,
                              RequestState::kDecoding,  RequestState::kFinished,
                              RequestState::kCancelled, RequestState::kFailed};
  constexpr bool expected[6][6] = {
      {false, true, false, false, true, true},    {false, false, true, true, true, true},
      {false, false, false, true, true, true},    {false, false, false, false, false, false},
      {false, false, false, false, false, false}, {false, false, false, false, false, false},
  };

  for (std::size_t from = 0; from < states.size(); ++from) {
    EXPECT_EQ(request_state_is_terminal(states[from]), from >= 3U);
    for (std::size_t to = 0; to < states.size(); ++to) {
      EXPECT_EQ(request_state_transition_allowed(states[from], states[to]), expected[from][to])
          << "from " << request_state_name(states[from]) << " to "
          << request_state_name(states[to]);
    }
  }
}

TEST(RequestTest, CancellationAndFailurePreserveContextFromEveryActiveState) {
  for (const RequestState target :
       {RequestState::kWaiting, RequestState::kPrefill, RequestState::kDecoding}) {
    Request cancelled = make_request(10 + static_cast<std::uint64_t>(target), 1, 2);
    Request failed = make_request(20 + static_cast<std::uint64_t>(target), 1, 2);
    if (target != RequestState::kWaiting) {
      cancelled.begin_prefill(kCreated);
      failed.begin_prefill(kCreated);
    }
    if (target == RequestState::kDecoding) {
      cancelled.record_prefill_progress(1, kCreated);
      failed.record_prefill_progress(1, kCreated);
    }

    cancelled.cancel(kCreated + 1ms, "client disconnected");
    EXPECT_EQ(cancelled.state(), RequestState::kCancelled);
    EXPECT_EQ(cancelled.finish_reason(), FinishReason::kCancelled);
    EXPECT_EQ(cancelled.terminal_context(), "client disconnected");

    failed.fail(kCreated + 1ms, "executor failed");
    EXPECT_EQ(failed.state(), RequestState::kFailed);
    EXPECT_EQ(failed.finish_reason(), FinishReason::kFailed);
    EXPECT_EQ(failed.terminal_context(), "executor failed");
  }
}

TEST(RequestTest, InvalidOperationsAndTimeRegressionAreTransactional) {
  EXPECT_THROW(static_cast<void>(Request(RequestId{}, prompt(1), 1, kCreated)), Error);
  EXPECT_THROW(static_cast<void>(Request(RequestId{1}, {}, 1, kCreated)), Error);

  Request request = make_request(30, 3, 2);
  EXPECT_THROW(request.record_prefill_progress(1, kCreated), Error);
  EXPECT_THROW(request.append_generated_token(TokenId{1}, kCreated), Error);
  EXPECT_THROW(request.finish_decoding(kCreated), Error);
  EXPECT_EQ(request.state(), RequestState::kWaiting);

  request.begin_prefill(kCreated + 5ms);
  EXPECT_THROW(request.begin_prefill(kCreated + 6ms), Error);
  EXPECT_THROW(request.record_prefill_progress(0, kCreated + 6ms), Error);
  EXPECT_THROW(request.record_prefill_progress(4, kCreated + 6ms), Error);
  EXPECT_THROW(request.record_prefill_progress(1, kCreated + 4ms), Error);
  EXPECT_EQ(request.prompt_tokens_processed(), 0U);
  EXPECT_EQ(request.state(), RequestState::kPrefill);

  EXPECT_THROW(request.cancel(kCreated + 6ms, ""), Error);
  EXPECT_THROW(request.fail(kCreated + 6ms, ""), Error);
  EXPECT_EQ(request.state(), RequestState::kPrefill);

  request.record_prefill_progress(3, kCreated + 7ms);
  request.cancel(kCreated + 8ms);
  EXPECT_THROW(request.cancel(kCreated + 9ms), Error);
  EXPECT_THROW(request.fail(kCreated + 9ms, "late failure"), Error);
  EXPECT_THROW(request.append_generated_token(TokenId{2}, kCreated + 9ms), Error);
  EXPECT_EQ(request.state(), RequestState::kCancelled);
  EXPECT_TRUE(request.generated_tokens().empty());
}

TEST(RequestTest, StateAndReasonNamesRejectUnknownValues) {
  EXPECT_EQ(request_state_name(RequestState::kDecoding), "decoding");
  EXPECT_EQ(finish_reason_name(FinishReason::kEndOfSequence), "end_of_sequence");
  EXPECT_THROW(static_cast<void>(request_state_name(static_cast<RequestState>(255))), Error);
  EXPECT_THROW(static_cast<void>(finish_reason_name(static_cast<FinishReason>(255))), Error);
}

TEST(RequestTest, RequestIsMoveOnlyAndRetainsSingleStateOwner) {
  static_assert(!std::is_copy_constructible_v<Request>);
  static_assert(!std::is_copy_assignable_v<Request>);
  static_assert(std::is_nothrow_move_constructible_v<Request>);
  static_assert(!std::is_move_assignable_v<Request>);

  Request source = make_request(40, 1, 1);
  source.begin_prefill(kCreated);
  Request destination(std::move(source));
  EXPECT_EQ(destination.id(), RequestId{40});
  EXPECT_EQ(destination.state(), RequestState::kPrefill);
  EXPECT_FALSE(source.id().valid());
  EXPECT_TRUE(source.terminal());
}

} // namespace
} // namespace helix::scheduler
