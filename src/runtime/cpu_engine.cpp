#include "helix/runtime/cpu_engine.hpp"

#include "helix/core/error.hpp"
#include "helix/model/generation.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace helix::runtime {
namespace {

std::string engine_context(std::string_view operation) {
  return "runtime::CpuRuntimeEngine::" + std::string(operation);
}

std::size_t blocks_for_tokens(std::size_t tokens, std::size_t block_tokens) noexcept {
  return tokens / block_tokens + static_cast<std::size_t>(tokens % block_tokens != 0);
}

kv_cache::PagedKvCacheConfig cache_config(const model::CpuLlamaModel& model,
                                          const CpuRuntimeConfig& config) {
  return kv_cache::PagedKvCacheConfig{
      model.config().layer_count, model.config().attention_head_count,
      model.config().head_dimension(), config.scheduler.kv_block_tokens, config.physical_kv_blocks};
}

std::string work_failure_context(scheduler::RequestId request_id, const std::exception& error) {
  std::ostringstream message;
  message << "CPU model execution failed for request " << request_id.value() << ": "
          << error.what();
  return message.str();
}

} // namespace

struct CpuRuntimeEngine::Impl {
  struct Session {
    kv_cache::PagedSequenceHandle sequence;
    std::optional<scheduler::TokenId> end_of_sequence;
    std::optional<Tensor> next_logits;
  };

  Impl(model::CpuLlamaModel owned_model, CpuRuntimeConfig runtime_config)
      : config(std::move(runtime_config)), model(std::move(owned_model)),
        cache(cache_config(model, config)), scheduler(config.scheduler) {
    HELIX_CHECK(config.prefill_mode == CpuPrefillMode::kSequentialDecode ||
                    config.prefill_mode == CpuPrefillMode::kFullPrompt,
                ErrorCode::kInvalidArgument,
                engine_context("constructor") + ": unknown CPU prefill mode");
    HELIX_CHECK(config.paged_decode_mode == model::PagedDecodeMode::kCompactCopy ||
                    config.paged_decode_mode == model::PagedDecodeMode::kDirectBlockTable,
                ErrorCode::kInvalidArgument,
                engine_context("constructor") + ": unknown paged decode mode");
  }

  void validate_submission(const scheduler::Request& request,
                           std::optional<scheduler::TokenId> end_of_sequence) const {
    constexpr std::string_view kOperation = "submit";
    HELIX_CHECK(request.state() == scheduler::RequestState::kWaiting, ErrorCode::kInvalidArgument,
                engine_context(kOperation) + ": request must be waiting");
    HELIX_CHECK(!sessions.contains(request.id()), ErrorCode::kInvalidArgument,
                engine_context(kOperation) + ": duplicate runtime request ID");
    for (const scheduler::TokenId token : request.prompt_tokens()) {
      HELIX_CHECK(token.value() < model.config().vocabulary_size, ErrorCode::kOutOfRange,
                  engine_context(kOperation) + ": prompt token is outside the vocabulary");
    }
    if (end_of_sequence.has_value()) {
      HELIX_CHECK(end_of_sequence->value() < model.config().vocabulary_size, ErrorCode::kOutOfRange,
                  engine_context(kOperation) + ": EOS token is outside the vocabulary");
    }

    const std::size_t prompt_tokens = request.prompt_tokens().size();
    HELIX_CHECK(prompt_tokens <= model.config().max_sequence_length, ErrorCode::kOutOfRange,
                engine_context(kOperation) + ": prompt exceeds model context capacity");
    HELIX_CHECK(request.max_new_tokens() <= model.config().max_sequence_length - prompt_tokens,
                ErrorCode::kOutOfRange,
                engine_context(kOperation) +
                    ": prompt plus generation budget exceeds model context capacity");
    // The last length-limited token is returned directly from existing logits and
    // is not fed back through the model. Every earlier generated token is cached
    // because it is needed to produce a successor.
    const std::size_t cached_generated_tokens =
        request.max_new_tokens() == 0 ? 0 : request.max_new_tokens() - 1;
    const std::size_t maximum_runtime_tokens = prompt_tokens + cached_generated_tokens;
    const std::size_t required_blocks =
        blocks_for_tokens(maximum_runtime_tokens, config.scheduler.kv_block_tokens);
    HELIX_CHECK(required_blocks <= config.physical_kv_blocks, ErrorCode::kResourceExhausted,
                engine_context(kOperation) +
                    ": request cannot fit in the runtime KV pool even when run alone");
  }

  [[nodiscard]] Session& session(scheduler::RequestId request_id) {
    const auto position = sessions.find(request_id);
    HELIX_CHECK(position != sessions.end(), ErrorCode::kInternal,
                engine_context("session") + ": active request has no model session");
    return position->second;
  }

  void release_session(scheduler::RequestId request_id) {
    const auto position = sessions.find(request_id);
    HELIX_CHECK(position != sessions.end(), ErrorCode::kInternal,
                engine_context("release_session") + ": request has no model session");
    cache.release_sequence(position->second.sequence);
    sessions.erase(position);
  }

  [[nodiscard]] scheduler::WorkCompletion execute_prefill(const scheduler::ScheduledWork& work) {
    Session& active = session(work.request_id);
    const scheduler::Request& request = scheduler.request(work.request_id);
    HELIX_CHECK(cache.length(active.sequence) == work.token_offset, ErrorCode::kInternal,
                engine_context("execute_prefill") +
                    ": physical KV length does not match scheduled prompt offset");
    HELIX_CHECK(work.token_offset <= request.prompt_tokens().size() &&
                    work.token_count <= request.prompt_tokens().size() - work.token_offset,
                ErrorCode::kInternal,
                engine_context("execute_prefill") + ": scheduled prompt range is invalid");

    if (config.prefill_mode == CpuPrefillMode::kFullPrompt && work.token_offset == 0 &&
        work.token_count == request.prompt_tokens().size()) {
      HELIX_CHECK(full_prompt_prefill_count != std::numeric_limits<std::uint64_t>::max(),
                  ErrorCode::kResourceExhausted,
                  engine_context("execute_prefill") + ": full-prefill counter exhausted");
      std::vector<std::uint32_t> token_ids;
      token_ids.reserve(request.prompt_tokens().size());
      for (const scheduler::TokenId token : request.prompt_tokens()) {
        token_ids.push_back(token.value());
      }
      model::CpuLlamaPrefillResult prefill = model.prefill_with_kv(token_ids);
      Tensor next_logits =
          Tensor::empty({1, static_cast<Dimension>(model.config().vocabulary_size)},
                        DType::kFloat32, Device::cpu());
      const std::size_t vocabulary = model.config().vocabulary_size;
      std::copy_n(prefill.logits.data_as<float>() +
                      (request.prompt_tokens().size() - 1) * vocabulary,
                  vocabulary, next_logits.data_as<float>());
      cache.append_layer_major(active.sequence, prefill.layer_keys, prefill.layer_values);
      active.next_logits = std::move(next_logits);
      ++full_prompt_prefill_count;
      return scheduler::WorkCompletion{
          work.request_id, scheduler::WorkStatus::kSucceeded, std::nullopt, false, {}};
    }

    for (std::size_t index = 0; index < work.token_count; ++index) {
      HELIX_CHECK(sequential_prefill_token_count != std::numeric_limits<std::uint64_t>::max(),
                  ErrorCode::kResourceExhausted,
                  engine_context("execute_prefill") + ": sequential-prefill counter exhausted");
      const scheduler::TokenId token = request.prompt_tokens()[work.token_offset + index];
      model::CpuLlamaDecodeResult decoded =
          model.decode_one(token.value(), cache.length(active.sequence), cache, active.sequence,
                           config.paged_decode_mode);
      cache.append(active.sequence, decoded.token_keys, decoded.token_values);
      active.next_logits = std::move(decoded.logits);
      ++sequential_prefill_token_count;
    }
    return scheduler::WorkCompletion{
        work.request_id, scheduler::WorkStatus::kSucceeded, std::nullopt, false, {}};
  }

  [[nodiscard]] scheduler::WorkCompletion execute_decode(const scheduler::ScheduledWork& work) {
    Session& active = session(work.request_id);
    const scheduler::Request& request = scheduler.request(work.request_id);
    HELIX_CHECK(active.next_logits.has_value(), ErrorCode::kInternal,
                engine_context("execute_decode") + ": request has no logits to sample");
    HELIX_CHECK(cache.length(active.sequence) == work.token_offset, ErrorCode::kInternal,
                engine_context("execute_decode") +
                    ": physical KV length does not match scheduled decode offset");

    const scheduler::TokenId generated(model::select_greedy_token(*active.next_logits));
    const bool end_of_sequence =
        active.end_of_sequence.has_value() && generated == *active.end_of_sequence;
    const bool terminal = end_of_sequence || request.remaining_generation_tokens() == 1;
    if (!terminal) {
      model::CpuLlamaDecodeResult decoded =
          model.decode_one(generated.value(), cache.length(active.sequence), cache, active.sequence,
                           config.paged_decode_mode);
      cache.append(active.sequence, decoded.token_keys, decoded.token_values);
      active.next_logits = std::move(decoded.logits);
    }
    return scheduler::WorkCompletion{
        work.request_id, scheduler::WorkStatus::kSucceeded, generated, end_of_sequence, {}};
  }

  [[nodiscard]] scheduler::WorkCompletion execute(const scheduler::ScheduledWork& work) {
    return work.kind == scheduler::WorkKind::kPrefill ? execute_prefill(work)
                                                      : execute_decode(work);
  }

  CpuRuntimeConfig config;
  model::CpuLlamaModel model;
  kv_cache::PagedKvCache cache;
  scheduler::FifoScheduler scheduler;
  std::map<scheduler::RequestId, Session> sessions;
  std::uint64_t executed_iteration_count = 0;
  std::uint64_t failed_work_item_count = 0;
  std::uint64_t full_prompt_prefill_count = 0;
  std::uint64_t sequential_prefill_token_count = 0;
};

CpuRuntimeEngine::CpuRuntimeEngine(model::CpuLlamaModel model, CpuRuntimeConfig config)
    : impl_(std::make_unique<Impl>(std::move(model), std::move(config))) {}

CpuRuntimeEngine::~CpuRuntimeEngine() = default;

void CpuRuntimeEngine::submit(scheduler::Request request,
                              std::optional<scheduler::TokenId> end_of_sequence) {
  impl_->validate_submission(request, end_of_sequence);
  const scheduler::RequestId request_id = request.id();
  const kv_cache::PagedSequenceHandle sequence = impl_->cache.create_sequence();
  try {
    const auto [position, inserted] =
        impl_->sessions.emplace(request_id, Impl::Session{sequence, end_of_sequence, std::nullopt});
    static_cast<void>(position);
    HELIX_ASSERT(inserted, "runtime duplicate session passed prevalidation");
  } catch (...) {
    impl_->cache.release_sequence(sequence);
    throw;
  }

  try {
    impl_->scheduler.submit(std::move(request));
  } catch (...) {
    impl_->cache.release_sequence(sequence);
    impl_->sessions.erase(request_id);
    throw;
  }
}

CpuRuntimeIterationResult CpuRuntimeEngine::run_iteration(scheduler::RequestTimePoint now) {
  CpuRuntimeIterationResult result;
  result.free_kv_blocks_before = impl_->cache.stats().free_blocks;
  // Reserve before scheduling because schedule_iteration opens a transactional
  // in-flight batch. Subsequent completion pushes cannot then strand that batch
  // merely because this public result vector needs to grow.
  result.completions.reserve(impl_->config.scheduler.max_running_requests);
  scheduler::ScheduleResult schedule =
      impl_->scheduler.schedule_iteration(result.free_kv_blocks_before, now);
  result.token_budget_exhausted = schedule.token_budget_exhausted;
  result.kv_capacity_limited = schedule.kv_capacity_limited;
  if (!schedule.batch.has_value()) {
    result.free_kv_blocks_after = impl_->cache.stats().free_blocks;
    return result;
  }

  HELIX_CHECK(impl_->executed_iteration_count != std::numeric_limits<std::uint64_t>::max(),
              ErrorCode::kResourceExhausted,
              engine_context("run_iteration") + ": execution counter exhausted");
  scheduler::IterationBatch& batch = *schedule.batch;
  result.iteration_id = batch.id;
  result.work = std::move(batch.work);
  for (const scheduler::ScheduledWork& work : result.work) {
    if (work.kind == scheduler::WorkKind::kPrefill) {
      result.scheduled_prefill_tokens += work.token_count;
    } else {
      result.scheduled_decode_tokens += work.token_count;
    }
    try {
      result.completions.push_back(impl_->execute(work));
    } catch (const std::exception& error) {
      HELIX_CHECK(impl_->failed_work_item_count != std::numeric_limits<std::uint64_t>::max(),
                  ErrorCode::kResourceExhausted,
                  engine_context("run_iteration") + ": failed-work counter exhausted");
      ++impl_->failed_work_item_count;
      result.completions.push_back(
          scheduler::WorkCompletion{work.request_id, scheduler::WorkStatus::kFailed, std::nullopt,
                                    false, work_failure_context(work.request_id, error)});
    }
  }

  scheduler::CompletionReport completion =
      impl_->scheduler.complete_iteration(batch.id, result.completions, now);
  for (const scheduler::RetiredRequest& retired : completion.retired_requests) {
    impl_->release_session(retired.request_id);
  }
  result.retired_requests = std::move(completion.retired_requests);
  ++impl_->executed_iteration_count;
  result.free_kv_blocks_after = impl_->cache.stats().free_blocks;
  return result;
}

scheduler::RetiredRequest CpuRuntimeEngine::cancel(scheduler::RequestId request_id,
                                                   scheduler::RequestTimePoint now,
                                                   std::string context) {
  scheduler::RetiredRequest retired = impl_->scheduler.cancel(request_id, now, std::move(context));
  impl_->release_session(request_id);
  return retired;
}

bool CpuRuntimeEngine::contains(scheduler::RequestId request_id) const noexcept {
  return impl_->scheduler.contains(request_id);
}

bool CpuRuntimeEngine::has_live_requests() const noexcept {
  return impl_->scheduler.has_live_requests();
}

const scheduler::Request& CpuRuntimeEngine::request(scheduler::RequestId request_id) const {
  return impl_->scheduler.request(request_id);
}

scheduler::SchedulerMetrics
CpuRuntimeEngine::metrics(scheduler::RequestTimePoint observed_at) const {
  return impl_->scheduler.metrics(observed_at);
}

CpuRuntimeStats CpuRuntimeEngine::stats() const {
  return CpuRuntimeStats{impl_->scheduler.stats(),
                         impl_->cache.stats(),
                         impl_->sessions.size(),
                         impl_->executed_iteration_count,
                         impl_->failed_work_item_count,
                         impl_->full_prompt_prefill_count,
                         impl_->sequential_prefill_token_count};
}

const CpuRuntimeConfig& CpuRuntimeEngine::config() const noexcept {
  return impl_->config;
}

const model::LlamaConfig& CpuRuntimeEngine::model_config() const noexcept {
  return impl_->model.config();
}

} // namespace helix::runtime
