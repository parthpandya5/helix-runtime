#include "helix/core/error.hpp"
#include "llama_detail.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

namespace helix::model::detail {
namespace {

Dimension rope_dimension(std::size_t value, const char* name) {
  HELIX_CHECK(value <= static_cast<std::size_t>(std::numeric_limits<Dimension>::max()),
              ErrorCode::kOutOfRange,
              std::string("model::make_rope_cache: ") + name + " exceeds tensor dimension range");
  return static_cast<Dimension>(value);
}

} // namespace

RopeCache make_rope_cache(std::size_t tokens, std::size_t head_dimension, float theta,
                          std::size_t position_offset) {
  HELIX_CHECK(head_dimension != 0 && head_dimension % 2 == 0, ErrorCode::kInvalidArgument,
              "model::make_rope_cache: head dimension must be positive and even");
  HELIX_CHECK(std::isfinite(theta) && theta > 0.0F, ErrorCode::kInvalidArgument,
              "model::make_rope_cache: theta must be finite and positive");
  HELIX_CHECK(position_offset <= std::numeric_limits<std::size_t>::max() - tokens,
              ErrorCode::kOutOfRange, "model::make_rope_cache: position range overflows");

  const std::size_t pairs = head_dimension / 2;
  Tensor cosine =
      Tensor::empty({rope_dimension(tokens, "token count"), rope_dimension(pairs, "pair count")});
  Tensor sine = Tensor::empty(cosine.shape());
  auto* cosine_values = cosine.data_as<float>();
  auto* sine_values = sine.data_as<float>();
  for (std::size_t token = 0; token < tokens; ++token) {
    for (std::size_t pair = 0; pair < pairs; ++pair) {
      const double exponent =
          -2.0 * static_cast<double>(pair) / static_cast<double>(head_dimension);
      const double frequency = std::pow(static_cast<double>(theta), exponent);
      const double angle = static_cast<double>(position_offset + token) * frequency;
      cosine_values[token * pairs + pair] = static_cast<float>(std::cos(angle));
      sine_values[token * pairs + pair] = static_cast<float>(std::sin(angle));
    }
  }
  return RopeCache{std::move(cosine), std::move(sine)};
}

} // namespace helix::model::detail
