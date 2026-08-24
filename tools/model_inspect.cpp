#include "helix/model/format.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::overflow_error("model parameter count exceeds this host's size range");
  }
  return lhs + rhs;
}

std::size_t parameter_count(const helix::model::LlamaWeights& weights) {
  std::size_t count = weights.token_embedding().numel();
  for (const helix::model::LlamaLayerWeights& layer : weights.layers()) {
    count = checked_add(count, layer.attention_norm_weight.numel());
    count = checked_add(count, layer.query_weight.numel());
    count = checked_add(count, layer.key_weight.numel());
    count = checked_add(count, layer.value_weight.numel());
    count = checked_add(count, layer.attention_output_weight.numel());
    count = checked_add(count, layer.ffn_norm_weight.numel());
    count = checked_add(count, layer.gate_weight.numel());
    count = checked_add(count, layer.up_weight.numel());
    count = checked_add(count, layer.down_weight.numel());
  }
  count = checked_add(count, weights.final_norm_weight().numel());
  return checked_add(count, weights.lm_head_weight().numel());
}

void print_usage(std::ostream& output) {
  output << "Usage: helix_model_inspect MODEL.helix "
            "[--smoke | --last-logits TOKEN[,TOKEN...]]\n"
            "Loads and validates the complete HELIXLM1 file. --smoke also runs a "
            "one-token CPU forward pass. --last-logits prints the final row as exact "
            "hexadecimal floats for validation tooling.\n";
}

std::vector<std::uint32_t> parse_tokens(std::string_view text) {
  std::vector<std::uint32_t> tokens;
  std::size_t offset = 0;
  while (offset <= text.size()) {
    const std::size_t separator = text.find(',', offset);
    const std::size_t end = separator == std::string_view::npos ? text.size() : separator;
    const std::string_view field = text.substr(offset, end - offset);
    std::uint32_t token = 0;
    const char* const begin = field.data();
    const char* const finish = begin + field.size();
    const auto [cursor, error] = std::from_chars(begin, finish, token);
    if (field.empty() || error != std::errc{} || cursor != finish) {
      throw std::invalid_argument("token list must contain comma-separated uint32 values");
    }
    tokens.push_back(token);
    if (separator == std::string_view::npos) {
      break;
    }
    offset = separator + 1;
  }
  return tokens;
}

} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4) {
      throw std::invalid_argument("expected a model path and optional validation mode");
    }
    const std::string_view first(argv[1]);
    if (first == "--help" || first == "-h") {
      print_usage(std::cout);
      return 0;
    }
    bool smoke = false;
    std::vector<std::uint32_t> requested_tokens;
    if (argc >= 3) {
      const std::string_view option(argv[2]);
      if (option == "--smoke" && argc == 3) {
        smoke = true;
      } else if (option == "--last-logits" && argc == 4) {
        requested_tokens = parse_tokens(argv[3]);
      } else {
        throw std::invalid_argument("invalid validation mode or argument count");
      }
    }
    helix::model::LoadedLlamaModel loaded =
        helix::model::read_llama_model_file(std::filesystem::path(argv[1]));
    const std::size_t parameters = parameter_count(loaded.weights);
    std::cout << "model_kind=llama\n"
              << "vocabulary_size=" << loaded.config.vocabulary_size << '\n'
              << "hidden_size=" << loaded.config.hidden_size << '\n'
              << "intermediate_size=" << loaded.config.intermediate_size << '\n'
              << "layer_count=" << loaded.config.layer_count << '\n'
              << "attention_head_count=" << loaded.config.attention_head_count << '\n'
              << "max_sequence_length=" << loaded.config.max_sequence_length << '\n'
              << "parameter_count=" << parameters << '\n';
    if (smoke || !requested_tokens.empty()) {
      helix::model::CpuLlamaModel model(loaded.config, std::move(loaded.weights));
      const std::vector<std::uint32_t> tokens =
          smoke ? std::vector<std::uint32_t>{0} : std::move(requested_tokens);
      const helix::Tensor logits = model.forward(tokens);
      const float* values = logits.data_as<float>();
      for (std::size_t index = 0; index < logits.numel(); ++index) {
        if (!std::isfinite(values[index])) {
          throw std::runtime_error("smoke forward produced non-finite logits");
        }
      }
      if (smoke) {
        std::cout << "smoke_logits=" << logits.shape()[0] << 'x' << logits.shape()[1] << '\n';
      } else {
        const std::size_t vocabulary = loaded.config.vocabulary_size;
        const std::size_t row_offset = (tokens.size() - 1) * vocabulary;
        std::cout << "last_logits_hex=" << std::hexfloat;
        for (std::size_t token = 0; token < vocabulary; ++token) {
          if (token != 0) {
            std::cout << ',';
          }
          std::cout << values[row_offset + token];
        }
        std::cout << std::defaultfloat << '\n';
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "helix_model_inspect: " << error.what() << '\n';
    print_usage(std::cerr);
    return 1;
  }
}
