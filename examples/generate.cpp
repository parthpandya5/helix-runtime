#include "helix/model/format.hpp"
#include "helix/model/generation.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

struct Arguments {
  std::filesystem::path model_path;
  std::size_t max_new_tokens = 32;
  std::optional<std::uint32_t> eos_token;
  std::vector<std::uint32_t> prompt_tokens;
};

void print_usage(std::ostream& output) {
  output << "Usage: helix_generate --model MODEL.helix [--max-new-tokens N] "
            "[--eos TOKEN] TOKEN [TOKEN ...]\n"
            "\n"
            "This M18 baseline accepts token IDs directly; it does not include a tokenizer.\n";
}

template <typename Integer> Integer parse_integer(std::string_view text, std::string_view name) {
  Integer value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [cursor, error] = std::from_chars(begin, end, value);
  if (error != std::errc{} || cursor != end) {
    throw std::invalid_argument(std::string(name) +
                                " must be a nonnegative integer: " + std::string(text));
  }
  return value;
}

Arguments parse_arguments(std::span<char*> raw_arguments) {
  Arguments arguments;
  for (std::size_t index = 1; index < raw_arguments.size(); ++index) {
    const std::string_view value(raw_arguments[index]);
    if (value == "--help" || value == "-h") {
      print_usage(std::cout);
      std::exit(0);
    }
    if (value == "--model") {
      if (++index == raw_arguments.size()) {
        throw std::invalid_argument("--model requires a path");
      }
      arguments.model_path = raw_arguments[index];
      continue;
    }
    if (value == "--max-new-tokens") {
      if (++index == raw_arguments.size()) {
        throw std::invalid_argument("--max-new-tokens requires a value");
      }
      arguments.max_new_tokens =
          parse_integer<std::size_t>(raw_arguments[index], "--max-new-tokens");
      continue;
    }
    if (value == "--eos") {
      if (++index == raw_arguments.size()) {
        throw std::invalid_argument("--eos requires a token ID");
      }
      arguments.eos_token = parse_integer<std::uint32_t>(raw_arguments[index], "--eos");
      continue;
    }
    if (!value.empty() && value.front() == '-') {
      throw std::invalid_argument("unknown option: " + std::string(value));
    }
    arguments.prompt_tokens.push_back(parse_integer<std::uint32_t>(value, "prompt token"));
  }

  if (arguments.model_path.empty()) {
    throw std::invalid_argument("--model is required");
  }
  if (arguments.prompt_tokens.empty()) {
    throw std::invalid_argument("at least one prompt token ID is required");
  }
  return arguments;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments =
        parse_arguments(std::span<char*>(argv, static_cast<std::size_t>(argc)));
    helix::model::LoadedLlamaModel loaded =
        helix::model::read_llama_model_file(arguments.model_path);
    helix::model::CpuLlamaGenerator generator(
        helix::model::CpuLlamaModel(loaded.config, std::move(loaded.weights)));
    const helix::model::GenerationResult result = generator.generate(
        arguments.prompt_tokens,
        helix::model::GenerationOptions{arguments.max_new_tokens, arguments.eos_token});

    std::cout << "generated_token_ids:";
    for (const std::uint32_t token : result.generated_tokens) {
      std::cout << ' ' << token;
    }
    std::cout << "\nstop_reason: " << helix::model::generation_stop_reason_name(result.stop_reason)
              << "\ncached_token_count: " << result.cached_token_count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "helix_generate: " << error.what() << '\n';
    print_usage(std::cerr);
    return 1;
  }
}
