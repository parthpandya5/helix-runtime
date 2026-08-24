#include "helix/model/format.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct Arguments final {
  std::filesystem::path output;
  std::filesystem::path tensor_directory;
  helix::model::LlamaConfig config;
};

void print_usage(std::ostream& output) {
  output << "Usage: helix_model_pack --output MODEL.helix --tensor-dir DIRECTORY "
            "--vocab-size N --hidden-size N --intermediate-size N --layers N "
            "--heads N --max-sequence-length N --rms-norm-epsilon F --rope-theta F\n\n"
            "DIRECTORY must contain canonical little-endian Float32 payloads named "
            "00000000.f32, 00000001.f32, ... in HELIXLM1 tensor order. Existing "
            "output files are never overwritten.\n";
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

float parse_float(std::string_view text, std::string_view name) {
  float value = 0.0F;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [cursor, error] = std::from_chars(begin, end, value);
  if (error != std::errc{} || cursor != end || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(name) +
                                " must be a finite number: " + std::string(text));
  }
  return value;
}

std::string_view option_value(std::span<char*> raw_arguments, std::size_t& index,
                              std::string_view option) {
  if (++index == raw_arguments.size()) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  return raw_arguments[index];
}

Arguments parse_arguments(std::span<char*> raw_arguments) {
  Arguments arguments;
  for (std::size_t index = 1; index < raw_arguments.size(); ++index) {
    const std::string_view option(raw_arguments[index]);
    if (option == "--help" || option == "-h") {
      print_usage(std::cout);
      std::exit(0);
    }
    if (option == "--output") {
      arguments.output = option_value(raw_arguments, index, option);
    } else if (option == "--tensor-dir") {
      arguments.tensor_directory = option_value(raw_arguments, index, option);
    } else if (option == "--vocab-size") {
      arguments.config.vocabulary_size =
          parse_integer<std::size_t>(option_value(raw_arguments, index, option), option);
    } else if (option == "--hidden-size") {
      arguments.config.hidden_size =
          parse_integer<std::size_t>(option_value(raw_arguments, index, option), option);
    } else if (option == "--intermediate-size") {
      arguments.config.intermediate_size =
          parse_integer<std::size_t>(option_value(raw_arguments, index, option), option);
    } else if (option == "--layers") {
      arguments.config.layer_count =
          parse_integer<std::size_t>(option_value(raw_arguments, index, option), option);
    } else if (option == "--heads") {
      arguments.config.attention_head_count =
          parse_integer<std::size_t>(option_value(raw_arguments, index, option), option);
    } else if (option == "--max-sequence-length") {
      arguments.config.max_sequence_length =
          parse_integer<std::size_t>(option_value(raw_arguments, index, option), option);
    } else if (option == "--rms-norm-epsilon") {
      arguments.config.rms_norm_epsilon =
          parse_float(option_value(raw_arguments, index, option), option);
    } else if (option == "--rope-theta") {
      arguments.config.rope_theta = parse_float(option_value(raw_arguments, index, option), option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (arguments.output.empty()) {
    throw std::invalid_argument("--output is required");
  }
  if (arguments.tensor_directory.empty()) {
    throw std::invalid_argument("--tensor-dir is required");
  }
  arguments.config.validate();
  return arguments;
}

std::vector<std::filesystem::path> tensor_paths(const Arguments& arguments) {
  if (arguments.config.layer_count > (std::numeric_limits<std::size_t>::max() - 3) / 9) {
    throw std::invalid_argument("layer count overflows the canonical tensor count");
  }
  const std::size_t count = 3 + arguments.config.layer_count * 9;
  std::vector<std::filesystem::path> paths;
  paths.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    std::ostringstream filename;
    filename << std::setfill('0') << std::setw(8) << index << ".f32";
    paths.push_back(arguments.tensor_directory / filename.str());
  }
  return paths;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments =
        parse_arguments(std::span<char*>(argv, static_cast<std::size_t>(argc)));
    if (std::filesystem::exists(arguments.output)) {
      throw std::invalid_argument("output already exists: " + arguments.output.string());
    }
    const std::vector<std::filesystem::path> paths = tensor_paths(arguments);
    struct PartialOutputCleanup final {
      const std::filesystem::path& path;
      bool complete = false;
      ~PartialOutputCleanup() {
        if (!complete) {
          std::error_code ignored;
          std::filesystem::remove(path, ignored);
        }
      }
    } cleanup{arguments.output};
    helix::model::write_llama_model_file_from_raw_float32_files(arguments.output, arguments.config,
                                                                paths);
    cleanup.complete = true;
    std::cout << "packed_model=" << arguments.output.string() << '\n'
              << "tensor_count=" << paths.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "helix_model_pack: " << error.what() << '\n';
    print_usage(std::cerr);
    return 1;
  }
}
