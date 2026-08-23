#pragma once

#include "helix/model/llama.hpp"

#include <filesystem>
#include <iosfwd>

namespace helix::model {

// The v1 container is little-endian and stores only the narrow Float32
// Llama-style architecture accepted by LlamaWeights::validate. Stream offsets
// are relative to the beginning of the stream; callers must provide a seekable
// stream.
struct LoadedLlamaModel final {
  LlamaConfig config;
  LlamaWeights weights;
};

void write_llama_model_file(std::ostream& output, const LlamaConfig& config,
                            const LlamaWeights& weights);
void write_llama_model_file(const std::filesystem::path& path, const LlamaConfig& config,
                            const LlamaWeights& weights);

[[nodiscard]] LoadedLlamaModel read_llama_model_file(std::istream& input);
[[nodiscard]] LoadedLlamaModel read_llama_model_file(const std::filesystem::path& path);

} // namespace helix::model
