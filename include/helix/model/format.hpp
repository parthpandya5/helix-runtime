#pragma once

#include "helix/model/llama.hpp"

#include <filesystem>
#include <iosfwd>
#include <span>

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

// Tooling path for streaming an already converted model without retaining a
// second in-memory LlamaWeights object. Each input file contains canonical
// little-endian Float32 payload bytes. Files are ordered as token embedding,
// nine tensors per layer (the LlamaLayerWeights member order), final norm, and
// LM head. Every file is size-checked, scanned for non-finite values, and
// checksummed before the output is opened.
void write_llama_model_file_from_raw_float32_files(
    const std::filesystem::path& path, const LlamaConfig& config,
    std::span<const std::filesystem::path> tensor_paths);

[[nodiscard]] LoadedLlamaModel read_llama_model_file(std::istream& input);
[[nodiscard]] LoadedLlamaModel read_llama_model_file(const std::filesystem::path& path);

// The mapped path is an opt-in POSIX/little-endian IEEE-754 optimization. It
// performs the same structural, checksum, padding, and finite-value validation
// as the copied loader, then exposes payloads through read-only Tensor storage
// whose lifetime owns the mapping. The source inode must not be modified while
// the model lives.
[[nodiscard]] bool mapped_model_loading_supported() noexcept;
[[nodiscard]] LoadedLlamaModel read_llama_model_file_mapped(const std::filesystem::path& path);

} // namespace helix::model
