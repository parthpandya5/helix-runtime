#!/usr/bin/env python3
"""Validate a small Hugging Face Llama checkpoint against converted Helix logits.

This is an offline correctness tool, not a runtime dependency or a performant
model implementation. It independently reads source-layout safetensors, evaluates
the supported Llama equations with NumPy, and compares the complete final logits
row with the C++ Helix inspector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError as error:  # pragma: no cover - CLI environment dependent
    raise SystemExit(
        "validate_hf_llama.py requires NumPy; install "
        "tools/requirements-convert.txt in a conversion-only environment"
    ) from error


MAX_HEADER_BYTES = 100_000_000
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}


class ValidationError(RuntimeError):
    """A contextual validation failure suitable for command-line output."""


def strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"could not parse JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root in {path} must be an object")
    return value


def positive_integer(document: dict[str, Any], name: str) -> int:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"config field {name!r} must be a positive integer")
    return value


def finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"config field {name!r} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValidationError(f"config field {name!r} must be positive and finite")
    return result


@dataclass(frozen=True)
class ReferenceConfig:
    vocabulary_size: int
    hidden_size: int
    intermediate_size: int
    layer_count: int
    head_count: int
    max_sequence_length: int
    rms_norm_epsilon: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.head_count

    @property
    def parameter_count(self) -> int:
        per_layer = (
            2 * self.hidden_size
            + 4 * self.hidden_size * self.hidden_size
            + 3 * self.hidden_size * self.intermediate_size
        )
        return (
            2 * self.vocabulary_size * self.hidden_size
            + self.layer_count * per_layer
            + self.hidden_size
        )


def parse_config(path: Path) -> ReferenceConfig:
    document = read_json(path)
    if document.get("model_type") != "llama":
        raise ValidationError("config field 'model_type' must be exactly 'llama'")
    vocabulary = positive_integer(document, "vocab_size")
    hidden = positive_integer(document, "hidden_size")
    intermediate = positive_integer(document, "intermediate_size")
    layers = positive_integer(document, "num_hidden_layers")
    heads = positive_integer(document, "num_attention_heads")
    max_sequence = positive_integer(document, "max_position_embeddings")
    if hidden % heads != 0 or (hidden // heads) % 2 != 0:
        raise ValidationError("hidden/head geometry must produce an even head dimension")
    key_value_heads = document.get("num_key_value_heads", heads)
    if key_value_heads is None:
        key_value_heads = heads
    if isinstance(key_value_heads, bool) or not isinstance(key_value_heads, int):
        raise ValidationError("config field 'num_key_value_heads' must be an integer")
    if key_value_heads != heads:
        raise ValidationError("the reference validator does not support GQA/MQA")
    configured_head_dimension = document.get("head_dim", hidden // heads)
    if configured_head_dimension is None:
        configured_head_dimension = hidden // heads
    if (
        isinstance(configured_head_dimension, bool)
        or not isinstance(configured_head_dimension, int)
        or configured_head_dimension != hidden // heads
    ):
        raise ValidationError("head_dim must equal hidden_size / num_attention_heads")
    if document.get("hidden_act", "silu") != "silu":
        raise ValidationError("the reference validator supports only SiLU/SwiGLU")
    if document.get("attention_bias", False) or document.get("mlp_bias", False):
        raise ValidationError("projection biases are not supported")
    if finite_positive(document.get("partial_rotary_factor", 1.0), "partial_rotary_factor") != 1.0:
        raise ValidationError("partial rotary dimensions are not supported")
    if document.get("rope_scaling") is not None:
        raise ValidationError("only default, unscaled RoPE is supported")
    rope_parameters = document.get("rope_parameters")
    parameter_theta: float | None = None
    if rope_parameters is not None:
        if not isinstance(rope_parameters, dict):
            raise ValidationError("config field 'rope_parameters' must be an object")
        unknown = set(rope_parameters) - {
            "rope_type",
            "rope_theta",
            "partial_rotary_factor",
        }
        if unknown:
            raise ValidationError(
                "unsupported rope_parameters fields: " + ", ".join(sorted(unknown))
            )
        if rope_parameters.get("rope_type", "default") != "default":
            raise ValidationError("only default, unscaled RoPE is supported")
        if (
            finite_positive(
                rope_parameters.get("partial_rotary_factor", 1.0),
                "rope_parameters.partial_rotary_factor",
            )
            != 1.0
        ):
            raise ValidationError("partial rotary dimensions are not supported")
        if "rope_theta" in rope_parameters:
            parameter_theta = finite_positive(
                rope_parameters["rope_theta"], "rope_parameters.rope_theta"
            )
    theta = finite_positive(document.get("rope_theta", 10_000.0), "rope_theta")
    if parameter_theta is not None and parameter_theta != theta:
        raise ValidationError("rope_theta and rope_parameters.rope_theta disagree")
    tied = document.get("tie_word_embeddings", False)
    if not isinstance(tied, bool):
        raise ValidationError("config field 'tie_word_embeddings' must be a boolean")
    return ReferenceConfig(
        vocabulary,
        hidden,
        intermediate,
        layers,
        heads,
        max_sequence,
        finite_positive(document.get("rms_norm_eps", 1.0e-6), "rms_norm_eps"),
        theta,
        tied,
    )


@dataclass(frozen=True)
class TensorRecord:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


class SafeTensorFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            file_size = path.stat().st_size
            with path.open("rb") as source:
                length_data = source.read(8)
                if len(length_data) != 8:
                    raise ValidationError(f"{path} is missing a safetensors header")
                header_bytes = struct.unpack("<Q", length_data)[0]
                if header_bytes > MAX_HEADER_BYTES or header_bytes > file_size - 8:
                    raise ValidationError(f"safetensors header in {path} exceeds bounds")
                encoded = source.read(header_bytes)
                if len(encoded) != header_bytes:
                    raise ValidationError(f"safetensors header in {path} is truncated")
        except OSError as error:
            raise ValidationError(f"could not read safetensors file {path}: {error}") from error
        try:
            header = json.loads(encoded.decode("utf-8"), object_pairs_hook=strict_object)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"invalid safetensors header in {path}: {error}") from error
        if not isinstance(header, dict):
            raise ValidationError(f"safetensors header in {path} must be an object")
        metadata = header.pop("__metadata__", None)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationError(f"safetensors metadata in {path} must be an object")

        data_bytes = file_size - 8 - header_bytes
        records: dict[str, TensorRecord] = {}
        ranges: list[tuple[int, int, str]] = []
        for name, value in header.items():
            if not isinstance(name, str) or not name or not isinstance(value, dict):
                raise ValidationError(f"invalid tensor record in {path}")
            if set(value) != {"dtype", "shape", "data_offsets"}:
                raise ValidationError(f"tensor {name!r} has noncanonical metadata")
            dtype = value["dtype"]
            shape = value["shape"]
            offsets = value["data_offsets"]
            if dtype not in DTYPE_BYTES:
                raise ValidationError(f"tensor {name!r} uses unsupported dtype {dtype!r}")
            if not isinstance(shape, list) or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            ):
                raise ValidationError(f"tensor {name!r} has an invalid shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
            ):
                raise ValidationError(f"tensor {name!r} has invalid offsets")
            start, end = offsets
            expected_bytes = math.prod(shape) * DTYPE_BYTES[dtype]
            if start < 0 or end < start or end > data_bytes or end - start != expected_bytes:
                raise ValidationError(f"tensor {name!r} has an invalid byte range")
            records[name] = TensorRecord(dtype, tuple(shape), start, end)
            ranges.append((start, end, name))
        cursor = 0
        for start, end, name in sorted(ranges):
            if start != cursor:
                relationship = "overlaps prior data" if start < cursor else "leaves a data hole"
                raise ValidationError(f"tensor {name!r} in {path} {relationship}")
            cursor = end
        if cursor != data_bytes:
            raise ValidationError(f"safetensors data in {path} is not exactly covered")
        self.data_offset = 8 + header_bytes
        self.records = records

    def load(self, name: str, expected_shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
        record = self.records.get(name)
        if record is None:
            raise ValidationError(f"{self.path} is missing required tensor {name!r}")
        if record.shape != expected_shape:
            raise ValidationError(
                f"source tensor {name!r} shape {record.shape} does not match {expected_shape}"
            )
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[record.dtype]
        try:
            source = np.memmap(
                self.path,
                dtype=np.dtype(dtype),
                mode="r",
                offset=self.data_offset + record.start,
                shape=record.shape,
                order="C",
            )
        except (OSError, ValueError) as error:
            raise ValidationError(f"could not map source tensor {name!r}: {error}") from error
        if record.dtype == "BF16":
            bits = np.asarray(source, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
            result = bits.view(np.float32)
        else:
            result = np.asarray(source, dtype=np.float32)
        if not bool(np.isfinite(result).all()):
            raise ValidationError(f"source tensor {name!r} contains a non-finite value")
        return result


class Checkpoint:
    def __init__(self, directory: Path) -> None:
        try:
            self.directory = directory.resolve(strict=True)
        except OSError as error:
            raise ValidationError(f"could not resolve checkpoint {directory}: {error}") from error
        if not self.directory.is_dir():
            raise ValidationError(f"checkpoint path is not a directory: {directory}")
        self._files: dict[Path, SafeTensorFile] = {}
        index_path = self.directory / "model.safetensors.index.json"
        single_path = self.directory / "model.safetensors"
        if index_path.exists():
            index = read_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or any(
                not isinstance(name, str)
                or not name
                or not isinstance(filename, str)
                or not filename
                for name, filename in weight_map.items()
            ):
                raise ValidationError("safetensors index has an invalid weight_map")
            self.weight_map: dict[str, str] | None = dict(weight_map)
            self.single: SafeTensorFile | None = None
        elif single_path.is_file():
            self.weight_map = None
            self.single = SafeTensorFile(single_path)
        else:
            raise ValidationError("checkpoint has no safetensors weights")

    def source_files(self) -> list[Path]:
        if self.weight_map is None:
            assert self.single is not None
            return [self.single.path]
        return sorted({self._resolve(filename) for filename in self.weight_map.values()})

    def _resolve(self, filename: str) -> Path:
        candidate = self.directory / filename
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.directory)
        except (OSError, ValueError) as error:
            raise ValidationError(f"shard {filename!r} is not inside the checkpoint") from error
        if not resolved.is_file():
            raise ValidationError(f"safetensors shard is not a regular file: {resolved}")
        return resolved

    def has(self, name: str) -> bool:
        if self.weight_map is not None:
            return name in self.weight_map
        assert self.single is not None
        return name in self.single.records

    def load(self, name: str, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
        if self.weight_map is None:
            assert self.single is not None
            source = self.single
        else:
            filename = self.weight_map.get(name)
            if filename is None:
                raise ValidationError(f"checkpoint is missing required tensor {name!r}")
            path = self._resolve(filename)
            source = self._files.get(path)
            if source is None:
                source = SafeTensorFile(path)
                self._files[path] = source
        return source.load(name, shape)


def reference_linear(
    inputs: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    rows = inputs.reshape(-1, inputs.shape[-1])
    outputs = np.empty((rows.shape[0], weights.shape[0]), dtype=np.float32)
    for row in range(rows.shape[0]):
        for column in range(weights.shape[0]):
            outputs[row, column] = np.float32(
                np.sum(
                    rows[row].astype(np.float64) * weights[column].astype(np.float64),
                    dtype=np.float64,
                )
            )
    return outputs.reshape(*inputs.shape[:-1], weights.shape[0])


def reference_rms_norm(
    inputs: np.ndarray[Any, Any], weight: np.ndarray[Any, Any], epsilon: float
) -> np.ndarray[Any, Any]:
    output = np.empty_like(inputs, dtype=np.float32)
    rows = inputs.reshape(-1, inputs.shape[-1])
    output_rows = output.reshape(rows.shape)
    for row in range(rows.shape[0]):
        wide = rows[row].astype(np.float64)
        inverse_rms = 1.0 / math.sqrt(
            float(np.sum(wide * wide, dtype=np.float64)) / rows.shape[1] + epsilon
        )
        output_rows[row] = (wide * inverse_rms * weight.astype(np.float64)).astype(np.float32)
    return output


def reference_rope(values: np.ndarray[Any, Any], theta: float) -> np.ndarray[Any, Any]:
    tokens, _, head_dimension = values.shape
    half = head_dimension // 2
    output = np.empty_like(values, dtype=np.float32)
    for token in range(tokens):
        for frequency in range(half):
            angle = token * math.pow(theta, -2.0 * frequency / head_dimension)
            cosine = np.float32(math.cos(angle))
            sine = np.float32(math.sin(angle))
            first = values[token, :, frequency]
            second = values[token, :, half + frequency]
            output[token, :, frequency] = first * cosine - second * sine
            output[token, :, half + frequency] = second * cosine + first * sine
    return output


def reference_attention(
    query: np.ndarray[Any, Any],
    key: np.ndarray[Any, Any],
    value: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    tokens, heads, head_dimension = query.shape
    output = np.empty_like(query, dtype=np.float32)
    scale = 1.0 / math.sqrt(head_dimension)
    for head in range(heads):
        for query_index in range(tokens):
            scores = np.empty(query_index + 1, dtype=np.float32)
            for key_index in range(query_index + 1):
                dot = np.sum(
                    query[query_index, head].astype(np.float64)
                    * key[key_index, head].astype(np.float64),
                    dtype=np.float64,
                )
                scores[key_index] = np.float32(float(dot) * scale)
            maximum = np.max(scores)
            exponentials = np.exp((scores - maximum).astype(np.float64))
            probabilities = (exponentials / np.sum(exponentials)).astype(np.float32)
            for dimension in range(head_dimension):
                output[query_index, head, dimension] = np.float32(
                    np.sum(
                        probabilities.astype(np.float64)
                        * value[: query_index + 1, head, dimension].astype(np.float64),
                        dtype=np.float64,
                    )
                )
    return output


def reference_swiglu(
    gate: np.ndarray[Any, Any], up: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    output = np.empty_like(gate, dtype=np.float32)
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    output_flat = output.reshape(-1)
    for index in range(gate_flat.size):
        value = float(gate_flat[index])
        if value >= 0.0:
            sigmoid = 1.0 / (1.0 + math.exp(-value))
        else:
            exponential = math.exp(value)
            sigmoid = exponential / (1.0 + exponential)
        output_flat[index] = np.float32(value * sigmoid) * up_flat[index]
    return output


def reference_last_logits(
    config: ReferenceConfig, checkpoint: Checkpoint, tokens: Sequence[int]
) -> np.ndarray[Any, Any]:
    hidden = checkpoint.load(
        "model.embed_tokens.weight", (config.vocabulary_size, config.hidden_size)
    )[list(tokens)].astype(np.float32)
    for layer in range(config.layer_count):
        prefix = f"model.layers.{layer}"
        normalized = reference_rms_norm(
            hidden,
            checkpoint.load(f"{prefix}.input_layernorm.weight", (config.hidden_size,)),
            config.rms_norm_epsilon,
        )
        projection_shape = (config.hidden_size, config.hidden_size)
        query = reference_linear(
            normalized,
            checkpoint.load(f"{prefix}.self_attn.q_proj.weight", projection_shape),
        ).reshape(len(tokens), config.head_count, config.head_dimension)
        key = reference_linear(
            normalized,
            checkpoint.load(f"{prefix}.self_attn.k_proj.weight", projection_shape),
        ).reshape(len(tokens), config.head_count, config.head_dimension)
        value = reference_linear(
            normalized,
            checkpoint.load(f"{prefix}.self_attn.v_proj.weight", projection_shape),
        ).reshape(len(tokens), config.head_count, config.head_dimension)
        query = reference_rope(query, config.rope_theta)
        key = reference_rope(key, config.rope_theta)
        context = reference_attention(query, key, value).reshape(
            len(tokens), config.hidden_size
        )
        attention_output = reference_linear(
            context,
            checkpoint.load(f"{prefix}.self_attn.o_proj.weight", projection_shape),
        )
        hidden = (hidden + attention_output).astype(np.float32)

        normalized = reference_rms_norm(
            hidden,
            checkpoint.load(
                f"{prefix}.post_attention_layernorm.weight", (config.hidden_size,)
            ),
            config.rms_norm_epsilon,
        )
        feed_forward_shape = (config.intermediate_size, config.hidden_size)
        gate = reference_linear(
            normalized,
            checkpoint.load(f"{prefix}.mlp.gate_proj.weight", feed_forward_shape),
        )
        up = reference_linear(
            normalized,
            checkpoint.load(f"{prefix}.mlp.up_proj.weight", feed_forward_shape),
        )
        activated = reference_swiglu(gate, up)
        down = reference_linear(
            activated,
            checkpoint.load(
                f"{prefix}.mlp.down_proj.weight",
                (config.hidden_size, config.intermediate_size),
            ),
        )
        hidden = (hidden + down).astype(np.float32)

    normalized = reference_rms_norm(
        hidden,
        checkpoint.load("model.norm.weight", (config.hidden_size,)),
        config.rms_norm_epsilon,
    )
    head_name = "lm_head.weight"
    if not checkpoint.has(head_name):
        if not config.tie_word_embeddings:
            raise ValidationError("checkpoint is missing required tensor 'lm_head.weight'")
        head_name = "model.embed_tokens.weight"
    head = checkpoint.load(head_name, (config.vocabulary_size, config.hidden_size))
    return reference_linear(normalized, head)[-1]


def parse_tokens(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(field, 10) for field in text.split(","))
    except ValueError as error:
        raise ValidationError("--tokens must contain comma-separated decimal integers") from error
    if not values or any(value < 0 for value in values):
        raise ValidationError("--tokens must contain nonnegative token ids")
    return values


def inspector_logits(inspector: Path, model: Path, tokens: Sequence[int]) -> np.ndarray[Any, Any]:
    command = [
        str(inspector),
        str(model),
        "--last-logits",
        ",".join(str(token) for token in tokens),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise ValidationError(f"could not execute Helix inspector: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValidationError(f"Helix inspector failed: {detail}")
    lines = [line for line in completed.stdout.splitlines() if line.startswith("last_logits_hex=")]
    if len(lines) != 1:
        raise ValidationError("Helix inspector did not emit exactly one last_logits_hex record")
    try:
        values = [float.fromhex(field) for field in lines[0].split("=", 1)[1].split(",")]
    except ValueError as error:
        raise ValidationError("Helix inspector emitted an invalid hexadecimal float") from error
    result = np.asarray(values, dtype=np.float32)
    if not bool(np.isfinite(result).all()):
        raise ValidationError("Helix inspector emitted non-finite logits")
    return result


def transformers_logits(
    checkpoint: Path, tokens: Sequence[int]
) -> tuple[np.ndarray[Any, Any], str, str]:
    try:
        import torch
        import transformers
        from transformers import LlamaForCausalLM
    except ImportError as error:
        raise ValidationError(
            "--transformers requires optional torch and transformers packages"
        ) from error
    try:
        torch.set_num_threads(1)
        model = LlamaForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float32,
            local_files_only=True,
            use_safetensors=True,
        )
        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=torch.tensor([tokens], dtype=torch.long),
                use_cache=False,
            ).logits[0, -1]
        result = output.detach().cpu().numpy().astype(np.float32, copy=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValidationError(f"Transformers reference execution failed: {error}") from error
    if not bool(np.isfinite(result).all()):
        raise ValidationError("Transformers reference emitted non-finite logits")
    return result, str(torch.__version__), str(transformers.__version__)


def comparison_errors(
    actual: np.ndarray[Any, Any],
    expected: np.ndarray[Any, Any],
    *,
    label: str,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    if actual.shape != expected.shape:
        raise ValidationError(
            f"{label} logit count mismatch: Helix emitted {actual.size}, expected {expected.size}"
        )
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float32).tiny)
    relative = absolute / denominator
    close = np.isclose(actual, expected, rtol=rtol, atol=atol)
    if not bool(close.all()):
        index = int(np.flatnonzero(~close)[0])
        raise ValidationError(
            f"{label} logit mismatch at index {index}: Helix={float(actual[index]).hex()}, "
            f"reference={float(expected[index]).hex()}, abs_error={absolute[index]:.9g}, "
            f"rel_error={relative[index]:.9g}"
        )
    return float(np.max(absolute)), float(np.max(relative))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValidationError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def validate(options: argparse.Namespace) -> None:
    checkpoint = Checkpoint(options.checkpoint)
    config_path = checkpoint.directory / "config.json"
    config = parse_config(config_path)
    if options.max_parameters <= 0:
        raise ValidationError("--max-parameters must be positive")
    if config.parameter_count > options.max_parameters:
        raise ValidationError(
            f"reference model has {config.parameter_count} parameters, exceeding "
            f"--max-parameters={options.max_parameters}; this bounded oracle is for small models"
        )
    tokens = parse_tokens(options.tokens)
    if len(tokens) > config.max_sequence_length:
        raise ValidationError("token sequence exceeds max_position_embeddings")
    if any(token >= config.vocabulary_size for token in tokens):
        raise ValidationError("token id is outside the checkpoint vocabulary")
    if (
        not math.isfinite(options.rtol)
        or not math.isfinite(options.atol)
        or options.rtol < 0.0
        or options.atol < 0.0
    ):
        raise ValidationError("--rtol and --atol must be finite and nonnegative")

    expected = reference_last_logits(config, checkpoint, tokens)
    actual = inspector_logits(options.inspector.resolve(), options.model.resolve(), tokens)
    source_max_absolute, source_max_relative = comparison_errors(
        actual,
        expected,
        label="source-layout oracle",
        rtol=options.rtol,
        atol=options.atol,
    )

    framework_result: tuple[float, float, str, str] | None = None
    if options.transformers:
        external, torch_version, transformers_version = transformers_logits(
            checkpoint.directory, tokens
        )
        framework_max_absolute, framework_max_relative = comparison_errors(
            actual,
            external,
            label="Transformers",
            rtol=options.rtol,
            atol=options.atol,
        )
        framework_result = (
            framework_max_absolute,
            framework_max_relative,
            torch_version,
            transformers_version,
        )

    print("validation=passed")
    print(f"tokens={','.join(str(token) for token in tokens)}")
    print(f"parameter_count={config.parameter_count}")
    print(f"logit_count={expected.size}")
    print(f"rtol={options.rtol:.9g}")
    print(f"atol={options.atol:.9g}")
    print(f"source_max_abs_error={source_max_absolute:.9g}")
    print(f"source_max_rel_error={source_max_relative:.9g}")
    if framework_result is not None:
        framework_max_absolute, framework_max_relative, torch_version, transformers_version = (
            framework_result
        )
        print("transformers_validation=passed")
        print(f"torch_version={torch_version}")
        print(f"transformers_version={transformers_version}")
        print(f"transformers_max_abs_error={framework_max_absolute:.9g}")
        print(f"transformers_max_rel_error={framework_max_relative:.9g}")
    print(f"config_sha256={sha256(config_path)}")
    for path in checkpoint.source_files():
        print(f"weights_sha256[{path.name}]={sha256(path)}")


def parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a small source-layout HF Llama checkpoint with Helix logits."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--inspector", required=True, type=Path)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    parser.add_argument("--atol", type=float, default=2.0e-6)
    parser.add_argument("--max-parameters", type=int, default=5_000_000)
    parser.add_argument(
        "--transformers",
        action="store_true",
        help="also compare with an optional local CPU Float32 Transformers reference",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        validate(parse_arguments(sys.argv[1:] if arguments is None else arguments))
        return 0
    except ValidationError as error:
        print(f"validate_hf_llama.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
