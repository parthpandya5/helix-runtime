#!/usr/bin/env python3
"""Convert a narrowly supported Hugging Face Llama checkpoint to HELIXLM1.

This is an offline conversion utility, not part of the Helix runtime. It parses
safetensors directly, uses NumPy for bounded-memory layout conversion, and asks
the C++ helix_model_pack tool to validate and serialize the canonical container.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ModuleNotFoundError as error:  # pragma: no cover - exercised by CLI environments
    raise SystemExit(
        "convert_hf_llama.py requires NumPy for vectorized checkpoint conversion; "
        "install tools/requirements-convert.txt in a conversion-only environment"
    ) from error


MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
SUPPORTED_DTYPES = {"F32", "F16", "BF16"}
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
RAW_FILENAME_WIDTH = 8


class ConversionError(RuntimeError):
    """A contextual, user-facing checkpoint conversion error."""


def _strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source, object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"could not parse JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConversionError(f"JSON root in {path} must be an object")
    return value


def _require_integer(document: dict[str, Any], name: str) -> int:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversionError(f"config field {name!r} must be a positive integer")
    return value


def _optional_boolean(document: dict[str, Any], name: str, default: bool) -> bool:
    value = document.get(name, default)
    if not isinstance(value, bool):
        raise ConversionError(f"config field {name!r} must be a boolean")
    return value


def _positive_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionError(f"config field {name!r} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConversionError(f"config field {name!r} must be positive and finite")
    return result


@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    vocabulary_size: int
    hidden_size: int
    intermediate_size: int
    layer_count: int
    attention_head_count: int
    max_sequence_length: int
    rms_norm_epsilon: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.attention_head_count


def _parse_config(path: Path) -> LlamaConfig:
    document = _read_json(path)
    if document.get("model_type") != "llama":
        raise ConversionError("config field 'model_type' must be exactly 'llama'")
    vocabulary = _require_integer(document, "vocab_size")
    hidden = _require_integer(document, "hidden_size")
    intermediate = _require_integer(document, "intermediate_size")
    layers = _require_integer(document, "num_hidden_layers")
    heads = _require_integer(document, "num_attention_heads")
    max_sequence = _require_integer(document, "max_position_embeddings")
    if hidden % heads != 0:
        raise ConversionError("hidden_size must be divisible by num_attention_heads")
    head_dimension = hidden // heads
    if head_dimension % 2 != 0:
        raise ConversionError("the full rotary head dimension must be even")

    key_value_heads = document.get("num_key_value_heads", heads)
    if key_value_heads is None:
        key_value_heads = heads
    if isinstance(key_value_heads, bool) or not isinstance(key_value_heads, int):
        raise ConversionError("config field 'num_key_value_heads' must be an integer")
    if key_value_heads != heads:
        raise ConversionError(
            "Helix currently requires num_key_value_heads == num_attention_heads; "
            "GQA/MQA checkpoints are not supported"
        )

    configured_head_dimension = document.get("head_dim", head_dimension)
    if configured_head_dimension is None:
        configured_head_dimension = head_dimension
    if configured_head_dimension != head_dimension:
        raise ConversionError(
            "Helix requires head_dim == hidden_size / num_attention_heads"
        )
    if document.get("hidden_act", "silu") != "silu":
        raise ConversionError("Helix currently supports only hidden_act='silu'")
    if _optional_boolean(document, "attention_bias", False):
        raise ConversionError("attention projection biases are not supported")
    if _optional_boolean(document, "mlp_bias", False):
        raise ConversionError("MLP projection biases are not supported")

    partial_rotary = _positive_finite_number(
        document.get("partial_rotary_factor", 1.0), "partial_rotary_factor"
    )
    if partial_rotary != 1.0:
        raise ConversionError("partial rotary dimensions are not supported")
    rope_scaling = document.get("rope_scaling")
    if rope_scaling is not None:
        raise ConversionError("scaled or dynamic RoPE configurations are not supported")

    rope_parameters = document.get("rope_parameters")
    parameter_theta: float | None = None
    if rope_parameters is not None:
        if not isinstance(rope_parameters, dict):
            raise ConversionError("config field 'rope_parameters' must be an object")
        unknown = set(rope_parameters) - {
            "rope_type",
            "rope_theta",
            "partial_rotary_factor",
        }
        if unknown:
            raise ConversionError(
                "unsupported rope_parameters fields: " + ", ".join(sorted(unknown))
            )
        if rope_parameters.get("rope_type", "default") != "default":
            raise ConversionError("only default, unscaled RoPE is supported")
        parameter_partial = _positive_finite_number(
            rope_parameters.get("partial_rotary_factor", 1.0),
            "rope_parameters.partial_rotary_factor",
        )
        if parameter_partial != 1.0:
            raise ConversionError("partial rotary dimensions are not supported")
        if "rope_theta" in rope_parameters:
            parameter_theta = _positive_finite_number(
                rope_parameters["rope_theta"], "rope_parameters.rope_theta"
            )

    theta = _positive_finite_number(document.get("rope_theta", 10_000.0), "rope_theta")
    if parameter_theta is not None and parameter_theta != theta:
        raise ConversionError("rope_theta and rope_parameters.rope_theta disagree")

    return LlamaConfig(
        vocabulary_size=vocabulary,
        hidden_size=hidden,
        intermediate_size=intermediate,
        layer_count=layers,
        attention_head_count=heads,
        max_sequence_length=max_sequence,
        rms_norm_epsilon=_positive_finite_number(
            document.get("rms_norm_eps", 1.0e-6), "rms_norm_eps"
        ),
        rope_theta=theta,
        tie_word_embeddings=_optional_boolean(document, "tie_word_embeddings", False),
    )


@dataclasses.dataclass(frozen=True)
class SafeTensorInfo:
    name: str
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
                length_bytes = source.read(8)
                if len(length_bytes) != 8:
                    raise ConversionError(f"safetensors file {path} is missing its header length")
                header_length = struct.unpack("<Q", length_bytes)[0]
                if header_length > MAX_SAFETENSORS_HEADER_BYTES:
                    raise ConversionError(
                        f"safetensors header in {path} exceeds "
                        f"{MAX_SAFETENSORS_HEADER_BYTES} bytes"
                    )
                if header_length > file_size - 8:
                    raise ConversionError(f"safetensors header in {path} exceeds file bounds")
                header_bytes = source.read(header_length)
                if len(header_bytes) != header_length:
                    raise ConversionError(f"safetensors header in {path} is truncated")
        except OSError as error:
            raise ConversionError(f"could not read safetensors file {path}: {error}") from error

        try:
            header = json.loads(
                header_bytes.decode("utf-8"), object_pairs_hook=_strict_object
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ConversionError(f"invalid safetensors JSON header in {path}: {error}") from error
        if not isinstance(header, dict):
            raise ConversionError(f"safetensors header in {path} must be an object")
        metadata = header.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items())
        ):
            raise ConversionError(f"safetensors __metadata__ in {path} must map strings to strings")

        data_bytes = file_size - 8 - header_length
        tensors: dict[str, SafeTensorInfo] = {}
        ranges: list[tuple[int, int, str]] = []
        for name, record in header.items():
            if not isinstance(name, str) or not name:
                raise ConversionError(f"safetensors tensor names in {path} must be nonempty strings")
            if not isinstance(record, dict):
                raise ConversionError(f"safetensors tensor {name!r} in {path} has invalid metadata")
            if set(record) != {"dtype", "shape", "data_offsets"}:
                raise ConversionError(
                    f"safetensors tensor {name!r} in {path} has noncanonical metadata fields"
                )
            dtype = record["dtype"]
            if dtype not in SUPPORTED_DTYPES:
                raise ConversionError(
                    f"safetensors tensor {name!r} uses unsupported dtype {dtype!r}; "
                    "expected F32, F16, or BF16"
                )
            shape_value = record["shape"]
            if not isinstance(shape_value, list) or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape_value
            ):
                raise ConversionError(f"safetensors tensor {name!r} has an invalid shape")
            offsets = record["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets)
            ):
                raise ConversionError(f"safetensors tensor {name!r} has invalid data offsets")
            start, end = offsets
            if start < 0 or end < start or end > data_bytes:
                raise ConversionError(f"safetensors tensor {name!r} exceeds data-buffer bounds")
            element_count = math.prod(shape_value)
            expected_bytes = element_count * DTYPE_BYTES[dtype]
            if end - start != expected_bytes:
                raise ConversionError(
                    f"safetensors tensor {name!r} byte range does not match its shape/dtype"
                )
            info = SafeTensorInfo(name, dtype, tuple(shape_value), start, end)
            tensors[name] = info
            ranges.append((start, end, name))

        cursor = 0
        for start, end, name in sorted(ranges):
            if start != cursor:
                relationship = "overlaps a prior tensor" if start < cursor else "leaves a data hole"
                raise ConversionError(f"safetensors tensor {name!r} in {path} {relationship}")
            cursor = end
        if cursor != data_bytes:
            raise ConversionError(f"safetensors data buffer in {path} has trailing unclaimed bytes")

        self.data_offset = 8 + header_length
        self.tensors = tensors

    def array(self, info: SafeTensorInfo) -> np.memmap[Any, Any]:
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[info.dtype]
        return np.memmap(
            self.path,
            dtype=np.dtype(dtype),
            mode="r",
            offset=self.data_offset + info.start,
            shape=info.shape,
            order="C",
        )


class Checkpoint:
    def __init__(self, directory: Path) -> None:
        try:
            self.directory = directory.resolve(strict=True)
        except OSError as error:
            raise ConversionError(f"could not resolve checkpoint directory {directory}: {error}") from error
        if not self.directory.is_dir():
            raise ConversionError(f"checkpoint path is not a directory: {directory}")

        self._files: dict[Path, SafeTensorFile] = {}
        index_path = self.directory / "model.safetensors.index.json"
        single_path = self.directory / "model.safetensors"
        if index_path.exists():
            index = _read_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or any(
                not isinstance(name, str)
                or not name
                or not isinstance(filename, str)
                or not filename
                for name, filename in weight_map.items()
            ):
                raise ConversionError("safetensors index weight_map must map tensor names to files")
            self._weight_map = dict(weight_map)
            self._single_file: SafeTensorFile | None = None
        elif single_path.exists():
            self._single_file = SafeTensorFile(single_path)
            self._weight_map = None
        else:
            raise ConversionError(
                f"{self.directory} contains neither model.safetensors nor "
                "model.safetensors.index.json"
            )

    @property
    def names(self) -> set[str]:
        if self._weight_map is not None:
            return set(self._weight_map)
        assert self._single_file is not None
        return set(self._single_file.tensors)

    def _resolve_shard(self, filename: str) -> Path:
        candidate = self.directory / filename
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.directory)
        except (OSError, ValueError) as error:
            raise ConversionError(
                f"safetensors index shard {filename!r} does not resolve inside the checkpoint"
            ) from error
        if not resolved.is_file():
            raise ConversionError(f"safetensors shard is not a regular file: {resolved}")
        return resolved

    def tensor(self, name: str) -> tuple[SafeTensorFile, SafeTensorInfo]:
        if self._weight_map is None:
            assert self._single_file is not None
            source = self._single_file
        else:
            filename = self._weight_map.get(name)
            if filename is None:
                raise ConversionError(f"checkpoint is missing required tensor {name!r}")
            path = self._resolve_shard(filename)
            source = self._files.get(path)
            if source is None:
                source = SafeTensorFile(path)
                self._files[path] = source
        info = source.tensors.get(name)
        if info is None:
            raise ConversionError(
                f"checkpoint metadata maps tensor {name!r} to {source.path}, but the shard lacks it"
            )
        return source, info


@dataclasses.dataclass(frozen=True)
class TensorPlan:
    target_name: str
    target_shape: tuple[int, ...]
    source_name: str
    source_shape: tuple[int, ...]
    transform: str


def _tensor_plans(config: LlamaConfig, checkpoint: Checkpoint) -> list[TensorPlan]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    plans = [
        TensorPlan(
            "token_embedding",
            (config.vocabulary_size, hidden),
            "model.embed_tokens.weight",
            (config.vocabulary_size, hidden),
            "identity",
        )
    ]
    bias_names: list[str] = []
    for layer in range(config.layer_count):
        prefix = f"model.layers.{layer}"
        attention = f"{prefix}.self_attn"
        mlp = f"{prefix}.mlp"
        plans.extend(
            [
                TensorPlan(
                    f"layer.{layer}.attention_norm",
                    (hidden,),
                    f"{prefix}.input_layernorm.weight",
                    (hidden,),
                    "identity",
                ),
                TensorPlan(
                    f"layer.{layer}.query",
                    (hidden, hidden),
                    f"{attention}.q_proj.weight",
                    (hidden, hidden),
                    "transpose_rotary",
                ),
                TensorPlan(
                    f"layer.{layer}.key",
                    (hidden, hidden),
                    f"{attention}.k_proj.weight",
                    (hidden, hidden),
                    "transpose_rotary",
                ),
                TensorPlan(
                    f"layer.{layer}.value",
                    (hidden, hidden),
                    f"{attention}.v_proj.weight",
                    (hidden, hidden),
                    "transpose",
                ),
                TensorPlan(
                    f"layer.{layer}.attention_output",
                    (hidden, hidden),
                    f"{attention}.o_proj.weight",
                    (hidden, hidden),
                    "transpose",
                ),
                TensorPlan(
                    f"layer.{layer}.ffn_norm",
                    (hidden,),
                    f"{prefix}.post_attention_layernorm.weight",
                    (hidden,),
                    "identity",
                ),
                TensorPlan(
                    f"layer.{layer}.gate",
                    (hidden, intermediate),
                    f"{mlp}.gate_proj.weight",
                    (intermediate, hidden),
                    "transpose",
                ),
                TensorPlan(
                    f"layer.{layer}.up",
                    (hidden, intermediate),
                    f"{mlp}.up_proj.weight",
                    (intermediate, hidden),
                    "transpose",
                ),
                TensorPlan(
                    f"layer.{layer}.down",
                    (intermediate, hidden),
                    f"{mlp}.down_proj.weight",
                    (hidden, intermediate),
                    "transpose",
                ),
            ]
        )
        bias_names.extend(
            [
                f"{attention}.{projection}_proj.bias"
                for projection in ("q", "k", "v", "o")
            ]
        )
        bias_names.extend(
            [f"{mlp}.{projection}_proj.bias" for projection in ("gate", "up", "down")]
        )

    plans.append(
        TensorPlan("final_norm", (hidden,), "model.norm.weight", (hidden,), "identity")
    )
    lm_source = "lm_head.weight"
    if lm_source not in checkpoint.names:
        if not config.tie_word_embeddings:
            raise ConversionError("checkpoint is missing required tensor 'lm_head.weight'")
        lm_source = "model.embed_tokens.weight"
    plans.append(
        TensorPlan(
            "lm_head",
            (hidden, config.vocabulary_size),
            lm_source,
            (config.vocabulary_size, hidden),
            "transpose",
        )
    )

    present_biases = sorted(set(bias_names).intersection(checkpoint.names))
    if present_biases:
        raise ConversionError(
            "checkpoint contains unsupported projection bias tensor(s): "
            + ", ".join(present_biases)
        )
    for plan in plans:
        _, info = checkpoint.tensor(plan.source_name)
        if info.shape != plan.source_shape:
            raise ConversionError(
                f"source tensor {plan.source_name!r} shape {info.shape} does not match "
                f"expected {plan.source_shape} for {plan.target_name}"
            )
    return plans


def _as_float32(values: np.ndarray[Any, Any], dtype: str) -> np.ndarray[Any, Any]:
    if dtype == "BF16":
        bits = values.astype(np.uint32, copy=False) << np.uint32(16)
        return bits.view(np.float32)
    return values.astype(np.float32, copy=False)


def _rotary_output_permutation(config: LlamaConfig) -> np.ndarray[Any, Any]:
    half = config.head_dimension // 2
    permutation: list[int] = []
    for head in range(config.attention_head_count):
        base = head * config.head_dimension
        for frequency in range(half):
            permutation.extend((base + frequency, base + half + frequency))
    return np.asarray(permutation, dtype=np.intp)


def _write_float32_payload(
    checkpoint: Checkpoint,
    plan: TensorPlan,
    destination: Path,
    config: LlamaConfig,
    chunk_bytes: int,
) -> None:
    source, info = checkpoint.tensor(plan.source_name)
    array = source.array(info)
    try:
        with destination.open("xb") as output:
            if plan.transform == "identity":
                flat = array.reshape(-1)
                elements_per_chunk = max(1, chunk_bytes // 4)
                for start in range(0, flat.size, elements_per_chunk):
                    converted = _as_float32(flat[start : start + elements_per_chunk], info.dtype)
                    if not bool(np.isfinite(converted).all()):
                        raise ConversionError(f"source tensor {plan.source_name!r} is non-finite")
                    canonical = np.ascontiguousarray(converted, dtype=np.dtype("<f4"))
                    output.write(canonical.tobytes(order="C"))
            elif plan.transform in {"transpose", "transpose_rotary"}:
                target_rows, target_columns = plan.target_shape
                rows_per_chunk = max(1, chunk_bytes // (target_columns * 4))
                permutation = (
                    _rotary_output_permutation(config)
                    if plan.transform == "transpose_rotary"
                    else None
                )
                for start in range(0, target_rows, rows_per_chunk):
                    end = min(target_rows, start + rows_per_chunk)
                    block = (
                        array[:, start:end].T
                        if permutation is None
                        else array[permutation, start:end].T
                    )
                    converted = _as_float32(block, info.dtype)
                    if not bool(np.isfinite(converted).all()):
                        raise ConversionError(f"source tensor {plan.source_name!r} is non-finite")
                    canonical = np.ascontiguousarray(converted, dtype=np.dtype("<f4"))
                    output.write(canonical.tobytes(order="C"))
            else:  # pragma: no cover - construction invariant
                raise AssertionError(f"unknown transform {plan.transform}")
            output.flush()
        expected_bytes = math.prod(plan.target_shape) * 4
        if destination.stat().st_size != expected_bytes:
            raise ConversionError(
                f"converted tensor {plan.target_name} has the wrong byte length"
            )
    except OSError as error:
        raise ConversionError(f"could not write converted tensor {destination}: {error}") from error
    finally:
        del array


def _find_packer(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    environment = os.environ.get("HELIX_MODEL_PACK")
    if environment:
        candidates.append(Path(environment))
    executable = shutil.which("helix_model_pack")
    if executable:
        candidates.append(Path(executable))
    repository = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            repository / "build" / "tools" / "helix_model_pack",
            repository / "build-release" / "tools" / "helix_model_pack",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise ConversionError(
        "could not find helix_model_pack; build it with CMake or pass --packer PATH"
    )


def _packer_command(
    packer: Path,
    staging: Path,
    output: Path,
    config: LlamaConfig,
) -> list[str]:
    return [
        str(packer),
        "--output",
        str(output),
        "--tensor-dir",
        str(staging),
        "--vocab-size",
        str(config.vocabulary_size),
        "--hidden-size",
        str(config.hidden_size),
        "--intermediate-size",
        str(config.intermediate_size),
        "--layers",
        str(config.layer_count),
        "--heads",
        str(config.attention_head_count),
        "--max-sequence-length",
        str(config.max_sequence_length),
        "--rms-norm-epsilon",
        format(config.rms_norm_epsilon, ".9g"),
        "--rope-theta",
        format(config.rope_theta, ".9g"),
    ]


def convert(
    input_directory: Path,
    output_path: Path,
    packer_path: Path | None,
    chunk_mebibytes: int,
    force: bool,
    quiet: bool = False,
) -> None:
    if chunk_mebibytes <= 0:
        raise ConversionError("--chunk-mib must be positive")
    output = output_path.expanduser().resolve()
    if output.exists() and not force:
        raise ConversionError(f"output already exists (pass --force to replace): {output}")
    try:
        output.parent.mkdir(parents=False, exist_ok=True)
    except OSError as error:
        raise ConversionError(f"could not prepare output directory {output.parent}: {error}") from error

    checkpoint = Checkpoint(input_directory)
    config = _parse_config(checkpoint.directory / "config.json")
    plans = _tensor_plans(config, checkpoint)
    packer = _find_packer(packer_path)
    chunk_bytes = chunk_mebibytes * 1024 * 1024
    raw_bytes = sum(math.prod(plan.target_shape) * 4 for plan in plans)
    if not quiet:
        print(
            f"validated Llama checkpoint: layers={config.layer_count}, "
            f"hidden={config.hidden_size}, tensors={len(plans)}"
        )
        print(
            f"Float32 payload={raw_bytes / (1024 ** 3):.3f} GiB; conversion staging "
            "temporarily requires roughly twice that space"
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.convert-", dir=output.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        for index, plan in enumerate(plans):
            if not quiet:
                print(f"[{index + 1}/{len(plans)}] {plan.target_name}")
            destination = staging / f"{index:0{RAW_FILENAME_WIDTH}d}.f32"
            _write_float32_payload(
                checkpoint, plan, destination, config, chunk_bytes
            )

        packed_output = staging / "converted.helix"
        command = _packer_command(packer, staging, packed_output, config)
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            raise ConversionError(f"could not execute helix_model_pack: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ConversionError(f"helix_model_pack failed: {detail}")
        try:
            os.replace(packed_output, output)
        except OSError as error:
            raise ConversionError(f"could not atomically install {output}: {error}") from error
    if not quiet:
        print(f"wrote {output}")


def _parse_arguments(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a supported Hugging Face Llama safetensors checkpoint to HELIXLM1."
    )
    parser.add_argument("--input", required=True, type=Path, help="checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="output .helix file")
    parser.add_argument(
        "--packer", type=Path, help="path to the built helix_model_pack executable"
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=64,
        help="approximate NumPy conversion chunk size (default: 64 MiB)",
    )
    parser.add_argument(
        "--force", action="store_true", help="atomically replace an existing output"
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        options = _parse_arguments(sys.argv[1:] if arguments is None else arguments)
        convert(
            options.input,
            options.output,
            options.packer,
            options.chunk_mib,
            options.force,
            options.quiet,
        )
        return 0
    except ConversionError as error:
        print(f"convert_hf_llama.py: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
