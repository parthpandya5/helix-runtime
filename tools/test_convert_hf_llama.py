#!/usr/bin/env python3
"""Synthetic end-to-end tests for convert_hf_llama.py."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CONVERTER = Path()
PACKER = Path()
INSPECTOR = Path()
FNV_OFFSET = 14_695_981_039_346_656_037
FNV_PRIME = 1_099_511_628_211
GLOBAL_LAYER = 0xFFFFFFFF


@dataclass(frozen=True)
class ExpectedTensor:
    name: str
    role: int
    layer: int
    shape: tuple[int, ...]
    values: np.ndarray[Any, Any]


def fnv(data: bytes, initial: int = FNV_OFFSET) -> int:
    result = initial
    for byte in data:
        result ^= byte
        result = (result * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return result


def base_config(tied: bool = False) -> dict[str, Any]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 7,
        "hidden_size": 8,
        "intermediate_size": 10,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 12,
        "rms_norm_eps": 1.0e-5,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": tied,
    }


def source_specs(config: dict[str, Any], include_lm_head: bool = True) -> list[tuple[str, tuple[int, ...]]]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    result: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (config["vocab_size"], hidden))
    ]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        result.extend(
            [
                (f"{prefix}.input_layernorm.weight", (hidden,)),
                (f"{prefix}.self_attn.q_proj.weight", (hidden, hidden)),
                (f"{prefix}.self_attn.k_proj.weight", (hidden, hidden)),
                (f"{prefix}.self_attn.v_proj.weight", (hidden, hidden)),
                (f"{prefix}.self_attn.o_proj.weight", (hidden, hidden)),
                (f"{prefix}.post_attention_layernorm.weight", (hidden,)),
                (f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden)),
                (f"{prefix}.mlp.up_proj.weight", (intermediate, hidden)),
                (f"{prefix}.mlp.down_proj.weight", (hidden, intermediate)),
            ]
        )
    result.append(("model.norm.weight", (hidden,)))
    if include_lm_head:
        result.append(("lm_head.weight", (config["vocab_size"], hidden)))
    return result


def deterministic_values(name: str, shape: tuple[int, ...], seed: int) -> np.ndarray[Any, Any]:
    count = math.prod(shape)
    if "layernorm" in name or name == "model.norm.weight":
        values = 0.75 + (np.arange(count, dtype=np.float32) + seed) % 11 / 64.0
    else:
        integers = ((np.arange(count, dtype=np.int32) * 3 + seed) % 29) - 14
        values = integers.astype(np.float32) / 64.0
    return values.reshape(shape).astype(np.float32)


def encode_tensor(values: np.ndarray[Any, Any], dtype: str) -> tuple[bytes, np.ndarray[Any, Any]]:
    if dtype == "F32":
        converted = values.astype(np.dtype("<f4"))
        return converted.tobytes(order="C"), converted.astype(np.float32)
    if dtype == "F16":
        converted = values.astype(np.dtype("<f2"))
        return converted.tobytes(order="C"), converted.astype(np.float32)
    if dtype == "BF16":
        float_values = values.astype(np.dtype("<f4"))
        bits = float_values.view(np.dtype("<u4"))
        short_bits = (bits >> np.uint32(16)).astype(np.dtype("<u2"))
        decoded = (short_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        return short_bits.tobytes(order="C"), decoded.reshape(values.shape)
    raise AssertionError(dtype)


def write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, np.ndarray[Any, Any]]],
) -> dict[str, np.ndarray[Any, Any]]:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    payloads: list[bytes] = []
    decoded: dict[str, np.ndarray[Any, Any]] = {}
    cursor = 0
    for name, dtype, values in tensors:
        payload, expected = encode_tensor(values, dtype)
        header[name] = {
            "dtype": dtype,
            "shape": list(values.shape),
            "data_offsets": [cursor, cursor + len(payload)],
        }
        cursor += len(payload)
        payloads.append(payload)
        decoded[name] = expected.copy()
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payloads))
    return decoded


def write_checkpoint(
    directory: Path,
    config: dict[str, Any],
    *,
    sharded: bool,
    include_lm_head: bool = True,
    shape_override: tuple[str, tuple[int, ...]] | None = None,
    non_finite_name: str | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: list[tuple[str, str, np.ndarray[Any, Any]]] = []
    dtype_cycle = ("F32", "F16", "BF16")
    for index, (name, original_shape) in enumerate(source_specs(config, include_lm_head)):
        shape = shape_override[1] if shape_override and name == shape_override[0] else original_shape
        values = deterministic_values(name, shape, index + 3)
        if non_finite_name == name:
            values.reshape(-1)[0] = np.inf
        tensors.append((name, dtype_cycle[index % len(dtype_cycle)], values))

    if not sharded:
        return write_safetensors(directory / "model.safetensors", tensors)

    first = tensors[::2]
    second = tensors[1::2]
    decoded = write_safetensors(directory / "model-00001-of-00002.safetensors", first)
    decoded.update(write_safetensors(directory / "model-00002-of-00002.safetensors", second))
    weight_map = {
        name: "model-00001-of-00002.safetensors" for name, _, _ in first
    }
    weight_map.update(
        {name: "model-00002-of-00002.safetensors" for name, _, _ in second}
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return decoded


def rotary_permutation(config: dict[str, Any]) -> np.ndarray[Any, Any]:
    head_dimension = config["hidden_size"] // config["num_attention_heads"]
    half = head_dimension // 2
    result: list[int] = []
    for head in range(config["num_attention_heads"]):
        base = head * head_dimension
        for frequency in range(half):
            result.extend((base + frequency, base + half + frequency))
    return np.asarray(result, dtype=np.intp)


def expected_tensors(
    config: dict[str, Any], sources: dict[str, np.ndarray[Any, Any]]
) -> list[ExpectedTensor]:
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    permutation = rotary_permutation(config)
    result = [
        ExpectedTensor(
            "token_embedding",
            1,
            GLOBAL_LAYER,
            (config["vocab_size"], hidden),
            sources["model.embed_tokens.weight"],
        )
    ]
    roles = {
        "attention_norm": 2,
        "query": 3,
        "key": 4,
        "value": 5,
        "attention_output": 6,
        "ffn_norm": 7,
        "gate": 8,
        "up": 9,
        "down": 10,
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        mappings = [
            ("attention_norm", f"{prefix}.input_layernorm.weight", (hidden,), "identity"),
            ("query", f"{prefix}.self_attn.q_proj.weight", (hidden, hidden), "rotary"),
            ("key", f"{prefix}.self_attn.k_proj.weight", (hidden, hidden), "rotary"),
            ("value", f"{prefix}.self_attn.v_proj.weight", (hidden, hidden), "transpose"),
            (
                "attention_output",
                f"{prefix}.self_attn.o_proj.weight",
                (hidden, hidden),
                "transpose",
            ),
            ("ffn_norm", f"{prefix}.post_attention_layernorm.weight", (hidden,), "identity"),
            ("gate", f"{prefix}.mlp.gate_proj.weight", (hidden, intermediate), "transpose"),
            ("up", f"{prefix}.mlp.up_proj.weight", (hidden, intermediate), "transpose"),
            ("down", f"{prefix}.mlp.down_proj.weight", (intermediate, hidden), "transpose"),
        ]
        for target, source, shape, transform in mappings:
            values = sources[source]
            if transform == "transpose":
                values = values.T
            elif transform == "rotary":
                values = values[permutation, :].T
            result.append(
                ExpectedTensor(f"layer.{layer}.{target}", roles[target], layer, shape, values)
            )
    result.append(
        ExpectedTensor("final_norm", 11, GLOBAL_LAYER, (hidden,), sources["model.norm.weight"])
    )
    lm_source = sources.get("lm_head.weight", sources["model.embed_tokens.weight"])
    result.append(
        ExpectedTensor(
            "lm_head",
            12,
            GLOBAL_LAYER,
            (hidden, config["vocab_size"]),
            lm_source.T,
        )
    )
    return result


def reference_linear(
    inputs: np.ndarray[Any, Any], output_input_weights: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    rows = inputs.reshape(-1, inputs.shape[-1])
    outputs = np.empty((rows.shape[0], output_input_weights.shape[0]), dtype=np.float32)
    for row in range(rows.shape[0]):
        for column in range(output_input_weights.shape[0]):
            outputs[row, column] = np.float32(
                np.sum(
                    rows[row].astype(np.float64)
                    * output_input_weights[column].astype(np.float64),
                    dtype=np.float64,
                )
            )
    return outputs.reshape(*inputs.shape[:-1], output_input_weights.shape[0])


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
        output_rows[row] = (wide * inverse_rms * weight.astype(np.float64)).astype(
            np.float32
        )
    return output


def reference_half_split_rope(
    values: np.ndarray[Any, Any], theta: float
) -> np.ndarray[Any, Any]:
    tokens, heads, head_dimension = values.shape
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
        silu = np.float32(value * sigmoid)
        output_flat[index] = silu * up_flat[index]
    return output


def reference_hf_last_logits(
    config: dict[str, Any],
    sources: dict[str, np.ndarray[Any, Any]],
    tokens: Sequence[int],
) -> np.ndarray[Any, Any]:
    hidden = sources["model.embed_tokens.weight"][list(tokens)].astype(np.float32)
    heads = config["num_attention_heads"]
    head_dimension = config["hidden_size"] // heads
    epsilon = float(np.float32(config["rms_norm_eps"]))
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        normalized = reference_rms_norm(
            hidden, sources[f"{prefix}.input_layernorm.weight"], epsilon
        )
        query = reference_linear(
            normalized, sources[f"{prefix}.self_attn.q_proj.weight"]
        ).reshape(len(tokens), heads, head_dimension)
        key = reference_linear(
            normalized, sources[f"{prefix}.self_attn.k_proj.weight"]
        ).reshape(len(tokens), heads, head_dimension)
        value = reference_linear(
            normalized, sources[f"{prefix}.self_attn.v_proj.weight"]
        ).reshape(len(tokens), heads, head_dimension)
        query = reference_half_split_rope(query, config["rope_theta"])
        key = reference_half_split_rope(key, config["rope_theta"])
        context = reference_attention(query, key, value).reshape(
            len(tokens), config["hidden_size"]
        )
        attention_output = reference_linear(
            context, sources[f"{prefix}.self_attn.o_proj.weight"]
        )
        hidden = (hidden + attention_output).astype(np.float32)

        normalized = reference_rms_norm(
            hidden, sources[f"{prefix}.post_attention_layernorm.weight"], epsilon
        )
        gate = reference_linear(normalized, sources[f"{prefix}.mlp.gate_proj.weight"])
        up = reference_linear(normalized, sources[f"{prefix}.mlp.up_proj.weight"])
        activated = reference_swiglu(gate, up)
        down = reference_linear(activated, sources[f"{prefix}.mlp.down_proj.weight"])
        hidden = (hidden + down).astype(np.float32)

    normalized = reference_rms_norm(hidden, sources["model.norm.weight"], epsilon)
    lm_head = sources.get("lm_head.weight", sources["model.embed_tokens.weight"])
    return reference_linear(normalized, lm_head)[-1]


def validate_helix_file(
    path: Path, config: dict[str, Any], expected: list[ExpectedTensor]
) -> None:
    content = path.read_bytes()
    fields = struct.unpack_from("<8sIIII11QffQ", content, 0)
    (
        magic,
        version,
        endian,
        header_bytes,
        model_kind,
        file_size,
        directory_offset,
        tensor_count,
        directory_entry_bytes,
        data_offset,
        vocabulary,
        hidden,
        intermediate,
        layers,
        heads,
        max_sequence,
        epsilon,
        theta,
        metadata_checksum,
    ) = fields
    assert magic == b"HELIXLM1"
    assert (version, endian, header_bytes, model_kind) == (1, 0x01020304, 128, 1)
    assert file_size == len(content)
    assert (directory_offset, directory_entry_bytes, tensor_count) == (128, 80, len(expected))
    assert data_offset % 64 == 0
    assert (vocabulary, hidden, intermediate, layers, heads, max_sequence) == (
        config["vocab_size"],
        config["hidden_size"],
        config["intermediate_size"],
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["max_position_embeddings"],
    )
    assert epsilon == np.float32(config["rms_norm_eps"])
    assert theta == np.float32(config["rope_theta"])
    metadata = bytearray(content[: 128 + 80 * len(expected)])
    metadata[120:128] = b"\0" * 8
    assert fnv(bytes(metadata)) == metadata_checksum

    for index, tensor in enumerate(expected):
        record = struct.unpack_from("<IIII8Q", content, 128 + index * 80)
        role, layer, dtype, rank = record[:4]
        dimensions = record[4:8]
        offset, byte_length, alignment, checksum = record[8:]
        assert (role, layer, dtype, rank) == (tensor.role, tensor.layer, 1, len(tensor.shape))
        assert dimensions == tensor.shape + (0,) * (4 - len(tensor.shape))
        assert alignment == 64 and offset % 64 == 0
        payload = content[offset : offset + byte_length]
        assert len(payload) == math.prod(tensor.shape) * 4
        assert fnv(payload) == checksum
        actual = np.frombuffer(payload, dtype=np.dtype("<f4")).reshape(tensor.shape)
        np.testing.assert_array_equal(actual, tensor.values.astype(np.float32))


def run_converter(
    checkpoint: Path, output: Path, *additional: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--input",
            str(checkpoint),
            "--output",
            str(output),
            "--packer",
            str(PACKER),
            "--chunk-mib",
            "1",
            "--quiet",
            *additional,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


class ConverterIntegrationTest(unittest.TestCase):
    def test_sharded_mixed_precision_layout_and_cpp_loader(self) -> None:
        with tempfile.TemporaryDirectory(prefix="helix-converter-test-") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            config = base_config()
            sources = write_checkpoint(checkpoint, config, sharded=True)
            output = root / "model.helix"
            result = run_converter(checkpoint, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = expected_tensors(config, sources)
            validate_helix_file(output, config, expected)

            query = next(tensor for tensor in expected if tensor.name == "layer.0.query")
            unpermuted = sources["model.layers.0.self_attn.q_proj.weight"].T
            self.assertFalse(np.array_equal(query.values, unpermuted))

            inspected = subprocess.run(
                [str(INSPECTOR), str(output), "--smoke"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            self.assertIn("vocabulary_size=7", inspected.stdout)
            self.assertIn("layer_count=2", inspected.stdout)
            self.assertIn("smoke_logits=1x7", inspected.stdout)

            tokens = (1, 4, 2)
            dumped = subprocess.run(
                [
                    str(INSPECTOR),
                    str(output),
                    "--last-logits",
                    ",".join(str(token) for token in tokens),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(dumped.returncode, 0, dumped.stderr)
            line = next(
                line for line in dumped.stdout.splitlines() if line.startswith("last_logits_hex=")
            )
            actual_logits = np.asarray(
                [float.fromhex(value) for value in line.split("=", 1)[1].split(",")],
                dtype=np.float32,
            )
            expected_logits = reference_hf_last_logits(config, sources, tokens)
            np.testing.assert_allclose(actual_logits, expected_logits, rtol=2.0e-5, atol=2.0e-6)

    def test_tied_embedding_fallback_and_atomic_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="helix-converter-tied-") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            config = base_config(tied=True)
            sources = write_checkpoint(
                checkpoint, config, sharded=False, include_lm_head=False
            )
            output = root / "model.helix"
            sentinel = b"existing output remains intact"
            output.write_bytes(sentinel)

            rejected = run_converter(checkpoint, output)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(output.read_bytes(), sentinel)

            converted = run_converter(checkpoint, output, "--force")
            self.assertEqual(converted.returncode, 0, converted.stderr)
            validate_helix_file(output, config, expected_tensors(config, sources))

    def test_rejects_unsupported_architecture_and_bad_tensor_without_output(self) -> None:
        cases: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
        gqa = base_config()
        gqa["num_key_value_heads"] = 1
        cases.append(("gqa", gqa, {}, "GQA/MQA"))
        wrong_shape = base_config()
        cases.append(
            (
                "shape",
                wrong_shape,
                {
                    "shape_override": (
                        "model.layers.0.self_attn.q_proj.weight",
                        (7, 8),
                    )
                },
                "shape",
            )
        )
        non_finite = base_config()
        cases.append(
            (
                "nonfinite",
                non_finite,
                {"non_finite_name": "model.layers.0.mlp.up_proj.weight"},
                "non-finite",
            )
        )

        for label, config, options, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"helix-converter-{label}-"
            ) as temporary:
                root = Path(temporary)
                checkpoint = root / "checkpoint"
                write_checkpoint(checkpoint, config, sharded=False, **options)
                output = root / "model.helix"
                result = run_converter(checkpoint, output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(output.exists())

    def test_rejects_index_path_escape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="helix-converter-escape-") as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            config = base_config()
            write_checkpoint(checkpoint, config, sharded=True)
            index_path = checkpoint / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["weight_map"]["model.embed_tokens.weight"] = "../outside.safetensors"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            (root / "outside.safetensors").write_bytes(b"not relevant")
            output = root / "model.helix"
            result = run_converter(checkpoint, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("inside the checkpoint", result.stderr)
            self.assertFalse(output.exists())


def parse_test_arguments(arguments: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--packer", required=True, type=Path)
    parser.add_argument("--inspector", required=True, type=Path)
    return parser.parse_known_args(arguments)


if __name__ == "__main__":
    options, remaining = parse_test_arguments(sys.argv[1:])
    CONVERTER = options.converter.resolve()
    PACKER = options.packer.resolve()
    INSPECTOR = options.inspector.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
