#!/usr/bin/env python3
"""Independent MLX affine safetensors reader/dequantiser.

The format is row-major U32 codes, packed LSB-first, with BF16 scales and
biases shaped [rows, cols/group_size].  Values are scale * code + bias.
This tool intentionally has no dependency on go-inference.
"""

import argparse
import json
import pathlib
import struct

import numpy as np


def _tensor_file(model_dir: pathlib.Path, name: str) -> pathlib.Path:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return model_dir / weight_map[name]
    return model_dir / "model.safetensors"


def read_tensor(model_dir: pathlib.Path, name: str):
    path = _tensor_file(model_dir, name)
    with path.open("rb") as src:
        header_len = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(header_len))
        entry = header[name]
        start, end = entry["data_offsets"]
        src.seek(8 + header_len + start)
        payload = src.read(end - start)
    return entry["dtype"], tuple(entry["shape"]), payload


def bf16(payload: bytes, shape):
    words = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return (words << 16).view(np.float32).reshape(shape)


def dense_tensor(model_dir: pathlib.Path, name: str):
    dtype, shape, payload = read_tensor(model_dir, name)
    if dtype == "BF16":
        return bf16(payload, shape)
    if dtype == "F32":
        return np.frombuffer(payload, dtype="<f4").reshape(shape)
    raise ValueError(f"{name}: unsupported dense dtype {dtype}")


def dequantise(model_dir: pathlib.Path, base: str, group_size: int, bits: int):
    dtype, packed_shape, payload = read_tensor(model_dir, base + ".weight")
    if dtype != "U32" or len(packed_shape) != 2:
        raise ValueError(f"{base}.weight: want rank-2 U32, got {dtype} {packed_shape}")
    scales = bf16(*_payload_shape(model_dir, base + ".scales"))
    biases = bf16(*_payload_shape(model_dir, base + ".biases"))
    rows, groups = scales.shape
    cols = groups * group_size
    if packed_shape != (rows, cols * bits // 32) or biases.shape != scales.shape:
        raise ValueError(f"{base}: inconsistent packed/scales/biases geometry")
    words = np.frombuffer(payload, dtype="<u4").reshape(packed_shape)
    per_word = 32 // bits
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    codes = ((words[..., None] >> shifts) & ((1 << bits) - 1)).reshape(rows, cols)
    return codes.astype(np.float32) * np.repeat(scales, group_size, axis=1) + np.repeat(biases, group_size, axis=1)


def _payload_shape(model_dir: pathlib.Path, name: str):
    dtype, shape, payload = read_tensor(model_dir, name)
    if dtype != "BF16":
        raise ValueError(f"{name}: want BF16, got {dtype}")
    return payload, shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=pathlib.Path)
    parser.add_argument("tensor", help="quantised base name, or a dense tensor name")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--row", type=int)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    values = dense_tensor(args.model_dir, args.tensor) if args.dense else dequantise(args.model_dir, args.tensor, args.group_size, args.bits)
    if args.row is not None:
        values = values[args.row]
    values = np.asarray(values, dtype="<f4")
    args.output.write_bytes(values.tobytes(order="C"))
    print(json.dumps({"tensor": args.tensor, "shape": list(values.shape), "count": int(values.size), "min": float(values.min()), "max": float(values.max()), "rms": float(np.sqrt(np.mean(values * values, dtype=np.float64)))}, sort_keys=True))


if __name__ == "__main__":
    main()
