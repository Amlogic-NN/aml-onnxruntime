#!/usr/bin/env python3
"""Compare ONNX, optional LiteRT/TFLite, and AMLNN outputs."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_dtype(path):
    meta = path.with_suffix(".meta")
    if not meta.exists():
        return np.dtype(np.float32)
    text = meta.read_text()
    try:
        value = json.loads(text).get("dtype", "float32")
    except json.JSONDecodeError:
        value = "float32"
        for line in text.splitlines():
            if line.startswith("dtype="):
                value = line.split("=", 1)[1].strip()
                break
    return np.dtype({"bfloat16": np.uint16, "bool": np.bool_}.get(value, value))


def compare(label, first, second):
    a = np.fromfile(first, dtype=read_dtype(first)).astype(np.float64)
    b = np.fromfile(second, dtype=read_dtype(second)).astype(np.float64)
    if a.size != b.size:
        print(f"{label}: element count mismatch {a.size} vs {b.size}")
        return False
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(np.dot(a, b) / norm) if norm else (1.0 if np.array_equal(a, b) else 0.0)
    distance = float(np.linalg.norm(a - b))
    print(f"{label}: cosine_similarity={cosine:.9f} euclidean_distance={distance:.9f}")
    return True


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent / "output"
    parser.add_argument("onnx_dir", nargs="?", default=str(root / "onnx_output"))
    parser.add_argument("amlnn_dir", nargs="?", default=str(root / "amlnn_output"))
    parser.add_argument("tflite_dir", nargs="?", default=str(root / "tflite_output"))
    args = parser.parse_args()
    onnx_dir, amlnn_dir, tflite_dir = map(Path, (args.onnx_dir, args.amlnn_dir, args.tflite_dir))

    onnx_files = sorted(onnx_dir.glob("output_*.bin"))
    if not onnx_files:
        raise SystemExit(f"no ONNX output files in {onnx_dir}")
    passed = True
    for onnx_file in onnx_files:
        name = onnx_file.name
        amlnn_file = amlnn_dir / name
        if not amlnn_file.exists():
            print(f"{name}: missing AMLNN output: {amlnn_file}")
            passed = False
            continue
        passed &= compare(f"{name} ONNX vs AMLNN", onnx_file, amlnn_file)
        tflite_file = tflite_dir / name
        if tflite_file.exists():
            passed &= compare(f"{name} ONNX vs TFLite", onnx_file, tflite_file)
            passed &= compare(f"{name} TFLite vs AMLNN", tflite_file, amlnn_file)
        else:
            print(f"{name}: TFLite output not found; comparing ONNX and AMLNN only")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
