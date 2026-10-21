#!/usr/bin/env python3
"""Run a TFLite model with LiteRT and save outputs in ORT/ONNX layout."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError as exc:
    raise SystemExit(
        "LiteRT is required: install ai-edge-litert (from ai_edge_litert.interpreter import Interpreter)"
    ) from exc


def nchw_shape_from_tflite(shape):
    shape = [int(x) for x in shape]
    return [shape[0], shape[3], shape[1], shape[2]] if len(shape) == 4 else shape


def nchw_to_nhwc(value):
    return np.transpose(value, (0, 2, 3, 1)) if value.ndim == 4 else value


def nhwc_to_nchw(value):
    return np.transpose(value, (0, 3, 1, 2)) if value.ndim == 4 else value


def quantize_input(value, detail):
    dtype = np.dtype(detail["dtype"])
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale and np.issubdtype(dtype, np.integer) and np.issubdtype(value.dtype, np.floating):
        value = np.round(value / scale + zero_point)
    return value.astype(dtype)


def dequantize_output(value, detail):
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale and np.issubdtype(value.dtype, np.integer):
        return (value.astype(np.float32) - zero_point) * scale
    return value


def print_top5(value):
    flat = np.asarray(value).reshape(-1)
    if flat.size == 0 or not np.issubdtype(flat.dtype, np.number):
        print("  TFLite Top5: skipped for non-numeric or empty output")
        return
    indices = np.argsort(flat)[::-1][:5]
    print("  TFLite Top5:", [(int(i), float(flat[i])) for i in indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tflite_model")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "output" / "tflite_output"))
    args = parser.parse_args()

    interpreter = Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(args.inputs) != len(input_details):
        raise SystemExit(f"expected {len(input_details)} input files, got {len(args.inputs)}")

    for index, (detail, filename) in enumerate(zip(input_details, args.inputs)):
        tflite_shape = [int(x) for x in detail["shape"]]
        ort_shape = nchw_shape_from_tflite(tflite_shape)
        dtype = np.dtype(detail["dtype"])
        data = np.fromfile(filename, dtype=dtype)
        expected = int(np.prod(ort_shape))
        if data.size != expected:
            raise SystemExit(f"input {index}: expected {expected} elements for NCHW shape {ort_shape}, got {data.size}")
        value = quantize_input(nchw_to_nhwc(data.reshape(ort_shape)), detail)
        if tuple(value.shape) != tuple(tflite_shape):
            raise SystemExit(f"input {index}: prepared shape {list(value.shape)} != TFLite shape {tflite_shape}")
        interpreter.set_tensor(detail["index"], value)

    interpreter.invoke()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, detail in enumerate(output_details):
        tflite_shape = [int(x) for x in detail["shape"]]
        value = nhwc_to_nchw(dequantize_output(interpreter.get_tensor(detail["index"]), detail))
        ort_shape = nchw_shape_from_tflite(tflite_shape)
        if value.size != int(np.prod(ort_shape)):
            raise SystemExit(f"output {index}: got {value.size} elements, expected {int(np.prod(ort_shape))}")
        value.tofile(out_dir / f"output_{index}.bin")
        value = value.reshape(-1)
        meta = {"name": detail.get("name", f"output_{index}"), "dtype": str(value.dtype), "shape": ort_shape, "elements": int(value.size)}
        (out_dir / f"output_{index}.meta").write_text(json.dumps(meta, indent=2) + "\n")
        manifest.append(meta)
        print(f"Output {index}: name={meta['name']} shape={ort_shape} dtype={value.dtype} bytes={value.nbytes}")
        print_top5(value)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
