#!/usr/bin/env python3
"""Run an ONNX model with the host ONNX Runtime CPU provider."""
import argparse
import json
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit("onnxruntime is required: python3 -m pip install onnxruntime") from exc


def shape_value(shape):
    result = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            result.append(dim)
        else:
            raise ValueError(f"dynamic or invalid input shape: {shape}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "output" / "onnx_output"))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    model_inputs = session.get_inputs()
    if len(args.inputs) != len(model_inputs):
        raise SystemExit(f"expected {len(model_inputs)} input files, got {len(args.inputs)}")
    feed = {}
    for spec, filename in zip(model_inputs, args.inputs):
        shape = shape_value(spec.shape)
        type_name = spec.type.replace("tensor(", "").rstrip(")")
        type_map = {"float": "float32", "double": "float64", "float16": "float16", "bfloat16": "uint16",
                    "int8": "int8", "uint8": "uint8", "int16": "int16", "uint16": "uint16",
                    "int32": "int32", "uint32": "uint32", "int64": "int64", "uint64": "uint64", "bool": "bool"}
        if type_name not in type_map:
            raise SystemExit(f"unsupported ONNX input type for {spec.name}: {spec.type}")
        dtype = np.dtype(type_map[type_name])
        data = np.fromfile(filename, dtype=dtype)
        expected = int(np.prod(shape))
        if data.size != expected:
            raise SystemExit(f"input {spec.name}: expected {expected} elements, got {data.size}")
        feed[spec.name] = data.reshape(shape)
    outputs = session.run(None, feed)
    manifest = []
    for index, (spec, value) in enumerate(zip(session.get_outputs(), outputs)):
        value = np.asarray(value)
        value.tofile(out_dir / f"output_{index}.bin")
        meta = {"name": spec.name, "dtype": str(value.dtype), "shape": list(value.shape), "elements": int(value.size)}
        (out_dir / f"output_{index}.meta").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        manifest.append(meta)
        print(f"Output {index}: name={spec.name} shape={list(value.shape)} dtype={value.dtype} bytes={value.nbytes}")
        if np.issubdtype(value.dtype, np.number) and value.size:
            flat = value.reshape(-1)
            indices = np.argsort(flat)[::-1][:5]
            print("  ONNX Top5:", [(int(i), float(flat[i])) for i in indices])
        else:
            print("  ONNX Top5: skipped for non-numeric or empty output")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    main()
