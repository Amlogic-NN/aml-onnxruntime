# Amlogic User Guide

English | [中文](README_CN.md)

## 1. Overview

This project integrates `AmlogicExecutionProvider` into ONNX Runtime to convert ONNX models and run inference on Amlogic NPUs.

The Android workflow is:

```text
Create Ort::Session: ONNX → graph export → AML IR → Core / ADLA compiler → ADLA
Call Session.Run: input tensors → NNSDK2 → ADLA runtime / NPU → output tensors
```

After registering the Amlogic provider, an application uses a single Session for conversion and inference. The conversion bridge is embedded in `libonnxruntime.so`. Enabling TFLite dumping also produces a TFLite model for CPU reference inference.

This directory provides SO build scripts for multiple platforms, an Android64 test program, and host scripts for reference inference and output comparison. The following sections cover the complete Android64 workflow, followed by SO builds for other architectures.

**Run all commands from the repository root**, replacing `/path/to/...` with actual paths. Use this directory together with the repository source code and `third_party/amlogic/`.

## 2. Build the Android64 SO

### 2.1 Prepare the environment

Use a Linux x86_64 host with Python 3, CMake, C/C++ build tools, Git, and Android NDK r25c. The initial build requires access to download ONNX Runtime's third-party dependencies.

The matching vendor SDKs are located in `third_party/amlogic/`:

| Directory | Purpose |
| --- | --- |
| `aml_compiler_core_v1.0.1_20261015` | Core headers and libraries for model conversion |
| `compiler_adla_v3.5.0` | ADLA compiler matching the Core library |
| `sdk_v1.1.0_v3.0.3_20260910_merged` | NNSDK2 headers and libraries for inference |

Verify the SDK files and set the build environment variables:

```bash
python3 third_party/amlogic/verify_sdk.py

export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
export AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR="$PWD/third_party/amlogic/aml_compiler_core_v1.0.1_20261015/include"
export AMLNN_NNSDK2_INCLUDE_DIR="$PWD/third_party/amlogic/sdk_v1.1.0_v3.0.3_20260910_merged/include"
```

`ANDROID_NDK_PATH` points to the NDK root. The other two variables point to the Core and NNSDK2 header directories. The build locates headers through these environment variables; the matching vendor libraries are loaded at runtime.

### 2.2 Run the build

```bash
bash tools/amlogic_release/build_so_android64.sh
```

Output file:

```text
build/AMLOGIC_PIPELINE-android64/Release/libonnxruntime.so
```

This SO enables the Amlogic provider and includes the conversion and inference integration code.

## 3. Build the Android64 Test Program

After building the SO in the previous section, run the following in the same terminal:

```bash
bash tools/amlogic_release/build_pipeline_demo_android64.sh
```

The script compiles `pipeline_demo/pipeline_demo.cc` and copies the ORT SO to the test program directory. Outputs:

```text
tools/amlogic_release/pipeline_demo/bin/android64/
    pipeline_demo
    libonnxruntime.so
```

The test program creates a Session, runs inference, and saves the output tensors as binary files.

## 4. Run the Android64 Test Program

### 4.1 Prepare the device and inputs

Install adb and `readelf` on the host. The device must have a compatible NPU driver and `/vendor/lib64/libadla.so`. It must also allow program execution through adb and changes to the `vendor.TFLITE_FILE_DUMP_LEVEL` property.

Prepare an ONNX model and its input binary files. Each input must contain raw tensor data after preprocessing. The input count, order, dtype, and shape must match the model. This example assumes a model with a fixed input shape, float32 data, and NCHW layout, with float32 inputs for both ONNX and the exported TFLite model.

```bash
export MODEL=/path/to/model.onnx
export INPUT=/path/to/input_fp32_nchw.bin
export MODEL_NAME="$(basename -- "$MODEL" .onnx)"
export RESULT_DIR="$PWD/tools/amlogic_release/output/$MODEL_NAME"

adb devices
# Select the target device when multiple devices are connected:
# export ANDROID_SERIAL=<device-serial>
```

Use a new `RESULT_DIR` for each test to avoid including stale outputs in the comparison.

### 4.2 Run conversion and inference

Keep `ANDROID_NDK_PATH` set as described earlier, then run:

```bash
bash tools/amlogic_release/run_pipeline_demo_android64.sh \
  "$MODEL" "$RESULT_DIR/npu" info --dump-tflite "$INPUT"
```

| Argument | Meaning |
| --- | --- |
| `"$MODEL"` | ONNX file on the host |
| `"$RESULT_DIR/npu"` | Host directory for NPU outputs and runtime logs |
| `info` | Log level: `error`, `info`, or `debug` |
| `--dump-tflite` | Export TFLite for the three-way comparison in the next section |
| `"$INPUT"` | Input binary file on the host; append additional files in model input order for models with multiple inputs |

The script automatically pushes the test program, ORT, Core, compiler, NNSDK, and input files to the device. It also pushes `libc++_shared.so` from the NDK when required. By default, it uses the Android arm64-v8a vendor libraries from `third_party/amlogic/` and places them alongside ORT in the device test directory, without overwriting vendor libraries.

The default device directory is `/data/local/tmp/amlogic_release_test`. **This directory is cleared on every run.** To use another directory, set `export REMOTE_DIR=/data/local/tmp/your_test_dir`. The TFLite dump property is restored when the script exits.

To use different matching libraries, set `AMLOGIC_CORE_SO`, `AMLOGIC_COMPILER_SO`, and `AMLNN_NNSDK_SO` to their full host paths. Core and compiler versions must match, and the ABI must be Android arm64-v8a.

### 4.3 Inspect the outputs

The log message `AMLNN inference succeeded` indicates that inference succeeded. The host output directory contains:

```text
tools/amlogic_release/output/<model-name>/npu/
    run.log                  Device runtime log
    output_0.bin             Raw data for output 0
    output_0.meta            Name, dtype, shape, and element count
    ...                      Additional outputs numbered sequentially
    <model-name>.adla        Converted ADLA model
    <model-name>.tflite      Generated when dumping is enabled and export succeeds
```

If TFLite reference inference is unnecessary, omit `--dump-tflite`. ONNX conversion and NPU inference still run.

## 5. Compare the Results

Run ONNX and TFLite CPU reference inference on the host, then compare their outputs with the device's NPU outputs. All three paths must use the same input data.

### 5.1 Install reference inference dependencies

Install the following packages, preferably in a separate Python environment:

```bash
python3 -m pip install numpy onnxruntime ai-edge-litert
```

These packages provide host CPU reference inference and are used separately from the ORT SO built for the device. Continue running commands from the repository root.

### 5.2 Run ONNX CPU reference inference

```bash
python3 tools/amlogic_release/run_onnx_reference.py \
  "$MODEL" "$INPUT" --output-dir "$RESULT_DIR/onnx"
```

Outputs are saved in `$RESULT_DIR/onnx/`, including `output_*.bin`, `output_*.meta`, and `manifest.json`.

### 5.3 Run TFLite CPU reference inference

```bash
python3 tools/amlogic_release/run_tflite_reference.py \
  "$RESULT_DIR/npu/$MODEL_NAME.tflite" "$INPUT" \
  --output-dir "$RESULT_DIR/tflite"
```

Outputs are saved in `$RESULT_DIR/tflite/`. The script converts four-dimensional inputs from NCHW to NHWC and converts four-dimensional outputs back to NCHW. The EP already handles NPU output layouts, so do not transpose NPU outputs again.

For models with multiple inputs, append binary files in the corresponding input order to both reference commands. The ONNX reference script requires fixed input shapes. The TFLite script reads binary files using the TFLite input dtype. For quantized inputs or models with different input dtypes, check the types and quantization parameters before using these commands; the float32 input in this example may not apply.

### 5.4 Run the comparison

```bash
python3 tools/amlogic_release/compare_onnx_tflite_amlnn.py \
  "$RESULT_DIR/onnx" "$RESULT_DIR/npu" "$RESULT_DIR/tflite"
```

The directory order is **ONNX outputs, NPU outputs, TFLite outputs**. For each output, the script prints:

```text
ONNX vs AMLNN:  cosine_similarity=... euclidean_distance=...
ONNX vs TFLite: cosine_similarity=... euclidean_distance=...
TFLite vs AMLNN: cosine_similarity=... euclidean_distance=...
```

`AMLNN` refers to NPU outputs. A cosine similarity closer to 1 and a Euclidean distance closer to 0 indicate closer numerical results. A cosine similarity of at least 0.99 can serve as an initial reference; final acceptance should use the model's application metrics.

**Exit code 0 from the comparison script only means that the comparison completed; it does not indicate that an accuracy threshold was met.** The script pairs outputs by index and compares flattened data. Check the `.meta` files to confirm output correspondence and shapes.

To compare only ONNX and NPU outputs, skip TFLite reference inference and pass a nonexistent TFLite output directory to the comparison script. It will skip TFLite comparisons automatically.

## 6. Build SOs for Other Architectures

All platform builds require the Core and NNSDK2 header variables:

```bash
export AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR="$PWD/third_party/amlogic/aml_compiler_core_v1.0.1_20261015/include"
export AMLNN_NNSDK2_INCLUDE_DIR="$PWD/third_party/amlogic/sdk_v1.1.0_v3.0.3_20260910_merged/include"
```

### 6.1 Android32

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
bash tools/amlogic_release/build_so_android32.sh
```

The target ABI is `armeabi-v7a`.

### 6.2 Yocto32 / Yocto64

Set the root directory of each installed Yocto SDK:

```bash
export YOCTO32_PATH=/path/to/yocto-sdk-32
bash tools/amlogic_release/build_so_yocto32.sh

export YOCTO64_PATH=/path/to/yocto-sdk-64
bash tools/amlogic_release/build_so_yocto64.sh
```

Each SDK root must contain `environment-setup-*` and `sysroots/x86_64-pokysdk-linux/usr/share/cmake/OEToolchainConfig.cmake`. The scripts load the corresponding SDK environment automatically; no prior `source` command is required.

### 6.3 Buildroot32 / Buildroot64

```bash
export BUILDROOT32_PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf
bash tools/amlogic_release/build_so_buildroot32.sh

export BUILDROOT64_PATH=/path/to/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu
bash tools/amlogic_release/build_so_buildroot64.sh
```

The scripts expect the following toolchain directory structure:

| Platform | Compiler paths relative to toolchain root | Sysroot relative to toolchain root |
| --- | --- | --- |
| Buildroot32 | `bin/arm-none-linux-gnueabihf-gcc` and `g++` with the same prefix | `arm-none-linux-gnueabihf/libc` |
| Buildroot64 | `bin/aarch64-none-linux-gnu-gcc` and `g++` with the same prefix | `aarch64-none-linux-gnu/libc` |

### 6.4 Build all six platforms

Set the two header variables above and all five toolchain variables, then run:

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
export YOCTO32_PATH=/path/to/yocto-sdk-32
export YOCTO64_PATH=/path/to/yocto-sdk-64
export BUILDROOT32_PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf
export BUILDROOT64_PATH=/path/to/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu

bash tools/amlogic_release/build_so_total.sh
```

The script builds all six platforms sequentially and continues with the remaining platforms if a build fails. It returns a nonzero exit code if any build fails. Per-platform logs and `summary.log` are saved in `logs/amlogic_release/<timestamp>/`. This script builds SOs only; build the Android64 test program separately as described in Section 3.

### 6.5 Output paths and runtime support

| Platform | SO path |
| --- | --- |
| Android32 | `build/AMLOGIC_PIPELINE-android32/Release/libonnxruntime.so` |
| Android64 | `build/AMLOGIC_PIPELINE-android64/Release/libonnxruntime.so` |
| Yocto32 | `build/AMLOGIC_PIPELINE-yocto32/Release/libonnxruntime.so` |
| Yocto64 | `build/AMLOGIC_PIPELINE-yocto64/Release/libonnxruntime.so` |
| Buildroot32 | `build/AMLOGIC_PIPELINE-buildroot32/Release/libonnxruntime.so` |
| Buildroot64 | `build/AMLOGIC_PIPELINE-buildroot64/Release/libonnxruntime.so` |

The test program build, adb deployment, and three-way comparison workflow target Android64. For other platforms, build a suitable test application and use vendor runtime libraries matching the target system ABI.

On Yocto and Buildroot, the current EP uses `AMLNN_MODEL_PATH` to load a precompiled ADLA model. The corresponding ONNX model is still used to create the Session and provide graph and input/output information. These platforms do not run the Android model conversion workflow described above.
