# Amlogic 使用说明

[English](README.md) | 中文

## 1. 这是什么

本项目为 ONNX Runtime 接入了 `AmlogicExecutionProvider`，用于在 Amlogic 平台上转换 ONNX 模型并通过 NPU 推理。

Android 的处理流程为：

```text
创建 Ort::Session：ONNX → 图信息导出 → AML IR → Core / ADLA compiler → ADLA
调用 Session.Run：输入 tensor → NNSDK2 → ADLA runtime / NPU → 输出 tensor
```

应用注册 Amlogic provider 后，使用一个 Session 完成转换和推理。转换 bridge 已编入 `libonnxruntime.so`。开启 TFLite dump 后，可将转换生成的 TFLite 用于 CPU 参考推理。

本目录提供各平台 SO 编译脚本、Android64 测试程序，以及 PC 参考推理和结果比较脚本。下面先介绍 Android64 的完整操作，最后介绍其他架构的 SO 编译。

**所有命令均在本仓库根目录执行**，将 `/path/to/...` 替换为实际路径。本目录需要与仓库源码及 `third_party/amlogic/` 一起使用。

## 2. 编译 Android64 SO

### 2.1 准备环境

主机使用 Linux x86_64，安装 Python 3、CMake、C/C++ 构建工具和 Git，并准备 Android NDK r25c。首次构建需要能下载 ONNX Runtime 的第三方依赖。

配套厂商 SDK 位于 `third_party/amlogic/`：

| 目录 | 用途 |
| --- | --- |
| `aml_compiler_core_v1.0.1_20261015` | 模型转换的 Core 头文件和 SO |
| `compiler_adla_v3.5.0` | 与 Core 配套的 ADLA compiler |
| `sdk_v1.1.0_v3.0.3_20260910_merged` | NNSDK2 推理头文件和 SO |

校验 SDK 文件，并设置编译环境变量：

```bash
python3 third_party/amlogic/verify_sdk.py

export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
export AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR="$PWD/third_party/amlogic/aml_compiler_core_v1.0.1_20261015/include"
export AMLNN_NNSDK2_INCLUDE_DIR="$PWD/third_party/amlogic/sdk_v1.1.0_v3.0.3_20260910_merged/include"
```

`ANDROID_NDK_PATH` 指向 NDK 根目录，另外两个变量分别指向 Core 和 NNSDK2 的头文件目录。构建通过环境变量查找头文件，运行时再加载配套的厂商 SO。

### 2.2 执行编译

```bash
bash tools/amlogic_release/build_so_android64.sh
```

输出文件：

```text
build/AMLOGIC_PIPELINE-android64/Release/libonnxruntime.so
```

该 SO 已启用 Amlogic provider，包含转换和推理适配代码。

## 3. 编译 Android64 测试程序

完成上一节的 SO 编译后，在同一终端执行：

```bash
bash tools/amlogic_release/build_pipeline_demo_android64.sh
```

脚本编译 `pipeline_demo/pipeline_demo.cc`，并将 ORT SO 复制到测试程序目录。输出为：

```text
tools/amlogic_release/pipeline_demo/bin/android64/
    pipeline_demo
    libonnxruntime.so
```

测试程序创建 Session、执行推理，并将输出 tensor 保存为 bin 文件。

## 4. 运行 Android64 测试程序

### 4.1 准备板端与输入

主机需安装 adb 和 `readelf`。板端需具备匹配的 NPU 驱动及 `/vendor/lib64/libadla.so`，并允许 adb 执行程序和设置 `vendor.TFLITE_FILE_DUMP_LEVEL` 属性。

准备 ONNX 模型和输入 bin。输入必须是完成预处理后的原始 tensor 数据，数量、顺序、dtype 和 shape 与模型一致。下面以固定 shape、float32、NCHW 输入模型为例，ONNX 与导出的 TFLite 输入均为 float32。

```bash
export MODEL=/path/to/model.onnx
export INPUT=/path/to/input_fp32_nchw.bin
export MODEL_NAME="$(basename -- "$MODEL" .onnx)"
export RESULT_DIR="$PWD/tools/amlogic_release/output/$MODEL_NAME"

adb devices
# 多设备时指定目标设备：
# export ANDROID_SERIAL=<device-serial>
```

每次测试建议使用新的 `RESULT_DIR`，避免旧输出混入比较。

### 4.2 执行转换与推理

保持前文的 `ANDROID_NDK_PATH` 配置，执行：

```bash
bash tools/amlogic_release/run_pipeline_demo_android64.sh \
  "$MODEL" "$RESULT_DIR/npu" info --dump-tflite "$INPUT"
```

| 参数 | 含义 |
| --- | --- |
| `"$MODEL"` | 主机上的 ONNX 文件 |
| `"$RESULT_DIR/npu"` | 主机上保存 NPU 输出和运行日志的目录 |
| `info` | 日志等级，可选 `error`、`info`、`debug` |
| `--dump-tflite` | 导出 TFLite，供下一节进行三路比较 |
| `"$INPUT"` | 主机上的输入 bin；多输入时按模型输入顺序追加文件 |

脚本自动推送测试程序、ORT、Core、compiler、NNSDK 和输入文件；需要 `libc++_shared.so` 时从 NDK 推送。厂商库默认取自 `third_party/amlogic/` 中的 Android arm64-v8a 版本，与 ORT 放在同一板端测试目录，不覆盖 vendor 库。

默认板端目录为 `/data/local/tmp/amlogic_release_test`，**每次运行会清空该目录**。可通过 `export REMOTE_DIR=/data/local/tmp/your_test_dir` 更换。运行结束时恢复 TFLite dump 属性。

如需更换配套库，可设置 `AMLOGIC_CORE_SO`、`AMLOGIC_COMPILER_SO`、`AMLNN_NNSDK_SO` 为各自 SO 的完整主机路径；Core 与 compiler 必须配套，ABI 必须为 Android arm64-v8a。

### 4.3 查看输出

日志中出现 `AMLNN inference succeeded` 表示本次推理成功。主机输出目录为：

```text
tools/amlogic_release/output/<模型名>/npu/
    run.log                  板端运行日志
    output_0.bin             第 0 个输出的原始数据
    output_0.meta            对应的名称、dtype、shape、元素数
    ...                      多输出依次编号
    <模型名>.adla            转换生成的 ADLA
    <模型名>.tflite          开启 dump 且成功导出时生成
```

不需要 TFLite 参考推理时，可省略 `--dump-tflite`，仍执行 ONNX 转换与 NPU 推理。

## 5. 对比结果

在 PC 上分别运行 ONNX 和 TFLite 的 CPU 参考推理，再与板端 NPU 输出比较。三路必须使用同一份输入数据。

### 5.1 安装参考推理依赖

建议在独立 Python 环境中安装：

```bash
python3 -m pip install numpy onnxruntime ai-edge-litert
```

这里安装的是 PC 端 CPU 参考库，与板端编译的 ORT SO 分别使用。继续在仓库根目录执行后续命令。

### 5.2 运行 ONNX CPU 参考推理

```bash
python3 tools/amlogic_release/run_onnx_reference.py \
  "$MODEL" "$INPUT" --output-dir "$RESULT_DIR/onnx"
```

输出保存在 `$RESULT_DIR/onnx/`，包含 `output_*.bin`、`output_*.meta` 和 `manifest.json`。

### 5.3 运行 TFLite CPU 参考推理

```bash
python3 tools/amlogic_release/run_tflite_reference.py \
  "$RESULT_DIR/npu/$MODEL_NAME.tflite" "$INPUT" \
  --output-dir "$RESULT_DIR/tflite"
```

输出保存在 `$RESULT_DIR/tflite/`。脚本将四维输入由 NCHW 转为 NHWC，推理后将四维输出转回 NCHW；NPU 输出已由 EP 处理布局，无需再转置。

多输入模型在两个参考脚本中均按对应输入顺序追加 bin。ONNX 参考脚本要求固定输入 shape；TFLite 脚本按 TFLite 输入 dtype 读取 bin。量化输入或不同输入 dtype 的模型需先核对类型及量化参数，不能直接套用本例的 float32 输入。

### 5.4 执行比较

```bash
python3 tools/amlogic_release/compare_onnx_tflite_amlnn.py \
  "$RESULT_DIR/onnx" "$RESULT_DIR/npu" "$RESULT_DIR/tflite"
```

三个目录的顺序为 **ONNX 输出、NPU 输出、TFLite 输出**。脚本逐输出打印：

```text
ONNX vs AMLNN:  cosine_similarity=... euclidean_distance=...
ONNX vs TFLite: cosine_similarity=... euclidean_distance=...
TFLite vs AMLNN: cosine_similarity=... euclidean_distance=...
```

`AMLNN` 表示 NPU 输出。余弦相似度越接近 1、欧氏距离越接近 0，数值越接近。可将 cosine ≥ 0.99 作为初步参考，最终按模型业务指标验收。

**比较脚本退出 0 仅表示完成比较，不按精度阈值判定通过。** 脚本按输出编号配对并展平比较，需结合 `.meta` 核对输出对应关系和 shape。

只比较 ONNX/NPU 时，跳过 TFLite 参考推理，并给比较脚本传入一个不存在的 TFLite 输出目录；脚本会自动跳过 TFLite 比较。

## 6. 编译其他架构的 SO

各平台均需设置 Core、NNSDK2 头文件变量：

```bash
export AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR="$PWD/third_party/amlogic/aml_compiler_core_v1.0.1_20261015/include"
export AMLNN_NNSDK2_INCLUDE_DIR="$PWD/third_party/amlogic/sdk_v1.1.0_v3.0.3_20260910_merged/include"
```

### 6.1 Android32

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
bash tools/amlogic_release/build_so_android32.sh
```

目标 ABI 为 `armeabi-v7a`。

### 6.2 Yocto32 / Yocto64

分别指定已安装的 Yocto SDK 根目录：

```bash
export YOCTO32_PATH=/path/to/yocto-sdk-32
bash tools/amlogic_release/build_so_yocto32.sh

export YOCTO64_PATH=/path/to/yocto-sdk-64
bash tools/amlogic_release/build_so_yocto64.sh
```

SDK 根目录需包含 `environment-setup-*` 和 `sysroots/x86_64-pokysdk-linux/usr/share/cmake/OEToolchainConfig.cmake`。脚本自行加载对应 SDK 环境，无需提前 source。

### 6.3 Buildroot32 / Buildroot64

```bash
export BUILDROOT32_PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf
bash tools/amlogic_release/build_so_buildroot32.sh

export BUILDROOT64_PATH=/path/to/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu
bash tools/amlogic_release/build_so_buildroot64.sh
```

脚本使用以下工具链目录结构：

| 平台 | 编译器（相对工具链根目录） | sysroot（相对工具链根目录） |
| --- | --- | --- |
| Buildroot32 | `bin/arm-none-linux-gnueabihf-gcc`、同前缀 `g++` | `arm-none-linux-gnueabihf/libc` |
| Buildroot64 | `bin/aarch64-none-linux-gnu-gcc`、同前缀 `g++` | `aarch64-none-linux-gnu/libc` |

### 6.4 一次编译全部六个平台

设置上述两个头文件变量，以及全部五个工具链变量后执行：

```bash
export ANDROID_NDK_PATH=/path/to/android-ndk-r25c
export YOCTO32_PATH=/path/to/yocto-sdk-32
export YOCTO64_PATH=/path/to/yocto-sdk-64
export BUILDROOT32_PATH=/path/to/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf
export BUILDROOT64_PATH=/path/to/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu

bash tools/amlogic_release/build_so_total.sh
```

总脚本依次构建六个平台，单个平台失败后继续其他平台，最终存在任一失败则返回非零。逐平台日志和 `summary.log` 保存在 `logs/amlogic_release/<时间>/`。该脚本只编译 SO，Android64 测试程序仍按第 3 节单独编译。

### 6.5 输出路径与运行范围

| 平台 | SO 路径 |
| --- | --- |
| Android32 | `build/AMLOGIC_PIPELINE-android32/Release/libonnxruntime.so` |
| Android64 | `build/AMLOGIC_PIPELINE-android64/Release/libonnxruntime.so` |
| Yocto32 | `build/AMLOGIC_PIPELINE-yocto32/Release/libonnxruntime.so` |
| Yocto64 | `build/AMLOGIC_PIPELINE-yocto64/Release/libonnxruntime.so` |
| Buildroot32 | `build/AMLOGIC_PIPELINE-buildroot32/Release/libonnxruntime.so` |
| Buildroot64 | `build/AMLOGIC_PIPELINE-buildroot64/Release/libonnxruntime.so` |

测试程序的编译、adb 部署和三路比较流程针对 Android64。其他平台需另行构建对应测试应用，并使用匹配目标系统 ABI 的厂商运行库。

Yocto 和 Buildroot 的现有 EP 通过 `AMLNN_MODEL_PATH` 指定预编译 ADLA，同时使用对应 ONNX 创建 Session，提供图及输入输出信息；不执行本文 Android 的在线转换流程。
