#!/usr/bin/env bash
set -euo pipefail

: "${AMLNN_NNSDK2_INCLUDE_DIR:?AMLNN_NNSDK2_INCLUDE_DIR must be set}"
: "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR:?AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR must be set}"
: "${ANDROID_NDK_PATH:?ANDROID_NDK_PATH must be set}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

./build.sh \
  --build_shared_lib \
  --skip_tests \
  --android \
  --android_ndk_path "${ANDROID_NDK_PATH}" \
  --android_abi "arm64-v8a" \
  --android_api 21 \
  --config Release \
  --no_kleidiai \
  --no_sve \
  --build_dir "build/AMLOGIC_AMLNN_PIPELINE-android64" \
  --cmake_extra_defines \
  onnxruntime_USE_AMLOGIC=ON \
  onnxruntime_USE_AMLNN=ON \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR="${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}" \
  AMLNN_NNSDK2_INCLUDE_DIR="${AMLNN_NNSDK2_INCLUDE_DIR}"
