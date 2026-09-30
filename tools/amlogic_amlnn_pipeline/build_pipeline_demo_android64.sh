#!/usr/bin/env bash
set -euo pipefail

: "${ANDROID_NDK_PATH:?ANDROID_NDK_PATH must be set}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CXX="${ANDROID_NDK_PATH}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang++"
ORT_BUILD_DIR="${REPO_ROOT}/build/AMLOGIC_AMLNN_PIPELINE-android64/Release"
OUTPUT_DIR="${REPO_ROOT}/tools/amlogic_amlnn_pipeline/pipeline_demo/bin/android64"

[[ -x "${CXX}" ]] || { echo "Compiler not found: ${CXX}" >&2; exit 1; }
[[ -f "${ORT_BUILD_DIR}/libonnxruntime.so" ]] || {
  echo "ONNX Runtime library not found: ${ORT_BUILD_DIR}/libonnxruntime.so" >&2
  exit 1
}

mkdir -p "${OUTPUT_DIR}"
"${CXX}" \
  -std=c++17 \
  -O2 \
  -fexceptions \
  -frtti \
  -nostdlib++ \
  -stdlib=libc++ \
  -I"${REPO_ROOT}/include/onnxruntime" \
  -I"${REPO_ROOT}/include/onnxruntime/core/session" \
  "${REPO_ROOT}/tools/amlogic_amlnn_pipeline/pipeline_demo/pipeline_demo.cc" \
  -L"${ORT_BUILD_DIR}" \
  -lonnxruntime \
  -lc++_shared \
  -Wl,-rpath,'$ORIGIN' \
  -o "${OUTPUT_DIR}/pipeline_demo"

cp "${ORT_BUILD_DIR}/libonnxruntime.so" "${OUTPUT_DIR}/"
echo "Output: ${OUTPUT_DIR}"
