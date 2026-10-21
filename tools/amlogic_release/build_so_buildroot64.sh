#!/usr/bin/env bash
set -euo pipefail

: "${AMLNN_NNSDK2_INCLUDE_DIR:?AMLNN_NNSDK2_INCLUDE_DIR must be set}"
: "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR:?AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR must be set}"
: "${BUILDROOT64_PATH:?BUILDROOT64_PATH must be set to the Buildroot toolchain directory}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOLCHAIN_ROOT="${BUILDROOT64_PATH}"
CC="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc"
CXX="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++"
SYSROOT="${TOOLCHAIN_ROOT}/aarch64-none-linux-gnu/libc"
TOOLCHAIN_FILE="${REPO_ROOT}/cmake/linux_arm64_crosscompile_toolchain.cmake"

[[ -x "${CC}" ]] || { echo "C compiler was not found: ${CC}" >&2; exit 1; }
[[ -x "${CXX}" ]] || { echo "C++ compiler was not found: ${CXX}" >&2; exit 1; }
[[ -d "${SYSROOT}" ]] || { echo "Toolchain sysroot was not found: ${SYSROOT}" >&2; exit 1; }
[[ -f "${TOOLCHAIN_FILE}" ]] || { echo "CMake toolchain file was not found: ${TOOLCHAIN_FILE}" >&2; exit 1; }
export PATH="${TOOLCHAIN_ROOT}/bin:${PATH}"

cd "${REPO_ROOT}"

./build.sh \
  --build_shared_lib \
  --use_amlogic \
  --skip_tests \
  --compile_no_warning_as_error \
  --config Release \
  --no_kleidiai \
  --no_sve \
  --build_dir "build/AMLOGIC_PIPELINE-buildroot64" \
  --cmake_extra_defines \
  onnxruntime_AMLOGIC_ALLOW_GCC_10=ON \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  CMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  CMAKE_SYSTEM_PROCESSOR=aarch64 \
  CMAKE_C_COMPILER="${CC}" \
  CMAKE_CXX_COMPILER="${CXX}" \
  CMAKE_SYSROOT="${SYSROOT}" \
  AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR="${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}" \
  AMLNN_NNSDK2_INCLUDE_DIR="${AMLNN_NNSDK2_INCLUDE_DIR}"
