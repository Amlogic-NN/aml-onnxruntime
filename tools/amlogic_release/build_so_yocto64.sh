#!/usr/bin/env bash
set -euo pipefail

: "${AMLNN_NNSDK2_INCLUDE_DIR:?AMLNN_NNSDK2_INCLUDE_DIR must be set}"
: "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR:?AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR must be set}"
: "${YOCTO64_PATH:?YOCTO64_PATH must be set to the Yocto SDK directory}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

HOST_CMAKE="$(command -v cmake || true)"
[[ -n "${HOST_CMAKE}" ]] || { echo "cmake was not found before loading the Yocto SDK" >&2; exit 1; }

SDK_ROOT="${YOCTO64_PATH}"
[[ -d "${SDK_ROOT}" ]] || { echo "YOCTO64_PATH must point to an existing SDK directory: ${SDK_ROOT}" >&2; exit 1; }
TOOLCHAIN="${SDK_ROOT}/sysroots/x86_64-pokysdk-linux/usr/share/cmake/OEToolchainConfig.cmake"
[[ -f "${TOOLCHAIN}" ]] || { echo "Yocto toolchain file was not found: ${TOOLCHAIN}" >&2; exit 1; }
ENV_SETUP="$(find "${SDK_ROOT}" -maxdepth 1 -type f -name 'environment-setup-*' -print -quit)"
[[ -n "${ENV_SETUP}" ]] || { echo "Yocto environment setup script was not found under ${SDK_ROOT}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_SETUP}"

"${REPO_ROOT}/build.sh" \
  --cmake_path "${HOST_CMAKE}" \
  --build_shared_lib \
  --use_amlogic \
  --skip_tests \
  --config Release \
  --build_dir "build/AMLOGIC_PIPELINE-yocto64" \
  --cmake_extra_defines \
  CMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
  AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR="${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR}" \
  AMLNN_NNSDK2_INCLUDE_DIR="${AMLNN_NNSDK2_INCLUDE_DIR}"
