#!/usr/bin/env bash
set -euo pipefail

: "${AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR:?AMLOGIC_AML_COMPILER_CORE_INCLUDE_DIR must be set}"
: "${AMLNN_NNSDK2_INCLUDE_DIR:?AMLNN_NNSDK2_INCLUDE_DIR must be set}"
: "${ANDROID_NDK_PATH:?ANDROID_NDK_PATH must be set}"
: "${YOCTO64_PATH:?YOCTO64_PATH must be set}"
: "${YOCTO32_PATH:?YOCTO32_PATH must be set}"
: "${BUILDROOT64_PATH:?BUILDROOT64_PATH must be set}"
: "${BUILDROOT32_PATH:?BUILDROOT32_PATH must be set}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RELEASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${REPO_ROOT}/logs/amlogic_release/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"
SUMMARY_LOG="${LOG_ROOT}/summary.log"
OVERALL_STATUS=0

run_build() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  local start_time
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"

  printf '[%s] START %s\n' "${start_time}" "${name}" | tee -a "${SUMMARY_LOG}"
  if "$@" > >(tee "${log_file}") 2>&1; then
    printf '[%s] PASS  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${name}" | tee -a "${SUMMARY_LOG}"
  else
    local status=$?
    OVERALL_STATUS=1
    printf '[%s] FAIL  %s (exit %d)\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${name}" "${status}" | tee -a "${SUMMARY_LOG}"
  fi
}

run_build android64 bash "${RELEASE_DIR}/build_so_android64.sh"
run_build android32 bash "${RELEASE_DIR}/build_so_android32.sh"
run_build yocto64 bash "${RELEASE_DIR}/build_so_yocto64.sh"
run_build yocto32 bash "${RELEASE_DIR}/build_so_yocto32.sh"
run_build buildroot64 bash "${RELEASE_DIR}/build_so_buildroot64.sh"
run_build buildroot32 bash "${RELEASE_DIR}/build_so_buildroot32.sh"

printf '\nLogs: %s\nSummary: %s\n' "${LOG_ROOT}" "${SUMMARY_LOG}"
exit "${OVERALL_STATUS}"
