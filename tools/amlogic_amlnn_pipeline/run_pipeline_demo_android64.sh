#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model.onnx> <local_output_dir> [log_level] [--dump-tflite] [input0.bin ...]" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="${PACKAGE_DIR:-${REPO_ROOT}/tools/amlogic_amlnn_pipeline/pipeline_demo/bin/android64}"
ORT_BUILD="${ORT_BUILD:-${REPO_ROOT}/build/AMLOGIC_AMLNN_PIPELINE-android64/Release}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/amlogic_amlnn_pipeline_test}"

ONNX_MODEL="$1"
LOCAL_OUTPUT_DIR="$2"
shift 2

LOG_LEVEL="error"
if [[ $# -gt 0 ]]; then
  case "$1" in
    error|info|debug|0|1|2) LOG_LEVEL="$1"; shift ;;
  esac
fi

DUMP_TFLITE=0
INPUT_FILES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dump-tflite)
      DUMP_TFLITE=1
      shift
      ;;
    --no-dump-tflite)
      DUMP_TFLITE=0
      shift
      ;;
    *)
      INPUT_FILES+=("$1")
      shift
      ;;
  esac
done

RUNNER="${PACKAGE_DIR}/pipeline_demo"
ORT_SO="${ORT_BUILD}/libonnxruntime.so"
[[ -f "${ORT_SO}" ]] || ORT_SO="${PACKAGE_DIR}/libonnxruntime.so"

command -v adb >/dev/null || { echo "adb not found" >&2; exit 1; }
for file in "${RUNNER}" "${ORT_SO}" "${ONNX_MODEL}"; do
  [[ -f "${file}" ]] || { echo "File not found: ${file}" >&2; exit 1; }
done

# Check the full set of shipped ELF files. The bridge may use shared libc++
# even when the ORT library and the test executable do not.
NEED_LIBCXX=0
for file in "${RUNNER}" "${ORT_SO}"; do
  if readelf -d "${file}" | grep -q 'Shared library: \[libc++_shared\.so\]'; then
    NEED_LIBCXX=1
    break
  fi
done
if [[ ${NEED_LIBCXX} -eq 1 ]]; then
  NDK_ROOT="${ANDROID_NDK_PATH:?ANDROID_NDK_PATH is required by libc++_shared.so dependency}"
  LIBCXX_SO="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
  [[ -f "${LIBCXX_SO}" ]] || { echo "File not found: ${LIBCXX_SO}" >&2; exit 1; }
fi

mkdir -p "${LOCAL_OUTPUT_DIR}"
adb get-state >/dev/null
adb shell "rm -rf '${REMOTE_DIR}' && mkdir -p '${REMOTE_DIR}/output'"

adb push "${RUNNER}" "${REMOTE_DIR}/pipeline_demo" >/dev/null
adb push "${ORT_SO}" "${REMOTE_DIR}/libonnxruntime.so" >/dev/null
if [[ ${NEED_LIBCXX} -eq 1 ]]; then
  adb push "${LIBCXX_SO}" "${REMOTE_DIR}/libc++_shared.so" >/dev/null
fi

REMOTE_MODEL="${REMOTE_DIR}/$(basename -- "${ONNX_MODEL}")"
adb push "${ONNX_MODEL}" "${REMOTE_MODEL}" >/dev/null

REMOTE_INPUTS=""
for input_file in "${INPUT_FILES[@]}"; do
  [[ -f "${input_file}" ]] || { echo "Input file not found: ${input_file}" >&2; exit 1; }
  remote_input="${REMOTE_DIR}/$(basename -- "${input_file}")"
  adb push "${input_file}" "${remote_input}" >/dev/null
  REMOTE_INPUTS+=" '$(basename -- "${input_file}")'"
done

adb shell "chmod 755 '${REMOTE_DIR}/pipeline_demo'"
adb shell "setprop vendor.TFLITE_FILE_DUMP_LEVEL '${DUMP_TFLITE}'"
echo "Running single Session Android64 flow"
adb shell "cd '${REMOTE_DIR}' && export LD_LIBRARY_PATH='${REMOTE_DIR}' && ./pipeline_demo '$(basename -- "${ONNX_MODEL}")' output '${LOG_LEVEL}'${REMOTE_INPUTS}" \
  2>&1 | tee "${LOCAL_OUTPUT_DIR}/run.log"

remote_outputs="$(adb shell "cd '${REMOTE_DIR}/output' && for file in output_*.bin output_*.meta; do if [ -f \"\$file\" ]; then echo \"\$file\"; fi; done" | tr -d '\r')"
[[ -n "${remote_outputs}" ]] || { echo "No AMLNN outputs found in ${REMOTE_DIR}/output" >&2; exit 1; }
while IFS= read -r remote_file; do
  adb pull "${REMOTE_DIR}/output/${remote_file}" "${LOCAL_OUTPUT_DIR}/" >/dev/null
done <<< "${remote_outputs}"

MODEL_STEM="$(basename -- "${ONNX_MODEL}")"
MODEL_STEM="${MODEL_STEM%.onnx}"
for artifact in "${MODEL_STEM}.adla" "${MODEL_STEM}.tflite"; do
  if adb shell "test -f '${REMOTE_DIR}/output/${artifact}'"; then
    adb pull "${REMOTE_DIR}/output/${artifact}" "${LOCAL_OUTPUT_DIR}/" >/dev/null
    echo "Pulled conversion artifact: ${LOCAL_OUTPUT_DIR}/${artifact}"
  fi
done

if [[ ${DUMP_TFLITE} -eq 1 && ! -f "${LOCAL_OUTPUT_DIR}/${MODEL_STEM}.tflite" ]]; then
  echo "TFLite dump requested, but ${MODEL_STEM}.tflite was not generated" >&2
fi

echo "Outputs: ${LOCAL_OUTPUT_DIR}"
echo "Remote conversion artifacts: ${REMOTE_DIR}/output"
