// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_reference_style_tflite_bridge.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "core/common/common.h"
#include "core/platform/env.h"

#if defined(ORT_AMLOGIC_USE_TFLITE_REFACTOR)
#include "tflite_process_api.h"
#endif

namespace onnxruntime {
namespace amlogic {
namespace {

constexpr const char* kCompileEnvVar = "ORT_AMLOGIC_REFERENCE_STYLE_COMPILE_TFLITE";
constexpr const char* kOutputEnvVar = "ORT_AMLOGIC_REFERENCE_STYLE_TFLITE_OUTPUT";
constexpr const char* kVerboseEnvVar = "ORT_AMLOGIC_REFERENCE_STYLE_TFLITE_VERBOSE";

bool IsTruthyValue(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

#if defined(ORT_AMLOGIC_USE_TFLITE_REFACTOR)
std::string DeriveTfliteOutputPath(const std::string& json_output_path) {
  if (json_output_path.empty()) {
    return "amlogic_reference_style.tflite";
  }

  const std::string suffix = ".json";
  if (json_output_path.size() >= suffix.size() &&
      json_output_path.compare(json_output_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return json_output_path.substr(0, json_output_path.size() - suffix.size()) + ".tflite";
  }

  return json_output_path + ".tflite";
}
#endif

// The embedded refactorer emits per-op diagnostics directly to std::cout.
// Keep normal conversion output concise without hiding ORT logger warnings or
// errors; developers can restore those diagnostics with the verbose variable.
class ScopedRefactorerCoutSilencer {
 public:
  explicit ScopedRefactorerCoutSilencer(bool enabled) {
    if (enabled) {
      previous_buffer_ = std::cout.rdbuf(sink_.rdbuf());
    }
  }

  ~ScopedRefactorerCoutSilencer() {
    if (previous_buffer_ != nullptr) {
      std::cout.rdbuf(previous_buffer_);
    }
  }

 private:
  std::ostringstream sink_;
  std::streambuf* previous_buffer_ = nullptr;
};

}  // namespace

Status MaybeCompileReferenceStyleToTflite(const nlohmann::ordered_json& ops_json,
                                          const std::string& json_output_path,
                                          const logging::Logger& logger) {
  const std::string compile_env = Env::Default().GetEnvironmentVar(kCompileEnvVar);
  if (!IsTruthyValue(compile_env)) {
    return Status::OK();
  }

#if !defined(ORT_AMLOGIC_USE_TFLITE_REFACTOR)
  ORT_UNUSED_PARAMETER(ops_json);
  ORT_UNUSED_PARAMETER(json_output_path);
  ORT_UNUSED_PARAMETER(logger);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         kCompileEnvVar,
                         " is enabled, but this ONNX Runtime build was compiled without "
                         "onnxruntime_USE_AMLOGIC_TFLITE_REFACTOR.");
#else
  std::string output_tflite_path = Env::Default().GetEnvironmentVar(kOutputEnvVar);
  if (output_tflite_path.empty()) {
    output_tflite_path = DeriveTfliteOutputPath(json_output_path);
  }

  const bool verbose = IsTruthyValue(Env::Default().GetEnvironmentVar(kVerboseEnvVar));
  ScopedRefactorerCoutSilencer silence_refactorer_stdout(!verbose);
  const int exit_code = amlogic_tflite_refactor::RunOnnx2TfRefactorFromJsonText(
      output_tflite_path, ops_json.dump());
  ORT_RETURN_IF_NOT(exit_code == 0,
                    "Failed to compile Amlogic reference-style JSON to TFLite. Output path: ",
                    output_tflite_path, ", exit code: ", exit_code);

  // The refactorer currently reports success even if its output path cannot
  // be opened (for example, because a relative-path parent directory does
  // not exist). Verify the artifact before emitting a success log.
  std::ifstream output_file(output_tflite_path, std::ios::binary | std::ios::ate);
  ORT_RETURN_IF_NOT(output_file.good() && output_file.tellg() > 0,
                    "Amlogic reference-style TFLite compiler reported success but did not create a "
                    "non-empty output file: ",
                    output_tflite_path);

  LOGS(logger, WARNING) << "Amlogic compiled reference-style TFLite model to "
                        << output_tflite_path;
  return Status::OK();
#endif
}

}  // namespace amlogic
}  // namespace onnxruntime
