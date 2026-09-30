// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_model_info.h"
#include "core/providers/amlogic/amlogic_reference_style_compat.h"
#include "core/providers/amlogic/amlogic_reference_style_gate.h"
#include "core/providers/amlogic/amlogic_reference_style_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace amlogic {

namespace {

using ordered_json = nlohmann::ordered_json;

// The legacy qlinear exporter is physically split behind a single internal
// include entrypoint so this file can stay close to a thin orchestration
// wrapper, similar to reference_style_internal below.
#include "core/providers/amlogic/amlogic_qlinear_impl.h"

}  // namespace

namespace reference_style_internal {

using ordered_json = nlohmann::ordered_json;

// reference_style is physically split across shared/compat/collectors/
// postprocess modules and included through a single internal entrypoint.
#include "core/providers/amlogic/amlogic_reference_style_impl.h"

}  // namespace reference_style_internal

Status TryDumpQLinearGraphAsJson(const GraphViewer& graph_viewer,
                                 const std::string& output_path,
                                 const logging::Logger& logger,
                                 bool& dumped) {
  dumped = false;
  if (!reference_style_internal::IsQLinearGraph(graph_viewer)) {
    return Status::OK();
  }

  ordered_json ops_json;
  ORT_RETURN_IF_ERROR(BuildLegacyQLinearOpsJson(graph_viewer, ops_json));

  std::ofstream output(output_path, std::ios::out | std::ios::trunc);
  ORT_RETURN_IF_NOT(output.good(), "Failed to open Amlogic QLinear model info dump file: ", output_path);
  output << ops_json.dump(2) << '\n';
  output.close();

  dumped = true;
  LOGS(logger, WARNING) << "Amlogic dumped QLinear NHWC/TFLite-style ONNX op info to "
                        << output_path << ", ops: " << ops_json.size();
  return Status::OK();
}

}  // namespace amlogic
}  // namespace onnxruntime

