// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace amlogic {

// Build the maintained reference-style representation in memory. `built` is
// false when the graph is not eligible for the reference-style exporter.
Status BuildQLinearGraphAsReferenceStyleJson(const GraphViewer& graph_viewer,
                                              nlohmann::ordered_json& root_json,
                                              const logging::Logger& logger,
                                              bool& built);

Status TryDumpQLinearGraphAsReferenceStyleJson(const GraphViewer& graph_viewer,
                                               const std::string& output_path,
                                               const logging::Logger& logger,
                                               bool& dumped);

}  // namespace amlogic
}  // namespace onnxruntime
