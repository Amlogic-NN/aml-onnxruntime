// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace amlnn {
namespace reference_style_internal {

bool IsQLinearGraph(const GraphViewer& graph_viewer);

bool IsReferenceStyleSupportedOp(const std::string& op_type);

bool IsReferenceStyleGraph(const GraphViewer& graph_viewer);

}  // namespace reference_style_internal
}  // namespace amlnn
}  // namespace onnxruntime
