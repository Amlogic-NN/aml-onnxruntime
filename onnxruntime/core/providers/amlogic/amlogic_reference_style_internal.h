// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/amlogic/amlogic_reference_style_compat.h"
#include "core/providers/amlogic/amlogic_reference_style_gate.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace amlogic {

using ordered_json = nlohmann::ordered_json;

namespace reference_style_internal {

Status CollectInitialReferenceStyleOps(const GraphViewer& graph_viewer,
                                       ordered_json& ops_json);

Status PostprocessReferenceQuantScaleFill(ordered_json& ops_json);

void PostprocessReferenceOnnxInfo(ordered_json& ops_json);

void PostprocessReferenceQGemmBridges(ordered_json& ops_json);

void PostprocessReferenceConcatRequantize(ordered_json& ops_json);

void PostprocessReferenceUint8ToInt8(ordered_json& ops_json);

}  // namespace reference_style_internal

}  // namespace amlogic
}  // namespace onnxruntime
