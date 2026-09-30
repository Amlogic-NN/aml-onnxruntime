// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace amlogic {

Status MaybeCompileReferenceStyleToTflite(const nlohmann::ordered_json& ops_json,
                                          const std::string& json_output_path,
                                          const logging::Logger& logger);

}  // namespace amlogic
}  // namespace onnxruntime
