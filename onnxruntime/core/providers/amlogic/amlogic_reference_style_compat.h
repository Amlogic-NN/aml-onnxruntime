// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace onnxruntime {
namespace amlogic {

struct QGemmBridgeNames {
  std::string dequant_output_name;
  std::string requant_output_name;
};

namespace reference_style_internal {

std::string CompatibleInsertedQuantizeName(const std::string& tensor_name, int fallback_id);

bool TryGetCompatibleQGemmBridgeNames(const std::string& tensor_name, QGemmBridgeNames& names);

}  // namespace reference_style_internal

}  // namespace amlogic
}  // namespace onnxruntime
