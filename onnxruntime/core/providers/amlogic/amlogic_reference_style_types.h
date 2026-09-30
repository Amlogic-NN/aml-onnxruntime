// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace amlogic {

struct QuantInfo {
  nlohmann::ordered_json scale;
  nlohmann::ordered_json zero_point;
};

struct TensorState {
  std::string name;
  std::string tensor_type;
  std::vector<int> shape;
  bool is_constant{false};
  bool has_values{false};
  nlohmann::ordered_json values;
  bool has_quant{false};
  nlohmann::ordered_json quant_scale;
  nlohmann::ordered_json zero_point;
};

struct RequantRequest {
  QuantInfo target_quant;
  std::string output_name;
  TensorState output_state;
};

struct PendingPadRequest {
  int node_index{0};
  int conv_id{0};
  std::vector<int> pads;
  bool emitted{false};
};

struct PendingPreReshapeTransposeRequest {
  int node_index{0};
  bool emitted{false};
};

struct PendingPreTransposeRequest {
  int node_index{0};
  bool emitted{false};
};

}  // namespace amlogic
}  // namespace onnxruntime
