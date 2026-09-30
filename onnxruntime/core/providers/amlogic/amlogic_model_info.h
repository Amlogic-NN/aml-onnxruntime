// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace amlogic {

using json = nlohmann::json;

struct TensorValues {
  bool is_valid{false};
  std::vector<int> dims;
  std::vector<int> flat_values;
  std::vector<std::vector<std::vector<std::vector<int>>>> values_4d;
  std::vector<std::vector<std::vector<int>>> values_3d;
  std::vector<std::vector<int>> values_2d;
  std::vector<int> values_1d;

  std::vector<float> dims_float32;
  std::vector<float> flat_values_float32;
  std::vector<std::vector<std::vector<std::vector<float>>>> values_4d_float32;
  std::vector<std::vector<std::vector<float>>> values_3d_float32;
  std::vector<std::vector<float>> values_2d_float32;
  std::vector<float> values_1d_float32;
};

struct TensorInfo {
  std::string tensor_name;
  std::string tensor_type;
  std::vector<int> tensor_shape;
  bool is_constant{false};
  TensorValues values;
  std::vector<float> quant_scale;
  std::vector<int> zero_point;
};

struct OpInfo {
  std::string op_type;
  std::string op_name;
  std::vector<TensorInfo> input_tensors;
  TensorInfo input_weight;
  TensorInfo input_bias;
  std::vector<TensorInfo> output_tensors;
  std::unordered_map<std::string, json> attributes;
  bool has_weight{false};
  bool has_bias{false};
};

Status DumpGraphInfoAsJson(const GraphViewer& graph_viewer,
                           const std::string& output_path,
                           const logging::Logger& logger);

Status DumpGraphInfoPerOpAsJson(const GraphViewer& graph_viewer,
                                const std::string& output_path,
                                const logging::Logger& logger);

Status TryDumpQLinearGraphAsJson(const GraphViewer& graph_viewer,
                                 const std::string& output_path,
                                 const logging::Logger& logger,
                                 bool& dumped);

}  // namespace amlogic
}  // namespace onnxruntime
