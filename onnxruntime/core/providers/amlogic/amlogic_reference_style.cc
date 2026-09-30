// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <string>
#include <string_view>

#include "core/common/common.h"
#include "core/providers/amlogic/amlogic_reference_style.h"
#include "core/providers/amlogic/amlogic_reference_style_internal.h"
#include "core/providers/amlogic/amlogic_reference_style_tflite_bridge.h"

namespace onnxruntime {
namespace amlogic {

namespace reference_style_internal {

bool IsQLinearGraph(const GraphViewer& graph_viewer) {
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node != nullptr && node->OpType() == "QLinearConv") {
      return true;
    }
  }
  return false;
}

bool IsReferenceStyleSupportedOp(const std::string& op_type) {
  return op_type == "QuantizeLinear" ||
         op_type == "DequantizeLinear" ||
         op_type == "QLinearConv" ||
         op_type == "QLinearSigmoid" ||
         op_type == "QLinearSoftmax" ||
         op_type == "QLinearMul" ||
         op_type == "QLinearAdd" ||
         op_type == "QLinearGlobalAveragePool" ||
         op_type == "QLinearConcat" ||
         op_type == "QGemm" ||
         op_type == "Conv" ||
         op_type == "Identity" ||
         op_type == "Pad" ||
         op_type == "LRN" ||
         op_type == "PRelu" ||
         op_type == "LeakyRelu" ||
         op_type == "HardSwish" ||
         op_type == "HardSigmoid" ||
         op_type == "Clip" ||
         op_type == "Relu" ||
         op_type == "Sigmoid" ||
         op_type == "Tanh" ||
         op_type == "Erf" ||
         op_type == "Log" ||
         op_type == "Sqrt" ||
         op_type == "Softmax" ||
         op_type == "Exp" ||
         op_type == "Mul" ||
         op_type == "Add" ||
         op_type == "Pow" ||
         op_type == "Mod" ||
         op_type == "Concat" ||
         op_type == "BatchNormalization" ||
         op_type == "Dropout" ||
         op_type == "Constant" ||
         op_type == "AveragePool" ||
         op_type == "GlobalAveragePool" ||
         op_type == "InstanceNormalization" ||
         op_type == "Gemm" ||
         op_type == "MatMul" ||
         op_type == "ReduceMean" ||
         op_type == "ReduceMax" ||
         op_type == "ReduceSum" ||
         op_type == "Squeeze" ||
         op_type == "Unsqueeze" ||
         op_type == "Split" ||
         op_type == "Resize" ||
         op_type == "Upsample" ||
         op_type == "MaxPool" ||
         op_type == "ConvTranspose" ||
         op_type == "Flatten" ||
         op_type == "Reshape" ||
         op_type == "Transpose" ||
         op_type == "Slice" ||
         op_type == "Cast" ||
         op_type == "Gather" ||
         op_type == "GatherElements" ||
         op_type == "Tile" ||
         op_type == "Expand" ||
         op_type == "TopK" ||
         op_type == "Sub" ||
         op_type == "Div";
}

const Node* FindFirstUnsupportedReferenceStyleNode(const GraphViewer& graph_viewer) {
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node != nullptr && !IsReferenceStyleSupportedOp(node->OpType())) {
      return node;
    }
  }
  return nullptr;
}

bool IsReferenceStyleGraph(const GraphViewer& graph_viewer) {
  bool has_supported_node = false;
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node == nullptr) {
      continue;
    }

    if (!IsReferenceStyleSupportedOp(node->OpType())) {
      return false;
    }
    has_supported_node = true;
  }
  return has_supported_node;
}

}  // namespace reference_style_internal

namespace {


std::string ReferenceStyleTensorName(std::string_view onnx_name) {
  if (!onnx_name.empty() && onnx_name.front() == '/') {
    return std::string("wa") + std::string(onnx_name);
  }
  return std::string(onnx_name);
}

ordered_json BuildReferenceStyleRoot(const GraphViewer& graph_viewer,
                                     const ordered_json& ops_json) {
  ordered_json graph_inputs = ordered_json::array();
  for (const NodeArg* input : graph_viewer.GetInputs()) {
    if (input != nullptr && input->Exists()) {
      graph_inputs.push_back(ReferenceStyleTensorName(input->Name()));
    }
  }

  ordered_json graph_outputs = ordered_json::array();
  for (const NodeArg* output : graph_viewer.GetOutputs()) {
    if (output != nullptr && output->Exists()) {
      graph_outputs.push_back(ReferenceStyleTensorName(output->Name()));
    }
  }

  ordered_json root;
  root["format"] = "amlogic_reference_style";
  root["version"] = 2;
  root["graph_inputs"] = std::move(graph_inputs);
  root["graph_outputs"] = std::move(graph_outputs);
  root["ops"] = ops_json;
  return root;
}

}  // namespace

// reference_style is the preserved collector + postprocess baseline aligned
// with the current onnx2tf-style reference JSON contract for models_dataset.
// Keep the stage ordering below stable unless the baseline regression in
// amlogic_reference_style_check.sh is intentionally updated together with the
// collector/postprocess docs.
Status BuildQLinearGraphAsReferenceStyleJson(const GraphViewer& graph_viewer,
                                              ordered_json& root_json,
                                              const logging::Logger& logger,
                                              bool& built) {
  built = false;
  if (!reference_style_internal::IsReferenceStyleGraph(graph_viewer)) {
    if (const Node* unsupported = reference_style_internal::FindFirstUnsupportedReferenceStyleNode(graph_viewer)) {
      LOGS(logger, WARNING) << "Amlogic reference-style export skipped: unsupported op "
                            << unsupported->OpType() << ", node: " << unsupported->Name();
    }
    return Status::OK();
  }

  ordered_json ops_json = ordered_json::array();
  ORT_RETURN_IF_ERROR(reference_style_internal::CollectInitialReferenceStyleOps(graph_viewer, ops_json));
  ORT_RETURN_IF_ERROR(reference_style_internal::PostprocessReferenceQuantScaleFill(ops_json));
  reference_style_internal::PostprocessReferenceOnnxInfo(ops_json);
  reference_style_internal::PostprocessReferenceQGemmBridges(ops_json);
  reference_style_internal::PostprocessReferenceConcatRequantize(ops_json);
  reference_style_internal::PostprocessReferenceUint8ToInt8(ops_json);

  root_json = BuildReferenceStyleRoot(graph_viewer, ops_json);
  built = true;
  return Status::OK();
}

Status TryDumpQLinearGraphAsReferenceStyleJson(const GraphViewer& graph_viewer,
                                               const std::string& output_path,
                                               const logging::Logger& logger,
                                               bool& dumped) {
  dumped = false;
  ordered_json root_json;
  ORT_RETURN_IF_ERROR(BuildQLinearGraphAsReferenceStyleJson(graph_viewer, root_json, logger, dumped));
  if (!dumped) {
    return Status::OK();
  }

  std::ofstream output(output_path, std::ios::out | std::ios::trunc);
  ORT_RETURN_IF_NOT(output.good(),
                    "Failed to open Amlogic reference-style model info dump file: ", output_path);
  output << root_json.dump(2) << '\n';
  output.close();

  ORT_RETURN_IF_ERROR(MaybeCompileReferenceStyleToTflite(root_json, output_path, logger));

  LOGS(logger, WARNING) << "Amlogic dumped reference-style ONNX->TFLite op info to "
                        << output_path << ", ops: " << root_json.at("ops").size();
  return Status::OK();
}

}  // namespace amlogic
}  // namespace onnxruntime
