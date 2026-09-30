// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside the anonymous namespace. It keeps the legacy qlinear exporter request
// planning helpers in one place without changing behavior.

std::string RequestKey(const std::string& tensor_name, const QuantInfo& quant) {
  return tensor_name + "|" + quant.scale.dump() + "|" + quant.zero_point.dump();
}

Status BuildRequantRequests(const GraphViewer& graph_viewer,
                            std::unordered_map<std::string, RequantRequest>& requests) {
  int insert_id = 0;
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node == nullptr || node->OpType() != "QLinearConcat") {
      continue;
    }

    const auto inputs = node->InputDefs();
    if (inputs.size() < 5 || inputs[0] == nullptr || inputs[1] == nullptr) {
      continue;
    }

    QuantInfo output_quant;
    ORT_RETURN_IF_ERROR(ReadQuantInfo(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), output_quant));

    for (size_t i = 2; i + 2 < inputs.size(); i += 3) {
      if (inputs[i] == nullptr || inputs[i + 1] == nullptr || inputs[i + 2] == nullptr) {
        continue;
      }

      QuantInfo input_quant;
      ORT_RETURN_IF_ERROR(ReadQuantInfo(graph_viewer, inputs[i + 1]->Name(), inputs[i + 2]->Name(), input_quant));
      if (JsonEqual(input_quant.scale, output_quant.scale) &&
          JsonEqual(input_quant.zero_point, output_quant.zero_point)) {
        continue;
      }

      RequantRequest request;
      request.target_quant = output_quant;
      request.output_name = reference_style_internal::CompatibleInsertedQuantizeName(inputs[i]->Name(),
                                                                                     ++insert_id);
      const std::string key = RequestKey(inputs[i]->Name(), output_quant);
      requests.emplace(key, std::move(request));
    }
  }
  return Status::OK();
}

Status BuildPendingPadRequests(const GraphViewer& graph_viewer,
                               std::unordered_map<std::string, std::vector<PendingPadRequest>>& requests) {
  int conv_id = 0;
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node == nullptr || node->OpType() != "QLinearConv") {
      continue;
    }

    ++conv_id;
    if (!IsPaddedQLinearConvNode(node)) {
      continue;
    }

    const auto inputs = node->InputDefs();
    if (inputs.empty() || inputs[0] == nullptr) {
      continue;
    }

    PendingPadRequest request;
    request.node_index = node_index;
    request.conv_id = conv_id;
    request.pads = GetIntsAttribute(*node, "pads", {0, 0, 0, 0});
    requests[inputs[0]->Name()].push_back(std::move(request));
  }
  return Status::OK();
}

Status BuildPendingPreReshapeTransposeRequests(
    const GraphViewer& graph_viewer,
    std::unordered_map<std::string, std::vector<PendingPreReshapeTransposeRequest>>& requests) {
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node == nullptr || node->OpType() != "Reshape") {
      continue;
    }

    const auto inputs = node->InputDefs();
    const auto outputs = node->OutputDefs();
    if (inputs.size() < 2 || outputs.empty() ||
        inputs[0] == nullptr || outputs[0] == nullptr) {
      continue;
    }

    const std::vector<int> input_shape = ShapeForTensor(graph_viewer, inputs[0]->Name(), true);
    const std::vector<int> output_shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
    if (input_shape.size() != 4 || output_shape.size() != 3) {
      continue;
    }

    PendingPreReshapeTransposeRequest request;
    request.node_index = node_index;
    requests[inputs[0]->Name()].push_back(std::move(request));
  }
  return Status::OK();
}

Status BuildPendingPreTransposeRequests(
    const GraphViewer& graph_viewer,
    std::unordered_map<std::string, std::vector<PendingPreTransposeRequest>>& requests) {
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node == nullptr || node->OpType() != "Transpose") {
      continue;
    }

    const auto inputs = node->InputDefs();
    const auto outputs = node->OutputDefs();
    if (inputs.empty() || outputs.empty() || inputs[0] == nullptr || outputs[0] == nullptr) {
      continue;
    }

    const std::vector<int> input_shape = ShapeForTensor(graph_viewer, inputs[0]->Name(), false);
    const std::vector<int> output_shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
    if (input_shape.size() != 4 || output_shape.size() != 4) {
      continue;
    }

    PendingPreTransposeRequest request;
    request.node_index = node_index;
    requests[inputs[0]->Name()].push_back(std::move(request));
  }
  return Status::OK();
}
