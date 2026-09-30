// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside the anonymous namespace. It keeps the legacy qlinear exporter
// Pad/Transpose insertion helpers outside the main translation unit body
// without changing behavior.

Status MakeQLinearConvInputState(const GraphViewer& graph_viewer,
                                 const Node& node,
                                 const std::unordered_map<std::string, TensorState>& states,
                                 TensorState& input_state) {
  const auto inputs = node.InputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 3 && inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr,
                    "Invalid QLinearConv input.");
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(),
                                           true, input_state));
  const auto state_it = states.find(inputs[0]->Name());
  if (state_it != states.end()) {
    input_state = state_it->second;
  }
  return Status::OK();
}

Status EmitPadForQLinearConv(const int conv_id,
                             const TensorState& original_input_state,
                             const std::vector<int>& pads,
                             ordered_json& ops_json,
                             TensorState& pad_output_state) {
  const bool resnet_stem_pad = pads.size() == 4 &&
                               std::all_of(pads.begin(), pads.end(), [](int pad) { return pad == 3; });
  const std::string output_prefix = resnet_stem_pad ? "Pad_insert_output" : "Conv_SAME_Pad_insert_output";
  const std::string values_prefix = resnet_stem_pad ? "Pad_insert_values" : "Conv_SAME_Pad_insert_values";

  TensorState pad_input = original_input_state;
  pad_input.shape = ApplyNhwcPads(original_input_state.shape, pads);

  pad_output_state = pad_input;
  pad_output_state.name = output_prefix + std::to_string(conv_id);
  pad_output_state.is_constant = false;
  pad_output_state.has_values = false;

  ordered_json pad_values = ordered_json::array({0, 0, pads[0], pads[1], pads[2], pads[3], 0, 0});
  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Pad");
  payload["input_tensor_info_1"] = TensorInfoJson(pad_input);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(values_prefix + std::to_string(conv_id),
                                                          "int32", {4, 2}, pad_values);
  payload["output_tensor_info"] = TensorInfoJson(pad_output_state);
  ops_json.push_back(WrapOp("Pad", std::move(payload)));
  return Status::OK();
}

Status EmitPadForMaxPool(const int pad_id,
                         const TensorState& original_input_state,
                         const std::vector<int>& pads,
                         ordered_json& ops_json,
                         TensorState& pad_output_state) {
  TensorState pad_input = original_input_state;
  pad_input.shape = ApplyNhwcPads(original_input_state.shape, pads);
  if (pad_input.tensor_type == "float32") {
    pad_input.has_quant = true;
    pad_input.quant_scale = "None";
    pad_input.zero_point = "None";
  }

  pad_output_state = pad_input;
  pad_output_state.name = std::string("SAME_Pad_insert_output") + std::to_string(pad_id);
  pad_output_state.is_constant = false;
  pad_output_state.has_values = false;

  ordered_json pad_values = ordered_json::array({0, 0, pads[0], pads[1], pads[2], pads[3], 0, 0});
  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Pad");
  payload["input_tensor_info_1"] = TensorInfoJson(pad_input);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(
      std::string("SAME_Pad_insert_values") + std::to_string(pad_id), "int32", {4, 2}, pad_values);
  payload["output_tensor_info"] = TensorInfoJson(pad_output_state);
  ops_json.push_back(WrapOp("Pad", std::move(payload)));
  return Status::OK();
}

Status EmitPendingPadsIfNeeded(const GraphViewer& graph_viewer,
                               const std::string& output_name,
                               std::unordered_map<std::string, TensorState>& states,
                               std::unordered_map<std::string, std::vector<PendingPadRequest>>& requests,
                               std::unordered_map<int, TensorState>& precomputed_pad_outputs,
                               ordered_json& ops_json) {
  auto request_it = requests.find(output_name);
  if (request_it == requests.end()) {
    return Status::OK();
  }

  auto& output_requests = request_it->second;
  for (size_t reverse_i = output_requests.size(); reverse_i > 0; --reverse_i) {
    PendingPadRequest& request = output_requests[reverse_i - 1];
    if (request.emitted) {
      continue;
    }

    const Node* conv_node = graph_viewer.GetNode(static_cast<NodeIndex>(request.node_index));
    ORT_RETURN_IF_NOT(conv_node != nullptr, "Missing pending padded QLinearConv node: ", request.node_index);

    TensorState input_state;
    ORT_RETURN_IF_ERROR(MakeQLinearConvInputState(graph_viewer, *conv_node, states, input_state));

    TensorState pad_output;
    ORT_RETURN_IF_ERROR(EmitPadForQLinearConv(request.conv_id, input_state, request.pads, ops_json, pad_output));
    precomputed_pad_outputs[request.node_index] = std::move(pad_output);
    request.emitted = true;
  }

  return Status::OK();
}

Status EmitPendingPreReshapeTransposesIfNeeded(
    const GraphViewer& graph_viewer,
    const std::string& output_name,
    int& transpose_id,
    std::unordered_map<std::string, TensorState>& states,
    std::unordered_map<std::string, std::vector<PendingPreReshapeTransposeRequest>>& requests,
    std::unordered_map<int, TensorState>& precomputed_reshape_inputs,
    ordered_json& ops_json) {
  auto request_it = requests.find(output_name);
  if (request_it == requests.end()) {
    return Status::OK();
  }

  auto& output_requests = request_it->second;
  for (PendingPreReshapeTransposeRequest& request : output_requests) {
    if (request.emitted) {
      continue;
    }

    const Node* reshape_node = graph_viewer.GetNode(static_cast<NodeIndex>(request.node_index));
    ORT_RETURN_IF_NOT(reshape_node != nullptr, "Missing pending pre-reshape transpose node: ", request.node_index);

    const auto inputs = reshape_node->InputDefs();
    ORT_RETURN_IF_NOT(!inputs.empty() && inputs[0] != nullptr, "Invalid pending pre-reshape transpose input.");

    TensorState transpose_input = LookupState(graph_viewer, states, inputs[0]->Name(), true);
    transpose_input.shape = ToNchwShape(transpose_input.shape);

    TensorState transpose_output = transpose_input;
    transpose_output.name = std::string("Transpose_insert_output") + std::to_string(++transpose_id);

    ordered_json transpose_payload;
    transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
    transpose_payload["input_tensor_info_1"] = TensorInfoJson(transpose_input);
    transpose_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        std::string("Transpose_insert_values") + std::to_string(transpose_id), "int32", {4},
        ordered_json::array({0, 3, 1, 2}));
    transpose_payload["output_tensor_info"] = TensorInfoJson(transpose_output);
    transpose_payload["is_insert"] = true;
    ops_json.push_back(WrapOp("Transpose", std::move(transpose_payload)));

    precomputed_reshape_inputs[request.node_index] = std::move(transpose_output);
    request.emitted = true;
  }

  return Status::OK();
}

Status EmitPendingPreTransposesIfNeeded(
    const GraphViewer& graph_viewer,
    const std::string& output_name,
    int& transpose_id,
    std::unordered_map<std::string, TensorState>& states,
    std::unordered_map<std::string, std::vector<PendingPreTransposeRequest>>& requests,
    std::unordered_map<int, TensorState>& precomputed_transpose_inputs,
    ordered_json& ops_json) {
  auto request_it = requests.find(output_name);
  if (request_it == requests.end()) {
    return Status::OK();
  }

  auto& output_requests = request_it->second;
  for (PendingPreTransposeRequest& request : output_requests) {
    if (request.emitted) {
      continue;
    }

    const Node* transpose_node = graph_viewer.GetNode(static_cast<NodeIndex>(request.node_index));
    ORT_RETURN_IF_NOT(transpose_node != nullptr, "Missing pending pre-transpose node: ", request.node_index);

    const auto inputs = transpose_node->InputDefs();
    ORT_RETURN_IF_NOT(!inputs.empty() && inputs[0] != nullptr, "Invalid pending pre-transpose input.");

    TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), false);
    input_state.shape = ToNchwShape(input_state.shape);
    TensorState pre_transpose_output = input_state;
    pre_transpose_output.name = std::string("Transpose_insert_output") + std::to_string(++transpose_id);

    ordered_json pre_payload;
    pre_payload["op_type_info"] = OpTypeInfo("Transpose");
    pre_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
    pre_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        std::string("Transpose_insert_values") + std::to_string(transpose_id), "int32", {4},
        ordered_json::array({0, 3, 1, 2}));
    pre_payload["output_tensor_info"] = TensorInfoJson(pre_transpose_output);
    pre_payload["is_insert"] = true;
    ops_json.push_back(WrapOp("Transpose", std::move(pre_payload)));

    precomputed_transpose_inputs[request.node_index] = std::move(pre_transpose_output);
    request.emitted = true;
  }

  return Status::OK();
}

Status EmitNodeOutputPostOpsIfNeeded(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     int& transpose_id,
                                     std::unordered_map<std::string, TensorState>& states,
                                     std::unordered_map<std::string, RequantRequest>& requant_requests,
                                     std::unordered_map<std::string, std::vector<PendingPadRequest>>& pad_requests,
                                     std::unordered_map<int, TensorState>& precomputed_pad_outputs,
                                     std::unordered_map<std::string, std::vector<PendingPreReshapeTransposeRequest>>&
                                         pre_reshape_transpose_requests,
                                     std::unordered_map<int, TensorState>& precomputed_reshape_inputs,
                                     std::unordered_map<std::string, std::vector<PendingPreTransposeRequest>>&
                                         pre_transpose_requests,
                                     std::unordered_map<int, TensorState>& precomputed_transpose_inputs,
                                     ordered_json& ops_json) {
  ORT_RETURN_IF_ERROR(EmitNodeOutputRequantsIfNeeded(node, states, requant_requests, ops_json));

  const auto outputs = node.OutputDefs();
  for (size_t i = outputs.size(); i > 0; --i) {
    const NodeArg* output = outputs[i - 1];
    if (output == nullptr) {
      continue;
    }
    ORT_RETURN_IF_ERROR(EmitPendingPadsIfNeeded(graph_viewer, output->Name(), states, pad_requests,
                                                precomputed_pad_outputs, ops_json));
    ORT_RETURN_IF_ERROR(EmitPendingPreReshapeTransposesIfNeeded(graph_viewer, output->Name(), transpose_id, states,
                                                                pre_reshape_transpose_requests,
                                                                precomputed_reshape_inputs, ops_json));
    ORT_RETURN_IF_ERROR(EmitPendingPreTransposesIfNeeded(graph_viewer, output->Name(), transpose_id, states,
                                                         pre_transpose_requests, precomputed_transpose_inputs,
                                                         ops_json));
  }

  return Status::OK();
}
