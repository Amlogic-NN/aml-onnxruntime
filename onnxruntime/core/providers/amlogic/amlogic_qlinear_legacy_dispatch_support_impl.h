Status EmitGroupedPaddedQLinearConvCluster(
    const GraphViewer& graph_viewer,
    int node_index,
    size_t grouped_conv_count,
    int& conv_id,
    int& transpose_id,
    std::unordered_map<std::string, TensorState>& states,
    std::unordered_map<std::string, RequantRequest>& requant_requests,
    std::unordered_map<std::string, std::vector<PendingPadRequest>>& pad_requests,
    std::unordered_map<int, TensorState>& precomputed_pad_outputs,
    std::unordered_map<std::string, std::vector<PendingPreReshapeTransposeRequest>>& pre_reshape_transpose_requests,
    std::unordered_map<int, TensorState>& precomputed_reshape_inputs,
    std::unordered_map<std::string, std::vector<PendingPreTransposeRequest>>& pre_transpose_requests,
    std::unordered_map<int, TensorState>& precomputed_transpose_inputs,
    ordered_json& ops_json) {
  std::vector<const Node*> group_nodes;
  group_nodes.reserve(grouped_conv_count);
  for (size_t i = 0; i < grouped_conv_count; ++i) {
    group_nodes.push_back(graph_viewer.GetNode(static_cast<NodeIndex>(node_index + static_cast<int>(i))));
  }

  const int first_group_conv_id = conv_id + 1;
  std::vector<TensorState> pad_outputs(grouped_conv_count);
  // Match the reference converter: sibling SAME pads are emitted in reverse,
  // while the corresponding convolutions keep ONNX node order.
  for (size_t reverse_i = grouped_conv_count; reverse_i > 0; --reverse_i) {
    const size_t i = reverse_i - 1;
    TensorState input_state;
    ORT_RETURN_IF_ERROR(MakeQLinearConvInputState(graph_viewer, *group_nodes[i], states, input_state));
    const std::vector<int> pads = GetIntsAttribute(*group_nodes[i], "pads", {0, 0, 0, 0});
    ORT_RETURN_IF_ERROR(
        EmitPadForQLinearConv(first_group_conv_id + static_cast<int>(i), input_state, pads, ops_json, pad_outputs[i]));
  }

  for (size_t i = 0; i < grouped_conv_count; ++i) {
    ORT_RETURN_IF_ERROR(
        EmitQLinearConv(graph_viewer, *group_nodes[i], conv_id, states, ops_json, &pad_outputs[i]));
    ORT_RETURN_IF_ERROR(
        EmitNodeOutputPostOpsIfNeeded(graph_viewer, *group_nodes[i], transpose_id, states, requant_requests,
                                      pad_requests, precomputed_pad_outputs, pre_reshape_transpose_requests,
                                      precomputed_reshape_inputs, pre_transpose_requests,
                                      precomputed_transpose_inputs, ops_json));
  }

  return Status::OK();
}

Status EmitLegacyNode(
    const GraphViewer& graph_viewer,
    const Node& node,
    int node_index,
    int max_node_index,
    int& conv_id,
    int& transpose_id,
    int& negative_transpose_id,
    int& maxpool_pad_id,
    std::unordered_map<std::string, TensorState>& states,
    std::unordered_map<std::string, RequantRequest>& requant_requests,
    std::unordered_map<std::string, std::vector<PendingPadRequest>>& pad_requests,
    std::unordered_map<int, TensorState>& precomputed_pad_outputs,
    std::unordered_map<std::string, std::vector<PendingPreReshapeTransposeRequest>>& pre_reshape_transpose_requests,
    std::unordered_map<int, TensorState>& precomputed_reshape_inputs,
    std::unordered_map<std::string, std::vector<PendingPreTransposeRequest>>& pre_transpose_requests,
    std::unordered_map<int, TensorState>& precomputed_transpose_inputs,
    ordered_json& ops_json,
    bool& emitted,
    int& consumed_extra_nodes) {
  emitted = true;
  consumed_extra_nodes = 0;

  const std::string& op_type = node.OpType();
  if (op_type == "QuantizeLinear") {
    ORT_RETURN_IF_ERROR(EmitQuantizeLinear(graph_viewer, node, states, ops_json));
  } else if (op_type == "QLinearConv") {
    const auto precomputed_pad_output = precomputed_pad_outputs.find(node_index);
    if (precomputed_pad_output != precomputed_pad_outputs.end()) {
      ORT_RETURN_IF_ERROR(
          EmitQLinearConv(graph_viewer, node, conv_id, states, ops_json, &precomputed_pad_output->second));
    } else {
      size_t grouped_conv_count = 1;
      const auto current_inputs = node.InputDefs();
      if (IsPaddedQLinearConvNode(&node) && !current_inputs.empty() && current_inputs[0] != nullptr) {
        for (int next_node_index = node_index + 1; next_node_index < max_node_index; ++next_node_index) {
          const Node* next_node = graph_viewer.GetNode(static_cast<NodeIndex>(next_node_index));
          if (next_node == nullptr || IsOrtInsertedQdqHelperNode(*next_node) || !IsPaddedQLinearConvNode(next_node)) {
            break;
          }

          const auto next_inputs = next_node->InputDefs();
          if (next_inputs.empty() || next_inputs[0] == nullptr ||
              next_inputs[0]->Name() != current_inputs[0]->Name()) {
            break;
          }
          ++grouped_conv_count;
        }
      }

      if (grouped_conv_count > 1) {
        ORT_RETURN_IF_ERROR(
            EmitGroupedPaddedQLinearConvCluster(graph_viewer, node_index, grouped_conv_count, conv_id, transpose_id,
                                                states, requant_requests, pad_requests, precomputed_pad_outputs,
                                                pre_reshape_transpose_requests, precomputed_reshape_inputs,
                                                pre_transpose_requests, precomputed_transpose_inputs, ops_json));
        consumed_extra_nodes = static_cast<int>(grouped_conv_count) - 1;
        return Status::OK();
      }

      ORT_RETURN_IF_ERROR(EmitQLinearConv(graph_viewer, node, conv_id, states, ops_json));
    }
  } else if (op_type == "QLinearSigmoid") {
    ORT_RETURN_IF_ERROR(EmitQLinearUnary(graph_viewer, node, "QLinearSigmoid", states, ops_json));
  } else if (op_type == "QLinearSoftmax") {
    ORT_RETURN_IF_ERROR(EmitQLinearUnary(graph_viewer, node, "QLinearSoftmax", states, ops_json));
  } else if (op_type == "QLinearMul") {
    ORT_RETURN_IF_ERROR(EmitQLinearBinary(graph_viewer, node, "QLinearMul", states, ops_json));
  } else if (op_type == "QLinearAdd") {
    ORT_RETURN_IF_ERROR(EmitQLinearBinary(graph_viewer, node, "QLinearAdd", states, ops_json));
  } else if (op_type == "QLinearGlobalAveragePool") {
    ORT_RETURN_IF_ERROR(EmitQLinearGlobalAveragePool(graph_viewer, node, states, ops_json));
  } else if (op_type == "ReduceMean") {
    ORT_RETURN_IF_ERROR(EmitReduceMean(graph_viewer, node, states, ops_json));
  } else if (op_type == "Split") {
    ORT_RETURN_IF_ERROR(EmitSplit(graph_viewer, node, states, ops_json));
  } else if (op_type == "QLinearConcat") {
    ORT_RETURN_IF_ERROR(EmitQLinearConcat(graph_viewer, node, states, requant_requests, ops_json));
  } else if (op_type == "Resize") {
    ORT_RETURN_IF_ERROR(EmitResizeNearest(graph_viewer, node, states, ops_json));
  } else if (op_type == "MaxPool") {
    ORT_RETURN_IF_ERROR(EmitMaxPool(graph_viewer, node, maxpool_pad_id, states, ops_json));
  } else if (op_type == "Flatten") {
    ORT_RETURN_IF_ERROR(EmitFlattenAsReshape(graph_viewer, node, transpose_id, states, ops_json));
  } else if (op_type == "QGemm") {
    ORT_RETURN_IF_ERROR(EmitQGemm(graph_viewer, node, states, ops_json));
  } else if (op_type == "Reshape") {
    const auto precomputed_reshape_input = precomputed_reshape_inputs.find(node_index);
    ORT_RETURN_IF_ERROR(
        EmitReshape(graph_viewer, node, transpose_id, negative_transpose_id, states, ops_json,
                    precomputed_reshape_input == precomputed_reshape_inputs.end()
                        ? nullptr
                        : &precomputed_reshape_input->second));
  } else if (op_type == "Transpose") {
    const auto precomputed_transpose_input = precomputed_transpose_inputs.find(node_index);
    ORT_RETURN_IF_ERROR(
        EmitTranspose(graph_viewer, node, transpose_id, negative_transpose_id, states, ops_json,
                      precomputed_transpose_input == precomputed_transpose_inputs.end()
                          ? nullptr
                          : &precomputed_transpose_input->second));
  } else if (op_type == "DequantizeLinear") {
    ORT_RETURN_IF_ERROR(EmitDequantizeLinear(graph_viewer, node, states, ops_json));
  } else if (op_type == "Slice") {
    ORT_RETURN_IF_ERROR(EmitSlice(graph_viewer, node, states, ops_json));
  } else if (op_type == "Sub") {
    ORT_RETURN_IF_ERROR(EmitPlainBinary(graph_viewer, node, "Sub", states, ops_json));
  } else if (op_type == "Div") {
    ORT_RETURN_IF_ERROR(EmitPlainBinary(graph_viewer, node, "Div", states, ops_json));
  } else {
    emitted = false;
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(EmitNodeOutputPostOpsIfNeeded(graph_viewer, node, transpose_id, states, requant_requests,
                                                    pad_requests, precomputed_pad_outputs,
                                                    pre_reshape_transpose_requests, precomputed_reshape_inputs,
                                                    pre_transpose_requests, precomputed_transpose_inputs, ops_json));
  return Status::OK();
}
