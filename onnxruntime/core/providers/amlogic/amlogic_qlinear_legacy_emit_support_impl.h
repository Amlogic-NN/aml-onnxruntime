Status EmitQuantizeLinear(const GraphViewer& graph_viewer,
                          const Node& node,
                          std::unordered_map<std::string, TensorState>& states,
                          ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 3 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        outputs[0] != nullptr,
                    "Invalid QuantizeLinear node.");

  TensorState input_state;
  if (states.find(inputs[0]->Name()) != states.end()) {
    input_state = states[inputs[0]->Name()];
  } else {
    input_state = MakeFloatTensorState(graph_viewer, inputs[0]->Name(), true);
  }

  TensorState output_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, outputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(),
                                           true, output_state));
  states[outputs[0]->Name()] = output_state;

  TensorState display_input = input_state;
  if (display_input.tensor_type == "float32") {
    display_input.has_quant = false;
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("QuantizeLinear");
  payload["input_tensor_info"] = TensorInfoJson(display_input);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("QuantizeLinear", std::move(payload)));
  return Status::OK();
}

Status EmitQLinearConv(const GraphViewer& graph_viewer,
                       const Node& node,
                       int& conv_id,
                       std::unordered_map<std::string, TensorState>& states,
                       ordered_json& ops_json,
                       const TensorState* precomputed_pad_output = nullptr) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 8 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        inputs[3] != nullptr && inputs[4] != nullptr && inputs[5] != nullptr &&
                        inputs[6] != nullptr && inputs[7] != nullptr && outputs[0] != nullptr,
                    "Invalid QLinearConv node.");

  ++conv_id;
  TensorState input_state;
  ORT_RETURN_IF_ERROR(MakeQLinearConvInputState(graph_viewer, node, states, input_state));

  const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const bool needs_pad = pads.size() == 4 && std::any_of(pads.begin(), pads.end(), [](int pad) { return pad != 0; });

  if (precomputed_pad_output != nullptr) {
    input_state = *precomputed_pad_output;
    if (needs_pad) {
      input_state.shape = ApplyNhwcPads(input_state.shape, pads);
    }
  } else if (needs_pad) {
    TensorState pad_output;
    ORT_RETURN_IF_ERROR(EmitPadForQLinearConv(conv_id, input_state, pads, ops_json, pad_output));
    input_state = pad_output;
    input_state.shape = ApplyNhwcPads(input_state.shape, pads);
  }

  TensorState weight_state;
  ORT_RETURN_IF_ERROR(MakeWeightTensorState(graph_viewer, inputs[3]->Name(), inputs[4]->Name(), inputs[5]->Name(),
                                            weight_state));

  TensorState bias_state;
  if (inputs.size() > 8 && inputs[8] != nullptr && inputs[8]->Exists()) {
    ORT_RETURN_IF_ERROR(MakeBiasTensorState(graph_viewer, inputs[8]->Name(), input_state.quant_scale,
                                            weight_state.quant_scale, bias_state));
  } else {
    bias_state = MakeSyntheticBiasTensorState(inputs[3]->Name());
  }

  TensorState output_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, outputs[0]->Name(), inputs[6]->Name(), inputs[7]->Name(),
                                           true, output_state));
  states[outputs[0]->Name()] = output_state;

  const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
  const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});

  ordered_json attributes;
  attributes["dilation_h_factor"] = dilations.empty() ? 1 : dilations[0];
  attributes["dilation_w_factor"] = dilations.size() > 1 ? dilations[1] : 1;
  attributes["fused_activation_function"] = "None";
  attributes["padding"] = "VALID";
  attributes["quantized_bias_type"] = "FLOAT32";
  attributes["stride_h"] = strides.empty() ? 1 : strides[0];
  attributes["stride_w"] = strides.size() > 1 ? strides[1] : 1;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("QLinearConv");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["input_weight_info"] = TensorInfoJson(weight_state);
  payload["input_bias_info"] = TensorInfoJson(bias_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("QLinearConv", std::move(payload)));
  return Status::OK();
}

Status EmitQLinearUnary(const GraphViewer& graph_viewer,
                        const Node& node,
                        const std::string& op_type,
                        std::unordered_map<std::string, TensorState>& states,
                        ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 5 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        inputs[3] != nullptr && inputs[4] != nullptr && outputs[0] != nullptr,
                    "Invalid QLinear unary node.");

  TensorState input_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(),
                                           true, input_state));
  if (states.find(inputs[0]->Name()) != states.end()) {
    input_state = states[inputs[0]->Name()];
  }

  TensorState output_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, outputs[0]->Name(), inputs[3]->Name(), inputs[4]->Name(),
                                           true, output_state));
  states[outputs[0]->Name()] = output_state;

  TensorState display_output = output_state;
  if (op_type == "QLinearSigmoid") {
    display_output.quant_scale = ordered_json::array({0.00390625f});
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(display_output);
  if (op_type == "QLinearSoftmax") {
    ordered_json attributes;
    attributes["beta"] = 1;
    payload["attributes_info"] = std::move(attributes);
  }
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

Status EmitQLinearBinary(const GraphViewer& graph_viewer,
                         const Node& node,
                         const std::string& op_type,
                         std::unordered_map<std::string, TensorState>& states,
                         ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 8 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        inputs[3] != nullptr && inputs[4] != nullptr && inputs[5] != nullptr &&
                        inputs[6] != nullptr && inputs[7] != nullptr && outputs[0] != nullptr,
                    "Invalid QLinear binary node.");

  TensorState input_1;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(),
                                           true, input_1));
  if (states.find(inputs[0]->Name()) != states.end()) {
    input_1 = states[inputs[0]->Name()];
  }

  TensorState input_2;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[3]->Name(), inputs[4]->Name(), inputs[5]->Name(),
                                           true, input_2));
  if (states.find(inputs[3]->Name()) != states.end()) {
    input_2 = states[inputs[3]->Name()];
  }

  TensorState output_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, outputs[0]->Name(), inputs[6]->Name(), inputs[7]->Name(),
                                           true, output_state));
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["fused_activation_function"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info_1"] = input_1.has_values && input_1.has_quant
                                       ? TensorInfoJsonQuantBeforeValues(input_1)
                                       : TensorInfoJson(input_1);
  payload["input_tensor_info_2"] = input_2.has_values && input_2.has_quant
                                       ? TensorInfoJsonQuantBeforeValues(input_2)
                                       : TensorInfoJson(input_2);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

Status EmitSplit(const GraphViewer& graph_viewer,
                 const Node& node,
                 std::unordered_map<std::string, TensorState>& states,
                 ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 2 && inputs[0] != nullptr,
                    "Invalid Split node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true);
  const int onnx_axis = GetIntAttribute(node, "axis", 0);
  const int axis = MapAxisToNhwc(onnx_axis, input_state.shape.size());

  ordered_json split_values_info;
  if (inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    ORT_RETURN_IF_ERROR(MakeInitializerTensorInfo(graph_viewer, inputs[1]->Name(),
                                                  input_state.name + "splitv_valuse",
                                                  "int32", false, split_values_info));
  } else {
    std::vector<int> split_values = GetIntsAttribute(node, "split", {});
    if (split_values.empty()) {
      const int axis_dim = axis >= 0 && static_cast<size_t>(axis) < input_state.shape.size()
                               ? input_state.shape[static_cast<size_t>(axis)]
                               : 0;
      ORT_RETURN_IF_NOT(axis_dim > 0 && axis_dim % static_cast<int>(outputs.size()) == 0,
                        "Split node is missing split sizes and cannot be evenly inferred.");
      split_values.assign(outputs.size(), axis_dim / static_cast<int>(outputs.size()));
    }

    ordered_json split_values_json = ordered_json::array();
    for (int value : split_values) {
      split_values_json.push_back(value);
    }
    split_values_info = ConstantTensorInfoJson(input_state.name + "splitv_valuse",
                                               "int32",
                                               {static_cast<int>(split_values.size())},
                                               split_values_json);
  }
  ordered_json axis_values = ordered_json::array({axis});

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Split");
  TensorState display_input_state = input_state;
  const Node* producer_node = graph_viewer.GetProducerNode(inputs[0]->Name());
  if (producer_node != nullptr && input_state.shape.size() == 4 &&
      (producer_node->OpType() == "Transpose" || producer_node->OpType() == "Reshape")) {
    display_input_state.quant_scale = "None";
    display_input_state.zero_point = "None";
  }
  payload["input_tensor_info_1"] = TensorInfoJson(display_input_state);
  payload["input_tensor_info_2"] = std::move(split_values_info);
  payload["input_tensor_info_3"] = ConstantTensorInfoJson(input_state.name + "splitv_axis", "int32", {1}, axis_values);

  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] == nullptr) {
      continue;
    }
    TensorState output_state = input_state;
    output_state.name = ReferenceTensorName(outputs[i]->Name());
    output_state.shape = ShapeForTensor(graph_viewer, outputs[i]->Name(), true);
    output_state.is_constant = false;
    output_state.has_values = false;
    states[outputs[i]->Name()] = output_state;
    payload[std::string("output_tensor_info_") + std::to_string(i + 1)] = TensorInfoJson(output_state);
  }

  ordered_json attributes;
  attributes["num_splits"] = static_cast<int>(outputs.size());
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Split", std::move(payload)));
  return Status::OK();
}

Status EmitQLinearConcat(const GraphViewer& graph_viewer,
                         const Node& node,
                         std::unordered_map<std::string, TensorState>& states,
                         std::unordered_map<std::string, RequantRequest>& requant_requests,
                         ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 5 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid QLinearConcat node.");

  QuantInfo output_quant;
  ORT_RETURN_IF_ERROR(ReadQuantInfo(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), output_quant));

  TensorState output_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.tensor_type = "int8";
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  output_state.is_constant = false;
  output_state.has_quant = true;
  output_state.quant_scale = output_quant.scale;
  output_state.zero_point = output_quant.zero_point;
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("QLinearConcat");
  payload["output_tensor_info"] = TensorInfoJson(output_state);

  const int onnx_axis = GetIntAttribute(node, "axis", 0);
  ordered_json attributes;
  attributes["axis"] = MapAxisToNhwc(onnx_axis, output_state.shape.size());
  attributes["fused_activation_function"] = "None";
  payload["attributes_info"] = std::move(attributes);

  int input_index = 1;
  for (size_t i = 2; i + 2 < inputs.size(); i += 3) {
    if (inputs[i] == nullptr || inputs[i + 1] == nullptr || inputs[i + 2] == nullptr) {
      continue;
    }

    TensorState input_state;
    ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[i]->Name(),
                                             inputs[i + 1]->Name(), inputs[i + 2]->Name(),
                                             true, input_state));
    const std::string key = RequestKey(inputs[i]->Name(), output_quant);
    const auto req_it = requant_requests.find(key);
    if (req_it != requant_requests.end() && !req_it->second.output_state.name.empty()) {
      input_state = req_it->second.output_state;
    }
    states[inputs[i]->Name()] = input_state;
    payload[std::string("input_tensor_info_") + std::to_string(input_index++)] = TensorInfoJson(input_state);
  }

  ops_json.push_back(WrapOp("QLinearConcat", std::move(payload)));
  return Status::OK();
}

Status EmitResizeNearest(const GraphViewer& graph_viewer,
                         const Node& node,
                         std::unordered_map<std::string, TensorState>& states,
                         ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Resize node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true);
  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.tensor_type == "float32") {
    output_state.has_quant = true;
    output_state.quant_scale = "None";
    output_state.zero_point = "None";
  }
  states[outputs[0]->Name()] = output_state;

  std::vector<int> resize_values;
  if (output_state.shape.size() == 4) {
    resize_values = {output_state.shape[1], output_state.shape[2]};
  }
  ordered_json resize_json = ordered_json::array();
  for (int value : resize_values) {
    resize_json.push_back(value);
  }

  ordered_json attributes;
  attributes["align_corners"] = false;
  attributes["half_pixel_centers"] = true;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Resize_nearest");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "resize_shape_values",
                                                          "int32", {2}, resize_json);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Reshape_nearest", std::move(payload)));
  return Status::OK();
}

Status EmitMaxPool(const GraphViewer& graph_viewer,
                   const Node& node,
                   int& maxpool_pad_id,
                   std::unordered_map<std::string, TensorState>& states,
                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid MaxPool node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true);
  const auto pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const auto strides = GetIntsAttribute(node, "strides", {1, 1});
  const bool has_pads = pads.size() == 4 &&
                        std::any_of(pads.begin(), pads.end(), [](int pad) { return pad != 0; });
  const bool explicit_pad = has_pads &&
                            std::any_of(strides.begin(), strides.end(), [](int stride) { return stride > 1; });
  if (explicit_pad) {
    TensorState pad_output;
    ORT_RETURN_IF_ERROR(EmitPadForMaxPool(++maxpool_pad_id, input_state, pads, ops_json, pad_output));
    input_state = pad_output;
    input_state.shape = ApplyNhwcPads(input_state.shape, pads);
  }

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.tensor_type == "float32") {
    output_state.has_quant = true;
    output_state.quant_scale = "None";
    output_state.zero_point = "None";
  }
  states[outputs[0]->Name()] = output_state;

  TensorState display_input_state = FloatDisplayTensorState(input_state);

  const auto kernel_shape = GetIntsAttribute(node, "kernel_shape", {1, 1});

  ordered_json attributes;
  attributes["filter_height"] = kernel_shape.empty() ? 1 : kernel_shape[0];
  attributes["filter_width"] = kernel_shape.size() > 1 ? kernel_shape[1] : 1;
  attributes["fused_activation_function"] = "None";
  attributes["padding"] = (has_pads && !explicit_pad) ? "add_pad_op2" : "VALID";
  attributes["stride_h"] = strides.empty() ? 1 : strides[0];
  attributes["stride_w"] = strides.size() > 1 ? strides[1] : 1;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("MaxPool");
  payload["input_tensor_info"] = TensorInfoJson(display_input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("MaxPool", std::move(payload)));
  return Status::OK();
}

Status EmitQLinearGlobalAveragePool(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    std::unordered_map<std::string, TensorState>& states,
                                    ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 5 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        inputs[3] != nullptr && inputs[4] != nullptr && outputs[0] != nullptr,
                    "Invalid QLinearGlobalAveragePool node.");

  TensorState input_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(),
                                           true, input_state));
  if (states.find(inputs[0]->Name()) != states.end()) {
    input_state = states[inputs[0]->Name()];
  }

  TensorState output_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, outputs[0]->Name(), inputs[3]->Name(), inputs[4]->Name(),
                                           true, output_state));
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["keep_dims"] = true;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("QLinearGlobalAveragePool");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("QLinearGlobalAveragePool", std::move(payload)));
  return Status::OK();
}

Status EmitReduceMean(const GraphViewer& graph_viewer,
                      const Node& node,
                      std::unordered_map<std::string, TensorState>& states,
                      ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid ReduceMean node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), false);
  input_state.has_quant = true;
  input_state.quant_scale = "None";
  input_state.zero_point = "None";
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), false);
  output_state.has_quant = true;
  output_state.quant_scale = "None";
  output_state.zero_point = "None";
  states[outputs[0]->Name()] = output_state;

  std::vector<int> axes = GetIntsAttribute(node, "axes", {});
  ordered_json axes_json = ordered_json::array();
  for (int axis : axes) {
    axes_json.push_back(MapAxisToNhwc(axis, input_state.shape.size()));
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Mean");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "_axes", "int32",
                                                          {static_cast<int>(axes.size())}, axes_json);
  payload["output_tensor_info"] = TensorInfoJson(output_state);

  ordered_json attributes;
  attributes["keep_dims"] = GetIntAttribute(node, "keepdims", 1) != 0;
  payload["attributes_info"] = std::move(attributes);

  ops_json.push_back(WrapOp("Mean", std::move(payload)));
  return Status::OK();
}

Status EmitFlattenAsReshape(const GraphViewer& graph_viewer,
                            const Node& node,
                            int& transpose_id,
                            std::unordered_map<std::string, TensorState>& states,
                            ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Flatten node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), false);
  input_state.has_quant = true;
  input_state.quant_scale = "None";
  input_state.zero_point = "None";
  states[inputs[0]->Name()] = input_state;

  TensorState reshape_input = input_state;
  if (input_state.shape.size() == 4) {
    TensorState transpose_input = input_state;
    transpose_input.shape = ToNchwShape(input_state.shape);

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

    reshape_input = std::move(transpose_output);
  }

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), false);
  output_state.has_quant = true;
  output_state.quant_scale = "None";
  output_state.zero_point = "None";
  states[outputs[0]->Name()] = output_state;

  ordered_json new_shape = ordered_json::array();
  for (int dim : output_state.shape) {
    new_shape.push_back(dim);
  }

  ordered_json shape_tensor_info = ConstantTensorInfoJson(input_state.name + "_1", "int32",
                                                          {static_cast<int>(output_state.shape.size())}, new_shape);
  shape_tensor_info["quant_scale"] = "None";
  shape_tensor_info["zero_point"] = "None";

  ordered_json attributes;
  attributes["new_shape"] = new_shape;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(reshape_input);
  payload["input_tensor_info_2"] = std::move(shape_tensor_info);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("reshape", std::move(payload)));
  return Status::OK();
}

Status MakeQGemmWeightTensorState(const GraphViewer& graph_viewer,
                                  const std::string& weight_name,
                                  const std::string& scale_name,
                                  const std::string& zero_point_name,
                                  TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(weight_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing QGemm weight initializer: ", weight_name);

  const std::vector<int> shape = TensorProtoShape(*tensor);
  ordered_json values;
  ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), shape, shape, nullptr, false, values));

  state.name = weight_name;
  state.tensor_type = "int8";
  state.shape = shape;
  state.is_constant = true;
  state.has_values = true;
  state.values = std::move(values);
  state.has_quant = true;
  ORT_RETURN_IF_ERROR(InitializerAsFloatJson(graph_viewer, scale_name, state.quant_scale));
  ORT_RETURN_IF_ERROR(InitializerAsZeroPointJson(graph_viewer, zero_point_name, state.zero_point));
  KeepAsArray(state.quant_scale);
  KeepAsArray(state.zero_point);
  return Status::OK();
}

Status EmitQGemm(const GraphViewer& graph_viewer,
                 const Node& node,
                 std::unordered_map<std::string, TensorState>& states,
                 ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 9 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        inputs[3] != nullptr && inputs[4] != nullptr && inputs[5] != nullptr &&
                        inputs[6] != nullptr && inputs[7] != nullptr && inputs[8] != nullptr &&
                        outputs[0] != nullptr,
                    "Invalid QGemm node.");

  TensorState input_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(),
                                           false, input_state));
  if (states.find(inputs[0]->Name()) != states.end()) {
    input_state = states[inputs[0]->Name()];
  }

  TensorState weight_state;
  ORT_RETURN_IF_ERROR(MakeQGemmWeightTensorState(graph_viewer, inputs[3]->Name(), inputs[4]->Name(),
                                                 inputs[5]->Name(), weight_state));

  TensorState bias_state;
  ORT_RETURN_IF_ERROR(MakeBiasTensorState(graph_viewer, inputs[6]->Name(), input_state.quant_scale,
                                          weight_state.quant_scale, bias_state));

  TensorState output_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, outputs[0]->Name(), inputs[7]->Name(), inputs[8]->Name(),
                                           false, output_state));
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["keep_num_dims"] = false;
  attributes["fused_activation_function"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("QGemm");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["input_weight_info"] = TensorInfoJson(weight_state);
  payload["input_bias_info"] = TensorInfoJson(bias_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("QGemm", std::move(payload)));

  return Status::OK();
}

Status EmitReshape(const GraphViewer& graph_viewer,
                   const Node& node,
                   int& transpose_id,
                   int& negative_transpose_id,
                   std::unordered_map<std::string, TensorState>& states,
                   ordered_json& ops_json,
                   const TensorState* precomputed_input = nullptr) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid Reshape node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true);
  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  states[outputs[0]->Name()] = output_state;

  bool add_post_reshape_transpose = false;
  TensorState reshape_output_state = output_state;
  if (precomputed_input != nullptr) {
    input_state = *precomputed_input;
  } else if (input_state.shape.size() == 4 && output_state.shape.size() != 4) {
    TensorState transpose_input = input_state;
    transpose_input.shape = ToNchwShape(input_state.shape);
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

    input_state = transpose_output;
  }

  if (input_state.shape.size() != 4 && output_state.shape.size() == 4) {
    add_post_reshape_transpose = true;
    reshape_output_state = output_state;
    reshape_output_state.name = std::string("Transpose_insert_output-") + std::to_string(++negative_transpose_id);
  }

  ordered_json shape_tensor_info;
  ORT_RETURN_IF_ERROR(MakeInitializerTensorInfo(graph_viewer, inputs[1]->Name(),
                                                ReferenceTensorName(inputs[0]->Name()) + "_1",
                                                "int32", false, shape_tensor_info));
  if (shape_tensor_info.contains("values") && shape_tensor_info["values"].is_array() &&
      shape_tensor_info["values"].size() == output_state.shape.size()) {
    for (size_t i = 0; i < output_state.shape.size(); ++i) {
      if (shape_tensor_info["values"][i].is_number_integer() &&
          shape_tensor_info["values"][i].get<int>() == -1 &&
          output_state.shape[i] > 0) {
        shape_tensor_info["values"][i] = output_state.shape[i];
      }
    }
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = std::move(shape_tensor_info);
  payload["output_tensor_info"] = TensorInfoJson(reshape_output_state);
  ops_json.push_back(WrapOp("reshape", std::move(payload)));

  if (add_post_reshape_transpose) {
    TensorState post_transpose_output = output_state;
    post_transpose_output.shape = ApplyPermToShape(reshape_output_state.shape, {0, 2, 3, 1});
    states[outputs[0]->Name()] = post_transpose_output;

    ordered_json transpose_payload;
    transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
    transpose_payload["input_tensor_info_1"] = TensorInfoJson(reshape_output_state);
    transpose_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        std::string("Transpose_insert_values-") + std::to_string(negative_transpose_id), "int32", {4},
        ordered_json::array({0, 2, 3, 1}));
    transpose_payload["output_tensor_info"] = TensorInfoJson(post_transpose_output);
    transpose_payload["is_insert"] = true;
    ops_json.push_back(WrapOp("Transpose", std::move(transpose_payload)));
  }
  return Status::OK();
}

Status EmitTranspose(const GraphViewer& graph_viewer,
                     const Node& node,
                     int& transpose_id,
                     int& negative_transpose_id,
                     std::unordered_map<std::string, TensorState>& states,
                     ordered_json& ops_json,
                     const TensorState* precomputed_input = nullptr) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Transpose node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), false);
  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  states[outputs[0]->Name()] = output_state;

  std::vector<int> perm = GetIntsAttribute(node, "perm", {});
  ordered_json perm_values = ordered_json::array();
  for (int value : perm) {
    perm_values.push_back(value);
  }

  if (input_state.shape.size() == 4 && output_state.shape.size() == 4) {
    TensorState pre_transpose_output;
    if (precomputed_input != nullptr) {
      pre_transpose_output = *precomputed_input;
    } else {
      TensorState pre_transpose_input = input_state;
      pre_transpose_input.shape = ToNchwShape(pre_transpose_input.shape);
      pre_transpose_output = pre_transpose_input;
      pre_transpose_output.name = std::string("Transpose_insert_output") + std::to_string(++transpose_id);

      ordered_json pre_payload;
      pre_payload["op_type_info"] = OpTypeInfo("Transpose");
      pre_payload["input_tensor_info_1"] = TensorInfoJson(pre_transpose_input);
      pre_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
          std::string("Transpose_insert_values") + std::to_string(transpose_id), "int32", {4},
          ordered_json::array({0, 3, 1, 2}));
      pre_payload["output_tensor_info"] = TensorInfoJson(pre_transpose_output);
      pre_payload["is_insert"] = true;
      ops_json.push_back(WrapOp("Transpose", std::move(pre_payload)));
    }

    TensorState original_transpose_output = output_state;
    original_transpose_output.name =
        std::string("Transpose_insert_output-") + std::to_string(++negative_transpose_id);

    ordered_json original_payload;
    original_payload["op_type_info"] = OpTypeInfo("Transpose");
    original_payload["input_tensor_info_1"] = TensorInfoJson(pre_transpose_output);
    original_payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "_perm_values", "int32",
                                                                     {static_cast<int>(perm.size())}, perm_values);
    original_payload["output_tensor_info"] = TensorInfoJson(original_transpose_output);
    original_payload["add_Transpose"] = true;
    ops_json.push_back(WrapOp("Transpose", std::move(original_payload)));

    TensorState post_transpose_output = output_state;
    post_transpose_output.shape = ApplyPermToShape(original_transpose_output.shape, {0, 2, 3, 1});
    states[outputs[0]->Name()] = post_transpose_output;

    ordered_json post_payload;
    post_payload["op_type_info"] = OpTypeInfo("Transpose");
    post_payload["input_tensor_info_1"] = TensorInfoJson(original_transpose_output);
    post_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        std::string("Transpose_insert_values-") + std::to_string(negative_transpose_id), "int32", {4},
        ordered_json::array({0, 2, 3, 1}));
    post_payload["output_tensor_info"] = TensorInfoJson(post_transpose_output);
    post_payload["is_insert"] = true;
    ops_json.push_back(WrapOp("Transpose", std::move(post_payload)));
    return Status::OK();
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Transpose");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "_perm_values", "int32",
                                                          {static_cast<int>(perm.size())}, perm_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("Transpose", std::move(payload)));
  return Status::OK();
}

Status EmitDequantizeLinear(const GraphViewer& graph_viewer,
                            const Node& node,
                            std::unordered_map<std::string, TensorState>& states,
                            ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 3 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        outputs[0] != nullptr,
                    "Invalid DequantizeLinear node.");

  TensorState input_state;
  ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(),
                                           true, input_state));
  if (states.find(inputs[0]->Name()) != states.end()) {
    input_state = states[inputs[0]->Name()];
  }

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (input_state.shape.size() == 4 && output_state.shape.size() == 4 &&
      input_state.shape[1] == 1 && input_state.shape[2] == 1 &&
      output_state.shape[2] == 1 && output_state.shape[3] == 1 &&
      output_state.shape[1] == input_state.shape[3]) {
    output_state.shape = input_state.shape;
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("DequantizeLinear");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("DequantizeLinear", std::move(payload)));
  return Status::OK();
}

Status EmitPlainBinary(const GraphViewer& graph_viewer,
                       const Node& node,
                       const std::string& op_type,
                       std::unordered_map<std::string, TensorState>& states,
                       ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid plain binary node.");

  auto make_input = [&](const NodeArg* input) -> Status {
    if (states.find(input->Name()) == states.end()) {
      TensorState state = MakeFloatTensorState(graph_viewer, input->Name(), false);
      const auto* initializer = graph_viewer.GetConstantInitializer(input->Name(), true);
      if (initializer != nullptr) {
        ordered_json values;
        const auto shape = TensorProtoShape(*initializer);
        ORT_RETURN_IF_ERROR(TensorValuesJson(*initializer, graph_viewer.ModelPath(), shape, shape, nullptr,
                                             false, values));
        state.is_constant = true;
        state.has_values = true;
        state.values = std::move(values);
        state.has_quant = true;
        state.quant_scale = "None";
        state.zero_point = "None";
      } else {
        state.has_quant = true;
        state.quant_scale = "None";
        state.zero_point = "None";
      }
      states[input->Name()] = std::move(state);
    } else if (!states[input->Name()].has_quant) {
      states[input->Name()].has_quant = true;
      states[input->Name()].quant_scale = "None";
      states[input->Name()].zero_point = "None";
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(make_input(inputs[0]));
  ORT_RETURN_IF_ERROR(make_input(inputs[1]));

  if (op_type == "Div") {
    TensorState& divisor_state = states[inputs[1]->Name()];
    if (divisor_state.is_constant && divisor_state.has_values && divisor_state.shape.empty() &&
        divisor_state.values.is_array() && divisor_state.values.size() == 1 &&
        divisor_state.values[0].is_number()) {
      const float divisor = divisor_state.values[0].get<float>();
      divisor_state.shape = {1};
      if (divisor != 0.0f) {
        divisor_state.values[0] = 1.0f / divisor;
      }
    }
  }

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), false);
  output_state.has_quant = true;
  output_state.quant_scale = "None";
  output_state.zero_point = "None";
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info_1"] = TensorInfoJson(states[inputs[0]->Name()]);
  payload["input_tensor_info_2"] = TensorInfoJson(states[inputs[1]->Name()]);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

Status EmitSlice(const GraphViewer& graph_viewer,
                 const Node& node,
                 std::unordered_map<std::string, TensorState>& states,
                 ordered_json& ops_json) {
  std::cout<<"amlogic_qlinear_legacy_emit_support_impl.h"<<std::endl;
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 3 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        outputs[0] != nullptr,
                    "Invalid Slice node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), false, "float32");
  input_state.has_quant = true;
  input_state.quant_scale = "None";
  input_state.zero_point = "None";
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), false);
  output_state.has_quant = true;
  output_state.quant_scale = "None";
  output_state.zero_point = "None";
  states[outputs[0]->Name()] = output_state;

  std::vector<int> starts;
  std::vector<int> ends;
  std::vector<int> axes;
  ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), starts));
  ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[2]->Name(), ends));
  if (inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[3]->Name(), axes));
  } else {
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  const size_t rank = input_state.shape.size();
  std::vector<int> begin(rank, 0);
  std::vector<int> size = input_state.shape;
  const size_t slice_dims = std::min({starts.size(), ends.size(), axes.size()});
  for (size_t i = 0; i < slice_dims; ++i) {
    int axis = axes[i];
    if (axis < 0) {
      axis += static_cast<int>(rank);
    }
    axis = MapAxisToNhwc(axis, rank);
    if (axis < 0 || axis >= static_cast<int>(rank)) {
      continue;
    }

    const int dim = input_state.shape[static_cast<size_t>(axis)];
    const int start = NormalizeSliceIndex(starts[i], dim);
    const int end = NormalizeSliceIndex(ends[i], dim);
    begin[static_cast<size_t>(axis)] = start;
    size[static_cast<size_t>(axis)] = dim > 0 ? std::max(0, end - start) : end - start;
  }

  ordered_json begin_values = ordered_json::array();
  ordered_json size_values = ordered_json::array();
  for (int value : begin) {
    begin_values.push_back(value);
  }
  for (int value : size) {
    size_values.push_back(value);
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Slice");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "begin_valuse", "int32",
                                                          {static_cast<int>(rank)}, begin_values);
  payload["input_tensor_info_3"] = ConstantTensorInfoJson(input_state.name + "size_values", "int32",
                                                          {static_cast<int>(rank)}, size_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("Slice", std::move(payload)));
  return Status::OK();
}
