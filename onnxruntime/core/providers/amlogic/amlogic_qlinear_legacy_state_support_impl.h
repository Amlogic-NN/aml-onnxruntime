// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside the anonymous namespace. It keeps the legacy qlinear exporter state
// lookup and inserted-requant helpers outside the main translation unit body
// without changing behavior.

Status EmitInsertedRequantIfNeeded(const std::string& output_name,
                                   const TensorState& input_state,
                                   std::unordered_map<std::string, TensorState>& states,
                                   std::unordered_map<std::string, RequantRequest>& requests,
                                   ordered_json& ops_json) {
  std::vector<std::string> keys;
  keys.reserve(requests.size());
  for (const auto& entry : requests) {
    if (entry.first.rfind(output_name + "|", 0) == 0) {
      keys.push_back(entry.first);
    }
  }
  std::sort(keys.begin(), keys.end(), std::greater<std::string>());

  for (const auto& key : keys) {
    auto it = requests.find(key);
    if (it == requests.end() || !it->second.output_state.name.empty()) {
      continue;
    }

    TensorState output_state = input_state;
    output_state.name = it->second.output_name;
    output_state.tensor_type = "int8";
    output_state.is_constant = false;
    output_state.has_values = false;
    output_state.has_quant = true;
    output_state.quant_scale = it->second.target_quant.scale;
    output_state.zero_point = it->second.target_quant.zero_point;
    it->second.output_state = output_state;
    states[it->second.output_name] = output_state;

    ordered_json payload;
    payload["op_type_info"] = OpTypeInfo("QuantizeLinear");
    payload["input_tensor_info"] = TensorInfoJson(input_state);
    payload["output_tensor_info"] = TensorInfoJson(output_state);
    ops_json.push_back(WrapOp("QuantizeLinear", std::move(payload)));
  }

  return Status::OK();
}

Status EmitNodeOutputRequantsIfNeeded(const Node& node,
                                      std::unordered_map<std::string, TensorState>& states,
                                      std::unordered_map<std::string, RequantRequest>& requests,
                                      ordered_json& ops_json) {
  const auto outputs = node.OutputDefs();
  for (size_t i = outputs.size(); i > 0; --i) {
    const NodeArg* output = outputs[i - 1];
    if (output == nullptr) {
      continue;
    }
    const auto state_it = states.find(output->Name());
    if (state_it != states.end()) {
      ORT_RETURN_IF_ERROR(EmitInsertedRequantIfNeeded(output->Name(), state_it->second,
                                                      states, requests, ops_json));
    }
  }
  return Status::OK();
}

TensorState LookupState(const GraphViewer& graph_viewer,
                        const std::unordered_map<std::string, TensorState>& states,
                        const std::string& tensor_name,
                        bool nhwc,
                        const std::string& tensor_type = "int8") {
  auto it = states.find(tensor_name);
  if (it != states.end()) {
    return it->second;
  }

  const std::string canonical_name = CanonicalOnnxTensorName(tensor_name);
  it = states.find(canonical_name);
  if (it != states.end()) {
    return it->second;
  }

  TensorState state;
  state.name = ReferenceTensorName(canonical_name);
  state.tensor_type = tensor_type;
  state.shape = ShapeForTensor(graph_viewer, canonical_name, nhwc);
  state.is_constant = false;
  return state;
}

const TensorState* FindExistingState(const std::unordered_map<std::string, TensorState>& states,
                                     const std::string& tensor_name) {
  auto it = states.find(tensor_name);
  if (it != states.end()) {
    return &it->second;
  }

  const std::string canonical_name = CanonicalOnnxTensorName(tensor_name);
  it = states.find(canonical_name);
  if (it != states.end()) {
    return &it->second;
  }

  return nullptr;
}

bool IsPaddedQLinearConvNode(const Node* node) {
  if (node == nullptr || node->OpType() != "QLinearConv") {
    return false;
  }
  const std::vector<int> pads = GetIntsAttribute(*node, "pads", {0, 0, 0, 0});
  return pads.size() == 4 && std::any_of(pads.begin(), pads.end(), [](int pad) { return pad != 0; });
}
