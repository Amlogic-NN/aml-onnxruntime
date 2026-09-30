// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside the anonymous namespace. It keeps the stable name/shape helpers that
// are shared by both the legacy qlinear exporter and reference_style support
// logic in one place without changing behavior.

std::string ReferenceTensorName(std::string_view onnx_name) {
  if (!onnx_name.empty() && onnx_name.front() == '/') {
    return std::string("wa") + std::string(onnx_name);
  }
  return std::string(onnx_name);
}

std::string OnnxTensorName(std::string_view reference_name) {
  if (reference_name.rfind("wa/", 0) == 0) {
    return std::string(reference_name.substr(2));
  }
  return std::string(reference_name);
}

std::string BaseValueInfoName(std::string_view name) {
  std::string result = OnnxTensorName(name);
  if (EndsWith(result, "_quantized")) {
    result.resize(result.size() - std::string_view("_quantized").size());
  } else if (EndsWith(result, "quantized")) {
    result.resize(result.size() - std::string_view("quantized").size());
  }
  return result;
}

std::string CanonicalOnnxTensorName(std::string_view name) {
  std::string result = OnnxTensorName(name);
  if (EndsWith(result, "/duplicated")) {
    result.resize(result.size() - std::string_view("/duplicated").size());
  }
  if (EndsWith(result, "_pre_q")) {
    result.resize(result.size() - std::string_view("_pre_q").size());
  }
  if (EndsWith(result, "_q_to_dq")) {
    result.resize(result.size() - std::string_view("_q_to_dq").size());
  }
  if (EndsWith(result, "_post_dq")) {
    result.resize(result.size() - std::string_view("_post_dq").size());
  }
  return result;
}

std::vector<int> TensorProtoShape(const ONNX_NAMESPACE::TensorProto& tensor) {
  std::vector<int> shape;
  shape.reserve(static_cast<size_t>(tensor.dims_size()));
  for (int i = 0; i < tensor.dims_size(); ++i) {
    shape.push_back(ToIntDim(tensor.dims(i)));
  }
  return shape;
}

std::vector<int> NodeArgShape(const NodeArg& node_arg) {
  std::vector<int> shape;
  const auto* tensor_shape = node_arg.Shape();
  if (tensor_shape == nullptr) {
    return shape;
  }

  shape.reserve(static_cast<size_t>(tensor_shape->dim_size()));
  for (int i = 0; i < tensor_shape->dim_size(); ++i) {
    const auto& dim = tensor_shape->dim(i);
    if (dim.has_dim_value()) {
      shape.push_back(ToIntDim(dim.dim_value()));
    } else {
      // The Amlogic reference converter materializes symbolic batch as 1
      // for these static NPU export records.
      shape.push_back(i == 0 ? 1 : -1);
    }
  }
  return shape;
}

std::vector<int> ToNhwcShape(const std::vector<int>& shape) {
  if (shape.size() != 4) {
    return shape;
  }
  return {shape[0], shape[2], shape[3], shape[1]};
}

std::vector<int> ToNchwShape(const std::vector<int>& shape) {
  if (shape.size() != 4) {
    return shape;
  }
  return {shape[0], shape[3], shape[1], shape[2]};
}

std::vector<int> ApplyPermToShape(const std::vector<int>& shape, const std::vector<int>& perm) {
  if (shape.size() != perm.size()) {
    return shape;
  }

  std::vector<int> result;
  result.reserve(perm.size());
  for (int axis : perm) {
    if (axis < 0) {
      axis += static_cast<int>(shape.size());
    }
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
      return shape;
    }
    result.push_back(shape[static_cast<size_t>(axis)]);
  }
  return result;
}

std::vector<int> ConvWeightOihwToOhwiShape(const std::vector<int>& shape) {
  if (shape.size() != 4) {
    return shape;
  }
  return {shape[0], shape[2], shape[3], shape[1]};
}

std::vector<int> ApplyNhwcPads(const std::vector<int>& shape, const std::vector<int>& pads) {
  if (shape.size() != 4 || pads.size() != 4) {
    return shape;
  }
  return {shape[0], shape[1] + pads[0] + pads[2], shape[2] + pads[1] + pads[3], shape[3]};
}

size_t ElementCount(const std::vector<int>& shape) {
  if (shape.empty()) {
    return 1;
  }
  size_t count = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<int> ShapeForTensor(const GraphViewer& graph_viewer, const std::string& tensor_name, bool nhwc) {
  const std::string onnx_name = CanonicalOnnxTensorName(tensor_name);
  const ONNX_NAMESPACE::TensorProto* initializer = graph_viewer.GetConstantInitializer(onnx_name, true);
  if (initializer != nullptr) {
    return TensorProtoShape(*initializer);
  }

  const NodeArg* node_arg = graph_viewer.GetNodeArg(onnx_name);
  if (node_arg == nullptr || NodeArgShape(*node_arg).empty()) {
    node_arg = graph_viewer.GetNodeArg(BaseValueInfoName(onnx_name));
  }

  std::vector<int> shape;
  if (node_arg != nullptr) {
    shape = NodeArgShape(*node_arg);
  }

  return nhwc ? ToNhwcShape(shape) : shape;
}

std::vector<int> GetIntsAttribute(const Node& node, const std::string& name, std::vector<int> default_value) {
  const auto& attributes = node.GetAttributes();
  const auto attr = attributes.find(name);
  if (attr == attributes.end() ||
      attr->second.type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INTS) {
    return default_value;
  }

  std::vector<int> values;
  values.reserve(static_cast<size_t>(attr->second.ints_size()));
  for (int i = 0; i < attr->second.ints_size(); ++i) {
    values.push_back(ToIntDim(attr->second.ints(i)));
  }
  return values;
}

int GetIntAttribute(const Node& node, const std::string& name, int default_value) {
  const auto& attributes = node.GetAttributes();
  const auto attr = attributes.find(name);
  if (attr == attributes.end() ||
      attr->second.type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT) {
    return default_value;
  }
  return ToIntDim(attr->second.i());
}

bool JsonEqual(const ordered_json& a, const ordered_json& b) {
  return a.dump() == b.dump();
}

void KeepAsArray(ordered_json& value) {
  if (!value.is_array()) {
    value = ordered_json::array({value});
  }
}

ordered_json TensorInfoJson(const TensorState& tensor) {
  ordered_json result;
  result["tensor_name"] = tensor.name;
  result["tensor_type"] = tensor.tensor_type;
  result["tensor_shape"] = tensor.shape;
  result["is_constant"] = tensor.is_constant;
  if (tensor.has_values) {
    result["values"] = tensor.values;
  }
  if (tensor.has_quant) {
    result["quant_scale"] = tensor.quant_scale;
    result["zero_point"] = tensor.zero_point;
  }
  return result;
}

ordered_json TensorInfoJsonQuantBeforeValues(const TensorState& tensor) {
  ordered_json result;
  result["tensor_name"] = tensor.name;
  result["tensor_type"] = tensor.tensor_type;
  result["tensor_shape"] = tensor.shape;
  result["is_constant"] = tensor.is_constant;
  if (tensor.has_quant) {
    result["quant_scale"] = tensor.quant_scale;
    result["zero_point"] = tensor.zero_point;
  }
  if (tensor.has_values) {
    result["values"] = tensor.values;
  }
  return result;
}

TensorState FloatDisplayTensorState(const TensorState& tensor) {
  TensorState display = tensor;
  if (display.tensor_type == "float32") {
    display.has_quant = true;
    display.quant_scale = "None";
    display.zero_point = "None";
  }
  return display;
}

ordered_json OpTypeInfo(std::string_view op_name) {
  ordered_json result;
  result["op_name"] = std::string(op_name);
  return result;
}

ordered_json WrapOp(std::string_view op_type, ordered_json payload) {
  ordered_json result;
  result[std::string(op_type)] = std::move(payload);
  return result;
}

ordered_json ConstantTensorInfoJson(const std::string& name,
                                    const std::string& tensor_type,
                                    const std::vector<int>& shape,
                                    const ordered_json& values) {
  ordered_json result;
  result["tensor_name"] = name;
  result["tensor_type"] = tensor_type;
  result["tensor_shape"] = shape;
  result["is_constant"] = true;
  result["values"] = values;
  return result;
}
