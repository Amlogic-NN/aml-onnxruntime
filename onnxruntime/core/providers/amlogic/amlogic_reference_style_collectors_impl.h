// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside namespace reference_style_internal. It keeps the reference_style
// collector logic in a dedicated module without changing behavior.

bool PadsAllEqual(const std::vector<int>& pads, int value) {
  if (pads.empty()) {
    return value == 0;
  }
  return std::all_of(pads.begin(), pads.end(), [value](int pad) {
    return pad == value;
  });
}

bool PadsAnyNonZero(const std::vector<int>& pads) {
  return std::any_of(pads.begin(), pads.end(), [](int pad) {
    return pad != 0;
  });
}

std::vector<int> ReferenceConvExplicitPads(const std::vector<int>& pads) {
  if (pads.size() == 8) {
    return {pads[2], pads[3], pads[6], pads[7]};
  }
  if (pads.size() == 4) {
    return pads;
  }
  return {};
}

ordered_json IntValuesJson(const std::vector<int>& values) {
  ordered_json result = ordered_json::array();
  for (int value : values) {
    result.push_back(value);
  }
  return result;
}

std::string ReferenceConvPaddingMode(const std::vector<int>& pads) {
  if (PadsAllEqual(pads, 0)) {
    return "VALID";
  }
  if (PadsAllEqual(pads, 1)) {
    return "add_pad_op1";
  }
  if (PadsAllEqual(pads, 2)) {
    return "add_pad_op2";
  }
  if (PadsAllEqual(pads, 3)) {
    return "add_pad_op3";
  }
  return "VALID";
}

std::string ReferenceConvPaddingModeWithExplicitPads(const std::vector<int>& pads) {
  const std::string legacy_mode = ReferenceConvPaddingMode(pads);
  if (legacy_mode != "VALID" || !PadsAnyNonZero(pads)) {
    return legacy_mode;
  }
  return ReferenceConvExplicitPads(pads).empty() ? "VALID" : "explicit_pad_op";
}

std::string ReferenceConvPaddingModeForShape(const std::vector<int>& pads,
                                             const TensorState& input_state,
                                             const TensorState& output_state,
                                             int kernel_h,
                                             int kernel_w,
                                             int stride_h,
                                             int stride_w,
                                             int dilation_h,
                                             int dilation_w) {
  if (PadsAllEqual(pads, 0) &&
      input_state.shape.size() == 4 &&
      output_state.shape.size() == 4 &&
      kernel_h > 0 && kernel_w > 0 &&
      stride_h > 0 && stride_w > 0 &&
      dilation_h > 0 && dilation_w > 0) {
    const int effective_kernel_h = dilation_h * (kernel_h - 1) + 1;
    const int effective_kernel_w = dilation_w * (kernel_w - 1) + 1;
    const int valid_h = (input_state.shape[1] - effective_kernel_h) / stride_h + 1;
    const int valid_w = (input_state.shape[2] - effective_kernel_w) / stride_w + 1;
    const int same_h = (input_state.shape[1] + stride_h - 1) / stride_h;
    const int same_w = (input_state.shape[2] + stride_w - 1) / stride_w;
    if (output_state.shape[1] == same_h && output_state.shape[2] == same_w &&
        (output_state.shape[1] != valid_h || output_state.shape[2] != valid_w)) {
      return "SAME";
    }
  }

  return ReferenceConvPaddingMode(pads);
}

std::string ReferenceConvPaddingModeForShapeWithExplicitPads(const std::vector<int>& pads,
                                                             const TensorState& input_state,
                                                             const TensorState& output_state,
                                                             int kernel_h,
                                                             int kernel_w,
                                                             int stride_h,
                                                             int stride_w,
                                                             int dilation_h,
                                                             int dilation_w) {
  const std::string shape_mode = ReferenceConvPaddingModeForShape(pads, input_state, output_state,
                                                                  kernel_h, kernel_w,
                                                                  stride_h, stride_w,
                                                                  dilation_h, dilation_w);
  if (shape_mode != "VALID" || !PadsAnyNonZero(pads)) {
    return shape_mode;
  }
  return ReferenceConvExplicitPads(pads).empty() ? "VALID" : "explicit_pad_op";
}

void AttachExplicitConvPadsIfNeeded(ordered_json& attributes, const std::vector<int>& pads) {
  if (!attributes.contains("padding") ||
      !attributes["padding"].is_string() ||
      attributes["padding"].get<std::string>() != "explicit_pad_op") {
    return;
  }
  const std::vector<int> explicit_pads = ReferenceConvExplicitPads(pads);
  if (!explicit_pads.empty()) {
    attributes["explicit_pads"] = IntValuesJson(explicit_pads);
  }
}

float GetFloatAttribute(const Node& node, const std::string& name, float default_value) {
  const auto& attributes = node.GetAttributes();
  const auto attr = attributes.find(name);
  if (attr == attributes.end() ||
      attr->second.type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT) {
    return default_value;
  }
  return attr->second.f();
}

std::string GetStringAttribute(const Node& node, const std::string& name, const std::string& default_value) {
  const auto& attributes = node.GetAttributes();
  const auto attr = attributes.find(name);
  if (attr == attributes.end() ||
      attr->second.type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRING) {
    return default_value;
  }
  return attr->second.s();
}

ordered_json FloatValuesJson(const std::vector<float>& values) {
  ordered_json result = ordered_json::array();
  for (float value : values) {
    result.push_back(value);
  }
  return result;
}

bool FlattenJsonScalars(const ordered_json& source, ordered_json& target);
bool TryJsonToIntVector(const ordered_json& value, std::vector<int>& result);

TensorState MakeFloatConstantTensorState(const std::string& name,
                                         const std::vector<int>& shape,
                                         const std::vector<float>& values) {
  TensorState state;
  state.name = ReferenceTensorName(name);
  state.tensor_type = "float32";
  state.shape = shape.empty() ? std::vector<int>{1} : shape;
  state.is_constant = true;
  state.has_values = true;
  state.values = FloatValuesJson(values);
  state.has_quant = false;
  return state;
}

bool IsReferenceGraphInput(const GraphViewer& graph_viewer, const std::string& tensor_name) {
  return std::any_of(graph_viewer.GetInputs().begin(), graph_viewer.GetInputs().end(),
                     [&tensor_name](const NodeArg* input) {
                       return input != nullptr && input->Name() == tensor_name;
                     });
}

int64_t StaticElementCount(const std::vector<int>& shape) {
  int64_t count = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    count *= dim;
  }
  return count;
}

void MarkFloatTensorNoQuant(TensorState& state) {
  if (state.tensor_type == "float32") {
    state.has_quant = false;
  }
}

bool IsReferenceIntegerTensorType(const std::string& tensor_type) {
  return tensor_type == "int32" || tensor_type == "int8" || tensor_type == "uint8";
}

bool JsonNumberToInt32(const ordered_json& value, int& result) {
  if (value.is_number_unsigned()) {
    const uint64_t as_uint64 = value.get<uint64_t>();
    if (as_uint64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    result = static_cast<int>(as_uint64);
    return true;
  }
  if (value.is_number_integer()) {
    const int64_t as_int64 = value.get<int64_t>();
    if (as_int64 < std::numeric_limits<int>::min() ||
        as_int64 > std::numeric_limits<int>::max()) {
      return false;
    }
    result = static_cast<int>(as_int64);
    return true;
  }
  if (!value.is_number_float()) {
    return false;
  }
  const double as_double = value.get<double>();
  if (as_double < static_cast<double>(std::numeric_limits<int>::min()) ||
      as_double > static_cast<double>(std::numeric_limits<int>::max())) {
    return false;
  }
  const double rounded = std::round(as_double);
  if (std::fabs(as_double - rounded) > 1.0e-5) {
    return false;
  }
  result = static_cast<int>(rounded);
  return true;
}

bool CoerceConstantTensorValuesToInt32(TensorState& state) {
  if (!state.is_constant || !state.has_values) {
    return true;
  }
  ordered_json flat_values = ordered_json::array();
  if (!FlattenJsonScalars(state.values, flat_values)) {
    return false;
  }
  std::vector<int> int_values;
  int_values.reserve(flat_values.size());
  for (const auto& value : flat_values) {
    int converted = 0;
    if (!JsonNumberToInt32(value, converted)) {
      return false;
    }
    int_values.push_back(converted);
  }
  state.values = ValuesToJson(int_values, state.shape);
  return true;
}

bool InvertSingleFloatConstant(TensorState& state) {
  if (!state.is_constant || !state.has_values) {
    return false;
  }
  ordered_json flat_values = ordered_json::array();
  if (!FlattenJsonScalars(state.values, flat_values) || flat_values.size() != 1 ||
      !flat_values[0].is_number()) {
    return false;
  }
  const float divisor = flat_values[0].get<float>();
  if (divisor == 0.0f) {
    return false;
  }
  if (state.shape.empty()) {
    state.shape = {1};
  }
  state.values = ValuesToJson(std::vector<float>{1.0f / divisor}, state.shape);
  return true;
}

bool IsStaticPositiveShape(const std::vector<int>& shape) {
  return std::all_of(shape.begin(), shape.end(), [](int dim) { return dim > 0; });
}

std::vector<int> ResolveExpandOutputShape(const GraphViewer& graph_viewer,
                                          const Node& node,
                                          const std::unordered_map<std::string, TensorState>& states,
                                          const std::vector<int>& source_shape) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  if (!outputs.empty() && outputs[0] != nullptr) {
    std::vector<int> output_shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
    if (!output_shape.empty() && IsStaticPositiveShape(output_shape)) {
      return output_shape;
    }
  }

  std::vector<int> requested_shape;
  if (inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    if (const auto* shape_state = FindExistingState(states, inputs[1]->Name())) {
      TryJsonToIntVector(shape_state->values, requested_shape);
    }
  }

  if (requested_shape.empty() && inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    const std::string canonical_shape_name = CanonicalOnnxTensorName(inputs[1]->Name());
    Status status = InitializerAsIntVector(graph_viewer, canonical_shape_name, requested_shape);
    if (!status.IsOK()) {
      requested_shape.clear();
    }
  }

  if (requested_shape.empty()) {
    return source_shape.empty() ? std::vector<int>{1} : source_shape;
  }

  const size_t rank = std::max(source_shape.size(), requested_shape.size());
  std::vector<int> result(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int source_dim = i < rank - source_shape.size() ? 1 : source_shape[i - (rank - source_shape.size())];
    const int requested_dim =
        i < rank - requested_shape.size() ? 1 : requested_shape[i - (rank - requested_shape.size())];
    if (requested_dim <= 0) {
      result[i] = source_dim > 0 ? source_dim : 1;
    } else if (source_dim <= 0 || source_dim == 1 || source_dim == requested_dim) {
      result[i] = requested_dim;
    } else if (requested_dim == 1) {
      result[i] = source_dim;
    } else {
      result[i] = std::max(source_dim, requested_dim);
    }
  }
  return result;
}

template <typename T>
Status BroadcastFlatValues(const std::vector<T>& source_values,
                           const std::vector<int>& source_shape,
                           const std::vector<int>& target_shape,
                           std::vector<T>& target_values) {
  const size_t source_count = ElementCount(source_shape);
  const size_t target_count = ElementCount(target_shape);
  ORT_RETURN_IF_NOT(source_count == source_values.size(),
                    "Expand source initializer size does not match source shape.");
  ORT_RETURN_IF_NOT(target_count > 0,
                    "Expand target shape must be fully static and positive.");

  if (source_shape == target_shape || (source_count == 1 && target_count == 1)) {
    target_values = source_values;
    return Status::OK();
  }

  const size_t rank = target_shape.size();
  ORT_RETURN_IF_NOT(source_shape.size() <= rank,
                    "Expand source rank is larger than target rank.");

  std::vector<int> padded_source_shape(rank, 1);
  const size_t source_offset = rank - source_shape.size();
  for (size_t i = 0; i < source_shape.size(); ++i) {
    padded_source_shape[source_offset + i] = source_shape[i];
  }

  for (size_t axis = 0; axis < rank; ++axis) {
    const int source_dim = padded_source_shape[axis];
    const int target_dim = target_shape[axis];
    ORT_RETURN_IF_NOT(source_dim == 1 || source_dim == target_dim,
                      "Expand initializer shape is not broadcastable to target shape.");
  }

  std::vector<size_t> source_strides(rank, 1);
  std::vector<size_t> target_strides(rank, 1);
  for (size_t i = rank; i-- > 1;) {
    source_strides[i - 1] = source_strides[i] * static_cast<size_t>(padded_source_shape[i]);
    target_strides[i - 1] = target_strides[i] * static_cast<size_t>(target_shape[i]);
  }

  target_values.resize(target_count);
  for (size_t target_offset = 0; target_offset < target_count; ++target_offset) {
    size_t source_index = 0;
    for (size_t axis = 0; axis < rank; ++axis) {
      const size_t target_axis_index =
          rank == 0 ? 0 : (target_offset / target_strides[axis]) % static_cast<size_t>(target_shape[axis]);
      const size_t source_axis_index = padded_source_shape[axis] == 1 ? 0 : target_axis_index;
      source_index += source_axis_index * source_strides[axis];
    }
    target_values[target_offset] = source_values[source_index];
  }
  return Status::OK();
}

template <typename T>
Status BroadcastTypedInitializerValuesJson(const ONNX_NAMESPACE::TensorProto& tensor,
                                           const std::filesystem::path& model_path,
                                           const std::vector<int>& source_shape,
                                           const std::vector<int>& target_shape,
                                           ordered_json& values) {
  std::vector<T> source_values;
  ORT_RETURN_IF_ERROR(UnpackVector<T>(tensor, model_path, source_values));
  std::vector<T> target_values;
  ORT_RETURN_IF_ERROR(BroadcastFlatValues(source_values, source_shape, target_shape, target_values));
  values = ValuesToJson(target_values, target_shape);
  return Status::OK();
}

Status BroadcastInitializerValuesJson(const ONNX_NAMESPACE::TensorProto& tensor,
                                      const std::filesystem::path& model_path,
                                      const std::vector<int>& source_shape,
                                      const std::vector<int>& target_shape,
                                      ordered_json& values) {
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return BroadcastTypedInitializerValuesJson<float>(tensor, model_path, source_shape, target_shape, values);
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      std::vector<int8_t> source_values;
      ORT_RETURN_IF_ERROR(UnpackVector<int8_t>(tensor, model_path, source_values));
      std::vector<int8_t> target_values;
      ORT_RETURN_IF_ERROR(BroadcastFlatValues(source_values, source_shape, target_shape, target_values));
      values = ValuesToJson(ToIntVector(target_values), target_shape);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      std::vector<uint8_t> source_values;
      ORT_RETURN_IF_ERROR(UnpackVector<uint8_t>(tensor, model_path, source_values));
      std::vector<uint8_t> target_values;
      ORT_RETURN_IF_ERROR(BroadcastFlatValues(source_values, source_shape, target_shape, target_values));
      values = ValuesToJson(ToIntVector(target_values), target_shape);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      std::vector<int32_t> source_values;
      ORT_RETURN_IF_ERROR(UnpackVector<int32_t>(tensor, model_path, source_values));
      std::vector<int32_t> target_values;
      ORT_RETURN_IF_ERROR(BroadcastFlatValues(source_values, source_shape, target_shape, target_values));
      values = ValuesToJson(ToIntVector(target_values), target_shape);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      std::vector<int64_t> source_values;
      ORT_RETURN_IF_ERROR(UnpackVector<int64_t>(tensor, model_path, source_values));
      std::vector<int64_t> target_values;
      ORT_RETURN_IF_ERROR(BroadcastFlatValues(source_values, source_shape, target_shape, target_values));
      values = ValuesToJson(ToIntVector(target_values), target_shape);
      return Status::OK();
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Unsupported Expand initializer data type for constant folding.");
  }
}

Status BroadcastStateValuesJson(const TensorState& source_state,
                                const std::vector<int>& target_shape,
                                ordered_json& values) {
  ordered_json flat_json = ordered_json::array();
  ORT_RETURN_IF_NOT(source_state.has_values &&
                        FlattenJsonScalars(source_state.values, flat_json),
                    "Expand source TensorState does not contain foldable scalar values.");

  if (source_state.tensor_type == "float32" || source_state.tensor_type == "float16") {
    std::vector<float> source_values;
    source_values.reserve(flat_json.size());
    for (const auto& value : flat_json) {
      ORT_RETURN_IF_NOT(value.is_number(), "Expand float source contains a non-numeric value.");
      source_values.push_back(value.get<float>());
    }
    std::vector<float> target_values;
    ORT_RETURN_IF_ERROR(BroadcastFlatValues(source_values, source_state.shape, target_shape, target_values));
    values = ValuesToJson(target_values, target_shape);
    return Status::OK();
  }

  std::vector<int> source_values;
  source_values.reserve(flat_json.size());
  for (const auto& value : flat_json) {
    ORT_RETURN_IF_NOT(value.is_number_integer() || value.is_number_unsigned(),
                      "Expand integer source contains a non-integer value.");
    source_values.push_back(value.get<int>());
  }
  std::vector<int> target_values;
  ORT_RETURN_IF_ERROR(BroadcastFlatValues(source_values, source_state.shape, target_shape, target_values));
  values = ValuesToJson(target_values, target_shape);
  return Status::OK();
}

std::string ReferenceTensorTypeFromOnnxDataType(int data_type,
                                                const std::string& fallback_type) {
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return "float32";
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return "int8";
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return "uint8";
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return "int32";
    default:
      return fallback_type;
  }
}

std::string ReferenceTensorTypeFromNodeArg(const NodeArg* node_arg,
                                           const std::string& fallback_type) {
  if (node_arg == nullptr) {
    return fallback_type;
  }

  const auto* type_proto = node_arg->TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type() ||
      !type_proto->tensor_type().has_elem_type()) {
    return fallback_type;
  }

  return ReferenceTensorTypeFromOnnxDataType(type_proto->tensor_type().elem_type(),
                                             fallback_type);
}

Status MakeReferenceInputTensorState(const GraphViewer& graph_viewer,
                                     const NodeArg* input,
                                     bool nhwc,
                                     const std::string& fallback_type,
                                     std::unordered_map<std::string, TensorState>& states,
                                     TensorState& state) {
  ORT_RETURN_IF_NOT(input != nullptr, "Invalid null reference-style input.");

  auto existing = states.find(input->Name());
  if (existing != states.end()) {
    state = existing->second;
    return Status::OK();
  }

  const std::string tensor_type = ReferenceTensorTypeFromNodeArg(input, fallback_type);
  const std::string canonical_name = CanonicalOnnxTensorName(input->Name());
  const auto* initializer = graph_viewer.GetConstantInitializer(canonical_name, true);
  if (initializer != nullptr) {
    const std::vector<int> shape = TensorProtoShape(*initializer);
    ordered_json values;
    ORT_RETURN_IF_ERROR(TensorValuesJson(*initializer, graph_viewer.ModelPath(), shape, shape, nullptr,
                                         false, values));

    state.name = ReferenceTensorName(canonical_name);
    state.tensor_type = ReferenceTensorTypeFromOnnxDataType(initializer->data_type(), tensor_type);
    state.shape = shape.empty() ? std::vector<int>{1} : shape;
    state.is_constant = true;
    state.has_values = true;
    state.values = std::move(values);
    MarkFloatTensorNoQuant(state);
    states[input->Name()] = state;
    return Status::OK();
  }

  state = LookupState(graph_viewer, states, input->Name(), nhwc, tensor_type);
  state.tensor_type = tensor_type;
  MarkFloatTensorNoQuant(state);
  states[input->Name()] = state;
  return Status::OK();
}

TensorState MakeReferenceOutputTensorState(const GraphViewer& graph_viewer,
                                           const NodeArg* output,
                                           bool nhwc,
                                           const std::string& fallback_type) {
  TensorState state;
  if (output == nullptr) {
    return state;
  }

  const std::string canonical_name = CanonicalOnnxTensorName(output->Name());
  state.name = ReferenceTensorName(canonical_name);
  state.tensor_type = ReferenceTensorTypeFromNodeArg(output, fallback_type);
  state.shape = ShapeForTensor(graph_viewer, canonical_name, nhwc);
  state.is_constant = false;
  state.has_values = false;
  MarkFloatTensorNoQuant(state);
  return state;
}

bool Rank2ValuesToRank4BroadcastJson(const ordered_json& source,
                                      int rows,
                                      int cols,
                                      bool transpose_2d,
                                      ordered_json& target) {
  if (!source.is_array() || rows <= 0 || cols <= 0) {
    return false;
  }

  const bool nested_rank2 =
      source.size() == static_cast<size_t>(rows) &&
      rows > 0 && source[0].is_array();
  const bool flat_rank1 =
      source.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (!nested_rank2 && !flat_rank1) {
    return false;
  }

  auto value_at = [&](int row, int col, ordered_json& value) -> bool {
    if (nested_rank2) {
      if (source[row].size() != static_cast<size_t>(cols)) {
        return false;
      }
      value = source[row][col];
      return true;
    }
    value = source[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    return true;
  };

  const int out_dim_1 = transpose_2d ? cols : rows;
  const int out_dim_2 = transpose_2d ? rows : cols;
  ordered_json batch = ordered_json::array();
  for (int i = 0; i < out_dim_1; ++i) {
    ordered_json plane = ordered_json::array();
    for (int j = 0; j < out_dim_2; ++j) {
      const int src_row = transpose_2d ? j : i;
      const int src_col = transpose_2d ? i : j;
      ordered_json value;
      if (!value_at(src_row, src_col, value)) {
        return false;
      }
      ordered_json last_dim = ordered_json::array();
      last_dim.push_back(std::move(value));
      plane.push_back(std::move(last_dim));
    }
    batch.push_back(std::move(plane));
  }

  target = ordered_json::array();
  target.push_back(std::move(batch));
  return true;
}

bool Rank2ValuesToYoloPoseGridBroadcastJson(const ordered_json& source,
                                             int rows,
                                             int cols,
                                             ordered_json& target) {
  if (!source.is_array() || rows <= 0 || cols <= 0) {
    return false;
  }

  const bool nested_rank2 =
      source.size() == static_cast<size_t>(rows) &&
      rows > 0 && source[0].is_array();
  const bool flat_rank1 =
      source.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (!nested_rank2 && !flat_rank1) {
    return false;
  }

  auto value_at = [&](int row, int col, ordered_json& value) -> bool {
    if (nested_rank2) {
      if (source[row].size() != static_cast<size_t>(cols)) {
        return false;
      }
      value = source[row][col];
      return true;
    }
    value = source[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    return true;
  };

  ordered_json row_axis = ordered_json::array();
  for (int row = 0; row < rows; ++row) {
    ordered_json col_axis = ordered_json::array();
    for (int col = 0; col < cols; ++col) {
      ordered_json value;
      if (!value_at(row, col, value)) {
        return false;
      }
      ordered_json singleton = ordered_json::array();
      singleton.push_back(std::move(value));
      col_axis.push_back(std::move(singleton));
    }
    row_axis.push_back(std::move(col_axis));
  }

  target = ordered_json::array();
  target.push_back(std::move(row_axis));
  return true;
}

bool Rank2ValuesToYoloPoseGridBroadcastJsonTransposed(const ordered_json& source,
                                                      int rows,
                                                      int cols,
                                                      ordered_json& target) {
  if (!source.is_array() || rows <= 0 || cols <= 0) {
    return false;
  }

  const bool nested_rank2 =
      source.size() == static_cast<size_t>(rows) &&
      rows > 0 && source[0].is_array();
  const bool flat_rank1 =
      source.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (!nested_rank2 && !flat_rank1) {
    return false;
  }

  auto value_at = [&](int row, int col, ordered_json& value) -> bool {
    if (nested_rank2) {
      if (source[row].size() != static_cast<size_t>(cols)) {
        return false;
      }
      value = source[row][col];
      return true;
    }
    value = source[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    return true;
  };

  ordered_json coord_axis = ordered_json::array();
  for (int col = 0; col < cols; ++col) {
    ordered_json anchor_axis = ordered_json::array();
    for (int row = 0; row < rows; ++row) {
      ordered_json value;
      if (!value_at(row, col, value)) {
        return false;
      }
      ordered_json singleton = ordered_json::array();
      singleton.push_back(std::move(value));
      anchor_axis.push_back(std::move(singleton));
    }
    coord_axis.push_back(std::move(anchor_axis));
  }

  target = ordered_json::array();
  target.push_back(std::move(coord_axis));
  return true;
}

bool RetargetRank2BroadcastConstantToRank4(TensorState& constant_state,
                                           const TensorState& reference_state) {
  if (!constant_state.is_constant ||
      !constant_state.has_values ||
      constant_state.shape.size() != 2 ||
      reference_state.shape.size() != 4) {
    return false;
  }

  const int rows = constant_state.shape[0];
  const int cols = constant_state.shape[1];
  const bool direct_match =
      rows == reference_state.shape[1] && cols == reference_state.shape[2];
  const bool transposed_match =
      rows == reference_state.shape[2] && cols == reference_state.shape[1];
  const bool yolo_pose_grid_channel_major =
      rows <= 4 && cols > rows &&
      (direct_match ||
       transposed_match ||
       (rows == reference_state.shape[2] && cols == reference_state.shape[3]));
  const bool yolo_pose_grid_anchor_major =
      cols <= 4 && rows > cols &&
      (direct_match ||
       transposed_match ||
       (cols == reference_state.shape[2] && rows == reference_state.shape[3]));
  if (!direct_match && !transposed_match &&
      !yolo_pose_grid_channel_major && !yolo_pose_grid_anchor_major) {
    return false;
  }
  if (yolo_pose_grid_channel_major) {
    ordered_json retargeted_values;
    if (!Rank2ValuesToYoloPoseGridBroadcastJson(constant_state.values,
                                                rows,
                                                cols,
                                                retargeted_values)) {
      return false;
    }
    constant_state.shape = {1, rows, cols, 1};
    constant_state.values = std::move(retargeted_values);
    return true;
  }
  if (yolo_pose_grid_anchor_major) {
    ordered_json retargeted_values;
    if (!Rank2ValuesToYoloPoseGridBroadcastJsonTransposed(constant_state.values,
                                                          rows,
                                                          cols,
                                                          retargeted_values)) {
      return false;
    }
    constant_state.shape = {1, cols, rows, 1};
    constant_state.values = std::move(retargeted_values);
    return true;
  }

  // Only transpose when the reference tensor really has the opposite middle
  // dimensions.
  const bool transpose_values = !direct_match && transposed_match;

  ordered_json retargeted_values;
  if (!Rank2ValuesToRank4BroadcastJson(constant_state.values,
                                       rows,
                                       cols,
                                       transpose_values,
                                       retargeted_values)) {
    return false;
  }

  if (transpose_values) {
    constant_state.shape = {1, cols, rows, 1};
  } else {
    constant_state.shape = {1, rows, cols, 1};
  }
  constant_state.values = std::move(retargeted_values);
  return true;
}

bool RetargetRank3BroadcastConstantToRank4(TensorState& constant_state,
                                           const TensorState& reference_state) {
  if (!constant_state.is_constant ||
      !constant_state.has_values ||
      constant_state.shape.size() != 3 ||
      reference_state.shape.size() != 4) {
    return false;
  }

  const int batch = constant_state.shape[0];
  const int rows = constant_state.shape[1];
  const int cols = constant_state.shape[2];
  if (batch == reference_state.shape[3] && rows == 1 && cols == 1) {
    if (!constant_state.values.is_array() || batch <= 0) {
      return false;
    }

    const bool flat_channels =
        constant_state.values.size() == static_cast<size_t>(batch);
    const bool nested_channels =
        flat_channels && constant_state.values[0].is_array();
    if (!flat_channels) {
      return false;
    }

    ordered_json channel_values = ordered_json::array();
    for (int c = 0; c < batch; ++c) {
      const auto& src = constant_state.values[static_cast<size_t>(c)];
      if (nested_channels) {
        if (src.empty()) {
          return false;
        }
        if (src[0].is_array()) {
          if (src[0].empty()) {
            return false;
          }
          channel_values.push_back(src[0][0]);
        } else {
          channel_values.push_back(src[0]);
        }
      } else {
        channel_values.push_back(src);
      }
    }

    ordered_json width = ordered_json::array();
    width.push_back(std::move(channel_values));
    ordered_json height = ordered_json::array();
    height.push_back(std::move(width));
    ordered_json retargeted_values = ordered_json::array();
    retargeted_values.push_back(std::move(height));

    constant_state.shape = {1, 1, 1, batch};
    constant_state.values = std::move(retargeted_values);
    return true;
  }

  const bool direct_match =
      batch == reference_state.shape[0] &&
      rows == reference_state.shape[1] &&
      cols == reference_state.shape[2];
  const bool transposed_match =
      batch == reference_state.shape[0] &&
      rows == reference_state.shape[2] &&
      cols == reference_state.shape[1];
  if (!direct_match && !transposed_match) {
    return false;
  }
  if (!constant_state.values.is_array() || batch <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }
  const bool nested_rank3 =
      constant_state.values.size() == static_cast<size_t>(batch) &&
      constant_state.values[0].is_array();
  const bool flat_rank1 =
      constant_state.values.size() ==
      static_cast<size_t>(batch) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (!nested_rank3 && !flat_rank1) {
    return false;
  }

  const bool looks_like_yolo_anchor_grid =
      direct_match && rows <= 4 && cols > rows && reference_state.shape[3] > 1;
  const bool transpose_values =
      looks_like_yolo_anchor_grid || (!direct_match && transposed_match);
  const int out_h = transpose_values ? cols : rows;
  const int out_w = transpose_values ? rows : cols;

  ordered_json retargeted_values = ordered_json::array();
  auto value_at = [&](int b, int row, int col, ordered_json& value) -> bool {
    if (nested_rank3) {
      if (!constant_state.values[static_cast<size_t>(b)].is_array() ||
          constant_state.values[static_cast<size_t>(b)].size() != static_cast<size_t>(rows)) {
        return false;
      }
      const auto& src_row_values = constant_state.values[static_cast<size_t>(b)][static_cast<size_t>(row)];
      if (!src_row_values.is_array() || src_row_values.size() != static_cast<size_t>(cols)) {
        return false;
      }
      value = src_row_values[static_cast<size_t>(col)];
      return true;
    }

    const size_t offset =
        static_cast<size_t>(b) * static_cast<size_t>(rows) * static_cast<size_t>(cols) +
        static_cast<size_t>(row) * static_cast<size_t>(cols) +
        static_cast<size_t>(col);
    value = constant_state.values[offset];
    return true;
  };

  for (int b = 0; b < batch; ++b) {
    ordered_json out_batch = ordered_json::array();
    for (int h = 0; h < out_h; ++h) {
      ordered_json out_row = ordered_json::array();
      for (int w = 0; w < out_w; ++w) {
        const int src_row = transpose_values ? w : h;
        const int src_col = transpose_values ? h : w;
        ordered_json value;
        if (!value_at(b, src_row, src_col, value)) {
          return false;
        }
        ordered_json channel = ordered_json::array();
        channel.push_back(std::move(value));
        out_row.push_back(std::move(channel));
      }
      out_batch.push_back(std::move(out_row));
    }
    retargeted_values.push_back(std::move(out_batch));
  }

  constant_state.shape = {batch, out_h, out_w, 1};
  constant_state.values = std::move(retargeted_values);
  return true;
}

bool RetargetRank1ChannelConstantToRank4(TensorState& constant_state,
                                         const TensorState& reference_state) {
  if (!constant_state.is_constant ||
      !constant_state.has_values ||
      constant_state.shape.size() != 1 ||
      reference_state.shape.size() != 4) {
    return false;
  }

  const int channels = constant_state.shape[0];
  if (channels <= 0 || channels != reference_state.shape[3] ||
      !constant_state.values.is_array() ||
      constant_state.values.size() != static_cast<size_t>(channels)) {
    return false;
  }

  ordered_json channel_values = ordered_json::array();
  for (int i = 0; i < channels; ++i) {
    channel_values.push_back(constant_state.values[static_cast<size_t>(i)]);
  }

  ordered_json width = ordered_json::array();
  width.push_back(std::move(channel_values));
  ordered_json height = ordered_json::array();
  height.push_back(std::move(width));
  ordered_json batch = ordered_json::array();
  batch.push_back(std::move(height));

  constant_state.shape = {1, 1, 1, channels};
  constant_state.values = std::move(batch);
  return true;
}

bool FlattenJsonScalars(const ordered_json& source, ordered_json& target) {
  if (source.is_array()) {
    for (const auto& value : source) {
      if (!FlattenJsonScalars(value, target)) {
        return false;
      }
    }
    return true;
  }
  if (source.is_object()) {
    return false;
  }
  target.push_back(source);
  return true;
}

bool RetargetRank4NchwChannelConstantToRank4(TensorState& constant_state,
                                             const TensorState& reference_state) {
  if (!constant_state.is_constant ||
      !constant_state.has_values ||
      constant_state.shape.size() != 4 ||
      reference_state.shape.size() != 4) {
    return false;
  }

  const int batch = constant_state.shape[0];
  const int channels = constant_state.shape[1];
  const int height = constant_state.shape[2];
  const int width = constant_state.shape[3];
  if (batch != 1 ||
      channels <= 0 ||
      height != 1 ||
      width != 1 ||
      channels != reference_state.shape[3] ||
      !constant_state.values.is_array()) {
    return false;
  }

  ordered_json channel_values = ordered_json::array();
  if (!FlattenJsonScalars(constant_state.values, channel_values) ||
      channel_values.size() != static_cast<size_t>(channels)) {
    return false;
  }

  ordered_json width_axis = ordered_json::array();
  width_axis.push_back(std::move(channel_values));
  ordered_json height_axis = ordered_json::array();
  height_axis.push_back(std::move(width_axis));
  ordered_json batch_axis = ordered_json::array();
  batch_axis.push_back(std::move(height_axis));

  constant_state.shape = {1, 1, 1, channels};
  constant_state.values = std::move(batch_axis);
  return true;
}

Status EmitReferenceGenericInitial(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   const std::string& op_type,
                                   bool nhwc,
                                   const ordered_json& attributes,
                                   std::unordered_map<std::string, TensorState>& states,
                                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(!inputs.empty() && !outputs.empty(),
                    "Invalid generic reference-style node: ", op_type);

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);

  int input_index = 1;
  for (const NodeArg* input : inputs) {
    if (input == nullptr || !input->Exists()) {
      continue;
    }

    TensorState input_state;
    ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, input, nhwc, "float32",
                                                      states, input_state));
    payload[std::string("input_tensor_info_") + std::to_string(input_index++)] =
        TensorInfoJson(input_state);
  }
  ORT_RETURN_IF_NOT(input_index > 1, "Generic reference-style node has no valid inputs: ", op_type);

  int output_index = 1;
  for (const NodeArg* output : outputs) {
    if (output == nullptr || !output->Exists()) {
      continue;
    }

    TensorState output_state =
        MakeReferenceOutputTensorState(graph_viewer, output, nhwc, "float32");
    states[output->Name()] = output_state;
    const std::string output_key = outputs.size() == 1
                                       ? "output_tensor_info"
                                       : std::string("output_tensor_info_") + std::to_string(output_index);
    payload[output_key] = TensorInfoJson(output_state);
    ++output_index;
  }
  ORT_RETURN_IF_NOT(output_index > 1, "Generic reference-style node has no valid outputs: ", op_type);

  if (!attributes.empty()) {
    payload["attributes_info"] = attributes;
  }
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

bool TryGetSingleIntJsonValue(const ordered_json& value_json, int& value) {
  if (value_json.is_number_integer() || value_json.is_number_unsigned()) {
    value = value_json.get<int>();
    return true;
  }

  if (!value_json.is_array() || value_json.size() != 1) {
    return false;
  }

  return TryGetSingleIntJsonValue(value_json[0], value);
}

Status EmitReferenceGatherElementsRank3Axis1AsGatherNd(
    const GraphViewer& graph_viewer,
    const Node& node,
    const TensorState& data_state,
    const TensorState& indices_state,
    TensorState output_state,
    std::unordered_map<std::string, TensorState>& states,
    ordered_json& ops_json) {
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(!outputs.empty() && outputs[0] != nullptr,
                    "Invalid GatherElements output.");
  ORT_RETURN_IF_NOT(data_state.shape.size() == 3 && indices_state.shape.size() == 3,
                    "GatherElements GatherND lowering requires rank-3 tensors. Node: ", node.Name());

  const int axis = GetIntAttribute(node, "axis", 0) < 0
                       ? GetIntAttribute(node, "axis", 0) + 3
                       : GetIntAttribute(node, "axis", 0);
  ORT_RETURN_IF_NOT(axis == 1,
                    "GatherElements GatherND lowering only handles axis=1 rank-3 tensors. Node: ", node.Name());

  const int batch = indices_state.shape[0];
  const int gathered_count = indices_state.shape[1];
  const int feature_count = indices_state.shape[2];
  ORT_RETURN_IF_NOT(batch > 0 && gathered_count > 0 && feature_count > 0 &&
                        data_state.shape[0] == batch &&
                        data_state.shape[2] == feature_count,
                    "GatherElements rank-3 shape mismatch. Node: ", node.Name());

  if (output_state.shape.empty()) {
    output_state.shape = indices_state.shape;
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  const std::string base_name = ReferenceTensorName(outputs[0]->Name());
  TensorState axis_indices_state = indices_state;
  axis_indices_state.name = base_name + "_gather_elements_axis_indices";
  axis_indices_state.shape = {batch, gathered_count, feature_count, 1};
  axis_indices_state.is_constant = false;
  axis_indices_state.has_values = false;

  ordered_json axis_shape_values = ordered_json::array();
  for (int dim : axis_indices_state.shape) {
    axis_shape_values.push_back(dim);
  }
  ordered_json axis_reshape_payload;
  axis_reshape_payload["op_type_info"] = OpTypeInfo("reshape");
  axis_reshape_payload["skip_transpose_insert_after"] = true;
  axis_reshape_payload["input_tensor_info_1"] = TensorInfoJson(indices_state);
  axis_reshape_payload["input_tensor_info_2"] =
      ConstantTensorInfoJson(axis_indices_state.name + "_shape", "int32",
                             {static_cast<int>(axis_indices_state.shape.size())},
                             axis_shape_values);
  axis_reshape_payload["output_tensor_info"] = TensorInfoJson(axis_indices_state);
  ops_json.push_back(WrapOp("reshape", std::move(axis_reshape_payload)));

  ordered_json batch_values = ordered_json::array();
  ordered_json feature_values = ordered_json::array();
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < gathered_count; ++i) {
      for (int c = 0; c < feature_count; ++c) {
        batch_values.push_back(b);
        feature_values.push_back(c);
      }
    }
  }

  const std::vector<int> coordinate_shape = {batch, gathered_count, feature_count, 1};
  TensorState gather_nd_indices = axis_indices_state;
  gather_nd_indices.name = base_name + "_gather_nd_indices";
  gather_nd_indices.shape = {batch, gathered_count, feature_count, 3};
  gather_nd_indices.is_constant = false;
  gather_nd_indices.has_values = false;

  ordered_json concat_attributes;
  concat_attributes["axis"] = 3;
  concat_attributes["fused_activation_function"] = "None";

  ordered_json concat_payload;
  concat_payload["op_type_info"] = OpTypeInfo("Concat");
  concat_payload["input_tensor_info_1"] =
      ConstantTensorInfoJson(base_name + "_gather_elements_batch_grid", "int32",
                             coordinate_shape, batch_values);
  concat_payload["input_tensor_info_2"] = TensorInfoJson(axis_indices_state);
  concat_payload["input_tensor_info_3"] =
      ConstantTensorInfoJson(base_name + "_gather_elements_feature_grid", "int32",
                             coordinate_shape, feature_values);
  concat_payload["output_tensor_info"] = TensorInfoJson(gather_nd_indices);
  concat_payload["attributes_info"] = std::move(concat_attributes);
  ops_json.push_back(WrapOp("Concat", std::move(concat_payload)));

  ordered_json gather_nd_payload;
  gather_nd_payload["op_type_info"] = OpTypeInfo("GatherND");
  gather_nd_payload["input_tensor_info_1"] = TensorInfoJson(data_state);
  gather_nd_payload["input_tensor_info_2"] = TensorInfoJson(gather_nd_indices);
  gather_nd_payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("GatherND", std::move(gather_nd_payload)));
  (void)graph_viewer;
  return Status::OK();
}

Status EmitReferenceGatherInitial(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  const std::string& op_type,
                                  std::unordered_map<std::string, TensorState>& states,
                                  ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && inputs[0] != nullptr && inputs[1] != nullptr &&
                        !outputs.empty() && outputs[0] != nullptr,
                    "Invalid Gather node.");

  TensorState data_state;
  ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, inputs[0], true, "float32",
                                                    states, data_state));

  TensorState indices_state;
  ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, inputs[1], true, "int32",
                                                    states, indices_state));

  TensorState output_state = MakeReferenceOutputTensorState(graph_viewer, outputs[0], true, "float32");
  MarkFloatTensorNoQuant(output_state);

  const int mapped_axis = MapAxisToNhwc(GetIntAttribute(node, "axis", 0), data_state.shape.size());
  if (op_type == "GatherElements" &&
      data_state.shape.size() == 3 &&
      indices_state.shape.size() == 3 &&
      mapped_axis == 1) {
    return EmitReferenceGatherElementsRank3Axis1AsGatherNd(
        graph_viewer, node, data_state, indices_state, output_state, states, ops_json);
  }

  int scalar_index = 0;
  if (op_type == "Gather" &&
      indices_state.is_constant &&
      indices_state.has_values &&
      TryGetSingleIntJsonValue(indices_state.values, scalar_index) &&
      !data_state.shape.empty() &&
      output_state.shape.size() + 1 == data_state.shape.size()) {
    int axis = mapped_axis;
    if (axis < 0) {
      axis += static_cast<int>(data_state.shape.size());
    }
    if (axis >= 0 && static_cast<size_t>(axis) < data_state.shape.size()) {
      const int axis_dim = data_state.shape[static_cast<size_t>(axis)];
      if (scalar_index < 0 && axis_dim > 0) {
        scalar_index += axis_dim;
      }
      if (axis_dim <= 0 || (scalar_index >= 0 && scalar_index < axis_dim)) {
        std::vector<int> begin(data_state.shape.size(), 0);
        std::vector<int> size = data_state.shape;
        begin[static_cast<size_t>(axis)] = scalar_index;
        size[static_cast<size_t>(axis)] = 1;

        ordered_json begin_values = ordered_json::array();
        ordered_json size_values = ordered_json::array();
        for (size_t dim = 0; dim < data_state.shape.size(); ++dim) {
          begin_values.push_back(begin[dim]);
          size_values.push_back(size[dim]);
        }

        TensorState slice_state = output_state;
        slice_state.name = ReferenceTensorName(std::string(outputs[0]->Name()) + "_gather_slice");
        slice_state.shape = size;
        slice_state.is_constant = false;
        slice_state.has_values = false;
        MarkFloatTensorNoQuant(slice_state);

        ordered_json slice_payload;
        slice_payload["op_type_info"] = OpTypeInfo("Slice");
        slice_payload["input_tensor_info_1"] = TensorInfoJson(data_state);
        slice_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
            slice_state.name + "_begin_values", "int32",
            {static_cast<int>(begin.size())}, begin_values);
        slice_payload["input_tensor_info_3"] = ConstantTensorInfoJson(
            slice_state.name + "_size_values", "int32",
            {static_cast<int>(size.size())}, size_values);
        slice_payload["output_tensor_info"] = TensorInfoJson(slice_state);
        ops_json.push_back(WrapOp("Slice", std::move(slice_payload)));

        ordered_json output_shape = ordered_json::array();
        for (int dim : output_state.shape) {
          output_shape.push_back(dim);
        }

        ordered_json reshape_payload;
        reshape_payload["op_type_info"] = OpTypeInfo("reshape");
        reshape_payload["input_tensor_info_1"] = TensorInfoJson(slice_state);
        reshape_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
            output_state.name + "_gather_reshape_shape", "int32",
            {static_cast<int>(output_state.shape.size())}, output_shape);
        reshape_payload["output_tensor_info"] = TensorInfoJson(output_state);
        ordered_json reshape_attributes;
        reshape_attributes["new_shape"] = output_shape;
        reshape_payload["attributes_info"] = std::move(reshape_attributes);
        ops_json.push_back(WrapOp("reshape", std::move(reshape_payload)));

        states[outputs[0]->Name()] = output_state;
        return Status::OK();
      }
    }
  }

  ordered_json attributes;
  attributes["axis"] = mapped_axis;
  attributes["batch_dims"] = GetIntAttribute(node, "batch_dims", 0);
  return EmitReferenceGenericInitial(graph_viewer, node, op_type, true, attributes,
                                     states, ops_json);
}

std::vector<int> BroadcastShape(const std::vector<int>& lhs, const std::vector<int>& rhs) {
  if (lhs.empty() && rhs.empty()) {
    return {1};
  }
  if (lhs.empty()) {
    return rhs;
  }
  if (rhs.empty()) {
    return lhs;
  }

  const size_t rank = std::max(lhs.size(), rhs.size());
  std::vector<int> result(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int lhs_dim = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
    const int rhs_dim = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
    if (lhs_dim == rhs_dim || rhs_dim == 1) {
      result[i] = lhs_dim;
    } else if (lhs_dim == 1) {
      result[i] = rhs_dim;
    } else {
      result[i] = lhs_dim;
    }
  }
  return result;
}

void PreferPlainRank4OutputShape(TensorState& output_state,
                                 const std::vector<int>& plain_shape) {
  if (plain_shape.size() != 4 ||
      output_state.shape.empty() ||
      output_state.shape == plain_shape ||
      StaticElementCount(output_state.shape) != StaticElementCount(plain_shape)) {
    return;
  }

  if (output_state.shape == ToNhwcShape(plain_shape)) {
    output_state.shape = plain_shape;
  }
}

int NormalizeAxisForRank(int axis, size_t rank) {
  if (rank == 0) {
    return axis;
  }
  if (axis < 0) {
    axis += static_cast<int>(rank);
  }
  return axis;
}

bool ConcatAxisMatchesOutputShape(const std::vector<TensorState>& input_states,
                                  const std::vector<int>& output_shape,
                                  int axis) {
  if (input_states.empty() || output_shape.empty() ||
      axis < 0 || static_cast<size_t>(axis) >= output_shape.size()) {
    return false;
  }

  int concat_dim_sum = 0;
  for (const auto& input_state : input_states) {
    if (input_state.shape.size() != output_shape.size()) {
      return false;
    }
    for (size_t dim = 0; dim < output_shape.size(); ++dim) {
      if (static_cast<int>(dim) == axis) {
        continue;
      }
      if (output_shape[dim] > 0 && input_state.shape[dim] > 0 &&
          output_shape[dim] != input_state.shape[dim]) {
        return false;
      }
    }
    if (input_state.shape[static_cast<size_t>(axis)] <= 0) {
      return false;
    }
    concat_dim_sum += input_state.shape[static_cast<size_t>(axis)];
  }

  return output_shape[static_cast<size_t>(axis)] > 0 &&
         concat_dim_sum == output_shape[static_cast<size_t>(axis)];
}

bool LooksLikePoseHeadAnchorMajorConcat(const std::vector<TensorState>& input_states,
                                        const std::vector<int>& output_shape,
                                        int plain_axis) {
  if (plain_axis != 1 ||
      input_states.size() < 2 ||
      output_shape.size() != 4 ||
      output_shape[2] < 1024 ||
      output_shape[3] != 17) {
    return false;
  }

  int channel_sum = 0;
  for (const auto& input_state : input_states) {
    if (input_state.shape.size() != 4 ||
        input_state.shape[0] != output_shape[0] ||
        input_state.shape[2] != output_shape[2] ||
        input_state.shape[3] != output_shape[3] ||
        input_state.shape[1] <= 0 ||
        input_state.shape[1] > 8) {
      return false;
    }
    channel_sum += input_state.shape[1];
  }

  return channel_sum == output_shape[1];
}

std::vector<int> InferConcatShape(const std::vector<TensorState>& input_states, int axis) {
  if (input_states.empty()) {
    return {};
  }

  std::vector<int> result = input_states.front().shape;
  if (result.empty() || axis < 0 || static_cast<size_t>(axis) >= result.size()) {
    return result;
  }

  for (size_t i = 1; i < input_states.size(); ++i) {
    if (input_states[i].shape.size() == result.size()) {
      result[static_cast<size_t>(axis)] += input_states[i].shape[static_cast<size_t>(axis)];
    }
  }
  return result;
}

int ResolveReferenceConcatAxis(int onnx_axis,
                               size_t rank,
                               const std::vector<TensorState>& input_states,
                               const std::vector<int>& output_shape) {
  const int plain_axis = NormalizeAxisForRank(onnx_axis, rank);
  const int nhwc_axis = MapAxisToNhwc(onnx_axis, rank);
  if (LooksLikePoseHeadAnchorMajorConcat(input_states, output_shape, plain_axis)) {
    return 2;
  }
  if (ConcatAxisMatchesOutputShape(input_states, output_shape, plain_axis)) {
    return plain_axis;
  }
  if (ConcatAxisMatchesOutputShape(input_states, output_shape, nhwc_axis)) {
    return nhwc_axis;
  }
  return nhwc_axis;
}

ordered_json NhwcPadValuesFromOnnxPads(const std::vector<int>& pads) {
  ordered_json values = ordered_json::array();
  if (pads.size() == 8) {
    values.push_back(pads[0]);
    values.push_back(pads[4]);
    values.push_back(pads[2]);
    values.push_back(pads[6]);
    values.push_back(pads[3]);
    values.push_back(pads[7]);
    values.push_back(pads[1]);
    values.push_back(pads[5]);
  } else if (pads.size() == 4) {
    values.push_back(0);
    values.push_back(0);
    values.push_back(pads[0]);
    values.push_back(pads[2]);
    values.push_back(pads[1]);
    values.push_back(pads[3]);
    values.push_back(0);
    values.push_back(0);
  } else {
    for (int i = 0; i < 8; ++i) {
      values.push_back(0);
    }
  }
  return values;
}

std::vector<int> NhwcPadsForShapeFromOnnxPads(const std::vector<int>& pads) {
  if (pads.size() == 8) {
    return {pads[2], pads[3], pads[6], pads[7]};
  }
  if (pads.size() == 4) {
    return pads;
  }
  return {0, 0, 0, 0};
}

void PushReferenceUnaryOp(ordered_json& ops_json,
                          const std::string& op_type,
                          const TensorState& input_state,
                          const TensorState& output_state) {
  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
}

void PushReferenceBinaryOp(ordered_json& ops_json,
                           const std::string& op_type,
                           const TensorState& input_1,
                           const TensorState& input_2,
                           const TensorState& output_state) {
  ordered_json attributes;
  attributes["fused_activation_function"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info_1"] = TensorInfoJson(input_1);
  payload["input_tensor_info_2"] = TensorInfoJson(input_2);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
}

bool CollectorRank3ToRank4BroadcastTargetShape(const std::vector<int>& rank3_shape,
                                               const std::vector<int>& rank4_shape,
                                               std::vector<int>& target_shape) {
  if (rank3_shape.size() != 3 || rank4_shape.size() != 4) {
    return false;
  }

  // NHWC-style prefix broadcast: [N, H, W] * [N, H, W, C].
  if (rank3_shape[0] == rank4_shape[0] &&
      rank3_shape[1] == rank4_shape[1] &&
      rank3_shape[2] == rank4_shape[2] &&
      rank4_shape[3] > 1) {
    target_shape = {rank3_shape[0], rank3_shape[1], rank3_shape[2], 1};
    return true;
  }

  // NCHW-style channel broadcast: [N, H, W] * [N, C, H, W].
  // After the normal NCHW->NHWC layout insertion this becomes
  // [N, H, W, 1] * [N, H, W, C], which TFLite can broadcast.
  if (rank3_shape[0] == rank4_shape[0] &&
      rank3_shape[1] == rank4_shape[2] &&
      rank3_shape[2] == rank4_shape[3] &&
      rank4_shape[1] > 1) {
    target_shape = {rank3_shape[0], 1, rank3_shape[1], rank3_shape[2]};
    return true;
  }

  return false;
}

bool CollectorEmitRank3ToRank4BroadcastReshape(ordered_json& ops_json,
                                               TensorState& input_state,
                                               const TensorState& other_state) {
  std::vector<int> target_shape;
  if (!CollectorRank3ToRank4BroadcastTargetShape(input_state.shape, other_state.shape, target_shape)) {
    return false;
  }

  ordered_json shape_values = ordered_json::array();
  for (int dim : target_shape) {
    shape_values.push_back(dim);
  }

  TensorState reshaped_state = input_state;
  reshaped_state.name += "_rank3_to_rank4_broadcast_" + std::to_string(ops_json.size());
  reshaped_state.shape = target_shape;
  reshaped_state.is_constant = false;
  reshaped_state.has_values = false;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(
      reshaped_state.name + "_shape", "int32",
      {static_cast<int>(target_shape.size())}, shape_values);
  payload["output_tensor_info"] = TensorInfoJson(reshaped_state);
  payload["skip_transpose_insert_before"] = true;
  ordered_json attributes;
  attributes["new_shape"] = shape_values;
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("reshape", std::move(payload)));

  input_state = std::move(reshaped_state);
  return true;
}

void CollectorPrepareRank3ToRank4BroadcastInputs(ordered_json& ops_json,
                                                 TensorState& input_1,
                                                 TensorState& input_2) {
  if (input_1.shape.size() == 3 && input_2.shape.size() == 4) {
    (void)CollectorEmitRank3ToRank4BroadcastReshape(ops_json, input_1, input_2);
  } else if (input_2.shape.size() == 3 && input_1.shape.size() == 4) {
    (void)CollectorEmitRank3ToRank4BroadcastReshape(ops_json, input_2, input_1);
  }
}

bool TryJsonToIntVector(const ordered_json& value, std::vector<int>& result) {
  if (!value.is_array()) {
    return false;
  }
  result.clear();
  for (const auto& item : value) {
    if (item.is_number_integer()) {
      result.push_back(item.get<int>());
    } else if (item.is_number()) {
      result.push_back(static_cast<int>(item.get<float>()));
    } else {
      return false;
    }
  }
  return true;
}

std::vector<int> ResolveReshapeShape(const std::vector<int>& input_shape,
                                     const std::vector<int>& requested_shape,
                                     bool allow_zero) {
  std::vector<int> result = requested_shape;
  int infer_index = -1;
  int64_t known_product = 1;
  const int64_t input_product = static_cast<int64_t>(ElementCount(input_shape));

  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i] == 0 && !allow_zero && i < input_shape.size()) {
      result[i] = input_shape[i];
    }
    if (result[i] == -1) {
      infer_index = static_cast<int>(i);
      continue;
    }
    if (result[i] > 0) {
      known_product *= result[i];
    }
  }

  if (infer_index >= 0 && input_product > 0 && known_product > 0) {
    result[static_cast<size_t>(infer_index)] = static_cast<int>(input_product / known_product);
  }
  return result;
}

Status EmitReferenceQLinearConvInitial(const GraphViewer& graph_viewer,
                                       const Node& node,
                                       std::unordered_map<std::string, TensorState>& states,
                                       ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 8 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        inputs[3] != nullptr && inputs[4] != nullptr && inputs[5] != nullptr &&
                        inputs[6] != nullptr && inputs[7] != nullptr && outputs[0] != nullptr,
                    "Invalid QLinearConv node.");

  TensorState input_state;
  ORT_RETURN_IF_ERROR(MakeQLinearConvInputState(graph_viewer, node, states, input_state));

  TensorState weight_state;
  ORT_RETURN_IF_ERROR(MakeWeightTensorState(graph_viewer, inputs[3]->Name(), inputs[4]->Name(),
                                            inputs[5]->Name(), weight_state));

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

  const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
  const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});

  ordered_json attributes;
  attributes["dilation_h_factor"] = dilations.empty() ? 1 : dilations[0];
  attributes["dilation_w_factor"] = dilations.size() > 1 ? dilations[1] : 1;
  attributes["fused_activation_function"] = "None";
  attributes["padding"] = ReferenceConvPaddingModeWithExplicitPads(pads);
  AttachExplicitConvPadsIfNeeded(attributes, pads);
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

Status EmitReferencePadInitial(const GraphViewer& graph_viewer,
                               const Node& node,
                               std::unordered_map<std::string, TensorState>& states,
                               ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Pad node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  std::vector<int> pads = GetIntsAttribute(node, "pads", {});
  if (pads.empty() && inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), pads));
  }

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = ApplyNhwcPads(input_state.shape, NhwcPadsForShapeFromOnnxPads(pads));
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  const std::string mode = GetStringAttribute(node, "mode", "constant");
  const bool mirror_pad = mode == "reflect" || mode == "symmetric";
  const std::string op_type = mirror_pad ? "MirrorPad" : "Pad";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "_pad_values", "int32",
                                                          {4, 2}, NhwcPadValuesFromOnnxPads(pads));
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  if (mirror_pad) {
    ordered_json attributes;
    attributes["mode"] = mode == "symmetric" ? 1 : 0;
    payload["attributes_info"] = std::move(attributes);
  }
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

Status EmitReferenceGroupedConvInitial(const GraphViewer& graph_viewer,
                                       const Node& node,
                                       int group,
                                       const std::vector<int>& source_weight_shape,
                                       std::unordered_map<std::string, TensorState>& states,
                                       ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid grouped Conv node.");
  ORT_RETURN_IF_NOT(source_weight_shape.size() == 4,
                    "Grouped Conv weight must be OIHW rank 4. Node: ", node.Name());
  ORT_RETURN_IF_NOT(group > 1 &&
                        source_weight_shape[0] % group == 0,
                    "Grouped Conv output channels must be divisible by group. Node: ", node.Name());

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;
  ORT_RETURN_IF_NOT(input_state.shape.size() == 4,
                    "Grouped Conv lowering requires a rank-4 NHWC input. Node: ", node.Name());

  const int in_per_group = source_weight_shape[1];
  const int out_per_group = source_weight_shape[0] / group;
  const int input_channels = input_state.shape[3];
  ORT_RETURN_IF_NOT(input_channels == in_per_group * group,
                    "Grouped Conv input channel mismatch. Node: ", node.Name());

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
    const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
    const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});
    const int kernel_h = source_weight_shape[2];
    const int kernel_w = source_weight_shape[3];
    const int stride_h = strides.empty() ? 1 : strides[0];
    const int stride_w = strides.size() > 1 ? strides[1] : 1;
    const int dilation_h = dilations.empty() ? 1 : dilations[0];
    const int dilation_w = dilations.size() > 1 ? dilations[1] : 1;
    const int pad_top = pads.empty() ? 0 : pads[0];
    const int pad_left = pads.size() > 1 ? pads[1] : 0;
    const int pad_bottom = pads.size() > 2 ? pads[2] : pad_top;
    const int pad_right = pads.size() > 3 ? pads[3] : pad_left;
    const int out_h = (input_state.shape[1] + pad_top + pad_bottom -
                       dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (input_state.shape[2] + pad_left + pad_right -
                       dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    output_state.shape = {input_state.shape[0], out_h, out_w, source_weight_shape[0]};
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json split_values = ordered_json::array();
  for (int i = 0; i < group; ++i) {
    split_values.push_back(in_per_group);
  }
  ordered_json split_axis = ordered_json::array({3});

  std::vector<TensorState> split_states;
  split_states.reserve(static_cast<size_t>(group));
  ordered_json split_payload;
  split_payload["op_type_info"] = OpTypeInfo("Split");
  split_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  split_payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "_group_split_values",
                                                                "int32", {group}, split_values);
  split_payload["input_tensor_info_3"] = ConstantTensorInfoJson(input_state.name + "_group_split_axis",
                                                                "int32", {1}, split_axis);
  for (int i = 0; i < group; ++i) {
    TensorState split_state = input_state;
    split_state.name = input_state.name + "_group_split_" + std::to_string(i);
    split_state.shape[3] = in_per_group;
    split_state.is_constant = false;
    split_state.has_values = false;
    MarkFloatTensorNoQuant(split_state);
    split_states.push_back(split_state);
    split_payload[std::string("output_tensor_info_") + std::to_string(i + 1)] = TensorInfoJson(split_state);
  }
  ordered_json split_attributes;
  split_attributes["num_splits"] = group;
  split_payload["attributes_info"] = std::move(split_attributes);
  ops_json.push_back(WrapOp("Split", std::move(split_payload)));

  const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
  const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});
  const int kernel_h = source_weight_shape.size() > 2 ? source_weight_shape[2] : 1;
  const int kernel_w = source_weight_shape.size() > 3 ? source_weight_shape[3] : 1;
  const int stride_h = strides.empty() ? 1 : strides[0];
  const int stride_w = strides.size() > 1 ? strides[1] : 1;
  const int dilation_h = dilations.empty() ? 1 : dilations[0];
  const int dilation_w = dilations.size() > 1 ? dilations[1] : 1;
  ordered_json conv_attributes;
  conv_attributes["dilation_h_factor"] = dilation_h;
  conv_attributes["dilation_w_factor"] = dilation_w;
  conv_attributes["fused_activation_function"] = "None";
  conv_attributes["padding"] = ReferenceConvPaddingModeForShapeWithExplicitPads(pads, input_state, output_state,
                                                                                kernel_h, kernel_w,
                                                                                stride_h, stride_w,
                                                                                dilation_h, dilation_w);
  AttachExplicitConvPadsIfNeeded(conv_attributes, pads);
  conv_attributes["stride_h"] = stride_h;
  conv_attributes["stride_w"] = stride_w;

  std::vector<TensorState> conv_outputs;
  conv_outputs.reserve(static_cast<size_t>(group));
  for (int i = 0; i < group; ++i) {
    TensorState weight_state;
    ORT_RETURN_IF_ERROR(MakeGroupedConvFloatWeightTensorState(graph_viewer, inputs[1]->Name(), i, group,
                                                              weight_state));

    TensorState bias_state;
    if (inputs.size() > 2 && inputs[2] != nullptr && inputs[2]->Exists()) {
      ORT_RETURN_IF_ERROR(MakeGroupedFloatBiasTensorState(graph_viewer, inputs[2]->Name(), i, group,
                                                          bias_state));
    } else {
      bias_state = MakeSyntheticFloatBiasTensorState(inputs[1]->Name() + "_group_" + std::to_string(i),
                                                     out_per_group);
    }

    TensorState group_output = output_state;
    group_output.name = output_state.name + "_group_conv_" + std::to_string(i);
    if (group_output.shape.size() == 4) {
      group_output.shape[3] = out_per_group;
    }
    group_output.is_constant = false;
    group_output.has_values = false;
    MarkFloatTensorNoQuant(group_output);
    conv_outputs.push_back(group_output);

    ordered_json conv_payload;
    conv_payload["op_type_info"] = OpTypeInfo("Conv");
    conv_payload["input_tensor_info"] = TensorInfoJson(split_states[static_cast<size_t>(i)]);
    conv_payload["input_weight_info"] = TensorInfoJson(weight_state);
    conv_payload["input_bias_info"] = TensorInfoJson(bias_state);
    conv_payload["output_tensor_info"] = TensorInfoJson(group_output);
    conv_payload["attributes_info"] = conv_attributes;
    ops_json.push_back(WrapOp("Conv", std::move(conv_payload)));
  }

  ordered_json concat_payload;
  concat_payload["op_type_info"] = OpTypeInfo("Concat");
  for (int i = 0; i < group; ++i) {
    concat_payload[std::string("input_tensor_info_") + std::to_string(i + 1)] =
        TensorInfoJson(conv_outputs[static_cast<size_t>(i)]);
  }
  concat_payload["output_tensor_info"] = TensorInfoJson(output_state);
  ordered_json concat_attributes;
  concat_attributes["axis"] = 3;
  concat_attributes["fused_activation_function"] = "None";
  concat_payload["attributes_info"] = std::move(concat_attributes);
  ops_json.push_back(WrapOp("Concat", std::move(concat_payload)));
  return Status::OK();
}

Status MakeConv1DFloatWeightTensorState(const GraphViewer& graph_viewer,
                                        const std::string& name,
                                        const std::vector<int>& source_weight_shape,
                                        TensorState& state) {
  ORT_RETURN_IF_NOT(source_weight_shape.size() == 3,
                    "Conv1D weight must be OIK rank 3: ", name);
  const int output_channels = source_weight_shape[0];
  const int input_channels = source_weight_shape[1];
  const int kernel_width = source_weight_shape[2];
  ORT_RETURN_IF_NOT(output_channels > 0 && input_channels > 0 && kernel_width > 0,
                    "Conv1D weight shape must be static positive: ", name);

  std::vector<float> source_values;
  ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, name, source_values));
  const size_t expected_size = static_cast<size_t>(output_channels) *
                               static_cast<size_t>(input_channels) *
                               static_cast<size_t>(kernel_width);
  ORT_RETURN_IF_NOT(source_values.size() == expected_size,
                    "Conv1D weight value count mismatch: ", name);

  std::vector<float> values(expected_size);
  for (int oc = 0; oc < output_channels; ++oc) {
    for (int kw = 0; kw < kernel_width; ++kw) {
      for (int ic = 0; ic < input_channels; ++ic) {
        const size_t source_index =
            (static_cast<size_t>(oc) * input_channels + ic) * kernel_width + kw;
        const size_t target_index =
            (static_cast<size_t>(oc) * kernel_width + kw) * input_channels + ic;
        values[target_index] = source_values[source_index];
      }
    }
  }

  state = MakeFloatConstantTensorState(name + "_conv1d_filter_4d",
                                       {output_channels, 1, kernel_width, input_channels},
                                       values);
  return Status::OK();
}

Status EmitReferenceConv1DInitial(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  const std::vector<int>& source_weight_shape,
                                  const TensorState& source_input_state,
                                  std::unordered_map<std::string, TensorState>& states,
                                  ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid Conv1D node.");
  ORT_RETURN_IF_NOT(source_input_state.shape.size() == 3 && source_weight_shape.size() == 3,
                    "Conv1D lowering requires rank-3 input and weight. Node: ", node.Name());

  const int batch = source_input_state.shape[0];
  const int input_channels = source_input_state.shape[1];
  const int input_length = source_input_state.shape[2];
  const int output_channels = source_weight_shape[0];
  const int weight_channels = source_weight_shape[1];
  const int kernel_width = source_weight_shape[2];
  ORT_RETURN_IF_NOT(batch > 0 && input_channels > 0 && input_length > 0 &&
                        output_channels > 0 && weight_channels > 0 && kernel_width > 0,
                    "Conv1D lowering requires static positive shapes. Node: ", node.Name());
  ORT_RETURN_IF_NOT(input_channels == weight_channels,
                    "Conv1D input channel mismatch. Node: ", node.Name());

  const std::vector<int> strides = GetIntsAttribute(node, "strides", {1});
  const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1});
  const int stride = strides.empty() ? 1 : strides[0];
  const int dilation = dilations.empty() ? 1 : dilations[0];
  ORT_RETURN_IF_NOT(stride > 0 && dilation > 0,
                    "Conv1D stride and dilation must be positive. Node: ", node.Name());
  const int effective_kernel = dilation * (kernel_width - 1) + 1;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  std::vector<int> pads = GetIntsAttribute(node, "pads", {});
  int pad_begin = 0;
  int pad_end = 0;
  if (pads.size() >= 2) {
    pad_begin = std::max(0, pads[0]);
    pad_end = std::max(0, pads[1]);
  } else if (pads.size() == 1) {
    pad_begin = std::max(0, pads[0]);
    pad_end = pad_begin;
  }

  const std::string auto_pad = GetStringAttribute(node, "auto_pad", "NOTSET");
  int output_length = -1;
  if (output_state.shape.size() == 3 && output_state.shape[2] > 0) {
    output_length = output_state.shape[2];
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    output_length = (input_length + stride - 1) / stride;
  }

  if (output_length > 0 && pad_begin == 0 && pad_end == 0) {
    const int total_pad = std::max(0, (output_length - 1) * stride +
                                      effective_kernel - input_length);
    if (auto_pad == "SAME_LOWER") {
      pad_begin = (total_pad + 1) / 2;
      pad_end = total_pad / 2;
    } else {
      pad_begin = total_pad / 2;
      pad_end = total_pad - pad_begin;
    }
  }

  if (output_length <= 0) {
    output_length = (input_length + pad_begin + pad_end - effective_kernel) / stride + 1;
  }
  ORT_RETURN_IF_NOT(output_length > 0,
                    "Conv1D computed non-positive output length. Node: ", node.Name());
  if (output_state.shape.empty()) {
    output_state.shape = {batch, output_channels, output_length};
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  TensorState input_state = source_input_state;
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState weight_state;
  ORT_RETURN_IF_ERROR(MakeConv1DFloatWeightTensorState(graph_viewer, inputs[1]->Name(),
                                                       source_weight_shape, weight_state));
  TensorState bias_state;
  if (inputs.size() > 2 && inputs[2] != nullptr && inputs[2]->Exists()) {
    ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[2]->Name(), false, bias_state));
  } else {
    bias_state = MakeSyntheticFloatBiasTensorState(inputs[1]->Name(), output_channels);
  }

  const std::string base_name = ReferenceTensorName(outputs[0]->Name());
  TensorState input_nlc = input_state;
  input_nlc.name = base_name + "_conv1d_input_nlc";
  input_nlc.shape = {batch, input_length, input_channels};
  input_nlc.is_constant = false;
  input_nlc.has_values = false;
  MarkFloatTensorNoQuant(input_nlc);

  ordered_json input_perm_values = ordered_json::array({0, 2, 1});
  ordered_json input_transpose_payload;
  input_transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
  input_transpose_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  input_transpose_payload["input_tensor_info_2"] =
      ConstantTensorInfoJson(input_nlc.name + "_perm", "int32", {3}, input_perm_values);
  input_transpose_payload["output_tensor_info"] = TensorInfoJson(input_nlc);
  ops_json.push_back(WrapOp("Transpose", std::move(input_transpose_payload)));

  TensorState input_4d = input_state;
  input_4d.name = base_name + "_conv1d_input_4d";
  input_4d.shape = {batch, 1, input_length, input_channels};
  input_4d.is_constant = false;
  input_4d.has_values = false;
  MarkFloatTensorNoQuant(input_4d);

  ordered_json input_shape_values = ordered_json::array();
  for (int dim : input_4d.shape) {
    input_shape_values.push_back(dim);
  }
  ordered_json input_reshape_payload;
  input_reshape_payload["op_type_info"] = OpTypeInfo("reshape");
  input_reshape_payload["skip_transpose_insert_after"] = true;
  input_reshape_payload["input_tensor_info_1"] = TensorInfoJson(input_nlc);
  input_reshape_payload["input_tensor_info_2"] =
      ConstantTensorInfoJson(input_4d.name + "_shape", "int32",
                             {static_cast<int>(input_4d.shape.size())}, input_shape_values);
  input_reshape_payload["output_tensor_info"] = TensorInfoJson(input_4d);
  ops_json.push_back(WrapOp("reshape", std::move(input_reshape_payload)));

  TensorState conv_input = input_4d;
  if (pad_begin > 0 || pad_end > 0) {
    TensorState padded_input = input_4d;
    padded_input.name = base_name + "_conv1d_padded";
    padded_input.shape = {batch, 1, input_length + pad_begin + pad_end, input_channels};
    padded_input.is_constant = false;
    padded_input.has_values = false;
    MarkFloatTensorNoQuant(padded_input);

    ordered_json pad_values = ordered_json::array({0, 0, 0, 0, pad_begin, pad_end, 0, 0});
    ordered_json pad_payload;
    pad_payload["op_type_info"] = OpTypeInfo("Pad");
    pad_payload["input_tensor_info_1"] = TensorInfoJson(input_4d);
    pad_payload["input_tensor_info_2"] =
        ConstantTensorInfoJson(input_4d.name + "_pad_values", "int32", {4, 2}, pad_values);
    pad_payload["output_tensor_info"] = TensorInfoJson(padded_input);
    ops_json.push_back(WrapOp("Pad", std::move(pad_payload)));
    conv_input = padded_input;
  }

  TensorState conv_output = output_state;
  conv_output.name = base_name + "_conv1d_output_4d";
  conv_output.shape = {batch, 1, output_length, output_channels};
  conv_output.is_constant = false;
  conv_output.has_values = false;
  MarkFloatTensorNoQuant(conv_output);

  ordered_json conv_attributes;
  conv_attributes["dilation_h_factor"] = 1;
  conv_attributes["dilation_w_factor"] = dilation;
  conv_attributes["fused_activation_function"] = "None";
  conv_attributes["padding"] = "VALID";
  conv_attributes["stride_h"] = 1;
  conv_attributes["stride_w"] = stride;

  ordered_json conv_payload;
  conv_payload["op_type_info"] = OpTypeInfo("Conv");
  conv_payload["input_tensor_info"] = TensorInfoJson(conv_input);
  conv_payload["input_weight_info"] = TensorInfoJson(weight_state);
  conv_payload["input_bias_info"] = TensorInfoJson(bias_state);
  conv_payload["output_tensor_info"] = TensorInfoJson(conv_output);
  conv_payload["attributes_info"] = std::move(conv_attributes);
  ops_json.push_back(WrapOp("Conv", std::move(conv_payload)));

  TensorState output_nlo = output_state;
  output_nlo.name = base_name + "_conv1d_output_nlo";
  output_nlo.shape = {batch, output_length, output_channels};
  output_nlo.is_constant = false;
  output_nlo.has_values = false;
  MarkFloatTensorNoQuant(output_nlo);

  ordered_json output_nlo_shape_values = ordered_json::array();
  for (int dim : output_nlo.shape) {
    output_nlo_shape_values.push_back(dim);
  }
  ordered_json output_reshape_payload;
  output_reshape_payload["op_type_info"] = OpTypeInfo("reshape");
  output_reshape_payload["input_tensor_info_1"] = TensorInfoJson(conv_output);
  output_reshape_payload["input_tensor_info_2"] =
      ConstantTensorInfoJson(output_nlo.name + "_shape", "int32",
                             {static_cast<int>(output_nlo.shape.size())}, output_nlo_shape_values);
  output_reshape_payload["output_tensor_info"] = TensorInfoJson(output_nlo);
  ops_json.push_back(WrapOp("reshape", std::move(output_reshape_payload)));

  ordered_json output_perm_values = ordered_json::array({0, 2, 1});
  ordered_json output_transpose_payload;
  output_transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
  output_transpose_payload["input_tensor_info_1"] = TensorInfoJson(output_nlo);
  output_transpose_payload["input_tensor_info_2"] =
      ConstantTensorInfoJson(output_state.name + "_perm", "int32", {3}, output_perm_values);
  output_transpose_payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("Transpose", std::move(output_transpose_payload)));
  return Status::OK();
}

Status EmitReferenceConvInitial(const GraphViewer& graph_viewer,
                                const Node& node,
                                std::unordered_map<std::string, TensorState>& states,
                                ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid Conv node.");

  const int group = GetIntAttribute(node, "group", 1);
  const auto* weight_tensor = graph_viewer.GetConstantInitializer(inputs[1]->Name(), true);
  ORT_RETURN_IF_NOT(weight_tensor != nullptr, "Missing Conv weight initializer: ", inputs[1]->Name());
  const std::vector<int> source_weight_shape = TensorProtoShape(*weight_tensor);
  const bool is_depthwise = group > 1 &&
                            source_weight_shape.size() == 4 &&
                            source_weight_shape[1] == 1 &&
                            source_weight_shape[0] % group == 0;
  if (group > 1 && !is_depthwise) {
    return EmitReferenceGroupedConvInitial(graph_viewer, node, group, source_weight_shape, states, ops_json);
  }
  ORT_RETURN_IF_NOT(group == 1 || is_depthwise,
                    "Amlogic reference_style Conv supports group=1 or depthwise Conv only. Node: ",
                    node.Name());

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  if (input_state.tensor_type == "float32") {
    input_state.has_quant = false;
  }
  states[inputs[0]->Name()] = input_state;

  if (group == 1 && source_weight_shape.size() == 3 && input_state.shape.size() == 3) {
    return EmitReferenceConv1DInitial(graph_viewer, node, source_weight_shape, input_state, states, ops_json);
  }

  TensorState weight_state;
  if (is_depthwise) {
    ORT_RETURN_IF_ERROR(MakeDepthwiseFloatWeightTensorState(graph_viewer, inputs[1]->Name(), weight_state));
  } else {
    ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[1]->Name(), true, weight_state));
  }

  TensorState bias_state;
  if (inputs.size() > 2 && inputs[2] != nullptr && inputs[2]->Exists()) {
    ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[2]->Name(), false, bias_state));
  } else {
    const int output_channels = is_depthwise
                                    ? source_weight_shape[0]
                                    : (weight_state.shape.empty() ? 1 : weight_state.shape[0]);
    bias_state = MakeSyntheticFloatBiasTensorState(inputs[1]->Name(), output_channels);
  }

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty() && input_state.shape.size() == 4 && source_weight_shape.size() == 4) {
    const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
    const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
    const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});
    const int kernel_h = source_weight_shape[2];
    const int kernel_w = source_weight_shape[3];
    const int stride_h = strides.empty() ? 1 : strides[0];
    const int stride_w = strides.size() > 1 ? strides[1] : 1;
    const int dilation_h = dilations.empty() ? 1 : dilations[0];
    const int dilation_w = dilations.size() > 1 ? dilations[1] : 1;
    const int pad_top = pads.empty() ? 0 : pads[0];
    const int pad_left = pads.size() > 1 ? pads[1] : 0;
    const int pad_bottom = pads.size() > 2 ? pads[2] : pad_top;
    const int pad_right = pads.size() > 3 ? pads[3] : pad_left;
    const int out_h = (input_state.shape[1] + pad_top + pad_bottom -
                       dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_w = (input_state.shape[2] + pad_left + pad_right -
                       dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    output_state.shape = {input_state.shape[0], out_h, out_w, source_weight_shape[0]};
  }
  output_state.has_quant = false;
  states[outputs[0]->Name()] = output_state;

  const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
  const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});
  const int kernel_h = source_weight_shape.size() > 2 ? source_weight_shape[2] : 1;
  const int kernel_w = source_weight_shape.size() > 3 ? source_weight_shape[3] : 1;
  const int stride_h = strides.empty() ? 1 : strides[0];
  const int stride_w = strides.size() > 1 ? strides[1] : 1;
  const int dilation_h = dilations.empty() ? 1 : dilations[0];
  const int dilation_w = dilations.size() > 1 ? dilations[1] : 1;

  ordered_json attributes;
  attributes["dilation_h_factor"] = dilation_h;
  attributes["dilation_w_factor"] = dilation_w;
  attributes["fused_activation_function"] = "None";
  attributes["padding"] = ReferenceConvPaddingModeForShapeWithExplicitPads(pads, input_state, output_state,
                                                                           kernel_h, kernel_w,
                                                                           stride_h, stride_w,
                                                                           dilation_h, dilation_w);
  AttachExplicitConvPadsIfNeeded(attributes, pads);
  attributes["stride_h"] = stride_h;
  attributes["stride_w"] = stride_w;
  if (is_depthwise) {
    attributes["depth_multiplier"] = group == 0 ? 1 : source_weight_shape[0] / group;
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(is_depthwise ? "DepthwiseConv" : "Conv");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["input_weight_info"] = TensorInfoJson(weight_state);
  payload["input_bias_info"] = TensorInfoJson(bias_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp(is_depthwise ? "DepthwiseConv" : "Conv", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceConvTransposeInitial(const GraphViewer& graph_viewer,
                                         const Node& node,
                                         std::unordered_map<std::string, TensorState>& states,
                                         ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid ConvTranspose node.");
  const int group = GetIntAttribute(node, "group", 1);
  ORT_RETURN_IF_NOT(group == 1,
                    "Amlogic reference_style ConvTranspose supports group=1 only. Node: ", node.Name());

  const auto* weight_tensor = graph_viewer.GetConstantInitializer(inputs[1]->Name(), true);
  ORT_RETURN_IF_NOT(weight_tensor != nullptr, "Missing ConvTranspose weight initializer: ", inputs[1]->Name());
  const std::vector<int> source_weight_shape = TensorProtoShape(*weight_tensor);
  ORT_RETURN_IF_NOT(source_weight_shape.size() == 4,
                    "ConvTranspose weight must be rank 4 for reference_style export: ", inputs[1]->Name());

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState weight_state;
  ORT_RETURN_IF_ERROR(MakeDepthwiseFloatWeightTensorState(graph_viewer, inputs[1]->Name(), weight_state));

  const int output_channels = source_weight_shape[1] * group;
  TensorState bias_state;
  if (inputs.size() > 2 && inputs[2] != nullptr && inputs[2]->Exists()) {
    ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[2]->Name(), false, bias_state));
  } else {
    bias_state = MakeSyntheticFloatBiasTensorState(inputs[1]->Name(), output_channels);
  }

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty() && input_state.shape.size() == 4) {
    const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
    const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
    const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});
    const std::vector<int> output_padding = GetIntsAttribute(node, "output_padding", {0, 0});
    const int kernel_h = source_weight_shape[2];
    const int kernel_w = source_weight_shape[3];
    const int stride_h = strides.empty() ? 1 : strides[0];
    const int stride_w = strides.size() > 1 ? strides[1] : 1;
    const int dilation_h = dilations.empty() ? 1 : dilations[0];
    const int dilation_w = dilations.size() > 1 ? dilations[1] : 1;
    const int pad_top = pads.empty() ? 0 : pads[0];
    const int pad_left = pads.size() > 1 ? pads[1] : 0;
    const int pad_bottom = pads.size() > 2 ? pads[2] : pad_top;
    const int pad_right = pads.size() > 3 ? pads[3] : pad_left;
    const int out_pad_h = output_padding.empty() ? 0 : output_padding[0];
    const int out_pad_w = output_padding.size() > 1 ? output_padding[1] : 0;
    const int out_h = stride_h * (input_state.shape[1] - 1) + out_pad_h +
                      dilation_h * (kernel_h - 1) + 1 - pad_top - pad_bottom;
    const int out_w = stride_w * (input_state.shape[2] - 1) + out_pad_w +
                      dilation_w * (kernel_w - 1) + 1 - pad_left - pad_right;
    output_state.shape = {input_state.shape[0], out_h, out_w, output_channels};
  }
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
  const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});

  ordered_json attributes;
  attributes["dilation_h_factor"] = dilations.empty() ? 1 : dilations[0];
  attributes["dilation_w_factor"] = dilations.size() > 1 ? dilations[1] : 1;
  attributes["fused_activation_function"] = "None";
  std::string transpose_padding = ReferenceConvPaddingMode(pads);
  if (transpose_padding == "add_pad_op1") {
    transpose_padding = "SAME";
  }
  attributes["padding"] = transpose_padding;
  attributes["stride_h"] = strides.empty() ? 1 : strides[0];
  attributes["stride_w"] = strides.size() > 1 ? strides[1] : 1;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("ConvTranspose");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["input_weight_info"] = TensorInfoJson(weight_state);
  payload["input_bias_info"] = TensorInfoJson(bias_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("ConvTranspose", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceQuantizeLinearInitial(const GraphViewer& graph_viewer,
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

Status EmitReferenceQLinearUnaryInitial(const GraphViewer& graph_viewer,
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
  if (op_type == "QLinearSigmoid") {
    // ONNX QLinearSigmoid accepts its own output scale/zero point, while a
    // quantized TFLite LOGISTIC requires the fixed int8 pair (1/256, -128).
    // Lower through float so the requested ONNX output quantization remains
    // exact for downstream QLinear arithmetic.
    TensorState float_input = input_state;
    float_input.name = ReferenceTensorName(outputs[0]->Name()) + "_sigmoid_dequantized";
    float_input.tensor_type = "float32";
    float_input.is_constant = false;
    float_input.has_values = false;
    MarkFloatTensorNoQuant(float_input);

    TensorState float_output = float_input;
    float_output.name = ReferenceTensorName(outputs[0]->Name()) + "_sigmoid_float";

    ordered_json dequantize_payload;
    dequantize_payload["op_type_info"] = OpTypeInfo("DequantizeLinear");
    dequantize_payload["input_tensor_info"] = TensorInfoJson(input_state);
    dequantize_payload["output_tensor_info"] = TensorInfoJson(float_input);
    ops_json.push_back(WrapOp("DequantizeLinear", std::move(dequantize_payload)));

    ordered_json sigmoid_payload;
    sigmoid_payload["op_type_info"] = OpTypeInfo("Sigmoid");
    sigmoid_payload["input_tensor_info"] = TensorInfoJson(float_input);
    sigmoid_payload["output_tensor_info"] = TensorInfoJson(float_output);
    ops_json.push_back(WrapOp("Sigmoid", std::move(sigmoid_payload)));

    ordered_json quantize_payload;
    quantize_payload["op_type_info"] = OpTypeInfo("QuantizeLinear");
    quantize_payload["input_tensor_info"] = TensorInfoJson(float_output);
    quantize_payload["output_tensor_info"] = TensorInfoJson(output_state);
    ops_json.push_back(WrapOp("QuantizeLinear", std::move(quantize_payload)));

    states[outputs[0]->Name()] = output_state;
    return Status::OK();
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  if (op_type == "QLinearSoftmax") {
    ordered_json attributes;
    attributes["beta"] = 1;
    payload["attributes_info"] = std::move(attributes);
  }
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

Status EmitReferenceReluInitial(const GraphViewer& graph_viewer,
                                const Node& node,
                                std::unordered_map<std::string, TensorState>& states,
                                ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Relu node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = input_state.shape;
  if (output_state.shape.empty()) {
    output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Relu");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("Relu", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceClipInitial(const GraphViewer& graph_viewer,
                                const Node& node,
                                std::unordered_map<std::string, TensorState>& states,
                                ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Clip node.");

  const auto& clip_attributes = node.GetAttributes();
  const bool has_min_attr = clip_attributes.find("min") != clip_attributes.end();
  const bool has_max_attr = clip_attributes.find("max") != clip_attributes.end();
  bool has_min_max = has_min_attr && has_max_attr;
  float min_value = GetFloatAttribute(node, "min", 0.0f);
  float max_value = GetFloatAttribute(node, "max", 6.0f);
  if (inputs.size() >= 3 && inputs[1] != nullptr && inputs[2] != nullptr &&
      inputs[1]->Exists() && inputs[2]->Exists()) {
    std::vector<float> min_values;
    std::vector<float> max_values;
    ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, inputs[1]->Name(), min_values));
    ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, inputs[2]->Name(), max_values));
    ORT_RETURN_IF_NOT(min_values.size() == 1 && max_values.size() == 1,
                      "Amlogic reference_style Clip min/max must be scalar initializers. Node: ",
                      node.Name());
    min_value = min_values[0];
    max_value = max_values[0];
    has_min_max = true;
  }
  ORT_RETURN_IF_NOT(has_min_max && min_value == 0.0f && max_value == 6.0f,
                    "Amlogic reference_style Clip currently maps only Clip(0, 6) to TFLite RELU6. Node: ",
                    node.Name());

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  if (input_state.tensor_type == "float32") {
    input_state.has_quant = false;
  }
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  } else {
    PreferPlainRank4OutputShape(output_state, input_state.shape);
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  if (output_state.tensor_type == "float32") {
    output_state.has_quant = false;
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Clip");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("Clip", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceQLinearBinaryInitial(const GraphViewer& graph_viewer,
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

Status EmitReferenceFloatUnaryInitial(const GraphViewer& graph_viewer,
                                      const Node& node,
                                      const std::string& op_type,
                                      std::unordered_map<std::string, TensorState>& states,
                                      ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid float unary node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  } else {
    PreferPlainRank4OutputShape(output_state, input_state.shape);
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  if (op_type == "Softmax") {
    ordered_json attributes;
    attributes["beta"] = 1;
    payload["attributes_info"] = std::move(attributes);
  }
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

// ONNX HardSigmoid is clip(alpha * x + beta, 0, 1), not the logistic
// sigmoid. Lower it to basic float operators because TFLite has no distinct
// HARD_SIGMOID builtin. The refactorer represents Clip as TFLite RELU6, so
// use the exact identity clip(s, 0, 1) = RELU6(6 * s) / 6.
Status EmitReferenceHardSigmoidInitial(const GraphViewer& graph_viewer,
                                       const Node& node,
                                       std::unordered_map<std::string, TensorState>& states,
                                       ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid HardSigmoid node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  } else {
    PreferPlainRank4OutputShape(output_state, input_state.shape);
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  const std::string base = std::string(outputs[0]->Name()) + "_hardsigmoid";
  TensorState alpha_state = MakeFloatConstantTensorState(base + "_alpha", {1},
                                                         {GetFloatAttribute(node, "alpha", 0.2f)});
  TensorState beta_state = MakeFloatConstantTensorState(base + "_beta", {1},
                                                        {GetFloatAttribute(node, "beta", 0.5f)});
  TensorState six_state = MakeFloatConstantTensorState(base + "_six", {1}, {6.0f});
  TensorState one_sixth_state = MakeFloatConstantTensorState(base + "_one_sixth", {1}, {1.0f / 6.0f});
  TensorState scaled_state = output_state;
  scaled_state.name = ReferenceTensorName(base + "_scaled");
  scaled_state.is_constant = false;
  scaled_state.has_values = false;
  MarkFloatTensorNoQuant(scaled_state);
  TensorState affine_state = scaled_state;
  affine_state.name = ReferenceTensorName(base + "_affine");
  TensorState relu6_input_state = scaled_state;
  relu6_input_state.name = ReferenceTensorName(base + "_relu6_input");
  TensorState relu6_output_state = scaled_state;
  relu6_output_state.name = ReferenceTensorName(base + "_relu6_output");

  PushReferenceBinaryOp(ops_json, "Mul", input_state, alpha_state, scaled_state);
  PushReferenceBinaryOp(ops_json, "Add", scaled_state, beta_state, affine_state);
  PushReferenceBinaryOp(ops_json, "Mul", affine_state, six_state, relu6_input_state);

  ordered_json attributes;
  attributes["min"] = 0.0f;
  attributes["max"] = 6.0f;
  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Clip");
  payload["input_tensor_info"] = TensorInfoJson(relu6_input_state);
  payload["output_tensor_info"] = TensorInfoJson(relu6_output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Clip", std::move(payload)));
  PushReferenceBinaryOp(ops_json, "Mul", relu6_output_state, one_sixth_state, output_state);
  return Status::OK();
}

Status EmitReferenceSoftmaxInitial(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   std::unordered_map<std::string, TensorState>& states,
                                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Softmax node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = input_state.shape;
  if (output_state.shape.empty()) {
    output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  const size_t rank = input_state.shape.size();
  int axis = GetIntAttribute(node, "axis", -1);
  axis = NormalizeAxisForRank(axis, rank);
  const std::vector<int> onnx_input_shape = ShapeForTensor(graph_viewer, inputs[0]->Name(), false);
  if (rank == 4 &&
      onnx_input_shape.size() == rank &&
      input_state.shape == ToNhwcShape(onnx_input_shape) &&
      input_state.shape != onnx_input_shape &&
      static_cast<size_t>(axis) + 1 != rank) {
    axis = MapAxisToNhwc(axis, rank);
  }
  ORT_RETURN_IF_NOT(rank == 0 || (axis >= 0 && static_cast<size_t>(axis) < rank),
                    "Invalid Softmax axis: ", axis, " rank: ", rank, " node: ", node.Name());

  auto push_softmax = [&](const TensorState& input, const TensorState& output) {
    ordered_json attributes;
    attributes["beta"] = 1;

    ordered_json payload;
    payload["op_type_info"] = OpTypeInfo("Softmax");
    payload["input_tensor_info"] = TensorInfoJson(input);
    payload["output_tensor_info"] = TensorInfoJson(output);
    payload["attributes_info"] = std::move(attributes);
    ops_json.push_back(WrapOp("Softmax", std::move(payload)));
  };

  if (rank <= 1 || static_cast<size_t>(axis) + 1 == rank) {
    push_softmax(input_state, output_state);
    return Status::OK();
  }

  std::vector<int> axis_to_last_perm;
  axis_to_last_perm.reserve(rank);
  for (size_t dim = 0; dim < rank; ++dim) {
    if (dim != static_cast<size_t>(axis)) {
      axis_to_last_perm.push_back(static_cast<int>(dim));
    }
  }
  axis_to_last_perm.push_back(axis);

  std::vector<int> last_to_axis_perm(rank, 0);
  for (size_t dim = 0; dim < rank; ++dim) {
    last_to_axis_perm[static_cast<size_t>(axis_to_last_perm[dim])] = static_cast<int>(dim);
  }

  auto make_perm_values = [](const std::vector<int>& perm) {
    ordered_json values = ordered_json::array();
    for (int value : perm) {
      values.push_back(value);
    }
    return values;
  };

  TensorState softmax_input = input_state;
  softmax_input.name = output_state.name + "_softmax_axis_to_last";
  softmax_input.shape = ApplyPermToShape(input_state.shape, axis_to_last_perm);
  softmax_input.is_constant = false;
  softmax_input.has_values = false;
  MarkFloatTensorNoQuant(softmax_input);

  TensorState softmax_output = softmax_input;
  softmax_output.name = output_state.name + "_softmax_axis_last_output";
  softmax_output.is_constant = false;
  softmax_output.has_values = false;
  MarkFloatTensorNoQuant(softmax_output);

  ordered_json to_last_payload;
  to_last_payload["op_type_info"] = OpTypeInfo("Transpose");
  to_last_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  to_last_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
      softmax_input.name + "_perm_values", "int32",
      {static_cast<int>(axis_to_last_perm.size())}, make_perm_values(axis_to_last_perm));
  to_last_payload["output_tensor_info"] = TensorInfoJson(softmax_input);
  to_last_payload["skip_transpose_insert_before"] = true;
  to_last_payload["skip_transpose_insert_after"] = true;
  ops_json.push_back(WrapOp("Transpose", std::move(to_last_payload)));

  push_softmax(softmax_input, softmax_output);

  ordered_json from_last_payload;
  from_last_payload["op_type_info"] = OpTypeInfo("Transpose");
  from_last_payload["input_tensor_info_1"] = TensorInfoJson(softmax_output);
  from_last_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
      output_state.name + "_softmax_axis_from_last_perm_values", "int32",
      {static_cast<int>(last_to_axis_perm.size())}, make_perm_values(last_to_axis_perm));
  from_last_payload["output_tensor_info"] = TensorInfoJson(output_state);
  from_last_payload["skip_transpose_insert_before"] = true;
  from_last_payload["skip_transpose_insert_after"] = true;
  ops_json.push_back(WrapOp("Transpose", std::move(from_last_payload)));

  return Status::OK();
}

Status EmitReferenceErfInitial(const GraphViewer& graph_viewer,
                               const Node& node,
                               std::unordered_map<std::string, TensorState>& states,
                               ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Erf node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState mul_output = input_state;
  mul_output.name = ReferenceTensorName(std::string(outputs[0]->Name()) + "_erf_tanh_input");
  mul_output.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (mul_output.shape.empty()) {
    mul_output.shape = input_state.shape;
  }
  mul_output.is_constant = false;
  mul_output.has_values = false;
  MarkFloatTensorNoQuant(mul_output);

  TensorState scale_state = MakeFloatConstantTensorState(
      std::string(outputs[0]->Name()) + "_erf_tanh_scale", {1}, {1.128379167f});

  ordered_json mul_attributes;
  mul_attributes["fused_activation_function"] = "None";
  ordered_json mul_payload;
  mul_payload["op_type_info"] = OpTypeInfo("Mul");
  mul_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  mul_payload["input_tensor_info_2"] = TensorInfoJson(scale_state);
  mul_payload["output_tensor_info"] = TensorInfoJson(mul_output);
  mul_payload["attributes_info"] = std::move(mul_attributes);
  ops_json.push_back(WrapOp("Mul", std::move(mul_payload)));

  TensorState output_state = mul_output;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = mul_output.shape;
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json tanh_payload;
  tanh_payload["op_type_info"] = OpTypeInfo("Tanh");
  tanh_payload["input_tensor_info"] = TensorInfoJson(mul_output);
  tanh_payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("Tanh", std::move(tanh_payload)));
  return Status::OK();
}

Status EmitReferenceIdentityInitial(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    std::unordered_map<std::string, TensorState>& states) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Identity node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;
  states[outputs[0]->Name()] = input_state;
  return Status::OK();
}

Status EmitReferencePReluInitial(const GraphViewer& graph_viewer,
                                 const Node& node,
                                 std::unordered_map<std::string, TensorState>& states,
                                 ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid PRelu node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState slope_state;
  ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[1]->Name(), false, slope_state));
  MarkFloatTensorNoQuant(slope_state);

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("PRelu");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = TensorInfoJson(slope_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("PRelu", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceLeakyReluInitial(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     std::unordered_map<std::string, TensorState>& states,
                                     ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid LeakyRelu node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["alpha"] = GetFloatAttribute(node, "alpha", 0.01f);

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("LeakyRelu");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("LeakyRelu", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceFloatBinaryInitial(const GraphViewer& graph_viewer,
                                       const Node& node,
                                       const std::string& op_type,
                                       std::unordered_map<std::string, TensorState>& states,
                                       ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid float binary node.");

  auto ensure_input = [&](const NodeArg* input) -> Status {
    if (states.find(input->Name()) != states.end()) {
      MarkFloatTensorNoQuant(states[input->Name()]);
      return Status::OK();
    }

    const auto* initializer = graph_viewer.GetConstantInitializer(input->Name(), true);
    if (initializer != nullptr) {
      const std::vector<int> shape = TensorProtoShape(*initializer);
      ordered_json values;
      ORT_RETURN_IF_ERROR(TensorValuesJson(*initializer, graph_viewer.ModelPath(), shape, shape, nullptr,
                                           false, values));
      TensorState state;
      state.name = ReferenceTensorName(input->Name());
      state.tensor_type = "float32";
      state.shape = shape.empty() ? std::vector<int>{1} : shape;
      state.is_constant = true;
      state.has_values = true;
      state.values = std::move(values);
      state.has_quant = false;
      states[input->Name()] = std::move(state);
      return Status::OK();
    }

    TensorState state = LookupState(graph_viewer, states, input->Name(), true, "float32");
    MarkFloatTensorNoQuant(state);
    states[input->Name()] = std::move(state);
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(ensure_input(inputs[0]));
  ORT_RETURN_IF_ERROR(ensure_input(inputs[1]));

  if (states[inputs[0]->Name()].is_constant) {
    RetargetRank1ChannelConstantToRank4(states[inputs[0]->Name()], states[inputs[1]->Name()]);
    RetargetRank4NchwChannelConstantToRank4(states[inputs[0]->Name()], states[inputs[1]->Name()]);
    RetargetRank3BroadcastConstantToRank4(states[inputs[0]->Name()], states[inputs[1]->Name()]);
    RetargetRank2BroadcastConstantToRank4(states[inputs[0]->Name()], states[inputs[1]->Name()]);
  }
  if (states[inputs[1]->Name()].is_constant) {
    RetargetRank1ChannelConstantToRank4(states[inputs[1]->Name()], states[inputs[0]->Name()]);
    RetargetRank4NchwChannelConstantToRank4(states[inputs[1]->Name()], states[inputs[0]->Name()]);
    RetargetRank3BroadcastConstantToRank4(states[inputs[1]->Name()], states[inputs[0]->Name()]);
    RetargetRank2BroadcastConstantToRank4(states[inputs[1]->Name()], states[inputs[0]->Name()]);
  }

  TensorState input_1 = states[inputs[0]->Name()];
  TensorState input_2 = states[inputs[1]->Name()];
  CollectorPrepareRank3ToRank4BroadcastInputs(ops_json, input_1, input_2);

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  const std::vector<int> broadcast_shape =
      BroadcastShape(input_1.shape, input_2.shape);
  if (output_state.shape.empty()) {
    output_state.shape = broadcast_shape;
  } else {
    PreferPlainRank4OutputShape(output_state, broadcast_shape);
  }
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["fused_activation_function"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(op_type);
  payload["input_tensor_info_1"] = TensorInfoJson(input_1);
  payload["input_tensor_info_2"] = TensorInfoJson(input_2);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp(op_type, std::move(payload)));
  return Status::OK();
}

Status EmitReferenceMatMulInitial(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  std::unordered_map<std::string, TensorState>& states,
                                  ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid MatMul node.");

  TensorState input_1;
  ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, inputs[0], false, "float32",
                                                    states, input_1));
  MarkFloatTensorNoQuant(input_1);
  states[inputs[0]->Name()] = input_1;

  TensorState input_2;
  ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, inputs[1], false, "float32",
                                                    states, input_2));
  MarkFloatTensorNoQuant(input_2);
  states[inputs[1]->Name()] = input_2;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), false);
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  auto align_matmul_input_to_onnx_layout = [&](const NodeArg* input_arg,
                                               TensorState& input_state,
                                               const std::string& suffix) -> Status {
    const std::vector<int> onnx_shape = ShapeForTensor(graph_viewer, input_arg->Name(), false);
    if (input_state.is_constant || onnx_shape.size() != 4 ||
        input_state.shape != ToNhwcShape(onnx_shape) ||
        input_state.shape == onnx_shape) {
      return Status::OK();
    }

    TensorState transposed_input = input_state;
    transposed_input.name = input_state.name + suffix;
    transposed_input.shape = onnx_shape;
    transposed_input.is_constant = false;
    transposed_input.has_values = false;
    MarkFloatTensorNoQuant(transposed_input);

    ordered_json transpose_payload;
    transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
    transpose_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
    transpose_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        input_state.name + suffix + "_perm", "int32", {4}, ordered_json::array({0, 3, 1, 2}));
    transpose_payload["output_tensor_info"] = TensorInfoJson(transposed_input);
    ops_json.push_back(WrapOp("Transpose", std::move(transpose_payload)));

    input_state = transposed_input;
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(align_matmul_input_to_onnx_layout(inputs[0], input_1, "_matmul_lhs_nhwc_to_onnx"));
  ORT_RETURN_IF_ERROR(align_matmul_input_to_onnx_layout(inputs[1], input_2, "_matmul_rhs_nhwc_to_onnx"));

  if (input_1.shape.size() == 3 && input_2.shape.size() == 2 && output_state.shape.size() == 3 &&
      input_1.shape[0] == output_state.shape[0] &&
      input_1.shape[1] == output_state.shape[1] &&
      input_1.shape[2] == input_2.shape[0] &&
      input_2.shape[1] == output_state.shape[2] &&
      input_2.is_constant) {
    const int flat_rows = input_1.shape[0] * input_1.shape[1];
    const int input_cols = input_1.shape[2];
    const int output_cols = output_state.shape[2];

    TensorState flat_input = input_1;
    flat_input.name = input_1.name + "_matmul_flatten";
    flat_input.shape = {flat_rows, input_cols};
    flat_input.is_constant = false;
    flat_input.has_values = false;
    MarkFloatTensorNoQuant(flat_input);

    ordered_json flat_shape = ordered_json::array({flat_rows, input_cols});
    ordered_json flatten_payload;
    flatten_payload["op_type_info"] = OpTypeInfo("reshape");
    flatten_payload["input_tensor_info_1"] = TensorInfoJson(input_1);
    flatten_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        input_1.name + "_matmul_flatten_shape", "int32", {2}, flat_shape);
    flatten_payload["output_tensor_info"] = TensorInfoJson(flat_input);
    ordered_json flatten_attributes;
    flatten_attributes["new_shape"] = flat_shape;
    flatten_payload["attributes_info"] = std::move(flatten_attributes);
    ops_json.push_back(WrapOp("reshape", std::move(flatten_payload)));

    TensorState weight_state;
    ORT_RETURN_IF_ERROR(MakeGemmFloatWeightTensorState(graph_viewer, inputs[1]->Name(), 0, weight_state));
    TensorState bias_state = MakeSyntheticFloatBiasTensorState(inputs[1]->Name(), output_cols);

    TensorState flat_output = output_state;
    flat_output.name = ReferenceTensorName(std::string(outputs[0]->Name()) + "_matmul_flat_output");
    flat_output.shape = {flat_rows, output_cols};
    flat_output.is_constant = false;
    flat_output.has_values = false;
    MarkFloatTensorNoQuant(flat_output);

    ordered_json gemm_attributes;
    gemm_attributes["keep_num_dims"] = false;
    gemm_attributes["fused_activation_function"] = "None";
    ordered_json gemm_payload;
    gemm_payload["op_type_info"] = OpTypeInfo("Gemm");
    gemm_payload["input_tensor_info"] = TensorInfoJson(flat_input);
    gemm_payload["input_weight_info"] = TensorInfoJson(weight_state);
    gemm_payload["input_bias_info"] = TensorInfoJson(bias_state);
    gemm_payload["output_tensor_info"] = TensorInfoJson(flat_output);
    gemm_payload["attributes_info"] = std::move(gemm_attributes);
    ops_json.push_back(WrapOp("Gemm", std::move(gemm_payload)));

    ordered_json output_shape = ordered_json::array();
    for (int dim : output_state.shape) {
      output_shape.push_back(dim);
    }
    ordered_json output_reshape_payload;
    output_reshape_payload["op_type_info"] = OpTypeInfo("reshape");
    output_reshape_payload["input_tensor_info_1"] = TensorInfoJson(flat_output);
    output_reshape_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        output_state.name + "_matmul_output_shape", "int32",
        {static_cast<int>(output_state.shape.size())}, output_shape);
    output_reshape_payload["output_tensor_info"] = TensorInfoJson(output_state);
    ordered_json output_reshape_attributes;
    output_reshape_attributes["new_shape"] = output_shape;
    output_reshape_payload["attributes_info"] = std::move(output_reshape_attributes);
    ops_json.push_back(WrapOp("reshape", std::move(output_reshape_payload)));
    return Status::OK();
  }

  if (input_1.shape.size() == 4 && input_2.shape.size() == 4 && output_state.shape.size() == 4 &&
      input_1.shape[0] == output_state.shape[0] &&
      input_2.shape[0] == output_state.shape[0] &&
      input_1.shape[1] == output_state.shape[2] &&
      input_1.shape[2] == input_2.shape[1] &&
      input_1.shape[3] == output_state.shape[1] &&
      input_2.shape[2] == output_state.shape[3] &&
      input_2.shape[3] == output_state.shape[1]) {
    ordered_json perm_values = ordered_json::array({0, 3, 1, 2});

    TensorState transposed_input_1 = input_1;
    transposed_input_1.name = input_1.name + "_batch_matmul_lhs";
    transposed_input_1.shape = {input_1.shape[0], input_1.shape[3], input_1.shape[1], input_1.shape[2]};
    transposed_input_1.is_constant = false;
    transposed_input_1.has_values = false;
    MarkFloatTensorNoQuant(transposed_input_1);

    ordered_json lhs_transpose_payload;
    lhs_transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
    lhs_transpose_payload["input_tensor_info_1"] = TensorInfoJson(input_1);
    lhs_transpose_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        input_1.name + "_batch_matmul_perm", "int32", {4}, perm_values);
    lhs_transpose_payload["output_tensor_info"] = TensorInfoJson(transposed_input_1);
    ops_json.push_back(WrapOp("Transpose", std::move(lhs_transpose_payload)));

    TensorState transposed_input_2 = input_2;
    transposed_input_2.name = input_2.name + "_batch_matmul_rhs";
    transposed_input_2.shape = {input_2.shape[0], input_2.shape[3], input_2.shape[1], input_2.shape[2]};
    transposed_input_2.is_constant = false;
    transposed_input_2.has_values = false;
    MarkFloatTensorNoQuant(transposed_input_2);

    ordered_json rhs_transpose_payload;
    rhs_transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
    rhs_transpose_payload["input_tensor_info_1"] = TensorInfoJson(input_2);
    rhs_transpose_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        input_2.name + "_batch_matmul_perm", "int32", {4}, std::move(perm_values));
    rhs_transpose_payload["output_tensor_info"] = TensorInfoJson(transposed_input_2);
    ops_json.push_back(WrapOp("Transpose", std::move(rhs_transpose_payload)));

    input_1 = transposed_input_1;
    input_2 = transposed_input_2;
  }

  if (input_1.shape.size() == 4 && input_2.shape.size() == 4 && output_state.shape.size() == 4 &&
      input_1.shape[0] == output_state.shape[0] &&
      input_2.shape[0] == output_state.shape[0] &&
      input_1.shape[1] == output_state.shape[1] &&
      input_1.shape[2] == output_state.shape[2] &&
      input_2.shape[1] == input_1.shape[3] &&
      input_2.shape[2] == output_state.shape[3] &&
      input_2.shape[3] == output_state.shape[1]) {
    ordered_json perm_values = ordered_json::array({0, 3, 1, 2});

    TensorState transposed_input_2 = input_2;
    transposed_input_2.name = input_2.name + "_batch_matmul_rhs_nhwc";
    transposed_input_2.shape = {input_2.shape[0], input_2.shape[3], input_2.shape[1], input_2.shape[2]};
    transposed_input_2.is_constant = false;
    transposed_input_2.has_values = false;
    MarkFloatTensorNoQuant(transposed_input_2);

    ordered_json rhs_transpose_payload;
    rhs_transpose_payload["op_type_info"] = OpTypeInfo("Transpose");
    rhs_transpose_payload["input_tensor_info_1"] = TensorInfoJson(input_2);
    rhs_transpose_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        input_2.name + "_batch_matmul_rhs_nhwc_perm", "int32", {4}, perm_values);
    rhs_transpose_payload["output_tensor_info"] = TensorInfoJson(transposed_input_2);
    ops_json.push_back(WrapOp("Transpose", std::move(rhs_transpose_payload)));

    input_2 = transposed_input_2;
  }

  if (input_1.shape.size() == 4 && input_2.shape.size() == 4 && output_state.shape.size() == 4 &&
      input_1.shape[0] == 1 &&
      input_2.shape[0] == 1 &&
      output_state.shape[0] == 1 &&
      input_1.shape[1] == input_2.shape[1] &&
      input_1.shape[1] == output_state.shape[1] &&
      input_1.shape[2] == output_state.shape[2] &&
      input_1.shape[3] == input_2.shape[2] &&
      input_2.shape[3] == output_state.shape[3]) {
    const int heads = input_1.shape[1];
    const int lhs_rows = input_1.shape[2];
    const int inner_dim = input_1.shape[3];
    const int rhs_cols = input_2.shape[3];

    TensorState flat_input_1 = input_1;
    flat_input_1.name = input_1.name + "_batch_flatten_lhs";
    flat_input_1.shape = {heads, lhs_rows, inner_dim};
    flat_input_1.is_constant = false;
    flat_input_1.has_values = false;
    MarkFloatTensorNoQuant(flat_input_1);

    ordered_json lhs_flat_shape = ordered_json::array({heads, lhs_rows, inner_dim});
    ordered_json lhs_flatten_payload;
    lhs_flatten_payload["op_type_info"] = OpTypeInfo("reshape");
    lhs_flatten_payload["input_tensor_info_1"] = TensorInfoJson(input_1);
    lhs_flatten_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        flat_input_1.name + "_shape", "int32", {3}, lhs_flat_shape);
    lhs_flatten_payload["output_tensor_info"] = TensorInfoJson(flat_input_1);
    lhs_flatten_payload["skip_transpose_insert_before"] = true;
    ordered_json lhs_flatten_attributes;
    lhs_flatten_attributes["new_shape"] = lhs_flat_shape;
    lhs_flatten_payload["attributes_info"] = std::move(lhs_flatten_attributes);
    ops_json.push_back(WrapOp("reshape", std::move(lhs_flatten_payload)));

    TensorState flat_input_2 = input_2;
    flat_input_2.name = input_2.name + "_batch_flatten_rhs";
    flat_input_2.shape = {heads, inner_dim, rhs_cols};
    flat_input_2.is_constant = false;
    flat_input_2.has_values = false;
    MarkFloatTensorNoQuant(flat_input_2);

    ordered_json rhs_flat_shape = ordered_json::array({heads, inner_dim, rhs_cols});
    ordered_json rhs_flatten_payload;
    rhs_flatten_payload["op_type_info"] = OpTypeInfo("reshape");
    rhs_flatten_payload["input_tensor_info_1"] = TensorInfoJson(input_2);
    rhs_flatten_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        flat_input_2.name + "_shape", "int32", {3}, rhs_flat_shape);
    rhs_flatten_payload["output_tensor_info"] = TensorInfoJson(flat_input_2);
    rhs_flatten_payload["skip_transpose_insert_before"] = true;
    ordered_json rhs_flatten_attributes;
    rhs_flatten_attributes["new_shape"] = rhs_flat_shape;
    rhs_flatten_payload["attributes_info"] = std::move(rhs_flatten_attributes);
    ops_json.push_back(WrapOp("reshape", std::move(rhs_flatten_payload)));

    TensorState flat_output = output_state;
    flat_output.name = output_state.name + "_batch_flatten_output";
    flat_output.shape = {heads, lhs_rows, rhs_cols};
    flat_output.is_constant = false;
    flat_output.has_values = false;
    MarkFloatTensorNoQuant(flat_output);

    ordered_json attributes;
    attributes["adj_x"] = false;
    attributes["adj_y"] = false;
    attributes["asymmetric_quantize_inputs"] = false;

    ordered_json payload;
    payload["op_type_info"] = OpTypeInfo("MatMul");
    payload["input_tensor_info_1"] = TensorInfoJson(flat_input_1);
    payload["input_tensor_info_2"] = TensorInfoJson(flat_input_2);
    payload["output_tensor_info"] = TensorInfoJson(flat_output);
    payload["attributes_info"] = std::move(attributes);
    ops_json.push_back(WrapOp("MatMul", std::move(payload)));

    ordered_json output_shape = ordered_json::array(
        {output_state.shape[0], output_state.shape[1], output_state.shape[2], output_state.shape[3]});
    ordered_json output_reshape_payload;
    output_reshape_payload["op_type_info"] = OpTypeInfo("reshape");
    output_reshape_payload["input_tensor_info_1"] = TensorInfoJson(flat_output);
    output_reshape_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        output_state.name + "_batch_flatten_output_shape", "int32",
        {static_cast<int>(output_state.shape.size())}, output_shape);
    output_reshape_payload["output_tensor_info"] = TensorInfoJson(output_state);
    output_reshape_payload["skip_transpose_insert_before"] = true;
    ordered_json output_reshape_attributes;
    output_reshape_attributes["new_shape"] = output_shape;
    output_reshape_payload["attributes_info"] = std::move(output_reshape_attributes);
    ops_json.push_back(WrapOp("reshape", std::move(output_reshape_payload)));
    return Status::OK();
  }

  ordered_json attributes;
  attributes["adj_x"] = false;
  attributes["adj_y"] = false;
  attributes["asymmetric_quantize_inputs"] = false;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("MatMul");
  payload["input_tensor_info_1"] = TensorInfoJson(input_1);
  payload["input_tensor_info_2"] = TensorInfoJson(input_2);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("MatMul", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceAddInitial(const GraphViewer& graph_viewer,
                               const Node& node,
                               std::unordered_map<std::string, TensorState>& states,
                               ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid Add node.");

  TensorState input_1;
  ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, inputs[0], true, "float32",
                                                    states, input_1));
  TensorState input_2;
  ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, inputs[1], true, "float32",
                                                    states, input_2));
  MarkFloatTensorNoQuant(input_1);
  MarkFloatTensorNoQuant(input_2);
  states[inputs[0]->Name()] = input_1;
  states[inputs[1]->Name()] = input_2;

  if (input_1.is_constant && RetargetRank1ChannelConstantToRank4(input_1, input_2)) {
    states[inputs[0]->Name()] = input_1;
  }
  if (input_2.is_constant && RetargetRank1ChannelConstantToRank4(input_2, input_1)) {
    states[inputs[1]->Name()] = input_2;
  }

  if (input_1.is_constant && RetargetRank4NchwChannelConstantToRank4(input_1, input_2)) {
    states[inputs[0]->Name()] = input_1;
  }
  if (input_2.is_constant && RetargetRank4NchwChannelConstantToRank4(input_2, input_1)) {
    states[inputs[1]->Name()] = input_2;
  }

  if (input_1.is_constant && RetargetRank3BroadcastConstantToRank4(input_1, input_2)) {
    states[inputs[0]->Name()] = input_1;
  }
  if (input_2.is_constant && RetargetRank3BroadcastConstantToRank4(input_2, input_1)) {
    states[inputs[1]->Name()] = input_2;
  }

  if (input_1.is_constant && RetargetRank2BroadcastConstantToRank4(input_1, input_2)) {
    states[inputs[0]->Name()] = input_1;
  }
  if (input_2.is_constant && RetargetRank2BroadcastConstantToRank4(input_2, input_1)) {
    states[inputs[1]->Name()] = input_2;
  }

  CollectorPrepareRank3ToRank4BroadcastInputs(ops_json, input_1, input_2);

  const std::string fallback_type =
      input_1.tensor_type == input_2.tensor_type ? input_1.tensor_type : "float32";
  TensorState output_state = MakeReferenceOutputTensorState(graph_viewer, outputs[0], true, fallback_type);
  const std::vector<int> broadcast_shape = BroadcastShape(input_1.shape, input_2.shape);
  if (output_state.shape.empty()) {
    output_state.shape = broadcast_shape;
  } else {
    PreferPlainRank4OutputShape(output_state, broadcast_shape);
  }
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["fused_activation_function"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Add");
  payload["input_tensor_info_1"] = TensorInfoJson(input_1);
  payload["input_tensor_info_2"] = TensorInfoJson(input_2);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Add", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceMaxPoolInitial(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   std::unordered_map<std::string, TensorState>& states,
                                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid MaxPool node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true);
  if (const auto* producer_tensor =
          FindProducerTensorInfo(ops_json, input_state.name, static_cast<int>(ops_json.size()))) {
    const auto producer_shape = JsonShapeToVector(*producer_tensor);
    if (producer_shape.size() == 4) {
      input_state.shape = producer_shape;
    }
  }
  if (input_state.tensor_type == "float32") {
    MarkFloatTensorNoQuant(input_state);
  } else if (!input_state.has_quant) {
    input_state.has_quant = true;
    input_state.quant_scale = "None";
    input_state.zero_point = "None";
  }
  states[inputs[0]->Name()] = input_state;

  TensorState output_state;
  if (const auto* existing_output_state = FindExistingState(states, outputs[0]->Name())) {
    output_state = *existing_output_state;
  } else {
    output_state = input_state;
    output_state.name = ReferenceTensorName(outputs[0]->Name());
    output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
    output_state.is_constant = false;
    output_state.has_values = false;
  }
  const auto pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const auto strides = GetIntsAttribute(node, "strides", {1, 1});
  const auto kernel_shape = GetIntsAttribute(node, "kernel_shape", {1, 1});
  const std::string auto_pad = GetStringAttribute(node, "auto_pad", "NOTSET");
  const int kernel_h = kernel_shape.empty() ? 1 : kernel_shape[0];
  const int kernel_w = kernel_shape.size() > 1 ? kernel_shape[1] : 1;
  const int stride_h = strides.empty() ? 1 : strides[0];
  const int stride_w = strides.size() > 1 ? strides[1] : 1;
  const int pad_top = pads.empty() ? 0 : pads[0];
  const int pad_left = pads.size() > 1 ? pads[1] : 0;
  const int pad_bottom = pads.size() > 2 ? pads[2] : pad_top;
  const int pad_right = pads.size() > 3 ? pads[3] : pad_left;

  if (output_state.shape.empty() && input_state.shape.size() == 4) {
    const int out_h = (input_state.shape[1] + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    const int out_w = (input_state.shape[2] + pad_left + pad_right - kernel_w) / stride_w + 1;
    output_state.shape = {input_state.shape[0], out_h, out_w, input_state.shape[3]};
  }
  if (output_state.tensor_type == "float32") {
    MarkFloatTensorNoQuant(output_state);
  } else if (!output_state.has_quant) {
    output_state.has_quant = input_state.has_quant;
    output_state.quant_scale = input_state.quant_scale;
    output_state.zero_point = input_state.zero_point;
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  bool output_matches_tflite_same = auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER";
  if (input_state.shape.size() == 4 && output_state.shape.size() == 4 &&
      stride_h > 0 && stride_w > 0) {
    const int valid_h = (input_state.shape[1] - kernel_h) / stride_h + 1;
    const int valid_w = (input_state.shape[2] - kernel_w) / stride_w + 1;
    const int same_h = (input_state.shape[1] + stride_h - 1) / stride_h;
    const int same_w = (input_state.shape[2] + stride_w - 1) / stride_w;
    output_matches_tflite_same =
        output_matches_tflite_same ||
        ((output_state.shape[1] == same_h && output_state.shape[2] == same_w) &&
         (output_state.shape[1] != valid_h || output_state.shape[2] != valid_w));
  }

  attributes["filter_height"] = kernel_h;
  attributes["filter_width"] = kernel_w;
  attributes["fused_activation_function"] = "None";
  attributes["padding"] = PadsAllEqual(pads, 1)
                              ? "add_pad_op1"
                              : (PadsAllEqual(pads, 2)
                                     ? "add_pad_op2"
                                     : (PadsAllEqual(pads, 3)
                                            ? "add_pad_op3"
                                            : (output_matches_tflite_same ? "SAME" : "VALID")));
  attributes["stride_h"] = stride_h;
  attributes["stride_w"] = stride_w;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("MaxPool");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("MaxPool", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceAveragePoolInitial(const GraphViewer& graph_viewer,
                                       const Node& node,
                                       std::unordered_map<std::string, TensorState>& states,
                                       ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid AveragePool node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state = input_state;
    output_state.name = ReferenceTensorName(outputs[0]->Name());
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  const auto pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
  const auto strides = GetIntsAttribute(node, "strides", {1, 1});
  const auto kernel_shape = GetIntsAttribute(node, "kernel_shape", {1, 1});
  int filter_height = kernel_shape.empty() ? 1 : kernel_shape[0];
  int filter_width = kernel_shape.size() > 1 ? kernel_shape[1] : 1;
  const std::string padding =
      pads == std::vector<int>{1, 1, 1, 1}
          ? "SAME"
          : (pads == std::vector<int>{2, 2, 2, 2} ? "add_pad_op2" : "VALID");

  if (padding == "VALID" &&
      input_state.shape.size() == 4 &&
      output_state.shape.size() == 4 &&
      output_state.shape[1] == 1 &&
      output_state.shape[2] == 1) {
    if (filter_height > input_state.shape[1]) {
      filter_height = input_state.shape[1];
    }
    if (filter_width > input_state.shape[2]) {
      filter_width = input_state.shape[2];
    }
  }

  ordered_json attributes;
  attributes["filter_height"] = filter_height;
  attributes["filter_width"] = filter_width;
  attributes["fused_activation_function"] = "None";
  attributes["padding"] = padding;
  attributes["stride_h"] = strides.empty() ? 1 : strides[0];
  attributes["stride_w"] = strides.size() > 1 ? strides[1] : 1;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("AveragePool");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("AveragePool", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceLRNInitial(const GraphViewer& graph_viewer,
                               const Node& node,
                               std::unordered_map<std::string, TensorState>& states,
                               ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid LRN node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state = input_state;
    output_state.name = ReferenceTensorName(outputs[0]->Name());
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  const int size = GetIntAttribute(node, "size", 1);
  ordered_json attributes;
  attributes["alpha"] = GetFloatAttribute(node, "alpha", 0.0001f);
  attributes["beta"] = GetFloatAttribute(node, "beta", 0.75f);
  attributes["bias"] = GetFloatAttribute(node, "bias", 1.0f);
  attributes["radius"] = std::max(size / 2, 0);

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("LRN");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("LRN", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceFlattenInitial(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   std::unordered_map<std::string, TensorState>& states,
                                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Flatten node.");

  TensorState input_state = MakeFloatTensorState(graph_viewer, inputs[0]->Name(), false);
  input_state.has_quant = true;
  input_state.quant_scale = "None";
  input_state.zero_point = "None";
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), false);
  output_state.has_quant = true;
  output_state.quant_scale = "None";
  output_state.zero_point = "None";
  states[outputs[0]->Name()] = output_state;

  ordered_json new_shape = ordered_json::array();
  for (int dim : output_state.shape) {
    new_shape.push_back(dim);
  }

  ordered_json shape_tensor_info = ConstantTensorInfoJson(
      input_state.name + "_1", "int32",
      {static_cast<int>(output_state.shape.size())}, new_shape);
  shape_tensor_info["quant_scale"] = "None";
  shape_tensor_info["zero_point"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = std::move(shape_tensor_info);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ordered_json attributes;
  attributes["new_shape"] = new_shape;
  payload["attributes_info"] = std::move(attributes);
  if (input_state.shape.size() == 4) {
    payload["is_flatten"] = true;
  }
  ops_json.push_back(WrapOp("reshape", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceQGemmInitial(const GraphViewer& graph_viewer,
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

Status EmitReferenceGemmInitial(const GraphViewer& graph_viewer,
                                const Node& node,
                                std::unordered_map<std::string, TensorState>& states,
                                ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid Gemm node.");

  const int trans_b = GetIntAttribute(node, "transB", 0);
  ORT_RETURN_IF_NOT(trans_b == 0 || trans_b == 1,
                    "Amlogic reference_style Gemm currently supports transB=0 or transB=1 only. Node: ",
                    node.Name());
  ORT_RETURN_IF_NOT(GetFloatAttribute(node, "alpha", 1.0f) == 1.0f &&
                        GetFloatAttribute(node, "beta", 1.0f) == 1.0f,
                    "Amlogic reference_style Gemm currently supports alpha=1 and beta=1 only. Node: ",
                    node.Name());

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), false, "float32");
  if (input_state.tensor_type == "float32") {
    input_state.has_quant = false;
  }
  states[inputs[0]->Name()] = input_state;

  TensorState weight_state;
  ORT_RETURN_IF_ERROR(MakeGemmFloatWeightTensorState(graph_viewer, inputs[1]->Name(), trans_b, weight_state));

  TensorState bias_state;
  if (inputs.size() > 2 && inputs[2] != nullptr && inputs[2]->Exists()) {
    ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[2]->Name(), false, bias_state));
  } else {
    const int output_channels = weight_state.shape.empty() ? 1 : weight_state.shape[0];
    bias_state = MakeSyntheticFloatBiasTensorState(inputs[1]->Name(), output_channels);
  }

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), false);
  output_state.has_quant = false;
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["keep_num_dims"] = false;
  attributes["fused_activation_function"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Gemm");
  payload["input_tensor_info"] = TensorInfoJson(input_state);
  payload["input_weight_info"] = TensorInfoJson(weight_state);
  payload["input_bias_info"] = TensorInfoJson(bias_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Gemm", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceSplitInitial(const GraphViewer& graph_viewer,
                                 const Node& node,
                                 std::unordered_map<std::string, TensorState>& states,
                                 ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 2 && inputs[0] != nullptr,
                    "Invalid Split node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true);
  const std::string input_declared_type =
      ReferenceTensorTypeFromNodeArg(inputs[0], input_state.tensor_type);
  if (input_declared_type != input_state.tensor_type) {
    input_state.tensor_type = input_declared_type;
    if (input_state.tensor_type == "float32") {
      MarkFloatTensorNoQuant(input_state);
    } else if (!input_state.has_quant) {
      input_state.has_quant = true;
      input_state.quant_scale = "None";
      input_state.zero_point = "None";
    }
  }
  states[inputs[0]->Name()] = input_state;
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
    output_state.tensor_type = ReferenceTensorTypeFromNodeArg(outputs[i], input_state.tensor_type);
    output_state.shape = ShapeForTensor(graph_viewer, outputs[i]->Name(), true);
    output_state.is_constant = false;
    output_state.has_values = false;
    if (output_state.tensor_type == "float32") {
      MarkFloatTensorNoQuant(output_state);
    } else if (!output_state.has_quant) {
      output_state.has_quant = true;
      output_state.quant_scale = "None";
      output_state.zero_point = "None";
    }
    states[outputs[i]->Name()] = output_state;
    payload[std::string("output_tensor_info_") + std::to_string(i + 1)] = TensorInfoJson(output_state);
  }

  ordered_json attributes;
  attributes["num_splits"] = static_cast<int>(outputs.size());
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Split", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceQLinearConcatInitial(const GraphViewer& graph_viewer,
                                         const Node& node,
                                         std::unordered_map<std::string, TensorState>& states,
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

  std::vector<TensorState> input_states;
  int input_index = 1;
  for (size_t i = 2; i + 2 < inputs.size(); i += 3) {
    if (inputs[i] == nullptr || inputs[i + 1] == nullptr || inputs[i + 2] == nullptr) {
      continue;
    }

    TensorState input_state;
    ORT_RETURN_IF_ERROR(MakeQuantTensorState(graph_viewer, inputs[i]->Name(),
                                             inputs[i + 1]->Name(), inputs[i + 2]->Name(),
                                             true, input_state));
    states[inputs[i]->Name()] = input_state;
    input_states.push_back(input_state);
    payload[std::string("input_tensor_info_") + std::to_string(input_index++)] = TensorInfoJson(input_state);
  }

  const int onnx_axis = GetIntAttribute(node, "axis", 0);
  ordered_json attributes;
  attributes["axis"] = ResolveReferenceConcatAxis(onnx_axis, output_state.shape.size(), input_states,
                                                  output_state.shape);
  attributes["fused_activation_function"] = "None";
  payload["attributes_info"] = std::move(attributes);

  ops_json.push_back(WrapOp("QLinearConcat", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceConcatInitial(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  std::unordered_map<std::string, TensorState>& states,
                                  ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && outputs[0] != nullptr,
                    "Invalid Concat node.");

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  MarkFloatTensorNoQuant(output_state);

  const int onnx_axis = GetIntAttribute(node, "axis", 0);

  // A four-way stride-2 spatial Slice on the same NCHW tensor followed by
  // Concat(axis=1) is equivalent to SPACE_TO_DEPTH(2), apart from a fixed
  // depth-block order. Recognize only this complete structural pattern; all
  // other Slice and Concat nodes retain their existing handling.
  const NodeArg* focus_source = nullptr;
  const bool is_stride2_focus_concat = [&]() {
    if (inputs.size() != 4 || onnx_axis != 1) {
      return false;
    }
    const std::vector<std::vector<int>> expected_starts = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == nullptr) {
        return false;
      }
      const Node* slice_node = graph_viewer.GetProducerNode(inputs[i]->Name());
      if (slice_node == nullptr || slice_node->OpType() != "Slice") {
        return false;
      }
      const auto slice_inputs = slice_node->InputDefs();
      if (slice_inputs.size() < 5 || slice_inputs[0] == nullptr ||
          slice_inputs[1] == nullptr || slice_inputs[2] == nullptr ||
          slice_inputs[3] == nullptr || slice_inputs[4] == nullptr ||
          !slice_inputs[1]->Exists() || !slice_inputs[2]->Exists() ||
          !slice_inputs[3]->Exists() || !slice_inputs[4]->Exists()) {
        return false;
      }
      if (focus_source == nullptr) {
        focus_source = slice_inputs[0];
      } else if (focus_source->Name() != slice_inputs[0]->Name()) {
        return false;
      }
      std::vector<int> starts;
      std::vector<int> ends;
      std::vector<int> axes;
      std::vector<int> steps;
      if (!InitializerAsIntVector(graph_viewer, slice_inputs[1]->Name(), starts).IsOK() ||
          !InitializerAsIntVector(graph_viewer, slice_inputs[2]->Name(), ends).IsOK() ||
          !InitializerAsIntVector(graph_viewer, slice_inputs[3]->Name(), axes).IsOK() ||
          !InitializerAsIntVector(graph_viewer, slice_inputs[4]->Name(), steps).IsOK() ||
          starts != expected_starts[i] || axes != std::vector<int>({2, 3}) ||
          steps != std::vector<int>({2, 2}) || ends.size() != 2) {
        return false;
      }
    }
    return focus_source != nullptr && IsReferenceGraphInput(graph_viewer, focus_source->Name());
  }();
  if (is_stride2_focus_concat) {
    TensorState focus_input_state = LookupState(graph_viewer, states, focus_source->Name(), true, "float32");
    MarkFloatTensorNoQuant(focus_input_state);
    const std::vector<int> onnx_focus_shape = ShapeForTensor(graph_viewer, focus_source->Name(), false);
    if (focus_input_state.shape == onnx_focus_shape && onnx_focus_shape.size() == 4) {
      focus_input_state.shape = ToNhwcShape(onnx_focus_shape);
    }
    states[focus_source->Name()] = focus_input_state;

    ordered_json attributes;
    attributes["block_size"] = 2;
    TensorState space_to_depth_state = output_state;
    const std::string focus_base = std::string(outputs[0]->Name()) + "_focus_space_to_depth";
    space_to_depth_state.name = ReferenceTensorName(focus_base);
    space_to_depth_state.is_constant = false;
    space_to_depth_state.has_values = false;
    MarkFloatTensorNoQuant(space_to_depth_state);
    ordered_json payload;
    payload["op_type_info"] = OpTypeInfo("SpaceToDepth");
    payload["input_tensor_info"] = TensorInfoJson(focus_input_state);
    payload["output_tensor_info"] = TensorInfoJson(space_to_depth_state);
    payload["attributes_info"] = std::move(attributes);
    ops_json.push_back(WrapOp("SpaceToDepth", std::move(payload)));

    // SPACE_TO_DEPTH orders 2x2 blocks as (00, 01, 10, 11), while the
    // original NCHW Slice/Concat sequence is (00, 10, 01, 11). Split its
    // contiguous channel groups and concatenate them in that order.
    ORT_RETURN_IF_NOT(space_to_depth_state.shape.size() == 4 &&
                          space_to_depth_state.shape[3] > 0 &&
                          space_to_depth_state.shape[3] % 4 == 0,
                      "Invalid stride-2 focus SPACE_TO_DEPTH output shape.");
    const int channels = space_to_depth_state.shape[3] / 4;
    std::vector<TensorState> focus_blocks;
    focus_blocks.reserve(4);
    for (int block_index = 0; block_index < 4; ++block_index) {
      TensorState block_state = space_to_depth_state;
      block_state.name = ReferenceTensorName(focus_base + "_block_" + std::to_string(block_index));
      block_state.shape[3] = channels;
      block_state.is_constant = false;
      block_state.has_values = false;
      MarkFloatTensorNoQuant(block_state);
      focus_blocks.push_back(block_state);

      ordered_json begin_values = ordered_json::array({0, 0, 0, block_index * channels});
      ordered_json size_values = ordered_json::array({space_to_depth_state.shape[0],
                                                       space_to_depth_state.shape[1],
                                                       space_to_depth_state.shape[2], channels});
      ordered_json slice_payload;
      slice_payload["op_type_info"] = OpTypeInfo("Slice");
      slice_payload["input_tensor_info_1"] = TensorInfoJson(space_to_depth_state);
      slice_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
          block_state.name + "_begin", "int32", {4}, begin_values);
      slice_payload["input_tensor_info_3"] = ConstantTensorInfoJson(
          block_state.name + "_size", "int32", {4}, size_values);
      slice_payload["output_tensor_info"] = TensorInfoJson(block_state);
      ops_json.push_back(WrapOp("Slice", std::move(slice_payload)));
    }

    const std::vector<int> focus_order = {0, 2, 1, 3};
    TensorState running_state = focus_blocks[focus_order[0]];
    for (size_t order_index = 1; order_index < focus_order.size(); ++order_index) {
      const bool is_last = order_index + 1 == focus_order.size();
      TensorState concat_state = output_state;
      if (!is_last) {
        concat_state.name = ReferenceTensorName(focus_base + "_concat_" + std::to_string(order_index));
        concat_state.shape = running_state.shape;
        concat_state.shape[3] += focus_blocks[focus_order[order_index]].shape[3];
        concat_state.is_constant = false;
        concat_state.has_values = false;
        MarkFloatTensorNoQuant(concat_state);
      }
      ordered_json concat_attributes;
      concat_attributes["axis"] = 3;
      concat_attributes["fused_activation_function"] = "None";
      ordered_json concat_payload;
      concat_payload["op_type_info"] = OpTypeInfo("Concat");
      concat_payload["input_tensor_info_1"] = TensorInfoJson(running_state);
      concat_payload["input_tensor_info_2"] = TensorInfoJson(focus_blocks[focus_order[order_index]]);
      concat_payload["output_tensor_info"] = TensorInfoJson(concat_state);
      concat_payload["attributes_info"] = std::move(concat_attributes);
      ops_json.push_back(WrapOp("Concat", std::move(concat_payload)));
      running_state = concat_state;
    }
    states[outputs[0]->Name()] = output_state;
    return Status::OK();
  }

  std::vector<TensorState> input_states;
  for (const NodeArg* input : inputs) {
    if (input == nullptr) {
      continue;
    }
    TensorState input_state = LookupState(graph_viewer, states, input->Name(), true, "float32");
    MarkFloatTensorNoQuant(input_state);
    const std::string canonical_input_name = CanonicalOnnxTensorName(input->Name());
    const auto* concat_initializer = graph_viewer.GetConstantInitializer(canonical_input_name, true);
    if (concat_initializer != nullptr) {
      const std::vector<int> initializer_shape = TensorProtoShape(*concat_initializer);
      const bool initializer_needs_nhwc =
          initializer_shape.size() == 4 &&
          !input_state.shape.empty() &&
          input_state.shape == ToNhwcShape(initializer_shape);
      ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, canonical_input_name,
                                                          initializer_needs_nhwc, input_state));
      MarkFloatTensorNoQuant(input_state);
    } else {
      const bool dangling_concat_input =
          graph_viewer.GetProducerNode(input->Name()) == nullptr &&
          !IsReferenceGraphInput(graph_viewer, input->Name());
      const int64_t element_count = StaticElementCount(input_state.shape);
      if (dangling_concat_input && element_count > 0 && element_count <= 1000000) {
        input_state = MakeFloatConstantTensorState(canonical_input_name,
                                                   input_state.shape,
                                                   std::vector<float>(static_cast<size_t>(element_count), 1.0f));
      }
    }
    states[input->Name()] = input_state;
    input_states.push_back(input_state);
  }

  ORT_RETURN_IF_NOT(!input_states.empty(), "Concat node has no valid inputs. Node: ", node.Name());
  const size_t concat_rank = !output_state.shape.empty() ? output_state.shape.size() : input_states.front().shape.size();
  const int axis = ResolveReferenceConcatAxis(onnx_axis, concat_rank, input_states, output_state.shape);
  if (output_state.shape.empty()) {
    output_state.shape = InferConcatShape(input_states, axis);
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["axis"] = axis;
  attributes["fused_activation_function"] = "None";

  if (input_states.size() <= 2) {
    ordered_json payload;
    payload["op_type_info"] = OpTypeInfo("Concat");
    int input_index = 1;
    for (const auto& input_state : input_states) {
      payload[std::string("input_tensor_info_") + std::to_string(input_index++)] = TensorInfoJson(input_state);
    }
    payload["output_tensor_info"] = TensorInfoJson(output_state);
    payload["attributes_info"] = attributes;
    ops_json.push_back(WrapOp("Concat", std::move(payload)));
    return Status::OK();
  }

  TensorState running_state = input_states.front();
  for (size_t i = 1; i < input_states.size(); ++i) {
    TensorState next_output = output_state;
    const bool is_last_concat = i + 1 == input_states.size();
    if (!is_last_concat) {
      next_output.name = ReferenceTensorName(std::string(outputs[0]->Name()) +
                                             "_concat_chain_" + std::to_string(i));
      next_output.shape = running_state.shape;
      if (!next_output.shape.empty() &&
          axis >= 0 && static_cast<size_t>(axis) < next_output.shape.size() &&
          input_states[i].shape.size() == next_output.shape.size()) {
        next_output.shape[static_cast<size_t>(axis)] += input_states[i].shape[static_cast<size_t>(axis)];
      }
      next_output.is_constant = false;
      next_output.has_values = false;
      MarkFloatTensorNoQuant(next_output);
    }

    ordered_json payload;
    payload["op_type_info"] = OpTypeInfo("Concat");
    payload["input_tensor_info_1"] = TensorInfoJson(running_state);
    payload["input_tensor_info_2"] = TensorInfoJson(input_states[i]);
    payload["output_tensor_info"] = TensorInfoJson(next_output);
    payload["attributes_info"] = attributes;
    ops_json.push_back(WrapOp("Concat", std::move(payload)));

    running_state = next_output;
  }
  return Status::OK();
}

Status EmitReferenceBatchNormalizationInitial(const GraphViewer& graph_viewer,
                                              const Node& node,
                                              std::unordered_map<std::string, TensorState>& states,
                                              ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 5 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        inputs[3] != nullptr && inputs[4] != nullptr && outputs[0] != nullptr,
                    "Invalid BatchNormalization node.");

  std::vector<float> scale_values;
  std::vector<float> bias_values;
  std::vector<float> mean_values;
  std::vector<float> var_values;
  ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, inputs[1]->Name(), scale_values));
  ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, inputs[2]->Name(), bias_values));
  ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, inputs[3]->Name(), mean_values));
  ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, inputs[4]->Name(), var_values));

  const size_t channels = scale_values.size();
  ORT_RETURN_IF_NOT(channels > 0 && bias_values.size() == channels &&
                        mean_values.size() == channels && var_values.size() == channels,
                    "BatchNormalization parameter size mismatch. Node: ", node.Name());

  const float epsilon = GetFloatAttribute(node, "epsilon", 1e-5f);
  std::vector<float> mul_values(channels);
  std::vector<float> add_values(channels);
  for (size_t i = 0; i < channels; ++i) {
    const float multiplier = scale_values[i] / std::sqrt(var_values[i] + epsilon);
    mul_values[i] = multiplier;
    add_values[i] = bias_values[i] - mean_values[i] * multiplier;
  }

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  }
  MarkFloatTensorNoQuant(output_state);

  TensorState mul_const = MakeFloatConstantTensorState(std::string(outputs[0]->Name()) + "_bn_mul_values",
                                                       {static_cast<int>(channels)}, mul_values);
  TensorState add_const = MakeFloatConstantTensorState(std::string(outputs[0]->Name()) + "_bn_add_values",
                                                       {static_cast<int>(channels)}, add_values);

  TensorState mul_output = output_state;
  mul_output.name = ReferenceTensorName(std::string(outputs[0]->Name()) + "_bn_mul_output");
  states[mul_output.name] = mul_output;

  ordered_json attributes;
  attributes["fused_activation_function"] = "None";

  ordered_json mul_payload;
  mul_payload["op_type_info"] = OpTypeInfo("Mul");
  mul_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  mul_payload["input_tensor_info_2"] = TensorInfoJson(mul_const);
  mul_payload["output_tensor_info"] = TensorInfoJson(mul_output);
  mul_payload["attributes_info"] = attributes;
  ops_json.push_back(WrapOp("Mul", std::move(mul_payload)));

  ordered_json add_payload;
  add_payload["op_type_info"] = OpTypeInfo("Add");
  add_payload["input_tensor_info_1"] = TensorInfoJson(mul_output);
  add_payload["input_tensor_info_2"] = TensorInfoJson(add_const);
  add_payload["output_tensor_info"] = TensorInfoJson(output_state);
  add_payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Add", std::move(add_payload)));

  states[outputs[0]->Name()] = output_state;
  return Status::OK();
}

Status EmitReferenceReduceMeanInitial(const GraphViewer& graph_viewer,
                                      const Node& node,
                                      std::unordered_map<std::string, TensorState>& states,
                                      ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid ReduceMean node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  input_state.tensor_type = "float32";
  input_state.has_quant = false;
  if (input_state.shape.empty()) {
    input_state.shape = {1};
  }
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  output_state.has_quant = false;

  std::vector<int> axes;
  if (inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), axes));
  } else {
    axes = GetIntsAttribute(node, "axes", {});
  }
  if (axes.empty() && !input_state.shape.empty()) {
    axes.resize(input_state.shape.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  ordered_json axis_values = ordered_json::array();
  for (int axis : axes) {
    axis_values.push_back(MapAxisToNhwc(axis, input_state.shape.size()));
  }

  ordered_json attributes;
  attributes["keepdims"] = GetIntAttribute(node, "keepdims", 1) != 0;
  if (output_state.shape.empty() && !input_state.shape.empty()) {
    std::vector<int> inferred = input_state.shape;
    std::vector<int> mapped_axes;
    for (int axis : axes) {
      mapped_axes.push_back(MapAxisToNhwc(axis, input_state.shape.size()));
    }
    if (attributes["keepdims"].get<bool>()) {
      for (int axis : mapped_axes) {
        if (axis >= 0 && static_cast<size_t>(axis) < inferred.size()) {
          inferred[static_cast<size_t>(axis)] = 1;
        }
      }
    } else {
      std::sort(mapped_axes.begin(), mapped_axes.end(), std::greater<int>());
      for (int axis : mapped_axes) {
        if (axis >= 0 && static_cast<size_t>(axis) < inferred.size()) {
          inferred.erase(inferred.begin() + axis);
        }
      }
      if (inferred.empty()) {
        inferred = {1};
      }
    }
    output_state.shape = inferred;
  }
  if (output_state.shape.empty()) {
    output_state.shape = {1};
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("ReduceMean");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_reduce_mean_axes", "int32",
                                                          {static_cast<int>(axis_values.size())}, axis_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("ReduceMean", std::move(payload)));
  return Status::OK();
}

std::vector<int> NormalizeReferenceReduceAxes(const std::vector<int>& axes, size_t rank);
std::vector<int> InferReferenceReducedShape(const std::vector<int>& input_shape,
                                            const std::vector<int>& axes,
                                            bool keepdims);
bool PreferPlainRank4AttentionReduceAxes(const std::vector<int>& input_shape,
                                         const std::vector<int>& onnx_axes,
                                         bool keepdims);
std::vector<int> ResolveReferenceReduceAxes(const std::vector<int>& input_shape,
                                            const std::vector<int>& output_shape,
                                            const std::vector<int>& onnx_axes,
                                            bool keepdims);
ordered_json MakeReferenceTransposeNchwToNhwcOp(const ordered_json& input_tensor,
                                                const ordered_json& output_tensor,
                                                const std::string& perm_tensor_name);

Status EmitReferenceReduceSumInitial(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     std::unordered_map<std::string, TensorState>& states,
                                     ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid ReduceSum node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  input_state.tensor_type = "float32";
  input_state.has_quant = false;
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  output_state.has_quant = false;

  std::vector<int> axes;
  if (inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), axes));
  } else {
    axes = GetIntsAttribute(node, "axes", {});
  }
  const bool noop_with_empty_axes = GetIntAttribute(node, "noop_with_empty_axes", 0) != 0;
  if (axes.empty() && noop_with_empty_axes && output_state.shape == input_state.shape) {
    states[outputs[0]->Name()] = input_state;
    return Status::OK();
  }
  if (axes.empty() && !input_state.shape.empty()) {
    axes.resize(input_state.shape.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  ordered_json attributes;
  attributes["keepdims"] = GetIntAttribute(node, "keepdims", 1) != 0;
  const bool keepdims = attributes["keepdims"].get<bool>();
  const bool prefer_plain_attention_reduce =
      PreferPlainRank4AttentionReduceAxes(input_state.shape, axes, keepdims);
  const std::vector<int> mapped_axes = prefer_plain_attention_reduce
                                           ? NormalizeReferenceReduceAxes(axes, input_state.shape.size())
                                           : ResolveReferenceReduceAxes(input_state.shape,
                                                                        output_state.shape,
                                                                        axes,
                                                                        keepdims);
  std::vector<int> reduced_shape =
      InferReferenceReducedShape(input_state.shape, mapped_axes, keepdims);

  ordered_json axis_values = ordered_json::array();
  for (int axis : mapped_axes) {
    axis_values.push_back(axis);
  }

  if (prefer_plain_attention_reduce && !reduced_shape.empty()) {
    output_state.shape = reduced_shape;
  } else if (output_state.shape.empty()) {
    output_state.shape = reduced_shape.empty() ? std::vector<int>{1} : reduced_shape;
  }

  if (axis_values.empty()) {
    ordered_json shape_values = ordered_json::array();
    for (int dim : output_state.shape) {
      shape_values.push_back(dim);
    }
    ordered_json payload;
    payload["op_type_info"] = OpTypeInfo("reshape");
    payload["input_tensor_info_1"] = TensorInfoJson(input_state);
    payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_reduce_sum_identity_shape",
                                                            "int32",
                                                            {static_cast<int>(output_state.shape.size())},
                                                            shape_values);
    payload["output_tensor_info"] = TensorInfoJson(output_state);
    ops_json.push_back(WrapOp("reshape", std::move(payload)));
    states[outputs[0]->Name()] = output_state;
    return Status::OK();
  }

  if (!reduced_shape.empty() &&
      reduced_shape.size() == 4 &&
      output_state.shape == ToNhwcShape(reduced_shape) &&
      output_state.shape != reduced_shape) {
    TensorState reduce_output_state = output_state;
    reduce_output_state.name = ReferenceTensorName(std::string(outputs[0]->Name()) + "_reduce_sum_nchw");
    reduce_output_state.shape = reduced_shape;
    MarkFloatTensorNoQuant(reduce_output_state);

    ordered_json reduce_payload;
    reduce_payload["op_type_info"] = OpTypeInfo("ReduceSum");
    reduce_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
    reduce_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        reduce_output_state.name + "_reduce_sum_axes", "int32",
        {static_cast<int>(axis_values.size())}, axis_values);
    reduce_payload["output_tensor_info"] = TensorInfoJson(reduce_output_state);
    reduce_payload["attributes_info"] = attributes;
    ops_json.push_back(WrapOp("ReduceSum", std::move(reduce_payload)));

    ops_json.push_back(MakeReferenceTransposeNchwToNhwcOp(
        TensorInfoJson(reduce_output_state),
        TensorInfoJson(output_state),
        output_state.name + "_reduce_sum_nchw_to_nhwc_perm"));
    states[outputs[0]->Name()] = output_state;
    return Status::OK();
  }

  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("ReduceSum");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_reduce_sum_axes", "int32",
                                                          {static_cast<int>(axis_values.size())}, axis_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("ReduceSum", std::move(payload)));
  return Status::OK();
}

std::vector<int> NormalizeReferenceReduceAxes(const std::vector<int>& axes, size_t rank) {
  std::vector<int> normalized;
  normalized.reserve(axes.size());
  for (int axis : axes) {
    int resolved_axis = axis;
    if (rank > 0 && resolved_axis < 0) {
      resolved_axis += static_cast<int>(rank);
    }
    normalized.push_back(resolved_axis);
  }
  return normalized;
}

std::vector<int> InferReferenceReducedShape(const std::vector<int>& input_shape,
                                            const std::vector<int>& axes,
                                            bool keepdims) {
  std::vector<int> inferred = input_shape;
  if (inferred.empty()) {
    return {1};
  }

  if (keepdims) {
    for (int axis : axes) {
      if (axis >= 0 && static_cast<size_t>(axis) < inferred.size()) {
        inferred[static_cast<size_t>(axis)] = 1;
      }
    }
  } else {
    std::vector<int> sorted_axes = axes;
    std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<int>());
    for (int axis : sorted_axes) {
      if (axis >= 0 && static_cast<size_t>(axis) < inferred.size()) {
        inferred.erase(inferred.begin() + axis);
      }
    }
    if (inferred.empty()) {
      inferred = {1};
    }
  }
  return inferred;
}

bool PreferPlainRank4AttentionReduceAxes(const std::vector<int>& input_shape,
                                         const std::vector<int>& onnx_axes,
                                         bool keepdims) {
  if (!keepdims || input_shape.size() != 4 || onnx_axes.size() != 1) {
    return false;
  }

  const std::vector<int> plain_axes = NormalizeReferenceReduceAxes(onnx_axes, input_shape.size());
  if (plain_axes.size() != 1 || plain_axes[0] != 3) {
    return false;
  }

  const int heads = input_shape[1];
  const int query_len = input_shape[2];
  const int key_len = input_shape[3];
  return heads > 0 && heads <= 16 &&
         query_len >= 64 &&
         query_len == key_len;
}

std::vector<int> ResolveReferenceReduceAxes(const std::vector<int>& input_shape,
                                            const std::vector<int>& output_shape,
                                            const std::vector<int>& onnx_axes,
                                            bool keepdims) {
  std::vector<int> plain_axes = NormalizeReferenceReduceAxes(onnx_axes, input_shape.size());
  std::vector<int> mapped_axes;
  mapped_axes.reserve(onnx_axes.size());
  for (int axis : onnx_axes) {
    mapped_axes.push_back(MapAxisToNhwc(axis, input_shape.size()));
  }

  if (!input_shape.empty() && !output_shape.empty()) {
    if (InferReferenceReducedShape(input_shape, plain_axes, keepdims) == output_shape) {
      return plain_axes;
    }
    if (InferReferenceReducedShape(input_shape, mapped_axes, keepdims) == output_shape) {
      return mapped_axes;
    }
    if (onnx_axes.size() == 1) {
      for (size_t axis = 0; axis < input_shape.size(); ++axis) {
        const std::vector<int> candidate_axes = {static_cast<int>(axis)};
        if (InferReferenceReducedShape(input_shape, candidate_axes, keepdims) == output_shape) {
          return candidate_axes;
        }
      }
    }
  }
  return mapped_axes;
}

Status EmitReferenceReduceMaxInitial(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     std::unordered_map<std::string, TensorState>& states,
                                     ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid ReduceMax node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  input_state.tensor_type = "float32";
  input_state.has_quant = false;
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  output_state.has_quant = false;

  std::vector<int> axes;
  if (inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), axes));
  } else {
    axes = GetIntsAttribute(node, "axes", {});
  }
  if (axes.empty() && !input_state.shape.empty()) {
    axes.resize(input_state.shape.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  ordered_json attributes;
  attributes["keepdims"] = GetIntAttribute(node, "keepdims", 1) != 0;
  const bool original_keepdims = attributes["keepdims"].get<bool>();
  const bool prefer_plain_attention_reduce =
      PreferPlainRank4AttentionReduceAxes(input_state.shape, axes, original_keepdims);
  const std::vector<int> mapped_axes = prefer_plain_attention_reduce
                                           ? NormalizeReferenceReduceAxes(axes, input_state.shape.size())
                                           : ResolveReferenceReduceAxes(input_state.shape,
                                                                        output_state.shape,
                                                                        axes,
                                                                        original_keepdims);
  const std::vector<int> reduced_shape =
      InferReferenceReducedShape(input_state.shape, mapped_axes, original_keepdims);

  ordered_json axis_values = ordered_json::array();
  for (int axis : mapped_axes) {
    axis_values.push_back(axis);
  }

  if (prefer_plain_attention_reduce && !reduced_shape.empty()) {
    output_state.shape = reduced_shape;
  } else if (output_state.shape.empty() && !input_state.shape.empty()) {
    output_state.shape = reduced_shape;
  }
  if (output_state.shape.empty()) {
    output_state.shape = {1};
  }

  if (!original_keepdims && !axis_values.empty() &&
      input_state.shape.size() > output_state.shape.size()) {
    TensorState keepdims_output_state = output_state;
    keepdims_output_state.name = ReferenceTensorName(std::string(outputs[0]->Name()) + "_reduce_max_keepdims");
    keepdims_output_state.shape = input_state.shape;
    for (const auto& axis_json : axis_values) {
      const int axis = axis_json.get<int>();
      if (axis >= 0 && static_cast<size_t>(axis) < keepdims_output_state.shape.size()) {
        keepdims_output_state.shape[static_cast<size_t>(axis)] = 1;
      }
    }
    MarkFloatTensorNoQuant(keepdims_output_state);

    ordered_json reduce_attributes = attributes;
    reduce_attributes["keepdims"] = true;
    ordered_json reduce_payload;
    reduce_payload["op_type_info"] = OpTypeInfo("ReduceMax");
    reduce_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
    reduce_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        keepdims_output_state.name + "_reduce_max_axes", "int32",
        {static_cast<int>(axis_values.size())}, axis_values);
    reduce_payload["output_tensor_info"] = TensorInfoJson(keepdims_output_state);
    reduce_payload["attributes_info"] = std::move(reduce_attributes);
    ops_json.push_back(WrapOp("ReduceMax", std::move(reduce_payload)));

    ordered_json shape_values = ordered_json::array();
    for (int dim : output_state.shape) {
      shape_values.push_back(dim);
    }
    ordered_json reshape_payload;
    reshape_payload["op_type_info"] = OpTypeInfo("reshape");
    reshape_payload["input_tensor_info_1"] = TensorInfoJson(keepdims_output_state);
    reshape_payload["input_tensor_info_2"] = ConstantTensorInfoJson(
        output_state.name + "_reduce_max_reshape_shape", "int32",
        {static_cast<int>(output_state.shape.size())}, shape_values);
    reshape_payload["output_tensor_info"] = TensorInfoJson(output_state);
    ordered_json reshape_attributes;
    reshape_attributes["new_shape"] = shape_values;
    reshape_payload["attributes_info"] = std::move(reshape_attributes);
    ops_json.push_back(WrapOp("reshape", std::move(reshape_payload)));

    states[outputs[0]->Name()] = output_state;
    return Status::OK();
  }

  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("ReduceMax");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_reduce_max_axes", "int32",
                                                          {static_cast<int>(axis_values.size())}, axis_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("ReduceMax", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceQLinearGlobalAveragePoolInitial(const GraphViewer& graph_viewer,
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
  attributes["keepdims"] = true;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("QLinearGlobalAveragePool");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_global_average_pool_axes",
                                                          "int32", {2}, ordered_json::array({1, 2}));
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("QLinearGlobalAveragePool", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceGlobalAveragePoolInitial(const GraphViewer& graph_viewer,
                                             const Node& node,
                                             std::unordered_map<std::string, TensorState>& states,
                                             ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid GlobalAveragePool node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  if (input_state.tensor_type == "float32") {
    input_state.has_quant = false;
  }
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty() && input_state.shape.size() == 4) {
    output_state.shape = {input_state.shape[0], 1, 1, input_state.shape[3]};
  }
  output_state.has_quant = false;
  states[outputs[0]->Name()] = output_state;

  ordered_json attributes;
  attributes["keepdims"] = true;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("GlobalAveragePool");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_global_average_pool_axes",
                                                          "int32", {2}, ordered_json::array({1, 2}));
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("GlobalAveragePool", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceInstanceNormalizationInitial(const GraphViewer& graph_viewer,
                                                 const Node& node,
                                                 std::unordered_map<std::string, TensorState>& states,
                                                 ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 3 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
                        outputs[0] != nullptr,
                    "Invalid InstanceNormalization node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState scale_state;
  ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[1]->Name(), false, scale_state));
  TensorState bias_state;
  ORT_RETURN_IF_ERROR(MakeFloatInitializerTensorState(graph_viewer, inputs[2]->Name(), false, bias_state));

  const std::string base = ReferenceTensorName(outputs[0]->Name());
  TensorState mean_state = input_state;
  mean_state.name = base + "_instance_mean";
  if (input_state.shape.size() == 4) {
    mean_state.shape = {input_state.shape[0], 1, 1, input_state.shape[3]};
  }
  mean_state.is_constant = false;
  mean_state.has_values = false;
  MarkFloatTensorNoQuant(mean_state);

  TensorState centered_state = input_state;
  centered_state.name = base + "_instance_centered";
  centered_state.is_constant = false;
  centered_state.has_values = false;
  MarkFloatTensorNoQuant(centered_state);

  TensorState square_state = centered_state;
  square_state.name = base + "_instance_square";

  TensorState var_state = mean_state;
  var_state.name = base + "_instance_var";

  TensorState epsilon_state = MakeFloatConstantTensorState(base + "_instance_epsilon", {1},
                                                          {GetFloatAttribute(node, "epsilon", 1.0e-5f)});

  TensorState var_eps_state = var_state;
  var_eps_state.name = base + "_instance_var_eps";

  TensorState inv_std_state = var_state;
  inv_std_state.name = base + "_instance_inv_std";

  TensorState normalized_state = input_state;
  normalized_state.name = base + "_instance_normalized";
  normalized_state.is_constant = false;
  normalized_state.has_values = false;
  MarkFloatTensorNoQuant(normalized_state);

  TensorState scaled_state = normalized_state;
  scaled_state.name = base + "_instance_scaled";

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json mean_attributes;
  mean_attributes["keepdims"] = true;
  ordered_json mean_payload;
  mean_payload["op_type_info"] = OpTypeInfo("ReduceMean");
  mean_payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  mean_payload["input_tensor_info_2"] = ConstantTensorInfoJson(base + "_instance_axes", "int32", {2},
                                                              ordered_json::array({1, 2}));
  mean_payload["output_tensor_info"] = TensorInfoJson(mean_state);
  mean_payload["attributes_info"] = mean_attributes;
  ops_json.push_back(WrapOp("ReduceMean", std::move(mean_payload)));

  PushReferenceBinaryOp(ops_json, "Sub", input_state, mean_state, centered_state);
  PushReferenceBinaryOp(ops_json, "Mul", centered_state, centered_state, square_state);

  ordered_json var_payload;
  var_payload["op_type_info"] = OpTypeInfo("ReduceMean");
  var_payload["input_tensor_info_1"] = TensorInfoJson(square_state);
  var_payload["input_tensor_info_2"] = ConstantTensorInfoJson(base + "_instance_var_axes", "int32", {2},
                                                             ordered_json::array({1, 2}));
  var_payload["output_tensor_info"] = TensorInfoJson(var_state);
  var_payload["attributes_info"] = mean_attributes;
  ops_json.push_back(WrapOp("ReduceMean", std::move(var_payload)));

  PushReferenceBinaryOp(ops_json, "Add", var_state, epsilon_state, var_eps_state);
  PushReferenceUnaryOp(ops_json, "Rsqrt", var_eps_state, inv_std_state);
  PushReferenceBinaryOp(ops_json, "Mul", centered_state, inv_std_state, normalized_state);
  PushReferenceBinaryOp(ops_json, "Mul", normalized_state, scale_state, scaled_state);
  PushReferenceBinaryOp(ops_json, "Add", scaled_state, bias_state, output_state);
  return Status::OK();
}

Status EmitReferenceResizeNearestInitial(const GraphViewer& graph_viewer,
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
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_resize_shape_values",
                                                          "int32", {2}, resize_json);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Reshape_nearest", std::move(payload)));
  return Status::OK();
}

// TFLite RESIZE_BILINEAR with half_pixel_centers is equivalent to the ONNX
// Resize combination used by common segmentation exports: mode=linear and
// coordinate_transformation_mode=half_pixel. Keep this separate from the
// established nearest path so existing Resize exports are unaffected.
Status EmitReferenceResizeLinearHalfPixelInitial(const GraphViewer& graph_viewer,
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
  payload["op_type_info"] = OpTypeInfo("Resize_bilinear");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_resize_shape_values",
                                                          "int32", {2}, resize_json);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("Reshape_bilinear", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceUpsampleNearestInitial(const GraphViewer& graph_viewer,
                                           const Node& node,
                                           std::unordered_map<std::string, TensorState>& states,
                                           ordered_json& ops_json) {
  ORT_RETURN_IF_NOT(GetStringAttribute(node, "mode", "nearest") == "nearest",
                    "Amlogic reference_style Upsample currently supports nearest mode only. Node: ",
                    node.Name());
  return EmitReferenceResizeNearestInitial(graph_viewer, node, states, ops_json);
}

Status EmitReferenceDequantizeLinearInitial(const GraphViewer& graph_viewer,
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

Status EmitReferenceSliceInitial(const GraphViewer& graph_viewer,
                                 const Node& node,
                                 std::unordered_map<std::string, TensorState>& states,
                                 ordered_json& ops_json) {
  // std::cout<<"amlogic_reference_style_collectors_impl.h"<<std::endl;
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && outputs[0] != nullptr,
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

  std::vector<int> starts;
  std::vector<int> ends;
  std::vector<int> axes;
  std::vector<int> steps;
  if (inputs.size() >= 3 && inputs[1] != nullptr && inputs[2] != nullptr &&
      inputs[1]->Exists() && inputs[2]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), starts));
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[2]->Name(), ends));
  } else {
    starts = GetIntsAttribute(node, "starts", {});
    ends = GetIntsAttribute(node, "ends", {});
    axes = GetIntsAttribute(node, "axes", {});
    steps = GetIntsAttribute(node, "steps", {});
  }
  if (axes.empty() && inputs.size() > 3 && inputs[3] != nullptr && inputs[3]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[3]->Name(), axes));
  }
  if (steps.empty() && inputs.size() > 4 && inputs[4] != nullptr && inputs[4]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[4]->Name(), steps));
  }
  if (axes.empty()) {
    axes.resize(starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  if (steps.empty()) {
    steps.assign(starts.size(), 1);
  }
  ORT_RETURN_IF_NOT(!starts.empty() && !ends.empty(), "Invalid Slice node: missing starts/ends.");

  const size_t rank = input_state.shape.size();
  const std::vector<int> onnx_input_shape = ShapeForTensor(graph_viewer, inputs[0]->Name(), false);
  const std::vector<int> onnx_output_shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  const size_t slice_dims = std::min({starts.size(), ends.size(), axes.size()});
  auto slice_step = [&](size_t index) {
    return index < steps.size() && steps[index] != 0 ? steps[index] : 1;
  };
  const bool has_focus_spatial_step = [&]() {
    for (size_t i = 0; i < slice_dims; ++i) {
      const int axis = NormalizeAxisForRank(axes[i], rank);
      if ((axis == 2 || axis == 3) && slice_step(i) == 2) {
        return true;
      }
    }
    return false;
  }();
  const bool input_is_nhwc =
      rank == 4 &&
      onnx_input_shape.size() == 4 &&
      input_state.shape == ToNhwcShape(onnx_input_shape) &&
      input_state.shape != onnx_input_shape;
  // YOLO Focus applies stride-2 spatial Slice ops directly to the NCHW graph input,
  // but the reference TFLite path expects the imported image tensor in NHWC layout.
  const bool focus_style_graph_input_slice =
      rank == 4 &&
      onnx_input_shape.size() == 4 &&
      onnx_output_shape.size() == 4 &&
      input_state.shape == onnx_input_shape &&
      IsReferenceGraphInput(graph_viewer, inputs[0]->Name()) &&
      onnx_input_shape[1] > 0 &&
      onnx_input_shape[1] <= 4 &&
      onnx_input_shape[2] > 0 &&
      onnx_input_shape[3] > 0 &&
      onnx_output_shape[0] == onnx_input_shape[0] &&
      onnx_output_shape[1] == onnx_input_shape[1] &&
      onnx_output_shape[2] > 0 &&
      onnx_output_shape[3] > 0 &&
      onnx_output_shape[2] * 2 == onnx_input_shape[2] &&
      onnx_output_shape[3] * 2 == onnx_input_shape[3] &&
      has_focus_spatial_step;
  const bool slice_uses_nhwc = input_is_nhwc || focus_style_graph_input_slice;
  if (focus_style_graph_input_slice) {
    input_state.shape = ToNhwcShape(onnx_input_shape);
    states[inputs[0]->Name()] = input_state;
  }

  std::vector<int> begin(rank, 0);
  std::vector<int> size = input_state.shape;
  for (size_t i = 0; i < slice_dims; ++i) {
    int axis = axes[i];
    if (axis < 0) {
      axis += static_cast<int>(rank);
    }
    if (slice_uses_nhwc) {
      axis = MapAxisToNhwc(axis, rank);
    }
    if (axis < 0 || axis >= static_cast<int>(rank)) {
      continue;
    }

    const int dim = input_state.shape[static_cast<size_t>(axis)];
    const int step = slice_step(i);
    const int start = NormalizeSliceIndex(starts[i], dim);
    const int end_index = NormalizeSliceIndex(ends[i], dim);
    begin[static_cast<size_t>(axis)] = start;
    const int abs_step = step < 0 ? -step : step;
    int extent = 0;
    if (step > 0) {
      extent = dim > 0 ? std::max(0, end_index - start) : end_index - start;
    } else {
      extent = dim > 0 ? std::max(0, start - end_index) : start - end_index;
    }
    size[static_cast<size_t>(axis)] =
        abs_step > 1 && extent > 0 ? (extent + abs_step - 1) / abs_step : extent;
  }

  ordered_json begin_values = ordered_json::array();
  ordered_json size_values = ordered_json::array();
  for (int value : begin) {
    begin_values.push_back(value);
  }
  for (int value : size) {
    size_values.push_back(value);
  }

  if (slice_uses_nhwc && output_state.shape.size() == 4 && input_state.shape.size() == 4) {
    output_state.shape = onnx_output_shape.size() == 4
                             ? ToNhwcShape(onnx_output_shape)
                             : ToNhwcShape(output_state.shape);
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  // The private refactorer's STRIDED_SLICE lowering still crashes after
  // receiving valid begin/end/stride tensors. Retain the established SLICE
  // fallback so batch conversion remains process-safe. Non-unit steps are
  // intentionally reported as numerical limitations until that refactorer
  // path is repaired and verified independently.
  payload["op_type_info"] = OpTypeInfo("Slice");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_begin_values", "int32",
                                                          {static_cast<int>(rank)}, begin_values);
  payload["input_tensor_info_3"] = ConstantTensorInfoJson(output_state.name + "_size_values", "int32",
                                                          {static_cast<int>(rank)}, size_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("Slice", std::move(payload)));
  return Status::OK();
}

Status EmitReferencePlainBinaryInitial(const GraphViewer& graph_viewer,
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
      TensorState state = MakeFloatTensorState(graph_viewer, input->Name(), true);
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

  TensorState& lhs_state = states[inputs[0]->Name()];
  TensorState& rhs_state = states[inputs[1]->Name()];
  TensorState lhs_payload_state = lhs_state;
  TensorState rhs_payload_state = rhs_state;
  const bool can_use_integer_binary = op_type == "Div" || op_type == "Mod";
  const std::string output_type = ReferenceTensorTypeFromNodeArg(
      outputs[0], can_use_integer_binary && IsReferenceIntegerTensorType(lhs_state.tensor_type)
                      ? lhs_state.tensor_type
                      : "float32");
  const bool integer_binary =
      can_use_integer_binary &&
      (IsReferenceIntegerTensorType(output_type) || IsReferenceIntegerTensorType(lhs_state.tensor_type));
  std::string emitted_op_type = op_type;

  if ((op_type == "Div" || op_type == "Mod") && integer_binary) {
    lhs_payload_state.tensor_type = "int32";
    rhs_payload_state.tensor_type = "int32";
    lhs_payload_state.has_quant = true;
    rhs_payload_state.has_quant = true;
    lhs_payload_state.quant_scale = "None";
    rhs_payload_state.quant_scale = "None";
    lhs_payload_state.zero_point = "None";
    rhs_payload_state.zero_point = "None";
    if (lhs_payload_state.is_constant) {
      lhs_payload_state.name += "_int32";
    }
    if (rhs_payload_state.is_constant) {
      rhs_payload_state.name += "_int32";
    }
    ORT_RETURN_IF_NOT(CoerceConstantTensorValuesToInt32(lhs_payload_state),
                      "Integer binary input contains non-integral constant values. Node: ", node.Name());
    ORT_RETURN_IF_NOT(CoerceConstantTensorValuesToInt32(rhs_payload_state),
                      "Integer binary input contains non-integral constant values. Node: ", node.Name());
  } else if (op_type == "Div") {
    if (InvertSingleFloatConstant(rhs_payload_state)) {
      rhs_payload_state.name += "_reciprocal";
      emitted_op_type = "Mul";
    }
  }

  CollectorPrepareRank3ToRank4BroadcastInputs(ops_json, lhs_payload_state, rhs_payload_state);

  TensorState output_state = MakeFloatTensorState(graph_viewer, outputs[0]->Name(), true);
  const std::vector<int> broadcast_shape =
      BroadcastShape(lhs_payload_state.shape, rhs_payload_state.shape);
  if (output_state.shape.empty()) {
    output_state.shape = broadcast_shape;
  } else {
    PreferPlainRank4OutputShape(output_state, broadcast_shape);
  }
  output_state.tensor_type = integer_binary ? "int32" : output_type;
  output_state.has_quant = true;
  output_state.quant_scale = "None";
  output_state.zero_point = "None";
  if (output_state.tensor_type == "float32") {
    MarkFloatTensorNoQuant(output_state);
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo(emitted_op_type);
  payload["input_tensor_info_1"] = TensorInfoJson(lhs_payload_state);
  payload["input_tensor_info_2"] = TensorInfoJson(rhs_payload_state);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp(emitted_op_type, std::move(payload)));
  return Status::OK();
}

Status EmitReferenceDropoutInitial(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   std::unordered_map<std::string, TensorState>& states,
                                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Dropout node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json new_shape = ordered_json::array();
  for (int dim : output_state.shape) {
    new_shape.push_back(dim);
  }

  ordered_json shape_tensor_info = ConstantTensorInfoJson(
      output_state.name + "_dropout_identity_shape", "int32",
      {static_cast<int>(output_state.shape.size())}, new_shape);
  shape_tensor_info["quant_scale"] = "None";
  shape_tensor_info["zero_point"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = std::move(shape_tensor_info);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ordered_json attributes;
  attributes["new_shape"] = new_shape;
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("reshape", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceSqueezeInitial(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   std::unordered_map<std::string, TensorState>& states,
                                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Squeeze node.");

  TensorState input_state = LookupState(graph_viewer, states, inputs[0]->Name(), true, "float32");
  MarkFloatTensorNoQuant(input_state);
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = {1};
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json new_shape = ordered_json::array();
  for (int dim : output_state.shape) {
    new_shape.push_back(dim);
  }

  ordered_json shape_tensor_info = ConstantTensorInfoJson(
      output_state.name + "_squeeze_shape", "int32",
      {static_cast<int>(output_state.shape.size())}, new_shape);
  shape_tensor_info["quant_scale"] = "None";
  shape_tensor_info["zero_point"] = "None";

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = std::move(shape_tensor_info);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ordered_json attributes;
  attributes["new_shape"] = new_shape;
  payload["attributes_info"] = std::move(attributes);
  ops_json.push_back(WrapOp("reshape", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceConstantInitial(const GraphViewer& graph_viewer,
                                    const Node& node,
                                    std::unordered_map<std::string, TensorState>& states) {
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(outputs.size() >= 1 && outputs[0] != nullptr, "Invalid Constant node.");

  const auto& attributes = node.GetAttributes();
  TensorState state;
  state.name = ReferenceTensorName(outputs[0]->Name());
  state.tensor_type = ReferenceTensorTypeFromNodeArg(outputs[0], "float32");
  state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  state.is_constant = true;
  state.has_values = true;

  auto shape_for_value_count = [&](size_t value_count) -> std::vector<int> {
    if (!state.shape.empty() &&
        StaticElementCount(state.shape) == static_cast<int64_t>(value_count)) {
      return state.shape;
    }
    if (value_count == 1) {
      return {};
    }
    return {static_cast<int>(value_count)};
  };

  const auto tensor_attr = attributes.find("value");
  const auto value_float_attr = attributes.find("value_float");
  const auto value_floats_attr = attributes.find("value_floats");
  const auto value_int_attr = attributes.find("value_int");
  const auto value_ints_attr = attributes.find("value_ints");
  if (tensor_attr != attributes.end() &&
      tensor_attr->second.type() ==
          ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_TENSOR) {
    const ONNX_NAMESPACE::TensorProto& tensor = tensor_attr->second.t();
    const std::vector<int> shape = TensorProtoShape(tensor);
    ordered_json values;
    ORT_RETURN_IF_ERROR(TensorValuesJson(tensor, graph_viewer.ModelPath(), shape, shape, nullptr, false, values));

    state.tensor_type = ReferenceTensorTypeFromOnnxDataType(tensor.data_type(), state.tensor_type);
    state.shape = shape;
    state.values = std::move(values);
  } else if (value_float_attr != attributes.end() &&
             value_float_attr->second.type() ==
                 ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT) {
    state.tensor_type = "float32";
    state.has_quant = false;
    state.shape = shape_for_value_count(1);
    state.values = ValuesToJson(std::vector<float>{value_float_attr->second.f()}, state.shape);
  } else if (value_floats_attr != attributes.end() &&
             value_floats_attr->second.type() ==
                 ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOATS) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(value_floats_attr->second.floats_size()));
    for (float value : value_floats_attr->second.floats()) {
      values.push_back(value);
    }
    state.tensor_type = "float32";
    state.has_quant = false;
    state.shape = shape_for_value_count(values.size());
    state.values = ValuesToJson(values, state.shape);
  } else if (value_int_attr != attributes.end() &&
             value_int_attr->second.type() ==
                 ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT) {
    state.tensor_type = "int32";
    state.shape = shape_for_value_count(1);
    state.values = ValuesToJson(std::vector<int>{static_cast<int>(value_int_attr->second.i())}, state.shape);
  } else if (value_ints_attr != attributes.end() &&
             value_ints_attr->second.type() ==
                 ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INTS) {
    std::vector<int> values;
    values.reserve(static_cast<size_t>(value_ints_attr->second.ints_size()));
    for (int64_t value : value_ints_attr->second.ints()) {
      values.push_back(static_cast<int>(value));
    }
    state.tensor_type = "int32";
    state.shape = shape_for_value_count(values.size());
    state.values = ValuesToJson(values, state.shape);
  } else {
    return Status::OK();
  }

  states[outputs[0]->Name()] = std::move(state);
  return Status::OK();
}

Status EmitReferenceUnsqueezeInitial(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     std::unordered_map<std::string, TensorState>& states,
                                     ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Unsqueeze node.");

  std::vector<int> axes;
  if (inputs.size() >= 2 && inputs[1] != nullptr && inputs[1]->Exists()) {
    ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), axes));
  } else {
    axes = GetIntsAttribute(node, "axes", {});
  }

  const auto* initializer = graph_viewer.GetConstantInitializer(inputs[0]->Name(), true);
  if (initializer != nullptr) {
    const std::vector<int> source_shape = TensorProtoShape(*initializer);
    std::vector<int> target_shape = source_shape;
    std::sort(axes.begin(), axes.end());
    for (int axis : axes) {
      int normalized_axis = axis < 0 ? axis + static_cast<int>(target_shape.size()) + 1 : axis;
      normalized_axis = std::max(0, std::min(normalized_axis, static_cast<int>(target_shape.size())));
      target_shape.insert(target_shape.begin() + normalized_axis, 1);
    }
    if (source_shape.size() == 1 && target_shape.size() == 3 &&
        target_shape[0] == source_shape[0] && target_shape[1] == 1 && target_shape[2] == 1) {
      target_shape = {1, 1, source_shape[0]};
    }

    TensorState state;
    state.name = ReferenceTensorName(outputs[0]->Name());
    state.tensor_type = ReferenceTensorTypeFromOnnxDataType(
        initializer->data_type(), ReferenceTensorTypeFromNodeArg(outputs[0], "float32"));
    state.shape = target_shape;
    state.is_constant = true;
    state.has_values = true;
    ORT_RETURN_IF_ERROR(TensorValuesJson(*initializer, graph_viewer.ModelPath(), source_shape, target_shape,
                                         nullptr, false, state.values));
    MarkFloatTensorNoQuant(state);
    states[outputs[0]->Name()] = std::move(state);
    return Status::OK();
  }

  TensorState input_state;
  ORT_RETURN_IF_ERROR(MakeReferenceInputTensorState(graph_viewer, inputs[0], true, "float32",
                                                   states, input_state));
  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.tensor_type = ReferenceTensorTypeFromNodeArg(outputs[0], input_state.tensor_type);
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), true);
  if (output_state.shape.empty()) {
    output_state.shape = input_state.shape;
    std::sort(axes.begin(), axes.end());
    for (int axis : axes) {
      int normalized_axis = axis < 0 ? axis + static_cast<int>(output_state.shape.size()) + 1 : axis;
      normalized_axis = std::max(0, std::min(normalized_axis, static_cast<int>(output_state.shape.size())));
      output_state.shape.insert(output_state.shape.begin() + normalized_axis, 1);
    }
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  MarkFloatTensorNoQuant(output_state);
  states[outputs[0]->Name()] = output_state;

  ordered_json shape_values = ordered_json::array();
  for (int dim : output_state.shape) {
    shape_values.push_back(dim);
  }
  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(output_state.name + "_unsqueeze_shape", "int32",
                                                          {static_cast<int>(output_state.shape.size())},
                                                          shape_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("reshape", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceExpandInitial(const GraphViewer& graph_viewer,
                                  const Node& node,
                                  std::unordered_map<std::string, TensorState>& states,
                                  ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid Expand node.");

  constexpr size_t kMaxFoldedExpandElements = 1000000;
  const std::string canonical_input_name = CanonicalOnnxTensorName(inputs[0]->Name());
  const auto* initializer = graph_viewer.GetConstantInitializer(canonical_input_name, true);
  if (initializer != nullptr) {
    const std::vector<int> source_shape = TensorProtoShape(*initializer);
    const std::vector<int> target_shape = ResolveExpandOutputShape(graph_viewer, node, states, source_shape);
    const size_t target_count = ElementCount(target_shape);
    if (target_count > 0 && target_count <= kMaxFoldedExpandElements) {
      ordered_json values;
      const Status status = BroadcastInitializerValuesJson(*initializer, graph_viewer.ModelPath(),
                                                           source_shape, target_shape, values);
      if (status.IsOK()) {
        TensorState state;
        state.name = ReferenceTensorName(outputs[0]->Name());
        state.tensor_type = ReferenceTensorTypeFromOnnxDataType(
            initializer->data_type(), ReferenceTensorTypeFromNodeArg(outputs[0], "float32"));
        state.shape = target_shape;
        state.is_constant = true;
        state.has_values = true;
        state.values = std::move(values);
        MarkFloatTensorNoQuant(state);
        states[outputs[0]->Name()] = std::move(state);
        return Status::OK();
      }
    }
  }

  const auto* input_state = FindExistingState(states, inputs[0]->Name());
  if (input_state != nullptr && input_state->is_constant && input_state->has_values) {
    const std::vector<int> target_shape = ResolveExpandOutputShape(graph_viewer, node, states, input_state->shape);
    const size_t target_count = ElementCount(target_shape);
    if (target_count > 0 && target_count <= kMaxFoldedExpandElements) {
      ordered_json values;
      const Status status = BroadcastStateValuesJson(*input_state, target_shape, values);
      if (status.IsOK()) {
        TensorState state = *input_state;
        state.name = ReferenceTensorName(outputs[0]->Name());
        state.tensor_type = ReferenceTensorTypeFromNodeArg(outputs[0], input_state->tensor_type);
        state.shape = target_shape;
        state.is_constant = true;
        state.has_values = true;
        state.values = std::move(values);
        MarkFloatTensorNoQuant(state);
        states[outputs[0]->Name()] = std::move(state);
        return Status::OK();
      }
    }
  }

  return EmitReferenceGenericInitial(graph_viewer, node, "BroadcastTo", true, ordered_json::object(),
                                     states, ops_json);
}

Status EmitReferenceReshapeInitial(const GraphViewer& graph_viewer,
                                   const Node& node,
                                   std::unordered_map<std::string, TensorState>& states,
                                   ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 2 && outputs.size() >= 1 &&
                        inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                    "Invalid Reshape node.");

  TensorState input_state;
  if (const auto* existing_input_state = FindExistingState(states, inputs[0]->Name())) {
    input_state = *existing_input_state;
  } else {
    input_state = MakeFloatTensorState(graph_viewer, inputs[0]->Name(), false);
    input_state.has_quant = true;
    input_state.quant_scale = "None";
    input_state.zero_point = "None";
  }
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  if (output_state.shape.empty()) {
    std::vector<int> requested_shape;
    if (const auto* shape_state = FindExistingState(states, inputs[1]->Name())) {
      TryJsonToIntVector(shape_state->values, requested_shape);
    }
    if (requested_shape.empty() && graph_viewer.GetConstantInitializer(inputs[1]->Name(), true) != nullptr) {
      ORT_RETURN_IF_ERROR(InitializerAsIntVector(graph_viewer, inputs[1]->Name(), requested_shape));
    }
    if (!requested_shape.empty()) {
      output_state.shape = ResolveReshapeShape(input_state.shape, requested_shape,
                                               GetIntAttribute(node, "allowzero", 0) != 0);
    }
  }
  output_state.is_constant = false;
  output_state.has_values = false;
  if (!output_state.has_quant) {
    output_state.has_quant = true;
    output_state.quant_scale = "None";
    output_state.zero_point = "None";
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json new_shape = ordered_json::array();
  for (int dim : output_state.shape) {
    new_shape.push_back(dim);
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Reshape");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(
      output_state.name + "_shape_values", "int32", {static_cast<int>(output_state.shape.size())}, new_shape);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  ops_json.push_back(WrapOp("reshape", std::move(payload)));
  return Status::OK();
}

Status EmitReferenceTransposeInitial(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     std::unordered_map<std::string, TensorState>& states,
                                     ordered_json& ops_json) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  ORT_RETURN_IF_NOT(inputs.size() >= 1 && outputs.size() >= 1 && inputs[0] != nullptr && outputs[0] != nullptr,
                    "Invalid Transpose node.");

  TensorState input_state;
  if (const auto* existing_input_state = FindExistingState(states, inputs[0]->Name())) {
    input_state = *existing_input_state;
  } else {
    input_state = MakeFloatTensorState(graph_viewer, inputs[0]->Name(), false);
    input_state.has_quant = true;
    input_state.quant_scale = "None";
    input_state.zero_point = "None";
  }
  states[inputs[0]->Name()] = input_state;

  TensorState output_state = input_state;
  output_state.name = ReferenceTensorName(outputs[0]->Name());
  output_state.shape = ShapeForTensor(graph_viewer, outputs[0]->Name(), false);
  output_state.is_constant = false;
  output_state.has_values = false;
  if (!output_state.has_quant) {
    output_state.has_quant = true;
    output_state.quant_scale = "None";
    output_state.zero_point = "None";
  }
  std::vector<int> perm = GetIntsAttribute(node, "perm", {});
  if (!input_state.shape.empty() && !perm.empty() && perm.size() == input_state.shape.size()) {
    output_state.shape = ApplyPermToShape(input_state.shape, perm);
  }
  states[outputs[0]->Name()] = output_state;

  ordered_json perm_values = ordered_json::array();
  for (int value : perm) {
    perm_values.push_back(value);
  }

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Transpose");
  payload["input_tensor_info_1"] = TensorInfoJson(input_state);
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(input_state.name + "_perm_values", "int32",
                                                          {static_cast<int>(perm.size())}, perm_values);
  payload["output_tensor_info"] = TensorInfoJson(output_state);
  if (input_state.shape.size() == 4) {
    payload["add_Transpose"] = true;
  }
  ops_json.push_back(WrapOp("Transpose", std::move(payload)));
  return Status::OK();
}

Status CollectInitialReferenceStyleOps(const GraphViewer& graph_viewer,
                                       ordered_json& ops_json) {
  std::unordered_map<std::string, TensorState> states;
  const int max_node_index = graph_viewer.MaxNodeIndex();
  for (int node_index = 0; node_index < max_node_index; ++node_index) {
    const Node* node = graph_viewer.GetNode(static_cast<NodeIndex>(node_index));
    if (node == nullptr || IsOrtInsertedQdqHelperNode(*node)) {
      continue;
    }

    const std::string& op_type = node->OpType();
    if (op_type == "QuantizeLinear") {
      ORT_RETURN_IF_ERROR(EmitReferenceQuantizeLinearInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "QLinearConv") {
      ORT_RETURN_IF_ERROR(EmitReferenceQLinearConvInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Identity") {
      ORT_RETURN_IF_ERROR(EmitReferenceIdentityInitial(graph_viewer, *node, states));
    } else if (op_type == "Pad") {
      ORT_RETURN_IF_ERROR(EmitReferencePadInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Conv") {
      ORT_RETURN_IF_ERROR(EmitReferenceConvInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "ConvTranspose") {
      ORT_RETURN_IF_ERROR(EmitReferenceConvTransposeInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "LRN") {
      ORT_RETURN_IF_ERROR(EmitReferenceLRNInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "QLinearSigmoid") {
      ORT_RETURN_IF_ERROR(EmitReferenceQLinearUnaryInitial(graph_viewer, *node, "QLinearSigmoid", states, ops_json));
    } else if (op_type == "QLinearSoftmax") {
      ORT_RETURN_IF_ERROR(EmitReferenceQLinearUnaryInitial(graph_viewer, *node, "QLinearSoftmax", states, ops_json));
    } else if (op_type == "Relu") {
      ORT_RETURN_IF_ERROR(EmitReferenceReluInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "PRelu") {
      ORT_RETURN_IF_ERROR(EmitReferencePReluInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "LeakyRelu") {
      ORT_RETURN_IF_ERROR(EmitReferenceLeakyReluInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "HardSwish") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatUnaryInitial(graph_viewer, *node, "HardSwish", states, ops_json));
    } else if (op_type == "HardSigmoid") {
      ORT_RETURN_IF_ERROR(EmitReferenceHardSigmoidInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Clip") {
      ORT_RETURN_IF_ERROR(EmitReferenceClipInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Sigmoid") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatUnaryInitial(graph_viewer, *node, "Sigmoid", states, ops_json));
    } else if (op_type == "Tanh") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatUnaryInitial(graph_viewer, *node, "Tanh", states, ops_json));
    } else if (op_type == "Erf") {
      ORT_RETURN_IF_ERROR(EmitReferenceErfInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Softmax") {
      ORT_RETURN_IF_ERROR(EmitReferenceSoftmaxInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Exp") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatUnaryInitial(graph_viewer, *node, "Exp", states, ops_json));
    } else if (op_type == "Log") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatUnaryInitial(graph_viewer, *node, "Log", states, ops_json));
    } else if (op_type == "Sqrt") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatUnaryInitial(graph_viewer, *node, "Sqrt", states, ops_json));
    } else if (op_type == "QLinearMul") {
      ORT_RETURN_IF_ERROR(EmitReferenceQLinearBinaryInitial(graph_viewer, *node, "QLinearMul", states, ops_json));
    } else if (op_type == "Mul") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatBinaryInitial(graph_viewer, *node, "Mul", states, ops_json));
    } else if (op_type == "Pow") {
      ORT_RETURN_IF_ERROR(EmitReferenceFloatBinaryInitial(graph_viewer, *node, "Pow", states, ops_json));
    } else if (op_type == "QLinearAdd") {
      ORT_RETURN_IF_ERROR(EmitReferenceQLinearBinaryInitial(graph_viewer, *node, "QLinearAdd", states, ops_json));
    } else if (op_type == "Add") {
      ORT_RETURN_IF_ERROR(EmitReferenceAddInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Concat") {
      ORT_RETURN_IF_ERROR(EmitReferenceConcatInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "BatchNormalization") {
      ORT_RETURN_IF_ERROR(EmitReferenceBatchNormalizationInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Dropout") {
      ORT_RETURN_IF_ERROR(EmitReferenceDropoutInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Constant") {
      ORT_RETURN_IF_ERROR(EmitReferenceConstantInitial(graph_viewer, *node, states));
    } else if (op_type == "Squeeze") {
      ORT_RETURN_IF_ERROR(EmitReferenceSqueezeInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Unsqueeze") {
      ORT_RETURN_IF_ERROR(EmitReferenceUnsqueezeInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "QLinearGlobalAveragePool") {
      ORT_RETURN_IF_ERROR(EmitReferenceQLinearGlobalAveragePoolInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "AveragePool") {
      ORT_RETURN_IF_ERROR(EmitReferenceAveragePoolInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "GlobalAveragePool") {
      ORT_RETURN_IF_ERROR(EmitReferenceGlobalAveragePoolInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "InstanceNormalization") {
      ORT_RETURN_IF_ERROR(EmitReferenceInstanceNormalizationInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "ReduceMean") {
      ORT_RETURN_IF_ERROR(EmitReferenceReduceMeanInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "ReduceMax") {
      ORT_RETURN_IF_ERROR(EmitReferenceReduceMaxInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "ReduceSum") {
      ORT_RETURN_IF_ERROR(EmitReferenceReduceSumInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Split") {
      ORT_RETURN_IF_ERROR(EmitReferenceSplitInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "QLinearConcat") {
      ORT_RETURN_IF_ERROR(EmitReferenceQLinearConcatInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Resize") {
      // Only switch the well-defined ONNX linear/half_pixel combination to
      // bilinear. Other variants retain the prior nearest export behavior.
      if (GetStringAttribute(*node, "mode", "nearest") == "linear" &&
          GetStringAttribute(*node, "coordinate_transformation_mode", "half_pixel") == "half_pixel") {
        ORT_RETURN_IF_ERROR(EmitReferenceResizeLinearHalfPixelInitial(graph_viewer, *node, states, ops_json));
      } else {
        ORT_RETURN_IF_ERROR(EmitReferenceResizeNearestInitial(graph_viewer, *node, states, ops_json));
      }
    } else if (op_type == "Upsample") {
      ORT_RETURN_IF_ERROR(EmitReferenceUpsampleNearestInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "MaxPool") {
      ORT_RETURN_IF_ERROR(EmitReferenceMaxPoolInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Flatten") {
      ORT_RETURN_IF_ERROR(EmitReferenceFlattenInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "QGemm") {
      ORT_RETURN_IF_ERROR(EmitReferenceQGemmInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Gemm") {
      ORT_RETURN_IF_ERROR(EmitReferenceGemmInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "MatMul") {
      ORT_RETURN_IF_ERROR(EmitReferenceMatMulInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Reshape") {
      ORT_RETURN_IF_ERROR(EmitReferenceReshapeInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Transpose") {
      ORT_RETURN_IF_ERROR(EmitReferenceTransposeInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "DequantizeLinear") {
      ORT_RETURN_IF_ERROR(EmitReferenceDequantizeLinearInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Slice") {
      ORT_RETURN_IF_ERROR(EmitReferenceSliceInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "Cast") {
      ORT_RETURN_IF_ERROR(EmitReferenceGenericInitial(graph_viewer, *node, "Cast", true, ordered_json::object(),
                                                     states, ops_json));
    } else if (op_type == "Gather" || op_type == "GatherElements") {
      ORT_RETURN_IF_ERROR(EmitReferenceGatherInitial(graph_viewer, *node, op_type, states, ops_json));
    } else if (op_type == "Tile") {
      ORT_RETURN_IF_ERROR(EmitReferenceGenericInitial(graph_viewer, *node, "Tile", true, ordered_json::object(),
                                                      states, ops_json));
    } else if (op_type == "Expand") {
      ORT_RETURN_IF_ERROR(EmitReferenceExpandInitial(graph_viewer, *node, states, ops_json));
    } else if (op_type == "TopK") {
      ORT_RETURN_IF_ERROR(EmitReferenceGenericInitial(graph_viewer, *node, "TopK", true, ordered_json::object(),
                                                      states, ops_json));
    } else if (op_type == "Mod") {
      ORT_RETURN_IF_ERROR(EmitReferencePlainBinaryInitial(graph_viewer, *node, "Mod", states, ops_json));
    } else if (op_type == "Sub") {
      ORT_RETURN_IF_ERROR(EmitReferencePlainBinaryInitial(graph_viewer, *node, "Sub", states, ops_json));
    } else if (op_type == "Div") {
      ORT_RETURN_IF_ERROR(EmitReferencePlainBinaryInitial(graph_viewer, *node, "Div", states, ops_json));
    } else if (!IsReferenceStyleSupportedOp(op_type)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op for Amlogic reference_style export: ",
                             op_type, ". Node: ", node->Name());
    }
  }

  return Status::OK();
}
