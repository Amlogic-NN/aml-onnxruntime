// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside the anonymous namespace. It keeps the stable quant/state construction
// helpers shared by the legacy qlinear exporter and reference_style support
// logic in one place without changing behavior.

Status ReadQuantInfo(const GraphViewer& graph_viewer,
                     const std::string& scale_name,
                     const std::string& zero_point_name,
                     QuantInfo& quant) {
  ORT_RETURN_IF_ERROR(InitializerAsFloatJson(graph_viewer, scale_name, quant.scale));
  ORT_RETURN_IF_ERROR(InitializerAsZeroPointJson(graph_viewer, zero_point_name, quant.zero_point));
  return Status::OK();
}

TensorState MakeFloatTensorState(const GraphViewer& graph_viewer,
                                 const std::string& tensor_name,
                                 bool nhwc) {
  const std::string canonical_name = CanonicalOnnxTensorName(tensor_name);
  TensorState state;
  state.name = ReferenceTensorName(canonical_name);
  state.tensor_type = "float32";
  state.shape = ShapeForTensor(graph_viewer, canonical_name, nhwc);
  state.is_constant = graph_viewer.GetConstantInitializer(canonical_name, true) != nullptr;
  return state;
}

Status MakeQuantTensorState(const GraphViewer& graph_viewer,
                            const std::string& tensor_name,
                            const std::string& scale_name,
                            const std::string& zero_point_name,
                            bool nhwc,
                            TensorState& state) {
  const std::string canonical_name = CanonicalOnnxTensorName(tensor_name);
  state.name = ReferenceTensorName(canonical_name);
  state.tensor_type = "int8";
  state.shape = ShapeForTensor(graph_viewer, canonical_name, nhwc);
  state.is_constant = graph_viewer.GetConstantInitializer(canonical_name, true) != nullptr;
  state.has_quant = true;
  ORT_RETURN_IF_ERROR(InitializerAsFloatJson(graph_viewer, scale_name, state.quant_scale));
  ORT_RETURN_IF_ERROR(InitializerAsZeroPointJson(graph_viewer, zero_point_name, state.zero_point));

  if (state.is_constant) {
    const auto* tensor = graph_viewer.GetConstantInitializer(canonical_name, true);
    ordered_json values;
    const auto shape = TensorProtoShape(*tensor);
    const bool uint8_to_int8 = tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), shape, shape, nullptr,
                                         uint8_to_int8, values));
    state.has_values = true;
    state.values = std::move(values);
  }
  return Status::OK();
}

Status MakeWeightTensorState(const GraphViewer& graph_viewer,
                             const std::string& weight_name,
                             const std::string& scale_name,
                             const std::string& zero_point_name,
                             TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(weight_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing QLinearConv weight initializer: ", weight_name);

  const std::vector<int> source_shape = TensorProtoShape(*tensor);
  const std::vector<int> target_shape = ConvWeightOihwToOhwiShape(source_shape);
  const std::array<int, 4> oihw_to_ohwi = {0, 2, 3, 1};
  const std::array<int, 4>* perm = source_shape.size() == 4 ? &oihw_to_ohwi : nullptr;

  ordered_json values;
  const bool uint8_to_int8 = tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8;
  ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), source_shape, target_shape, perm,
                                       uint8_to_int8, values));

  state.name = weight_name;
  state.tensor_type = "int8";
  state.shape = target_shape;
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

Status MakeFloatInitializerTensorState(const GraphViewer& graph_viewer,
                                       const std::string& tensor_name,
                                       bool conv_oihw_to_ohwi,
                                       TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(tensor_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing float initializer: ", tensor_name);

  const std::vector<int> source_shape = TensorProtoShape(*tensor);
  const std::vector<int> target_shape = conv_oihw_to_ohwi
                                            ? ConvWeightOihwToOhwiShape(source_shape)
                                            : source_shape;
  const std::array<int, 4> oihw_to_ohwi = {0, 2, 3, 1};
  const std::array<int, 4>* perm = conv_oihw_to_ohwi && source_shape.size() == 4 ? &oihw_to_ohwi : nullptr;

  ordered_json values;
  ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), source_shape, target_shape,
                                       perm, false, values));

  state.name = tensor_name;
  state.tensor_type = "float32";
  state.shape = target_shape;
  state.is_constant = true;
  state.has_values = true;
  state.values = std::move(values);
  return Status::OK();
}

Status MakeGemmFloatWeightTensorState(const GraphViewer& graph_viewer,
                                      const std::string& tensor_name,
                                      int trans_b,
                                      TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(tensor_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing Gemm float initializer: ", tensor_name);

  const std::vector<int> source_shape = TensorProtoShape(*tensor);
  ORT_RETURN_IF_NOT(source_shape.size() == 2,
                    "Gemm weight initializer must be rank 2 for reference_style export: ",
                    tensor_name);

  ordered_json values;
  std::vector<int> target_shape = source_shape;
  if (trans_b == 0) {
    std::vector<float> flat;
    ORT_RETURN_IF_ERROR(UnpackVector<float>(*tensor, graph_viewer.ModelPath(), flat));
    ORT_RETURN_IF_NOT(flat.size() == ElementCount(source_shape),
                      "Gemm weight initializer size mismatch: ", tensor_name);
    target_shape = {source_shape[1], source_shape[0]};
    std::vector<float> transposed(flat.size());
    for (int row = 0; row < source_shape[0]; ++row) {
      for (int col = 0; col < source_shape[1]; ++col) {
        transposed[static_cast<size_t>(col) * static_cast<size_t>(source_shape[0]) + static_cast<size_t>(row)] =
            flat[static_cast<size_t>(row) * static_cast<size_t>(source_shape[1]) + static_cast<size_t>(col)];
      }
    }
    values = ValuesToJson(transposed, target_shape);
  } else {
    ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), source_shape, target_shape,
                                         nullptr, false, values));
  }

  state.name = tensor_name;
  state.tensor_type = "float32";
  state.shape = target_shape;
  state.is_constant = true;
  state.has_values = true;
  state.values = std::move(values);
  return Status::OK();
}

Status MakeDepthwiseFloatWeightTensorState(const GraphViewer& graph_viewer,
                                           const std::string& tensor_name,
                                           TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(tensor_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing depthwise float initializer: ", tensor_name);

  const std::vector<int> source_shape = TensorProtoShape(*tensor);
  ORT_RETURN_IF_NOT(source_shape.size() == 4, "Depthwise Conv weight must be OIHW rank 4: ", tensor_name);

  const std::vector<int> target_shape = {source_shape[1], source_shape[2], source_shape[3], source_shape[0]};
  const std::array<int, 4> oihw_to_ihwo = {1, 2, 3, 0};

  ordered_json values;
  ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), source_shape, target_shape,
                                       &oihw_to_ihwo, false, values));

  state.name = tensor_name;
  state.tensor_type = "float32";
  state.shape = target_shape;
  state.is_constant = true;
  state.has_values = true;
  state.values = std::move(values);
  return Status::OK();
}

Status MakeGroupedConvFloatWeightTensorState(const GraphViewer& graph_viewer,
                                             const std::string& tensor_name,
                                             int group_index,
                                             int group_count,
                                             TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(tensor_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing grouped Conv float initializer: ", tensor_name);

  const std::vector<int> source_shape = TensorProtoShape(*tensor);
  ORT_RETURN_IF_NOT(source_shape.size() == 4,
                    "Grouped Conv weight must be OIHW rank 4: ", tensor_name);
  ORT_RETURN_IF_NOT(group_count > 1 && group_index >= 0 && group_index < group_count,
                    "Invalid grouped Conv group index/count for initializer: ", tensor_name);
  ORT_RETURN_IF_NOT(source_shape[0] % group_count == 0,
                    "Grouped Conv output channels must be divisible by group count: ", tensor_name);

  std::vector<float> flat;
  ORT_RETURN_IF_ERROR(UnpackVector<float>(*tensor, graph_viewer.ModelPath(), flat));
  ORT_RETURN_IF_NOT(flat.size() == ElementCount(source_shape),
                    "Grouped Conv weight initializer size mismatch: ", tensor_name);

  const int out_per_group = source_shape[0] / group_count;
  const int in_per_group = source_shape[1];
  const int kernel_h = source_shape[2];
  const int kernel_w = source_shape[3];
  const std::vector<int> target_shape = {out_per_group, kernel_h, kernel_w, in_per_group};
  std::vector<float> grouped(static_cast<size_t>(out_per_group) *
                             static_cast<size_t>(kernel_h) *
                             static_cast<size_t>(kernel_w) *
                             static_cast<size_t>(in_per_group));

  const auto source_offset = [&source_shape](int o, int i, int h, int w) {
    return (((static_cast<size_t>(o) * source_shape[1] + static_cast<size_t>(i)) *
             source_shape[2] + static_cast<size_t>(h)) *
            source_shape[3] + static_cast<size_t>(w));
  };
  const auto target_offset = [&target_shape](int o, int h, int w, int i) {
    return (((static_cast<size_t>(o) * target_shape[1] + static_cast<size_t>(h)) *
             target_shape[2] + static_cast<size_t>(w)) *
            target_shape[3] + static_cast<size_t>(i));
  };

  const int source_o_begin = group_index * out_per_group;
  for (int o = 0; o < out_per_group; ++o) {
    for (int h = 0; h < kernel_h; ++h) {
      for (int w = 0; w < kernel_w; ++w) {
        for (int i = 0; i < in_per_group; ++i) {
          grouped[target_offset(o, h, w, i)] =
              flat[source_offset(source_o_begin + o, i, h, w)];
        }
      }
    }
  }

  state.name = tensor_name + "_group_" + std::to_string(group_index);
  state.tensor_type = "float32";
  state.shape = target_shape;
  state.is_constant = true;
  state.has_values = true;
  state.values = ValuesToJson(grouped, target_shape);
  return Status::OK();
}

Status MakeGroupedFloatBiasTensorState(const GraphViewer& graph_viewer,
                                       const std::string& tensor_name,
                                       int group_index,
                                       int group_count,
                                       TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(tensor_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing grouped Conv bias initializer: ", tensor_name);

  const std::vector<int> source_shape = TensorProtoShape(*tensor);
  ORT_RETURN_IF_NOT(source_shape.size() == 1,
                    "Grouped Conv bias must be rank 1: ", tensor_name);
  ORT_RETURN_IF_NOT(group_count > 1 && group_index >= 0 && group_index < group_count,
                    "Invalid grouped Conv group index/count for bias: ", tensor_name);
  ORT_RETURN_IF_NOT(source_shape[0] % group_count == 0,
                    "Grouped Conv bias channels must be divisible by group count: ", tensor_name);

  std::vector<float> flat;
  ORT_RETURN_IF_ERROR(UnpackVector<float>(*tensor, graph_viewer.ModelPath(), flat));
  ORT_RETURN_IF_NOT(flat.size() == ElementCount(source_shape),
                    "Grouped Conv bias initializer size mismatch: ", tensor_name);

  const int out_per_group = source_shape[0] / group_count;
  const int begin = group_index * out_per_group;
  std::vector<float> grouped(flat.begin() + begin, flat.begin() + begin + out_per_group);

  state.name = tensor_name + "_group_" + std::to_string(group_index);
  state.tensor_type = "float32";
  state.shape = {out_per_group};
  state.is_constant = true;
  state.has_values = true;
  state.values = ValuesToJson(grouped, state.shape);
  return Status::OK();
}

TensorState MakeSyntheticFloatBiasTensorState(const std::string& weight_name,
                                              int output_channels) {
  const int bias_size = std::max(output_channels, 1);
  ordered_json values = ordered_json::array();
  for (int i = 0; i < bias_size; ++i) {
    values.push_back(0.0f);
  }

  TensorState state;
  state.name = weight_name + "_bias_values";
  state.tensor_type = "float32";
  state.shape = {bias_size};
  state.is_constant = true;
  state.has_values = true;
  state.values = std::move(values);
  return state;
}

Status MakeBiasTensorState(const GraphViewer& graph_viewer,
                           const std::string& bias_name,
                           const ordered_json& input_scale,
                           const ordered_json& weight_scale,
                           TensorState& state) {
  const auto* tensor = graph_viewer.GetConstantInitializer(bias_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing QLinearConv bias initializer: ", bias_name);

  const std::vector<int> shape = TensorProtoShape(*tensor);
  ordered_json values;
  ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), shape, shape, nullptr, false, values));

  state.name = bias_name;
  state.tensor_type = "int32";
  state.shape = shape;
  state.is_constant = true;
  state.has_values = true;
  state.values = std::move(values);
  state.has_quant = true;
  state.zero_point = ordered_json::array();

  if (weight_scale.is_array()) {
    float x_scale = input_scale.is_number() ? input_scale.get<float>() : 0.0f;
    state.quant_scale = ordered_json::array();
    for (const auto& ws : weight_scale) {
      state.quant_scale.push_back(static_cast<float>(x_scale * ws.get<float>()));
      state.zero_point.push_back(0);
    }
  } else {
    state.quant_scale = input_scale.is_number() && weight_scale.is_number()
                            ? ordered_json(static_cast<float>(input_scale.get<float>() * weight_scale.get<float>()))
                            : ordered_json("None");
    state.zero_point.push_back(0);
  }
  return Status::OK();
}

TensorState MakeSyntheticBiasTensorState(const std::string& weight_name) {
  TensorState state;
  state.name = weight_name + "_bias_values";
  state.tensor_type = "int32";
  state.shape = {1};
  state.is_constant = true;
  state.has_values = true;
  state.values = ordered_json::array({0});
  state.has_quant = true;
  state.quant_scale = "None";
  state.zero_point = "None";
  return state;
}

Status MakeInitializerTensorInfo(const GraphViewer& graph_viewer,
                                 const std::string& tensor_name,
                                 const std::string& reference_name,
                                 const std::string& tensor_type,
                                 bool quantized_uint8_to_int8,
                                 ordered_json& tensor_info) {
  const auto* tensor = graph_viewer.GetConstantInitializer(tensor_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing initializer: ", tensor_name);
  const auto shape = TensorProtoShape(*tensor);
  ordered_json values;
  ORT_RETURN_IF_ERROR(TensorValuesJson(*tensor, graph_viewer.ModelPath(), shape, shape, nullptr,
                                       quantized_uint8_to_int8, values));
  tensor_info = ConstantTensorInfoJson(reference_name, tensor_type, shape, values);
  return Status::OK();
}

Status InitializerAsIntVector(const GraphViewer& graph_viewer,
                              const std::string& tensor_name,
                              std::vector<int>& values) {
  const auto* tensor = graph_viewer.GetConstantInitializer(tensor_name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing int initializer: ", tensor_name);

  switch (tensor->data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      std::vector<int32_t> raw;
      ORT_RETURN_IF_ERROR(UnpackVector<int32_t>(*tensor, graph_viewer.ModelPath(), raw));
      values = ToIntVector(raw);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      std::vector<int64_t> raw;
      ORT_RETURN_IF_ERROR(UnpackVector<int64_t>(*tensor, graph_viewer.ModelPath(), raw));
      values = ToIntVector(raw);
      return Status::OK();
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported int initializer type for ", tensor_name);
  }
}
