// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside namespace reference_style_internal. It keeps the reference_style
// postprocess logic in a dedicated module without changing behavior.

ordered_json CloneTensorWithNameAndShape(const ordered_json& tensor_info,
                                         const std::string& tensor_name,
                                         const std::vector<int>& shape);
ordered_json MakeReferenceTransposeNchwToNhwcOp(const ordered_json& input_tensor,
                                                const ordered_json& output_tensor,
                                                const std::string& perm_tensor_name);
bool IsYoloAttentionPeDepthwiseConvNeedingNhwcInput(const std::string& op_name,
                                                    const ordered_json& payload,
                                                    const ordered_json& source_tensor);
void SyncQLinearConcatProducerQuantInfo(ordered_json& ops_json);

Status PostprocessReferenceQuantScaleFill(ordered_json& ops_json) {
  auto run_fill_pass = [&ops_json]() -> Status {
    for (size_t index = 0; index < ops_json.size(); ++index) {
      auto* payload = GetWrappedPayload(ops_json[index]);
      if (payload == nullptr) {
        continue;
      }

      for (const auto& input_key : InputTensorInfoKeys(*payload)) {
        auto input_it = payload->find(input_key);
        if (input_it != payload->end() &&
            input_it->is_object() &&
            input_it->contains("quant_scale") &&
            IsQuantNoneValue((*input_it)["quant_scale"])) {
          ORT_RETURN_IF_ERROR(MatchQuantScale(ops_json, "input", index, input_key));
        }
      }

      for (const auto& output_key : OutputTensorInfoKeys(*payload)) {
        auto output_it = payload->find(output_key);
        if (output_it != payload->end() &&
            output_it->is_object() &&
            output_it->contains("quant_scale") &&
            IsQuantNoneValue((*output_it)["quant_scale"])) {
          ORT_RETURN_IF_ERROR(MatchQuantScale(ops_json, "output", index, output_key));
        }
      }
    }

    for (auto& op_json : ops_json) {
      const std::string op_name = GetWrappedOpType(op_json);
      auto* payload = GetWrappedPayload(op_json);
      if (payload == nullptr) {
        continue;
      }

      if (op_name == "Split" &&
          payload->contains("output_tensor_info_1") &&
          payload->contains("output_tensor_info_2")) {
        ordered_json& output_1 = (*payload)["output_tensor_info_1"];
        ordered_json& output_2 = (*payload)["output_tensor_info_2"];
        if (output_1.contains("quant_scale") && output_2.contains("quant_scale")) {
          if (IsQuantNoneValue(output_1["quant_scale"]) &&
              !IsQuantNoneValue(output_2["quant_scale"])) {
            CopyQuantInfo(output_2, output_1);
          } else if (!IsQuantNoneValue(output_1["quant_scale"]) &&
                     IsQuantNoneValue(output_2["quant_scale"])) {
            CopyQuantInfo(output_1, output_2);
          }
        }
      }

      if ((op_name == "reshape" || op_name == "Transpose") &&
          payload->contains("input_tensor_info_1") &&
          payload->contains("output_tensor_info")) {
        ordered_json& input_tensor = (*payload)["input_tensor_info_1"];
        ordered_json& output_tensor = (*payload)["output_tensor_info"];
        if (input_tensor.contains("quant_scale") && output_tensor.contains("quant_scale")) {
          if (IsQuantNoneValue(input_tensor["quant_scale"]) &&
              !IsQuantNoneValue(output_tensor["quant_scale"])) {
            CopyQuantInfo(output_tensor, input_tensor);
          } else if (!IsQuantNoneValue(input_tensor["quant_scale"]) &&
                     IsQuantNoneValue(output_tensor["quant_scale"])) {
            CopyQuantInfo(input_tensor, output_tensor);
          }
        }
      }
    }

    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(run_fill_pass());
  ORT_RETURN_IF_ERROR(run_fill_pass());
  ORT_RETURN_IF_ERROR(run_fill_pass());
  return Status::OK();
}

void PostprocessReferenceMaxPoolTensorInfo(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        GetWrappedOpType(ops_json[index]) != "MaxPool" ||
        !payload->contains("input_tensor_info") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& input_tensor = (*payload)["input_tensor_info"];
    ordered_json& output_tensor = (*payload)["output_tensor_info"];

    if (static_cast<int>(index) > 0) {
      const auto* prev_payload = GetWrappedPayload(ops_json[index - 1]);
      if (prev_payload != nullptr && prev_payload->contains("output_tensor_info")) {
        const ordered_json& prev_output = (*prev_payload)["output_tensor_info"];
        CopyTensorType(prev_output, input_tensor);
        if (HasRealQuantInfo(prev_output)) {
          CopyQuantInfo(prev_output, input_tensor);
        }
      }
    }

    if (index + 1 < ops_json.size()) {
      const auto* next_payload = GetWrappedPayload(ops_json[index + 1]);
      if (next_payload != nullptr) {
        const auto input_keys = InputTensorInfoKeys(*next_payload);
        if (!input_keys.empty()) {
          auto next_input_it = next_payload->find(input_keys.front());
          if (next_input_it != next_payload->end() && next_input_it->is_object()) {
            const ordered_json& next_input = *next_input_it;
            CopyTensorType(next_input, output_tensor);
            if (HasRealQuantInfo(next_input)) {
              CopyQuantInfo(next_input, output_tensor);
            }
          }
        }
      }
    }
  }
}

void PostprocessReferenceMaxPoolSamePad(ordered_json& ops_json, int& same_pad_index) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        GetWrappedOpType(ops_json[index]) != "MaxPool" ||
        !payload->contains("attributes_info") ||
        !payload->contains("input_tensor_info")) {
      continue;
    }

    ordered_json& attributes = (*payload)["attributes_info"];
    ordered_json& input_tensor = (*payload)["input_tensor_info"];
    if (!attributes.contains("padding") || !input_tensor.contains("tensor_name")) {
      continue;
    }

    const std::string padding = attributes["padding"].get<std::string>();
    std::vector<int> pads;
    if (padding == "add_pad_op1") {
      pads = {1, 1, 1, 1};
    } else if (padding == "add_pad_op2") {
      pads = {2, 2, 2, 2};
    } else if (padding == "add_pad_op3") {
      pads = {3, 3, 3, 3};
    } else if (padding != "SAME") {
      continue;
    }

    if (!payload->contains("output_tensor_info")) {
      continue;
    }

    const std::vector<int> input_shape = JsonShapeToVector(input_tensor);
    const std::vector<int> output_shape = JsonShapeToVector((*payload)["output_tensor_info"]);
    if (input_shape.size() != 4 || output_shape.size() != 4 ||
        !attributes.contains("filter_height") ||
        !attributes.contains("filter_width") ||
        !attributes.contains("stride_h") ||
        !attributes.contains("stride_w")) {
      continue;
    }

    if (padding == "SAME") {
      const int filter_h = attributes["filter_height"].get<int>();
      const int filter_w = attributes["filter_width"].get<int>();
      const int stride_h = attributes["stride_h"].get<int>();
      const int stride_w = attributes["stride_w"].get<int>();
      const int total_pad_h = std::max((output_shape[1] - 1) * stride_h + filter_h - input_shape[1], 0);
      const int total_pad_w = std::max((output_shape[2] - 1) * stride_w + filter_w - input_shape[2], 0);
      if (total_pad_h == 0 && total_pad_w == 0) {
        attributes["padding"] = "VALID";
        continue;
      }

      const int pad_top = total_pad_h / 2;
      const int pad_bottom = total_pad_h - pad_top;
      const int pad_left = total_pad_w / 2;
      const int pad_right = total_pad_w - pad_left;
      pads = {pad_top, pad_left, pad_bottom, pad_right};
    }

    const std::string input_tensor_name = input_tensor["tensor_name"].get<std::string>();
    const int producer_index = FindProducerIndex(ops_json, input_tensor_name, static_cast<int>(index));
    const int insert_index = producer_index < 0 ? static_cast<int>(index) : producer_index + 1;

    const int current_pad_index = ++same_pad_index;
    const std::vector<int> updated_shape = ApplyNhwcPads(JsonShapeToVector(input_tensor), pads);
    const std::string updated_name = std::string("SAME_Pad_insert_output") + std::to_string(current_pad_index);

    ops_json.insert(ops_json.begin() + insert_index,
                    MakeReferencePadOp(input_tensor, "SAME_Pad_insert_values",
                                       "SAME_Pad_insert_output", current_pad_index, pads));
    auto* updated_payload = GetWrappedPayload(ops_json[index + 1]);
    if (updated_payload != nullptr) {
      auto& updated_input = (*updated_payload)["input_tensor_info"];
      updated_input["tensor_name"] = updated_name;
      updated_input["tensor_shape"] = updated_shape;
      (*updated_payload)["attributes_info"]["padding"] = "VALID";
    }
    ++index;
  }
}

std::vector<int> ComputeReferenceConvSamePads(const ordered_json& input_tensor,
                                              const ordered_json& weight_tensor,
                                              const ordered_json& output_tensor,
                                              const ordered_json& attributes) {
  const std::vector<int> input_shape = JsonShapeToVector(input_tensor);
  const std::vector<int> weight_shape = JsonShapeToVector(weight_tensor);
  const std::vector<int> output_shape = JsonShapeToVector(output_tensor);
  if (input_shape.size() != 4 || weight_shape.size() < 3 || output_shape.size() != 4) {
    return {1, 1, 1, 1};
  }

  const int stride_h = attributes.contains("stride_h") ? attributes["stride_h"].get<int>() : 1;
  const int stride_w = attributes.contains("stride_w") ? attributes["stride_w"].get<int>() : 1;
  const int dilation_h = attributes.contains("dilation_h_factor") ? attributes["dilation_h_factor"].get<int>() : 1;
  const int dilation_w = attributes.contains("dilation_w_factor") ? attributes["dilation_w_factor"].get<int>() : 1;
  const int kernel_h = weight_shape[1];
  const int kernel_w = weight_shape[2];
  if (stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0 ||
      kernel_h <= 0 || kernel_w <= 0) {
    return {1, 1, 1, 1};
  }

  const int effective_kernel_h = dilation_h * (kernel_h - 1) + 1;
  const int effective_kernel_w = dilation_w * (kernel_w - 1) + 1;
  const int total_pad_h = std::max((output_shape[1] - 1) * stride_h + effective_kernel_h - input_shape[1], 0);
  const int total_pad_w = std::max((output_shape[2] - 1) * stride_w + effective_kernel_w - input_shape[2], 0);
  const int pad_top = total_pad_h / 2;
  const int pad_bottom = total_pad_h - pad_top;
  const int pad_left = total_pad_w / 2;
  const int pad_right = total_pad_w - pad_left;
  return {pad_top, pad_left, pad_bottom, pad_right};
}

void PostprocessReferenceConvPad(ordered_json& ops_json, int& same_pad_index) {
  int conv_index = 0;
  for (size_t index = 0; index < ops_json.size(); ++index) {
    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr) {
      continue;
    }
    const std::string op_name = GetWrappedOpType(ops_json[index]);
    if (op_name == "QLinearConv" || op_name == "Conv" || op_name == "DepthwiseConv") {
      ++conv_index;
    }
    if ((op_name == "QLinearConv" || op_name == "Conv" || op_name == "DepthwiseConv") &&
        payload->contains("attributes_info") &&
        payload->contains("input_tensor_info")) {
      ordered_json& attributes = (*payload)["attributes_info"];
      ordered_json& input_tensor = (*payload)["input_tensor_info"];
      if (!attributes.contains("padding") || !input_tensor.contains("tensor_name")) {
        continue;
      }

      const std::string padding = attributes["padding"].get<std::string>();
      std::vector<int> pads;
      std::string tensor_name_prefix;
      std::string output_name_prefix;
      int current_pad_index = 0;
      if (padding == "add_pad_op1") {
        pads = {1, 1, 1, 1};
        tensor_name_prefix = "SAME_Pad_insert_values";
        output_name_prefix = "SAME_Pad_insert_output";
        current_pad_index = ++same_pad_index;
      } else if (padding == "add_pad_op3") {
        pads = {3, 3, 3, 3};
        tensor_name_prefix = "Pad_insert_values";
        output_name_prefix = "Pad_insert_output";
        current_pad_index = conv_index;
      } else if (padding == "add_pad_op2") {
        pads = {2, 2, 2, 2};
        tensor_name_prefix = "SAME_Pad_insert_values";
        output_name_prefix = "SAME_Pad_insert_output";
        current_pad_index = ++same_pad_index;
      } else if (padding == "explicit_pad_op") {
        if (!attributes.contains("explicit_pads") ||
            !attributes["explicit_pads"].is_array() ||
            attributes["explicit_pads"].size() != 4) {
          continue;
        }
        pads.clear();
        for (const auto& pad_value : attributes["explicit_pads"]) {
          if (!pad_value.is_number_integer() && !pad_value.is_number_unsigned()) {
            pads.clear();
            break;
          }
          pads.push_back(pad_value.get<int>());
        }
        if (pads.size() != 4) {
          continue;
        }
        tensor_name_prefix = "Pad_insert_values";
        output_name_prefix = "Pad_insert_output";
        current_pad_index = conv_index;
      } else if (padding == "SAME") {
        pads = {1, 1, 1, 1};
        tensor_name_prefix = "Conv_SAME_Pad_insert_values";
        output_name_prefix = "Conv_SAME_Pad_insert_output";
        current_pad_index = conv_index;
      } else {
        continue;
      }

      const std::string input_tensor_name = input_tensor["tensor_name"].get<std::string>();
      const int producer_index = FindProducerIndex(ops_json, input_tensor_name, static_cast<int>(index));
      if (producer_index < 0 && padding == "SAME") {
        continue;
      }

      const ordered_json* producer_tensor =
          FindProducerTensorInfo(ops_json, input_tensor_name, static_cast<int>(index));
      ordered_json pad_source_tensor = producer_tensor != nullptr ? *producer_tensor : input_tensor;
      int pad_producer_index = producer_index;
      int inserted_before_current = 0;

      if (producer_index >= 0 &&
          IsYoloAttentionPeDepthwiseConvNeedingNhwcInput(op_name, *payload, pad_source_tensor)) {
        const std::vector<int> source_shape = JsonShapeToVector(pad_source_tensor);
        const std::vector<int> nhwc_shape = {source_shape[0], source_shape[2], source_shape[3], source_shape[1]};
        ordered_json transpose_output = CloneTensorWithNameAndShape(
            pad_source_tensor,
            input_tensor_name + "_nchw_to_nhwc",
            nhwc_shape);
        ops_json.insert(ops_json.begin() + producer_index + 1,
                        MakeReferenceTransposeNchwToNhwcOp(
                            pad_source_tensor,
                            transpose_output,
                            input_tensor_name + "_nchw_to_nhwc_perm"));
        pad_source_tensor = std::move(transpose_output);
        pad_producer_index = producer_index + 1;
        inserted_before_current = 1;
      }

      if (padding == "SAME" &&
          payload->contains("input_weight_info") &&
          payload->contains("output_tensor_info")) {
        pads = ComputeReferenceConvSamePads(pad_source_tensor, (*payload)["input_weight_info"],
                                            (*payload)["output_tensor_info"], attributes);
      }

      const std::vector<int> updated_shape = ApplyNhwcPads(JsonShapeToVector(pad_source_tensor), pads);
      const std::string updated_name = output_name_prefix + std::to_string(current_pad_index);

      ordered_json pad_op = MakeReferencePadOp(pad_source_tensor, tensor_name_prefix, output_name_prefix,
                                               current_pad_index, pads);
      if (pad_producer_index < 0) {
        auto* pad_payload = GetWrappedPayload(pad_op);
        if (pad_payload != nullptr) {
          (*pad_payload)["input_tensor_info_1"] = input_tensor;
          (*pad_payload)["output_tensor_info"]["tensor_shape"] = updated_shape;
        }
        ops_json.insert(ops_json.begin() + static_cast<int>(index), std::move(pad_op));
      } else {
        ops_json.insert(ops_json.begin() + pad_producer_index + 1, std::move(pad_op));
      }

      auto* updated_payload = GetWrappedPayload(ops_json[index + inserted_before_current + 1]);
      if (updated_payload != nullptr) {
        auto& updated_input = (*updated_payload)["input_tensor_info"];
        updated_input["tensor_name"] = updated_name;
        updated_input["tensor_shape"] = updated_shape;
        (*updated_payload)["attributes_info"]["padding"] = "VALID";
      }
      index += inserted_before_current + 1;
    }
  }
}

bool IsMatMulLayoutHelperTensorName(const std::string& tensor_name) {
  return tensor_name.find("_matmul_") != std::string::npos ||
         tensor_name.find("_batch_matmul_") != std::string::npos;
}

bool IsTransposeOutputConsumedBySoftmax(ordered_json& ops_json,
                                        size_t transpose_index) {
  if (transpose_index >= ops_json.size() ||
      GetWrappedOpType(ops_json[transpose_index]) != "Transpose") {
    return false;
  }

  auto* transpose_payload = GetWrappedPayload(ops_json[transpose_index]);
  if (transpose_payload == nullptr ||
      !transpose_payload->contains("output_tensor_info") ||
      !(*transpose_payload)["output_tensor_info"].is_object() ||
      !(*transpose_payload)["output_tensor_info"].contains("tensor_name")) {
    return false;
  }

  const std::string output_name =
      (*transpose_payload)["output_tensor_info"]["tensor_name"].get<std::string>();
  const int consumer_index = FindFirstConsumerIndex(ops_json, output_name,
                                                    static_cast<int>(transpose_index));
  if (consumer_index < 0) {
    return false;
  }

  const std::string consumer_op =
      GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]);
  if (consumer_op == "Softmax") {
    return true;
  }
  if (consumer_op != "Transpose") {
    return false;
  }

  auto* consumer_payload = GetWrappedPayload(ops_json[static_cast<size_t>(consumer_index)]);
  if (consumer_payload == nullptr ||
      !consumer_payload->contains("output_tensor_info") ||
      !(*consumer_payload)["output_tensor_info"].is_object() ||
      !(*consumer_payload)["output_tensor_info"].contains("tensor_name")) {
    return false;
  }

  const std::string consumer_output_name =
      (*consumer_payload)["output_tensor_info"]["tensor_name"].get<std::string>();
  const int softmax_index = FindFirstConsumerIndex(
      ops_json, consumer_output_name, consumer_index);
  return softmax_index >= 0 &&
         GetWrappedOpType(ops_json[static_cast<size_t>(softmax_index)]) == "Softmax";
}

bool IsInsertedNhwcToNchwTransposePayload(const ordered_json& payload) {
  if (!payload.contains("is_insert") || !payload.contains("input_tensor_info_2")) {
    return false;
  }

  const auto& perm_tensor = payload["input_tensor_info_2"];
  return perm_tensor.is_object() &&
         perm_tensor.contains("values") &&
         perm_tensor["values"].is_array() &&
         perm_tensor["values"].size() == 4 &&
         perm_tensor["values"][0] == 0 &&
         perm_tensor["values"][1] == 3 &&
         perm_tensor["values"][2] == 1 &&
         perm_tensor["values"][3] == 2;
}

bool TensorValuesEqual(const ordered_json& tensor_info,
                       std::initializer_list<int> expected_values) {
  if (!tensor_info.is_object() ||
      !tensor_info.contains("values") ||
      !tensor_info["values"].is_array() ||
      tensor_info["values"].size() != expected_values.size()) {
    return false;
  }

  size_t index = 0;
  for (int expected : expected_values) {
    if (!tensor_info["values"][index].is_number_integer() ||
        tensor_info["values"][index].get<int>() != expected) {
      return false;
    }
    ++index;
  }
  return true;
}

bool IsReferenceAttentionScoreShape(const std::vector<int>& shape) {
  return shape.size() == 4 &&
         shape[1] > 0 &&
         shape[1] <= 16 &&
         shape[2] >= 64 &&
         shape[2] == shape[3];
}

bool IsReferenceAttentionScoreTranspose(ordered_json& ops_json,
                                        size_t transpose_index) {
  if (transpose_index >= ops_json.size() ||
      GetWrappedOpType(ops_json[transpose_index]) != "Transpose") {
    return false;
  }

  auto* payload = GetWrappedPayload(ops_json[transpose_index]);
  if (payload == nullptr ||
      !payload->contains("input_tensor_info_1") ||
      !payload->contains("input_tensor_info_2") ||
      !payload->contains("output_tensor_info")) {
    return false;
  }

  const ordered_json& input_tensor = (*payload)["input_tensor_info_1"];
  const ordered_json& perm_tensor = (*payload)["input_tensor_info_2"];
  const ordered_json& output_tensor = (*payload)["output_tensor_info"];
  if (!input_tensor.is_object() ||
      !output_tensor.is_object() ||
      !input_tensor.contains("tensor_name") ||
      !output_tensor.contains("tensor_name") ||
      !TensorValuesEqual(perm_tensor, {0, 1, 3, 2})) {
    return false;
  }

  const std::vector<int> input_shape = JsonShapeToVector(input_tensor);
  const std::vector<int> output_shape = JsonShapeToVector(output_tensor);
  if (!IsReferenceAttentionScoreShape(input_shape) ||
      output_shape.size() != 4 ||
      output_shape[0] != input_shape[0] ||
      output_shape[1] != input_shape[1] ||
      output_shape[2] != input_shape[3] ||
      output_shape[3] != input_shape[2]) {
    return false;
  }

  const std::string input_name = input_tensor["tensor_name"].get<std::string>();
  const int producer_index =
      FindProducerIndex(ops_json, input_name, static_cast<int>(transpose_index));
  if (producer_index < 0) {
    return false;
  }

  const std::string producer_op =
      GetWrappedOpType(ops_json[static_cast<size_t>(producer_index)]);
  if (producer_op != "Div" && producer_op != "Softmax") {
    return false;
  }

  const std::string output_name = output_tensor["tensor_name"].get<std::string>();
  const int consumer_index =
      FindFirstConsumerIndex(ops_json, output_name, static_cast<int>(transpose_index));
  return consumer_index >= 0 &&
         GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]) == "MatMul";
}

ordered_json ShapeVectorToJsonArray(const std::vector<int>& shape) {
  ordered_json shape_json = ordered_json::array();
  for (int dim : shape) {
    shape_json.push_back(dim);
  }
  return shape_json;
}

ordered_json MakeReferenceTransposeNchwToNhwcOp(const ordered_json& input_tensor,
                                                const ordered_json& output_tensor,
                                                const std::string& perm_tensor_name) {
  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("Transpose");
  payload["input_tensor_info_1"] = input_tensor;
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(
      perm_tensor_name, "int32", {4}, ordered_json::array({0, 2, 3, 1}));
  payload["output_tensor_info"] = output_tensor;
  payload["is_insert"] = true;
  payload["skip_transpose_insert_before"] = true;
  return WrapOp("Transpose", std::move(payload));
}

ordered_json CloneTensorWithNameAndShape(const ordered_json& tensor_info,
                                         const std::string& tensor_name,
                                         const std::vector<int>& shape) {
  ordered_json cloned = tensor_info;
  cloned["tensor_name"] = tensor_name;
  cloned["tensor_shape"] = ShapeVectorToJsonArray(shape);
  cloned["is_constant"] = false;
  cloned.erase("values");
  cloned.erase("flat_values");
  cloned.erase("values_1d");
  cloned.erase("values_2d");
  cloned.erase("values_3d");
  cloned.erase("values_4d");
  cloned.erase("flat_values_float32");
  cloned.erase("values_1d_float32");
  cloned.erase("values_2d_float32");
  cloned.erase("values_3d_float32");
  cloned.erase("values_4d_float32");
  return cloned;
}

bool IsLayoutTransformableTensor(const ordered_json& tensor_info) {
  if (!tensor_info.is_object() || !tensor_info.contains("tensor_type") ||
      !tensor_info["tensor_type"].is_string()) {
    return true;
  }

  const std::string tensor_type = tensor_info["tensor_type"].get<std::string>();
  return tensor_type == "float32" ||
         tensor_type == "float16" ||
         tensor_type == "int8" ||
         tensor_type == "uint8";
}

bool IsYoloAttentionPeDepthwiseConvNeedingNhwcInput(const std::string& op_name,
                                                    const ordered_json& payload,
                                                    const ordered_json& source_tensor) {
  if (op_name != "DepthwiseConv" ||
      !payload.contains("input_weight_info") ||
      !payload.contains("output_tensor_info")) {
    return false;
  }

  const auto& weight_tensor = payload["input_weight_info"];
  const auto& output_tensor = payload["output_tensor_info"];
  if (!weight_tensor.is_object() ||
      !weight_tensor.contains("tensor_name") ||
      !weight_tensor.contains("tensor_shape")) {
    return false;
  }

  const std::string weight_name = weight_tensor["tensor_name"].get<std::string>();
  if (weight_name.find(".attn.pe.conv.weight") == std::string::npos) {
    return false;
  }

  const std::vector<int> source_shape = JsonShapeToVector(source_tensor);
  const std::vector<int> weight_shape = JsonShapeToVector(weight_tensor);
  const std::vector<int> output_shape = JsonShapeToVector(output_tensor);
  return source_shape.size() == 4 &&
         weight_shape.size() == 4 &&
         output_shape.size() == 4 &&
         source_shape[0] == output_shape[0] &&
         source_shape[1] == weight_shape[3] &&
         output_shape[1] == source_shape[2] &&
         output_shape[2] == source_shape[3] &&
         output_shape[3] == source_shape[1];
}

ordered_json MakeReferenceReshapeOpFromJson(const ordered_json& input_tensor,
                                            const ordered_json& output_tensor,
                                            const std::string& shape_tensor_name,
                                            const std::vector<int>& output_shape) {
  ordered_json output_shape_json = ShapeVectorToJsonArray(output_shape);
  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("reshape");
  payload["input_tensor_info_1"] = input_tensor;
  payload["input_tensor_info_2"] = ConstantTensorInfoJson(
      shape_tensor_name, "int32", {static_cast<int>(output_shape.size())}, output_shape_json);
  payload["output_tensor_info"] = output_tensor;
  payload["skip_transpose_insert_before"] = true;

  ordered_json attributes;
  attributes["new_shape"] = std::move(output_shape_json);
  payload["attributes_info"] = std::move(attributes);
  return WrapOp("reshape", std::move(payload));
}

void PostprocessReferenceRank4BatchMatMulFlatten(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        GetWrappedOpType(ops_json[index]) != "MatMul" ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json input_1 = (*payload)["input_tensor_info_1"];
    ordered_json input_2 = (*payload)["input_tensor_info_2"];
    ordered_json output = (*payload)["output_tensor_info"];
    if (!input_1.contains("tensor_name") ||
        !input_2.contains("tensor_name") ||
        !output.contains("tensor_name")) {
      continue;
    }

    const std::vector<int> input_1_shape = JsonShapeToVector(input_1);
    const std::vector<int> input_2_shape = JsonShapeToVector(input_2);
    const std::vector<int> output_shape = JsonShapeToVector(output);
    if (input_1_shape.size() != 4 ||
        input_2_shape.size() != 4 ||
        output_shape.size() != 4 ||
        input_1_shape[0] != input_2_shape[0] ||
        input_1_shape[0] != output_shape[0] ||
        input_1_shape[1] != input_2_shape[1] ||
        input_1_shape[1] != output_shape[1] ||
        input_1_shape[2] != output_shape[2] ||
        input_1_shape[3] != input_2_shape[2] ||
        input_2_shape[3] != output_shape[3]) {
      continue;
    }

    const int batch_groups = input_1_shape[0] * input_1_shape[1];
    const int lhs_rows = input_1_shape[2];
    const int inner_dim = input_1_shape[3];
    const int rhs_cols = input_2_shape[3];

    const std::string input_1_name = input_1["tensor_name"].get<std::string>();
    const std::string input_2_name = input_2["tensor_name"].get<std::string>();
    const std::string output_name = output["tensor_name"].get<std::string>();
    const std::vector<int> flat_input_1_shape = {batch_groups, lhs_rows, inner_dim};
    const std::vector<int> flat_input_2_shape = {batch_groups, inner_dim, rhs_cols};
    const std::vector<int> flat_output_shape = {batch_groups, lhs_rows, rhs_cols};

    ordered_json flat_input_1 = CloneTensorWithNameAndShape(
        input_1, input_1_name + "_post_batch_flatten_lhs", flat_input_1_shape);
    ordered_json flat_input_2 = CloneTensorWithNameAndShape(
        input_2, input_2_name + "_post_batch_flatten_rhs", flat_input_2_shape);
    ordered_json flat_output = CloneTensorWithNameAndShape(
        output, output_name + "_post_batch_flatten_output", flat_output_shape);

    ordered_json matmul_attributes = payload->contains("attributes_info")
                                         ? (*payload)["attributes_info"]
                                         : ordered_json::object();
    matmul_attributes["adj_x"] = false;
    matmul_attributes["adj_y"] = false;
    if (!matmul_attributes.contains("asymmetric_quantize_inputs")) {
      matmul_attributes["asymmetric_quantize_inputs"] = false;
    }

    ordered_json matmul_payload;
    matmul_payload["op_type_info"] = OpTypeInfo("MatMul");
    matmul_payload["input_tensor_info_1"] = flat_input_1;
    matmul_payload["input_tensor_info_2"] = flat_input_2;
    matmul_payload["output_tensor_info"] = flat_output;
    matmul_payload["attributes_info"] = std::move(matmul_attributes);

    ops_json[index] = MakeReferenceReshapeOpFromJson(
        input_1, flat_input_1, input_1_name + "_post_batch_flatten_lhs_shape",
        flat_input_1_shape);
    ops_json.insert(ops_json.begin() + static_cast<int>(index + 1),
                    MakeReferenceReshapeOpFromJson(
                        input_2, flat_input_2,
                        input_2_name + "_post_batch_flatten_rhs_shape",
                        flat_input_2_shape));
    ops_json.insert(ops_json.begin() + static_cast<int>(index + 2),
                    WrapOp("MatMul", std::move(matmul_payload)));
    ops_json.insert(ops_json.begin() + static_cast<int>(index + 3),
                    MakeReferenceReshapeOpFromJson(
                        flat_output, output,
                        output_name + "_post_batch_flatten_output_shape",
                        output_shape));
    index += 3;
  }
}

void PostprocessReferenceTransposeInsertBefore(ordered_json& ops_json) {
  int transpose_before_index = 0;
  for (size_t index = 0; index < ops_json.size(); ++index) {
    const std::string op_name = GetWrappedOpType(ops_json[index]);
    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        (op_name != "reshape" && op_name != "Transpose") ||
        !payload->contains("input_tensor_info_1")) {
      continue;
    }
    if (payload->contains("skip_transpose_insert_before")) {
      continue;
    }

    ordered_json& input_tensor = (*payload)["input_tensor_info_1"];
    const auto input_shape = JsonShapeToVector(input_tensor);
    if (input_shape.size() != 4 || !input_tensor.contains("tensor_name") ||
        !IsLayoutTransformableTensor(input_tensor)) {
      continue;
    }

    const std::string input_tensor_name = input_tensor["tensor_name"].get<std::string>();
    if (op_name == "Transpose" &&
        payload->contains("output_tensor_info") &&
        (*payload)["output_tensor_info"].is_object() &&
        (*payload)["output_tensor_info"].contains("tensor_name") &&
        IsMatMulLayoutHelperTensorName(
            (*payload)["output_tensor_info"]["tensor_name"].get<std::string>())) {
      continue;
    }
    if (IsReferenceAttentionScoreTranspose(ops_json, index)) {
      continue;
    }

    const int producer_index = FindProducerIndex(ops_json, input_tensor_name, static_cast<int>(index));
    if (producer_index < 0) {
      continue;
    }
    const std::string producer_op =
        GetWrappedOpType(ops_json[static_cast<size_t>(producer_index)]);
    if (op_name == "Transpose" &&
        (producer_op == "Softmax" || IsTransposeOutputConsumedBySoftmax(ops_json, index))) {
      continue;
    }

    ++transpose_before_index;
    bool use_nchw_view = (op_name == "reshape");
    if (use_nchw_view && payload->contains("is_flatten")) {
      // onnx2tf keeps the inserted pre-reshape transpose in the same display
      // shape as the flatten input; it only rewires the tensor name.
      use_nchw_view = false;
    }
    ops_json.insert(ops_json.begin() + producer_index + 1,
                    MakeReferenceTransposeBeforeOp(input_tensor, transpose_before_index, use_nchw_view));
    auto* updated_payload = GetWrappedPayload(ops_json[index + 1]);
    if (updated_payload != nullptr) {
      (*updated_payload)["input_tensor_info_1"]["tensor_name"] =
          std::string("Transpose_insert_output") + std::to_string(transpose_before_index);
    }
    ++index;
  }
}

void PostprocessReferenceTransposeInsertAfter(ordered_json& ops_json) {
  int transpose_after_index = -1;
  for (size_t index = 0; index < ops_json.size(); ++index) {
    const std::string op_name = GetWrappedOpType(ops_json[index]);
    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        (op_name != "reshape" && op_name != "Transpose") ||
        payload->contains("is_insert") ||
        payload->contains("skip_transpose_insert_after") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& output_tensor = (*payload)["output_tensor_info"];
    const auto output_shape = JsonShapeToVector(output_tensor);
    if (output_shape.size() != 4 || !output_tensor.contains("tensor_name") ||
        !IsLayoutTransformableTensor(output_tensor)) {
      continue;
    }

    const std::string original_output_name = output_tensor["tensor_name"].get<std::string>();
    if (op_name == "Transpose" && IsMatMulLayoutHelperTensorName(original_output_name)) {
      continue;
    }
    if (IsReferenceAttentionScoreTranspose(ops_json, index)) {
      continue;
    }

    const int consumer_index = FindFirstConsumerIndex(ops_json, original_output_name,
                                                      static_cast<int>(index));
    if (consumer_index < 0) {
      continue;
    }
    const std::string consumer_op =
        GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]);
    if (op_name == "Transpose" &&
        (consumer_op == "Softmax" ||
         (consumer_op == "Transpose" &&
          IsTransposeOutputConsumedBySoftmax(ops_json, static_cast<size_t>(consumer_index))))) {
      continue;
    }
    if (op_name == "reshape" && consumer_op == "Transpose" &&
        IsTransposeOutputConsumedBySoftmax(ops_json, static_cast<size_t>(consumer_index))) {
      continue;
    }

    --transpose_after_index;
    ops_json.insert(ops_json.begin() + consumer_index,
                    MakeReferenceTransposeAfterOp(output_tensor, transpose_after_index));
    auto* updated_payload = GetWrappedPayload(ops_json[index]);
    if (updated_payload != nullptr) {
      (*updated_payload)["output_tensor_info"]["tensor_name"] =
          std::string("Transpose_insert_output") + std::to_string(transpose_after_index);
    }
    ++index;
  }
}

void PostprocessReferenceTransposeShapePropagation(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr) {
      continue;
    }

    for (const auto& input_key : InputTensorInfoKeys(*payload)) {
      auto input_it = payload->find(input_key);
      if (input_it == payload->end() || !input_it->is_object() || !input_it->contains("tensor_name")) {
        continue;
      }

      const std::string tensor_name = (*input_it)["tensor_name"].get<std::string>();
      const int producer_index = FindProducerIndex(ops_json, tensor_name, static_cast<int>(index));
      if (producer_index < 0) {
        continue;
      }

      const std::string producer_op = GetWrappedOpType(ops_json[producer_index]);
      if (producer_op != "Transpose") {
        continue;
      }

      if (GetWrappedOpType(ops_json[index]) == "Pad") {
        continue;
      }

      auto* producer_payload = GetWrappedPayload(ops_json[producer_index]);
      if (GetWrappedOpType(ops_json[index]) == "Transpose" &&
          IsInsertedNhwcToNchwTransposePayload(*payload)) {
        const bool producer_is_inserted_transpose =
            producer_op == "Transpose" &&
            producer_payload != nullptr &&
            producer_payload->contains("is_insert");
        if (!producer_is_inserted_transpose) {
          continue;
        }
      }

      if (producer_payload == nullptr || !producer_payload->contains("output_tensor_info")) {
        continue;
      }

      const ordered_json& producer_output = (*producer_payload)["output_tensor_info"];
      if (producer_output.contains("tensor_shape")) {
        (*input_it)["tensor_shape"] = producer_output["tensor_shape"];
      }
      if (producer_output.contains("tensor_type")) {
        (*input_it)["tensor_type"] = producer_output["tensor_type"];
      }
    }
  }
}

void PostprocessReferenceInsertedTransposeShapeNormalization(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    auto& op_json = ops_json[index];
    if (GetWrappedOpType(op_json) != "Transpose") {
      continue;
    }

    auto* payload = GetWrappedPayload(op_json);
    if (payload == nullptr ||
        !payload->contains("is_insert") ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    const auto& perm_tensor = (*payload)["input_tensor_info_2"];
    if (!perm_tensor.is_object() ||
        !perm_tensor.contains("values") ||
        !perm_tensor["values"].is_array() ||
        perm_tensor["values"].size() != 4 ||
        perm_tensor["values"][0] != 0 ||
        perm_tensor["values"][1] != 3 ||
        perm_tensor["values"][2] != 1 ||
        perm_tensor["values"][3] != 2) {
      continue;
    }

    auto& input_tensor = (*payload)["input_tensor_info_1"];
    auto& output_tensor = (*payload)["output_tensor_info"];
    const auto input_shape = JsonShapeToVector(input_tensor);
    if (input_shape.size() != 4) {
      continue;
    }

    const std::vector<int> transposed_shape = {input_shape[0], input_shape[3], input_shape[1], input_shape[2]};
    bool consumer_uses_transposed_shape = false;
    if (output_tensor.contains("tensor_name")) {
      const std::string output_name = output_tensor["tensor_name"].get<std::string>();
      const int consumer_index = FindFirstConsumerIndex(ops_json, output_name, static_cast<int>(index + 1));
      if (consumer_index >= 0) {
        const std::string consumer_op = GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]);
        auto* consumer_payload = GetWrappedPayload(ops_json[static_cast<size_t>(consumer_index)]);
        if (consumer_payload != nullptr) {
          const auto input_keys = InputTensorInfoKeys(*consumer_payload);
          auto* consumer_tensor = FindNamedTensorInfo(*consumer_payload, input_keys, output_name);
          consumer_uses_transposed_shape =
              consumer_op == "Transpose" ||
              (consumer_tensor != nullptr && JsonShapeToVector(*consumer_tensor) == transposed_shape);
          if (consumer_uses_transposed_shape && consumer_tensor != nullptr) {
            (*consumer_tensor)["tensor_shape"] = transposed_shape;
          }
        }
      }
    }

    output_tensor["tensor_shape"] = consumer_uses_transposed_shape ? transposed_shape
                                                                   : JsonShapeToVector(input_tensor);
  }
}

void PostprocessReferenceInsertedPadShapeNormalization(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "Pad") {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    auto& pad_input = (*payload)["input_tensor_info_1"];
    auto& pad_values = (*payload)["input_tensor_info_2"];
    auto& pad_output = (*payload)["output_tensor_info"];
    if (!pad_input.is_object() || !pad_values.is_object() || !pad_output.is_object() ||
        !pad_input.contains("tensor_name") || !pad_output.contains("tensor_name")) {
      continue;
    }

    const std::vector<int> pads = ReferencePadsFromPadValuesTensor(pad_values);
    if (pads.empty()) {
      continue;
    }

    const std::string pad_input_name = pad_input["tensor_name"].get<std::string>();
    const ordered_json* producer_tensor =
        FindProducerTensorInfo(ops_json, pad_input_name, static_cast<int>(index));
    if (producer_tensor == nullptr) {
      continue;
    }

    const std::vector<int> producer_shape = JsonShapeToVector(*producer_tensor);
    const std::vector<int> padded_shape = ApplyNhwcPads(producer_shape, pads);
    if (padded_shape.empty()) {
      continue;
    }

    pad_input["tensor_shape"] = producer_shape;
    pad_output["tensor_shape"] = padded_shape;

    const std::string pad_output_name = pad_output["tensor_name"].get<std::string>();
    const int consumer_index = FindFirstConsumerIndex(ops_json, pad_output_name, static_cast<int>(index + 1));
    if (consumer_index < 0) {
      continue;
    }

    const std::string consumer_op = GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]);
    if (consumer_op != "QLinearConv" &&
        consumer_op != "Conv" &&
        consumer_op != "DepthwiseConv") {
      continue;
    }

    auto* consumer_payload = GetWrappedPayload(ops_json[static_cast<size_t>(consumer_index)]);
    if (consumer_payload == nullptr) {
      continue;
    }

    const auto input_keys = InputTensorInfoKeys(*consumer_payload);
    auto* consumer_tensor = FindNamedTensorInfo(*consumer_payload, input_keys, pad_output_name);
    if (consumer_tensor == nullptr) {
      continue;
    }

    consumer_tensor->operator[]("tensor_shape") = padded_shape;
  }
}

void PostprocessReferenceSplitQuantOverride(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "Split") {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr || !payload->contains("input_tensor_info_1")) {
      continue;
    }

    auto& input_tensor = (*payload)["input_tensor_info_1"];
    if (!input_tensor.is_object() || !input_tensor.contains("tensor_name")) {
      continue;
    }

    const std::string tensor_name = input_tensor["tensor_name"].get<std::string>();
    const int producer_index = FindProducerIndex(ops_json, tensor_name, static_cast<int>(index));
    if (producer_index < 0) {
      continue;
    }

    const std::string producer_op = GetWrappedOpType(ops_json[static_cast<size_t>(producer_index)]);
    if (producer_op == "Transpose" || producer_op == "reshape") {
      input_tensor["quant_scale"] = "None";
      input_tensor["zero_point"] = "None";
    }
  }
}

int64_t ShapeElementCount(const std::vector<int>& shape) {
  if (shape.empty()) {
    return 0;
  }
  int64_t count = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    count *= dim;
  }
  return count;
}

std::vector<int> IntVectorFromValuesJson(const ordered_json& tensor_info) {
  std::vector<int> values;
  if (!tensor_info.is_object() ||
      !tensor_info.contains("values") ||
      !tensor_info["values"].is_array()) {
    return values;
  }
  for (const auto& value : tensor_info["values"]) {
    if (!value.is_number_integer()) {
      values.clear();
      return values;
    }
    values.push_back(value.get<int>());
  }
  return values;
}

std::vector<int> KeepDimsReducedShapeForAxes(const std::vector<int>& input_shape,
                                             const std::vector<int>& axes) {
  std::vector<int> reduced_shape = input_shape;
  for (int axis : axes) {
    int normalized_axis = axis;
    if (normalized_axis < 0) {
      normalized_axis += static_cast<int>(reduced_shape.size());
    }
    if (normalized_axis >= 0 && static_cast<size_t>(normalized_axis) < reduced_shape.size()) {
      reduced_shape[static_cast<size_t>(normalized_axis)] = 1;
    }
  }
  return reduced_shape;
}

void SetTensorShape(ordered_json& tensor_info, const std::vector<int>& shape) {
  tensor_info["tensor_shape"] = ShapeVectorToJsonArray(shape);
}

void SetTensorValues(ordered_json& tensor_info, const std::vector<int>& values) {
  tensor_info["values"] = ShapeVectorToJsonArray(values);
}

bool IsReferenceElementwiseBinaryOp(const std::string& op_name) {
  return op_name == "Add" ||
         op_name == "Mul" ||
         op_name == "Sub" ||
         op_name == "Div" ||
         op_name == "Pow" ||
         op_name == "Mod";
}

std::vector<int> BroadcastPostprocessShape(const std::vector<int>& lhs,
                                           const std::vector<int>& rhs) {
  if (lhs.empty()) {
    return rhs;
  }
  if (rhs.empty()) {
    return lhs;
  }

  const size_t rank = std::max(lhs.size(), rhs.size());
  std::vector<int> result(rank, 1);
  for (size_t index = 0; index < rank; ++index) {
    const int lhs_dim = index < rank - lhs.size() ? 1 : lhs[index - (rank - lhs.size())];
    const int rhs_dim = index < rank - rhs.size() ? 1 : rhs[index - (rank - rhs.size())];
    if (lhs_dim == rhs_dim || rhs_dim == 1) {
      result[index] = lhs_dim;
    } else if (lhs_dim == 1) {
      result[index] = rhs_dim;
    } else {
      return {};
    }
  }
  return result;
}

void PropagateTensorShapeToConsumers(ordered_json& ops_json,
                                     size_t producer_index,
                                     const std::string& tensor_name,
                                     const std::vector<int>& shape) {
  if (tensor_name.empty() || shape.empty()) {
    return;
  }

  for (size_t consumer = producer_index + 1; consumer < ops_json.size(); ++consumer) {
    auto* consumer_payload = GetWrappedPayload(ops_json[consumer]);
    if (consumer_payload == nullptr) {
      continue;
    }

    for (const auto& input_key : InputTensorInfoKeys(*consumer_payload)) {
      auto input_it = consumer_payload->find(input_key);
      if (input_it != consumer_payload->end() &&
          input_it->is_object() &&
          input_it->value("tensor_name", "") == tensor_name) {
        SetTensorShape(*input_it, shape);
      }
    }
  }
}

bool TensorProducerIsMatMul(ordered_json& ops_json,
                            size_t consumer_index,
                            const std::string& tensor_name) {
  if (tensor_name.empty()) {
    return false;
  }

  const int producer_index =
      FindProducerIndex(ops_json, tensor_name, static_cast<int>(consumer_index));
  if (producer_index < 0) {
    return false;
  }

  return GetWrappedOpType(ops_json[static_cast<size_t>(producer_index)]) == "MatMul";
}

void PostprocessReferenceElementwiseOutputShapes(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    const std::string op_name = GetWrappedOpType(ops_json[index]);
    if (!IsReferenceElementwiseBinaryOp(op_name)) {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& input_1 = (*payload)["input_tensor_info_1"];
    ordered_json& input_2 = (*payload)["input_tensor_info_2"];
    ordered_json& output = (*payload)["output_tensor_info"];
    const std::vector<int> original_output_shape = JsonShapeToVector(output);
    if (original_output_shape.size() != 4) {
      continue;
    }
    const bool matmul_sourced =
        TensorProducerIsMatMul(ops_json, index, input_1.value("tensor_name", "")) ||
        TensorProducerIsMatMul(ops_json, index, input_2.value("tensor_name", ""));
    if (!matmul_sourced) {
      continue;
    }

    const std::vector<int> output_shape =
        BroadcastPostprocessShape(JsonShapeToVector(input_1), JsonShapeToVector(input_2));
    if (output_shape.empty() || output_shape == original_output_shape) {
      continue;
    }

    SetTensorShape(output, output_shape);
    PropagateTensorShapeToConsumers(
        ops_json, index, output.value("tensor_name", ""), output_shape);
  }
}

bool Rank3ToRank4BroadcastTargetShape(const std::vector<int>& rank3_shape,
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
  if (rank3_shape[0] == rank4_shape[0] &&
      rank3_shape[1] == rank4_shape[2] &&
      rank3_shape[2] == rank4_shape[3] &&
      rank4_shape[1] > 1) {
    target_shape = {rank3_shape[0], 1, rank3_shape[1], rank3_shape[2]};
    return true;
  }

  return false;
}

bool InsertRank3ToRank4BroadcastReshape(ordered_json& ops_json,
                                        size_t op_index,
                                        const std::string& input_key,
                                        const std::vector<int>& other_shape,
                                        int& reshape_index) {
  auto* payload = GetWrappedPayload(ops_json[op_index]);
  if (payload == nullptr || !payload->contains(input_key)) {
    return false;
  }

  ordered_json& input = (*payload)[input_key];
  if (!input.is_object() || !input.contains("tensor_name")) {
    return false;
  }

  const std::vector<int> input_shape = JsonShapeToVector(input);
  std::vector<int> target_shape;
  if (!Rank3ToRank4BroadcastTargetShape(input_shape, other_shape, target_shape)) {
    return false;
  }

  const std::string input_name = input.value("tensor_name", "");
  if (input_name.empty()) {
    return false;
  }

  ++reshape_index;
  ordered_json reshaped_input = CloneTensorWithNameAndShape(
      input,
      input_name + "_rank3_to_rank4_broadcast_" + std::to_string(reshape_index),
      target_shape);
  ordered_json reshape_op = MakeReferenceReshapeOpFromJson(
      input,
      reshaped_input,
      input_name + "_rank3_to_rank4_broadcast_shape_" + std::to_string(reshape_index),
      target_shape);

  ops_json.insert(ops_json.begin() + static_cast<int>(op_index), std::move(reshape_op));
  auto* updated_payload = GetWrappedPayload(ops_json[op_index + 1]);
  if (updated_payload == nullptr) {
    return false;
  }
  (*updated_payload)[input_key] = std::move(reshaped_input);
  return true;
}

void PostprocessReferenceRank3ToRank4BroadcastInputs(ordered_json& ops_json) {
  int reshape_index = 0;
  for (size_t index = 0; index < ops_json.size(); ++index) {
    const std::string op_name = GetWrappedOpType(ops_json[index]);
    if (!IsReferenceElementwiseBinaryOp(op_name)) {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    const std::vector<int> input_1_shape =
        JsonShapeToVector((*payload)["input_tensor_info_1"]);
    const std::vector<int> input_2_shape =
        JsonShapeToVector((*payload)["input_tensor_info_2"]);

    bool inserted = false;
    if (input_1_shape.size() == 3 && input_2_shape.size() == 4) {
      inserted = InsertRank3ToRank4BroadcastReshape(
          ops_json, index, "input_tensor_info_1", input_2_shape, reshape_index);
    } else if (input_2_shape.size() == 3 && input_1_shape.size() == 4) {
      inserted = InsertRank3ToRank4BroadcastReshape(
          ops_json, index, "input_tensor_info_2", input_1_shape, reshape_index);
    }

    if (!inserted) {
      continue;
    }

    ++index;
    auto* updated_payload = GetWrappedPayload(ops_json[index]);
    if (updated_payload == nullptr) {
      continue;
    }

    ordered_json& output = (*updated_payload)["output_tensor_info"];
    const std::vector<int> output_shape = BroadcastPostprocessShape(
        JsonShapeToVector((*updated_payload)["input_tensor_info_1"]),
        JsonShapeToVector((*updated_payload)["input_tensor_info_2"]));
    if (output_shape.empty() || output_shape == JsonShapeToVector(output)) {
      continue;
    }

    SetTensorShape(output, output_shape);
    PropagateTensorShapeToConsumers(
        ops_json, index, output.value("tensor_name", ""), output_shape);
  }
}

bool IsReferenceShapePreservingUnaryOp(const std::string& op_name) {
  return op_name == "Softmax" ||
         op_name == "Exp";
}

void PostprocessReferenceShapePreservingUnaryOutputShapes(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    const std::string op_name = GetWrappedOpType(ops_json[index]);
    if (!IsReferenceShapePreservingUnaryOp(op_name)) {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr || !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json* input = nullptr;
    auto input_it = payload->find("input_tensor_info");
    if (input_it != payload->end() && input_it->is_object()) {
      input = &(*input_it);
    } else {
      input_it = payload->find("input_tensor_info_1");
      if (input_it != payload->end() && input_it->is_object()) {
        input = &(*input_it);
      }
    }
    if (input == nullptr) {
      continue;
    }

    ordered_json& output = (*payload)["output_tensor_info"];
    if (!output.is_object()) {
      continue;
    }

    const std::vector<int> input_shape = JsonShapeToVector(*input);
    const std::vector<int> output_shape = JsonShapeToVector(output);
    if (input_shape.empty() ||
        output_shape.empty() ||
        input_shape == output_shape ||
        input_shape.size() != output_shape.size()) {
      continue;
    }

    const int64_t input_count = ShapeElementCount(input_shape);
    if (input_count <= 0 || input_count != ShapeElementCount(output_shape)) {
      continue;
    }

    SetTensorShape(output, input_shape);
    PropagateTensorShapeToConsumers(
        ops_json, index, output.value("tensor_name", ""), input_shape);
  }
}

bool IsReferenceConvLikeOp(const std::string& op_name) {
  return op_name == "Conv" || op_name == "DepthwiseConv";
}

bool IsReferenceSoftmaxAxisRestoreTranspose(const ordered_json& payload) {
  if (!payload.contains("input_tensor_info_2")) {
    return false;
  }

  const auto& perm_tensor = payload["input_tensor_info_2"];
  if (!perm_tensor.is_object() ||
      !perm_tensor.contains("tensor_name") ||
      !perm_tensor["tensor_name"].is_string()) {
    return false;
  }

  const std::string perm_name = perm_tensor["tensor_name"].get<std::string>();
  return perm_name.find("_softmax_axis_from_last_perm_values") != std::string::npos;
}

bool ReferenceConvInputChannelsMatchLastDim(const ordered_json& conv_payload,
                                            const std::vector<int>& input_shape) {
  if (input_shape.size() != 4 ||
      !conv_payload.contains("input_weight_info") ||
      !conv_payload["input_weight_info"].is_object()) {
    return false;
  }

  const std::vector<int> weight_shape = JsonShapeToVector(conv_payload["input_weight_info"]);
  return weight_shape.size() == 4 && input_shape[3] == weight_shape[3];
}

void PostprocessReferenceSoftmaxAxisRestoreConvConsumer(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "Transpose") {
      continue;
    }

    auto* restore_payload = GetWrappedPayload(ops_json[index]);
    if (restore_payload == nullptr ||
        !IsReferenceSoftmaxAxisRestoreTranspose(*restore_payload) ||
        !restore_payload->contains("input_tensor_info_1") ||
        !restore_payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& restore_input = (*restore_payload)["input_tensor_info_1"];
    ordered_json& restore_output = (*restore_payload)["output_tensor_info"];
    if (!restore_input.is_object() ||
        !restore_output.is_object() ||
        !restore_input.contains("tensor_name") ||
        !restore_output.contains("tensor_name")) {
      continue;
    }

    const std::string restore_input_name = restore_input["tensor_name"].get<std::string>();
    const std::string restore_output_name = restore_output["tensor_name"].get<std::string>();
    const int softmax_index = FindProducerIndex(
        ops_json, restore_input_name, static_cast<int>(index));
    if (softmax_index < 0 ||
        GetWrappedOpType(ops_json[static_cast<size_t>(softmax_index)]) != "Softmax") {
      continue;
    }

    const int consumer_index = FindFirstConsumerIndex(
        ops_json, restore_output_name, static_cast<int>(index));
    if (consumer_index < 0 ||
        !IsReferenceConvLikeOp(GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]))) {
      continue;
    }

    auto* softmax_payload = GetWrappedPayload(ops_json[static_cast<size_t>(softmax_index)]);
    auto* conv_payload = GetWrappedPayload(ops_json[static_cast<size_t>(consumer_index)]);
    if (softmax_payload == nullptr ||
        conv_payload == nullptr ||
        !softmax_payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& softmax_output = (*softmax_payload)["output_tensor_info"];
    const std::vector<int> softmax_output_shape = JsonShapeToVector(softmax_output);
    if (!ReferenceConvInputChannelsMatchLastDim(*conv_payload, softmax_output_shape)) {
      continue;
    }

    const auto input_keys = InputTensorInfoKeys(*conv_payload);
    auto* conv_input = FindNamedTensorInfo(*conv_payload, input_keys, restore_output_name);
    if (conv_input == nullptr) {
      continue;
    }

    softmax_output["tensor_name"] = restore_output_name;
    SetTensorShape(softmax_output, softmax_output_shape);
    SetTensorShape(*conv_input, softmax_output_shape);
    PropagateTensorShapeToConsumers(
        ops_json, static_cast<size_t>(softmax_index), restore_output_name, softmax_output_shape);
    ops_json.erase(ops_json.begin() + static_cast<int>(index));
    --index;
  }
}

std::vector<int> PermuteVector(const std::vector<int>& values,
                               const std::vector<int>& perm) {
  std::vector<int> permuted;
  if (values.size() != perm.size()) {
    return permuted;
  }
  permuted.reserve(values.size());
  for (int axis : perm) {
    if (axis < 0 || static_cast<size_t>(axis) >= values.size()) {
      return {};
    }
    permuted.push_back(values[static_cast<size_t>(axis)]);
  }
  return permuted;
}

bool SliceSpecOutputShape(const std::vector<int>& input_shape,
                          const std::vector<int>& begin,
                          const std::vector<int>& size,
                          std::vector<int>& output_shape) {
  output_shape.clear();
  if (input_shape.size() != begin.size() || input_shape.size() != size.size()) {
    return false;
  }

  output_shape.reserve(input_shape.size());
  for (size_t axis = 0; axis < input_shape.size(); ++axis) {
    if (input_shape[axis] <= 0 || begin[axis] < 0 || begin[axis] > input_shape[axis]) {
      output_shape.clear();
      return false;
    }

    const int extent = size[axis] < 0 ? input_shape[axis] - begin[axis] : size[axis];
    if (extent < 0 || begin[axis] + extent > input_shape[axis]) {
      output_shape.clear();
      return false;
    }
    output_shape.push_back(extent);
  }
  return true;
}

bool TryPoseGridValuesFromAnchorMajorRank4(const ordered_json& source,
                                            int anchors,
                                            ordered_json& target) {
  if (!source.is_array() ||
      source.size() != 1 ||
      !source[0].is_array() ||
      source[0].size() != static_cast<size_t>(anchors)) {
    return false;
  }

  ordered_json coord_axis = ordered_json::array();
  for (int coord = 0; coord < 2; ++coord) {
    ordered_json anchor_axis = ordered_json::array();
    for (int anchor = 0; anchor < anchors; ++anchor) {
      const auto& anchor_value = source[0][static_cast<size_t>(anchor)];
      if (!anchor_value.is_array() ||
          anchor_value.size() != 2 ||
          !anchor_value[static_cast<size_t>(coord)].is_array() ||
          anchor_value[static_cast<size_t>(coord)].empty()) {
        return false;
      }
      ordered_json singleton = ordered_json::array();
      singleton.push_back(anchor_value[static_cast<size_t>(coord)][0]);
      anchor_axis.push_back(std::move(singleton));
    }
    coord_axis.push_back(std::move(anchor_axis));
  }

  target = ordered_json::array();
  target.push_back(std::move(coord_axis));
  return true;
}

bool TryPoseGridValuesFromLegacyCoordAnchorRank4(const ordered_json& source,
                                                 int anchors,
                                                 ordered_json& target) {
  if (!source.is_array() ||
      source.size() != 1 ||
      !source[0].is_array() ||
      source[0].size() != 1 ||
      !source[0][0].is_array() ||
      source[0][0].size() != 2) {
    return false;
  }

  ordered_json coord_axis = ordered_json::array();
  for (int coord = 0; coord < 2; ++coord) {
    const auto& source_anchor_axis = source[0][0][static_cast<size_t>(coord)];
    if (!source_anchor_axis.is_array() ||
        source_anchor_axis.size() != static_cast<size_t>(anchors)) {
      return false;
    }

    ordered_json anchor_axis = ordered_json::array();
    for (int anchor = 0; anchor < anchors; ++anchor) {
      ordered_json singleton = ordered_json::array();
      singleton.push_back(source_anchor_axis[static_cast<size_t>(anchor)]);
      anchor_axis.push_back(std::move(singleton));
    }
    coord_axis.push_back(std::move(anchor_axis));
  }

  target = ordered_json::array();
  target.push_back(std::move(coord_axis));
  return true;
}

bool FlattenScalarsJson(const ordered_json& source, ordered_json& target) {
  if (source.is_array()) {
    for (const auto& value : source) {
      if (!FlattenScalarsJson(value, target)) {
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

ordered_json MakeNhwcChannelBroadcastValues(ordered_json channel_values) {
  ordered_json width_axis = ordered_json::array();
  width_axis.push_back(std::move(channel_values));
  ordered_json height_axis = ordered_json::array();
  height_axis.push_back(std::move(width_axis));
  ordered_json batch_axis = ordered_json::array();
  batch_axis.push_back(std::move(height_axis));
  return batch_axis;
}

void PostprocessReferenceChannelBroadcastConstants(ordered_json& ops_json) {
  for (auto& op_json : ops_json) {
    const std::string op_name = GetWrappedOpType(op_json);
    if (op_name != "Mul" && op_name != "Add" && op_name != "Sub") {
      continue;
    }

    auto* payload = GetWrappedPayload(op_json);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2")) {
      continue;
    }

    auto try_retarget = [](ordered_json& constant_tensor,
                           const ordered_json& reference_tensor) -> bool {
      if (!constant_tensor.is_object() ||
          !reference_tensor.is_object() ||
          !constant_tensor.contains("values")) {
        return false;
      }

      const std::vector<int> constant_shape = JsonShapeToVector(constant_tensor);
      const std::vector<int> reference_shape = JsonShapeToVector(reference_tensor);
      if (reference_shape.size() != 4) {
        return false;
      }

      int channels = -1;
      if (constant_shape.size() == 3 &&
          constant_shape[0] > 0 &&
          constant_shape[1] == 1 &&
          constant_shape[2] == 1) {
        channels = constant_shape[0];
      } else if (constant_shape.size() == 4 &&
                 constant_shape[0] == 1 &&
                 constant_shape[1] > 0 &&
                 constant_shape[2] == 1 &&
                 constant_shape[3] == 1) {
        channels = constant_shape[1];
      } else {
        return false;
      }

      if (channels != reference_shape[3]) {
        return false;
      }

      ordered_json channel_values = ordered_json::array();
      if (!FlattenScalarsJson(constant_tensor["values"], channel_values) ||
          channel_values.size() != static_cast<size_t>(channels)) {
        return false;
      }

      SetTensorShape(constant_tensor, {1, 1, 1, channels});
      constant_tensor["values"] = MakeNhwcChannelBroadcastValues(std::move(channel_values));
      return true;
    };

    ordered_json& input_1 = (*payload)["input_tensor_info_1"];
    ordered_json& input_2 = (*payload)["input_tensor_info_2"];
    if (input_1.value("is_constant", false)) {
      try_retarget(input_1, input_2);
    }
    if (input_2.value("is_constant", false)) {
      try_retarget(input_2, input_1);
    }
  }
}

void PostprocessReferenceYoloPoseGridBroadcast(ordered_json& ops_json) {
  for (auto& op_json : ops_json) {
    const std::string op_name = GetWrappedOpType(op_json);
    if (op_name != "Mul" && op_name != "Add" && op_name != "Sub") {
      continue;
    }

    auto* payload = GetWrappedPayload(op_json);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2")) {
      continue;
    }

    auto try_retarget = [](ordered_json& constant_tensor,
                           const ordered_json& reference_tensor) -> bool {
      if (!constant_tensor.is_object() ||
          !reference_tensor.is_object() ||
          !constant_tensor.contains("values")) {
        return false;
      }

      const std::vector<int> constant_shape = JsonShapeToVector(constant_tensor);
      const std::vector<int> reference_shape = JsonShapeToVector(reference_tensor);
      const bool reference_has_coord_dim =
          std::find(reference_shape.begin(), reference_shape.end(), 2) != reference_shape.end();
      if (constant_shape.size() != 4 ||
          reference_shape.size() != 4 ||
          !reference_has_coord_dim) {
        return false;
      }

      ordered_json retargeted_values;
      int anchors = -1;
      if (constant_shape[0] == 1 &&
          constant_shape[1] > 4 &&
          constant_shape[2] == 2 &&
          constant_shape[3] == 1) {
        anchors = constant_shape[1];
        if (!TryPoseGridValuesFromAnchorMajorRank4(constant_tensor["values"],
                                                   anchors,
                                                   retargeted_values)) {
          return false;
        }
      } else if (constant_shape[0] == 1 &&
                 constant_shape[1] == 1 &&
                 constant_shape[2] == 2 &&
                 constant_shape[3] > 4) {
        anchors = constant_shape[3];
        if (!TryPoseGridValuesFromLegacyCoordAnchorRank4(constant_tensor["values"],
                                                         anchors,
                                                         retargeted_values)) {
          return false;
        }
      } else {
        return false;
      }
      if (std::find(reference_shape.begin(), reference_shape.end(), anchors) == reference_shape.end()) {
        return false;
      }

      SetTensorShape(constant_tensor, {1, 2, anchors, 1});
      constant_tensor["values"] = std::move(retargeted_values);
      return true;
    };

    ordered_json& input_1 = (*payload)["input_tensor_info_1"];
    ordered_json& input_2 = (*payload)["input_tensor_info_2"];
    if (input_1.value("is_constant", false)) {
      try_retarget(input_1, input_2);
    }
    if (input_2.value("is_constant", false)) {
      try_retarget(input_2, input_1);
    }
  }
}

void PostprocessReferenceReduceMaxReshapeAxis(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "ReduceMax") {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& input_tensor = (*payload)["input_tensor_info_1"];
    ordered_json& axes_tensor = (*payload)["input_tensor_info_2"];
    ordered_json& output_tensor = (*payload)["output_tensor_info"];
    const std::vector<int> input_shape = JsonShapeToVector(input_tensor);
    const std::vector<int> current_output_shape = JsonShapeToVector(output_tensor);
    if (input_shape.empty() || current_output_shape.empty()) {
      continue;
    }

    const std::string output_name = output_tensor.value("tensor_name", "");
    if (output_name.empty()) {
      continue;
    }

    const int consumer_index = FindFirstConsumerIndex(ops_json, output_name,
                                                      static_cast<int>(index + 1));
    const std::string consumer_op = consumer_index < 0
                                        ? std::string()
                                        : GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]);
    if (consumer_op != "reshape" && consumer_op != "Reshape") {
      continue;
    }

    auto* reshape_payload = GetWrappedPayload(ops_json[static_cast<size_t>(consumer_index)]);
    if (reshape_payload == nullptr ||
        !reshape_payload->contains("input_tensor_info_1") ||
        !reshape_payload->contains("input_tensor_info_2")) {
      continue;
    }

    ordered_json& reshape_input = (*reshape_payload)["input_tensor_info_1"];
    ordered_json& shape_tensor = (*reshape_payload)["input_tensor_info_2"];
    const std::vector<int> target_shape = IntVectorFromValuesJson(shape_tensor);
    const int64_t target_count = ShapeElementCount(target_shape);
    const std::vector<int> current_axes = IntVectorFromValuesJson(axes_tensor);
    const int64_t current_axis_count =
        ShapeElementCount(KeepDimsReducedShapeForAxes(input_shape, current_axes));
    if (target_count <= 0 || current_axis_count == target_count) {
      continue;
    }

    for (size_t axis = 0; axis < input_shape.size(); ++axis) {
      std::vector<int> candidate_shape = input_shape;
      candidate_shape[axis] = 1;
      if (ShapeElementCount(candidate_shape) != target_count) {
        continue;
      }

      axes_tensor["values"] = ordered_json::array({static_cast<int>(axis)});
      SetTensorShape(output_tensor, candidate_shape);
      SetTensorShape(reshape_input, candidate_shape);
      break;
    }
  }
}

void PostprocessReferenceSliceAfterInsertedTranspose(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "Slice") {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("input_tensor_info_3") ||
        !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& input_tensor = (*payload)["input_tensor_info_1"];
    ordered_json& begin_tensor = (*payload)["input_tensor_info_2"];
    ordered_json& size_tensor = (*payload)["input_tensor_info_3"];
    ordered_json& output_tensor = (*payload)["output_tensor_info"];
    if (!input_tensor.is_object() || !input_tensor.contains("tensor_name")) {
      continue;
    }

    const std::string input_name = input_tensor["tensor_name"].get<std::string>();
    const int producer_index = FindProducerIndex(ops_json, input_name, static_cast<int>(index));
    if (producer_index < 0 || GetWrappedOpType(ops_json[static_cast<size_t>(producer_index)]) != "Transpose") {
      continue;
    }

    auto* producer_payload = GetWrappedPayload(ops_json[static_cast<size_t>(producer_index)]);
    if (producer_payload == nullptr ||
        !producer_payload->contains("is_insert") ||
        !producer_payload->contains("input_tensor_info_1") ||
        !producer_payload->contains("input_tensor_info_2") ||
        !producer_payload->contains("output_tensor_info")) {
      continue;
    }

    const std::vector<int> producer_input_shape =
        JsonShapeToVector((*producer_payload)["input_tensor_info_1"]);
    const std::vector<int> transposed_input_shape = JsonShapeToVector(input_tensor);
    const std::vector<int> perm = IntVectorFromValuesJson((*producer_payload)["input_tensor_info_2"]);
    const std::vector<int> begin = IntVectorFromValuesJson(begin_tensor);
    const std::vector<int> size = IntVectorFromValuesJson(size_tensor);
    if (producer_input_shape.size() != 4 ||
        transposed_input_shape.size() != 4 ||
        perm.size() != 4 ||
        begin.size() != 4 ||
        size.size() != 4) {
      continue;
    }

    std::vector<int> old_layout_output_shape;
    std::vector<int> current_layout_output_shape;
    std::vector<int> permuted_layout_output_shape;
    if (!SliceSpecOutputShape(producer_input_shape, begin, size, old_layout_output_shape) ||
        SliceSpecOutputShape(transposed_input_shape, begin, size, current_layout_output_shape)) {
      continue;
    }

    const std::vector<int> permuted_begin = PermuteVector(begin, perm);
    const std::vector<int> permuted_size = PermuteVector(size, perm);
    if (permuted_begin.empty() ||
        permuted_size.empty() ||
        !SliceSpecOutputShape(transposed_input_shape, permuted_begin, permuted_size,
                              permuted_layout_output_shape)) {
      continue;
    }

    SetTensorValues(begin_tensor, permuted_begin);
    SetTensorValues(size_tensor, permuted_size);
    SetTensorShape(output_tensor, permuted_layout_output_shape);

    const std::string output_name = output_tensor.value("tensor_name", "");
    if (output_name.empty()) {
      continue;
    }
    for (size_t consumer = index + 1; consumer < ops_json.size(); ++consumer) {
      auto* consumer_payload = GetWrappedPayload(ops_json[consumer]);
      if (consumer_payload == nullptr) {
        continue;
      }
      for (const auto& input_key : InputTensorInfoKeys(*consumer_payload)) {
        auto input_it = consumer_payload->find(input_key);
        if (input_it != consumer_payload->end() &&
            input_it->is_object() &&
            input_it->value("tensor_name", "") == output_name) {
          SetTensorShape(*input_it, permuted_layout_output_shape);
        }
      }
    }
  }
}

ordered_json MakeReferenceMaxPoolOpFromJson(const ordered_json& input_tensor,
                                            const ordered_json& output_tensor,
                                            int filter_height,
                                            int filter_width,
                                            int stride_h,
                                            int stride_w) {
  ordered_json attributes;
  attributes["filter_height"] = filter_height;
  attributes["filter_width"] = filter_width;
  attributes["fused_activation_function"] = "None";
  attributes["padding"] = "VALID";
  attributes["stride_h"] = stride_h;
  attributes["stride_w"] = stride_w;

  ordered_json payload;
  payload["op_type_info"] = OpTypeInfo("MaxPool");
  payload["input_tensor_info"] = input_tensor;
  payload["output_tensor_info"] = output_tensor;
  payload["attributes_info"] = std::move(attributes);
  payload["is_insert"] = true;
  payload["skip_transpose_insert_before"] = true;
  payload["skip_transpose_insert_after"] = true;
  return WrapOp("MaxPool", std::move(payload));
}

void PostprocessReferenceReduceMaxToMaxPool(ordered_json& ops_json) {
  ordered_json rewritten = ordered_json::array();

  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "ReduceMax") {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr ||
        !payload->contains("input_tensor_info_1") ||
        !payload->contains("input_tensor_info_2") ||
        !payload->contains("output_tensor_info")) {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    const auto& input_tensor = (*payload)["input_tensor_info_1"];
    const auto& axes_tensor = (*payload)["input_tensor_info_2"];
    const auto& keepdims_output_tensor = (*payload)["output_tensor_info"];
    const std::vector<int> input_shape = JsonShapeToVector(input_tensor);
    const std::vector<int> keepdims_output_shape = JsonShapeToVector(keepdims_output_tensor);
    std::vector<int> axes = IntVectorFromValuesJson(axes_tensor);
    if (input_shape.size() != 3 ||
        keepdims_output_shape.size() != 3 ||
        axes.size() != 1 ||
        !payload->contains("attributes_info") ||
        !(*payload)["attributes_info"].is_object() ||
        !(*payload)["attributes_info"].value("keepdims", false)) {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    int axis = axes[0];
    if (axis < 0) {
      axis += static_cast<int>(input_shape.size());
    }
    if (axis != 2 ||
        input_shape[0] <= 0 ||
        input_shape[1] <= 0 ||
        input_shape[2] <= 1 ||
        keepdims_output_shape[0] != input_shape[0] ||
        keepdims_output_shape[1] != input_shape[1] ||
        keepdims_output_shape[2] != 1) {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    const std::string keepdims_output_name = keepdims_output_tensor.value("tensor_name", "");
    if (keepdims_output_name.empty() ||
        index + 1 >= ops_json.size() ||
        (GetWrappedOpType(ops_json[index + 1]) != "reshape" &&
         GetWrappedOpType(ops_json[index + 1]) != "Reshape")) {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    auto* reshape_payload = GetWrappedPayload(ops_json[index + 1]);
    if (reshape_payload == nullptr ||
        !reshape_payload->contains("input_tensor_info_1") ||
        !reshape_payload->contains("input_tensor_info_2") ||
        !reshape_payload->contains("output_tensor_info")) {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    const auto& reshape_input_tensor = (*reshape_payload)["input_tensor_info_1"];
    const std::string reshape_input_name = reshape_input_tensor.value("tensor_name", "");
    if (reshape_input_name != keepdims_output_name) {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    const std::vector<int> target_shape = IntVectorFromValuesJson((*reshape_payload)["input_tensor_info_2"]);
    if (ShapeElementCount(target_shape) != input_shape[0] * input_shape[1]) {
      rewritten.push_back(ops_json[index]);
      continue;
    }

    const std::string base = keepdims_output_name;
    const std::vector<int> pool_input_shape{input_shape[0], input_shape[1], input_shape[2], 1};
    const std::vector<int> pool_output_shape{input_shape[0], input_shape[1], 1, 1};
    ordered_json pool_input_tensor =
        CloneTensorWithNameAndShape(input_tensor, base + "_as_pool4d", pool_input_shape);
    ordered_json pool_output_tensor =
        CloneTensorWithNameAndShape(keepdims_output_tensor, base + "_pool4d", pool_output_shape);

    rewritten.push_back(MakeReferenceReshapeOpFromJson(
        input_tensor,
        pool_input_tensor,
        base + "_as_pool4d_shape",
        pool_input_shape));
    rewritten.push_back(MakeReferenceMaxPoolOpFromJson(
        pool_input_tensor,
        pool_output_tensor,
        1,
        input_shape[2],
        1,
        1));
    rewritten.push_back(MakeReferenceReshapeOpFromJson(
        pool_output_tensor,
        (*reshape_payload)["output_tensor_info"],
        base + "_from_pool4d_shape",
        target_shape));
    ++index;
  }

  ops_json = std::move(rewritten);
}

void PostprocessReferenceOnnxInfo(ordered_json& ops_json) {
  int same_pad_index = 0;
  PostprocessReferenceMaxPoolTensorInfo(ops_json);
  PostprocessReferenceMaxPoolSamePad(ops_json, same_pad_index);
  PostprocessReferenceConvPad(ops_json, same_pad_index);
  PostprocessReferenceRank3ToRank4BroadcastInputs(ops_json);
  PostprocessReferenceTransposeInsertBefore(ops_json);
  PostprocessReferenceTransposeInsertAfter(ops_json);
  PostprocessReferenceTransposeShapePropagation(ops_json);
  PostprocessReferenceElementwiseOutputShapes(ops_json);
  PostprocessReferenceShapePreservingUnaryOutputShapes(ops_json);
  PostprocessReferenceSoftmaxAxisRestoreConvConsumer(ops_json);
  PostprocessReferenceRank4BatchMatMulFlatten(ops_json);
  PostprocessReferenceInsertedTransposeShapeNormalization(ops_json);
  PostprocessReferenceInsertedPadShapeNormalization(ops_json);
  PostprocessReferenceSliceAfterInsertedTranspose(ops_json);
  PostprocessReferenceSplitQuantOverride(ops_json);
  PostprocessReferenceChannelBroadcastConstants(ops_json);
  PostprocessReferenceYoloPoseGridBroadcast(ops_json);
  PostprocessReferenceReduceMaxReshapeAxis(ops_json);
  PostprocessReferenceReduceMaxToMaxPool(ops_json);
}

void SyncQLinearConcatProducerQuantInfo(ordered_json& ops_json) {
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "QLinearConcat") {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr) {
      continue;
    }

    for (const auto& input_key : InputTensorInfoKeys(*payload)) {
      auto input_it = payload->find(input_key);
      if (input_it == payload->end() ||
          !input_it->is_object() ||
          !input_it->contains("tensor_name") ||
          !input_it->contains("quant_scale") ||
          !input_it->contains("zero_point")) {
        continue;
      }

      const std::string tensor_name = (*input_it)["tensor_name"].get<std::string>();
      const int producer_index = FindProducerIndex(ops_json, tensor_name, static_cast<int>(index));
      if (producer_index < 0) {
        continue;
      }

      auto* producer_payload = GetWrappedPayload(ops_json[static_cast<size_t>(producer_index)]);
      if (producer_payload == nullptr || !producer_payload->contains("output_tensor_info")) {
        continue;
      }

      ordered_json& producer_output = (*producer_payload)["output_tensor_info"];
      if (!producer_output.is_object() ||
          !producer_output.contains("tensor_name") ||
          producer_output["tensor_name"] != tensor_name) {
        continue;
      }

      producer_output["quant_scale"] = (*input_it)["quant_scale"];
      producer_output["zero_point"] = (*input_it)["zero_point"];
      if (input_it->contains("tensor_type")) {
        producer_output["tensor_type"] = (*input_it)["tensor_type"];
      }
    }
  }
}

int MaxQuantizeLinearInsertOutputId(ordered_json& ops_json) {
  constexpr const char* prefix = "QuantizeLinear_insert_output";
  constexpr size_t prefix_len = 28;
  int max_id = 0;
  for (auto& op_json : ops_json) {
    const auto* payload = GetWrappedPayload(op_json);
    if (payload == nullptr) {
      continue;
    }

    const auto keys = OutputTensorInfoKeys(*payload);
    for (const auto& key : keys) {
      auto it = payload->find(key);
      if (it == payload->end() ||
          !it->is_object() ||
          !it->contains("tensor_name") ||
          !(*it)["tensor_name"].is_string()) {
        continue;
      }

      const std::string name = (*it)["tensor_name"].get<std::string>();
      if (name.rfind(prefix, 0) != 0 || name.size() <= prefix_len) {
        continue;
      }

      int id = 0;
      bool valid = true;
      for (size_t i = prefix_len; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
          valid = false;
          break;
        }
        id = id * 10 + (name[i] - '0');
      }
      if (valid) {
        max_id = std::max(max_id, id);
      }
    }
  }
  return max_id;
}

void PostprocessReferenceConcatRequantize(ordered_json& ops_json) {
  int concat_index = MaxQuantizeLinearInsertOutputId(ops_json);
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "QLinearConcat") {
      continue;
    }

    bool inserted_requant = false;
    do {
      inserted_requant = false;
      auto* payload = GetWrappedPayload(ops_json[index]);
      if (payload == nullptr || !payload->contains("output_tensor_info")) {
        break;
      }

      const ordered_json output_tensor = (*payload)["output_tensor_info"];
      for (int input_slot = 1; input_slot <= 9; ++input_slot) {
        ++concat_index;
        const std::string input_key = "input_tensor_info_" + std::to_string(input_slot);
        auto input_it = payload->find(input_key);
        if (input_it == payload->end() || !input_it->is_object()) {
          continue;
        }

        ordered_json input_tensor = *input_it;
        const std::string tensor_name = input_tensor["tensor_name"].get<std::string>();
        const int producer_index = FindProducerIndex(ops_json, tensor_name, static_cast<int>(index));
        if (producer_index < 0) {
          continue;
        }

        const ordered_json* producer_tensor =
            FindProducerTensorInfo(ops_json, tensor_name, static_cast<int>(index));
        if (producer_tensor != nullptr && HasRealQuantInfo(*producer_tensor)) {
          input_tensor = *producer_tensor;
        }

        if (!HasRealQuantInfo(input_tensor) || !HasRealQuantInfo(output_tensor) ||
            (JsonEqual(input_tensor["quant_scale"], output_tensor["quant_scale"]) &&
             JsonEqual(input_tensor["zero_point"], output_tensor["zero_point"]))) {
          continue;
        }

        const std::string updated_name =
            std::string("QuantizeLinear_insert_output") + std::to_string(concat_index);
        ops_json.insert(ops_json.begin() + producer_index + 1,
                        MakeReferenceRequantizeOp(input_tensor, output_tensor, updated_name));
        if (producer_index + 1 <= static_cast<int>(index)) {
          ++index;
        }

        auto* updated_payload = GetWrappedPayload(ops_json[index]);
        if (updated_payload != nullptr) {
          auto& updated_input = (*updated_payload)[input_key];
          updated_input["tensor_name"] = updated_name;
          updated_input["quant_scale"] = output_tensor["quant_scale"];
          updated_input["zero_point"] = output_tensor["zero_point"];
        }
        inserted_requant = true;
        break;
      }
    } while (inserted_requant);
  }

  SyncQLinearConcatProducerQuantInfo(ops_json);
}

void PostprocessReferenceQGemmBridges(ordered_json& ops_json) {
  // This pass keeps the collector generic by moving the VGG-specific dense
  // bridge materialization into a dedicated postprocess. The rule remains
  // name-driven because the reference JSON contract itself is name-driven
  // here.
  for (size_t index = 0; index < ops_json.size(); ++index) {
    if (GetWrappedOpType(ops_json[index]) != "QGemm") {
      continue;
    }

    auto* payload = GetWrappedPayload(ops_json[index]);
    if (payload == nullptr || !payload->contains("output_tensor_info")) {
      continue;
    }

    ordered_json& output_tensor = (*payload)["output_tensor_info"];
    if (!output_tensor.is_object() || !output_tensor.contains("tensor_name")) {
      continue;
    }

    const std::string output_tensor_name = output_tensor["tensor_name"].get<std::string>();
    QGemmBridgeNames bridge_names;
    if (!reference_style_internal::TryGetCompatibleQGemmBridgeNames(output_tensor_name, bridge_names)) {
      continue;
    }

    const int consumer_index = FindFirstConsumerIndex(ops_json, output_tensor_name, static_cast<int>(index + 1));
    if (consumer_index < 0 ||
        GetWrappedOpType(ops_json[static_cast<size_t>(consumer_index)]) != "QGemm") {
      continue;
    }

    ordered_json dequant_output = output_tensor;
    dequant_output["tensor_name"] = bridge_names.dequant_output_name;
    dequant_output["tensor_type"] = "float32";
    dequant_output.erase("quant_scale");
    dequant_output.erase("zero_point");

    ordered_json dequant_payload;
    dequant_payload["op_type_info"] = OpTypeInfo("DequantizeLinear");
    dequant_payload["input_tensor_info"] = output_tensor;
    dequant_payload["output_tensor_info"] = dequant_output;

    ordered_json requant_output = output_tensor;
    requant_output["tensor_name"] = bridge_names.requant_output_name;

    ordered_json quant_payload;
    quant_payload["op_type_info"] = OpTypeInfo("QuantizeLinear");
    quant_payload["input_tensor_info"] = dequant_output;
    quant_payload["output_tensor_info"] = requant_output;

    ops_json.insert(ops_json.begin() + static_cast<int>(index + 1),
                    WrapOp("DequantizeLinear", std::move(dequant_payload)));
    ops_json.insert(ops_json.begin() + static_cast<int>(index + 2),
                    WrapOp("QuantizeLinear", std::move(quant_payload)));

    for (size_t consumer_scan = index + 3; consumer_scan < ops_json.size(); ++consumer_scan) {
      auto* consumer_payload = GetWrappedPayload(ops_json[consumer_scan]);
      if (consumer_payload == nullptr) {
        continue;
      }

      for (const auto& input_key : InputTensorInfoKeys(*consumer_payload)) {
        auto input_it = consumer_payload->find(input_key);
        if (input_it == consumer_payload->end() || !input_it->is_object() ||
            !input_it->contains("tensor_name")) {
          continue;
        }

        if ((*input_it)["tensor_name"] == output_tensor_name) {
          (*input_it)["tensor_name"] = bridge_names.requant_output_name;
        }
      }
    }

    index += 2;
  }
}

void ConvertTensorInfoUint8ToInt8(ordered_json& tensor_info) {
  if (!tensor_info.is_object() ||
      !tensor_info.contains("tensor_type") ||
      !tensor_info.contains("quant_scale") ||
      tensor_info["tensor_type"] != "uint8" ||
      IsQuantNoneValue(tensor_info["quant_scale"])) {
    return;
  }

  tensor_info["tensor_type"] = "int8";
  if (!tensor_info.contains("zero_point") || IsQuantNoneValue(tensor_info["zero_point"])) {
    return;
  }

  if (tensor_info["zero_point"].is_array()) {
    for (auto& value : tensor_info["zero_point"]) {
      if (value.is_number_integer()) {
        value = std::clamp(value.get<int>() - 128, -128, 127);
      }
    }
  } else if (tensor_info["zero_point"].is_number_integer()) {
    tensor_info["zero_point"] =
        std::clamp(tensor_info["zero_point"].get<int>() - 128, -128, 127);
  }
}

void PostprocessReferenceUint8ToInt8(ordered_json& ops_json) {
  for (auto& op_json : ops_json) {
    auto* payload = GetWrappedPayload(op_json);
    if (payload == nullptr) {
      continue;
    }

    for (const auto& key : InputTensorInfoKeys(*payload)) {
      auto it = payload->find(key);
      if (it != payload->end()) {
        ConvertTensorInfoUint8ToInt8(*it);
      }
    }

    for (const auto& key : OutputTensorInfoKeys(*payload)) {
      auto it = payload->find(key);
      if (it != payload->end()) {
        ConvertTensorInfoUint8ToInt8(*it);
      }
    }

    auto weight_it = payload->find("input_weight_info");
    if (weight_it != payload->end()) {
      ConvertTensorInfoUint8ToInt8(*weight_it);
    }
    auto bias_it = payload->find("input_bias_info");
    if (bias_it != payload->end()) {
      ConvertTensorInfoUint8ToInt8(*bias_it);
    }
  }

  PostprocessReferenceConcatRequantize(ops_json);
}
