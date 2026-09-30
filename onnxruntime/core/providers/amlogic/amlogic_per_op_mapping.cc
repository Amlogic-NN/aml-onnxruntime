// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_model_info.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace amlogic {
namespace {

int ToIntDim(int64_t dim) {
  if (dim > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  if (dim < static_cast<int64_t>(std::numeric_limits<int>::min())) {
    return std::numeric_limits<int>::min();
  }
  return static_cast<int>(dim);
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
  for (const auto& dim : tensor_shape->dim()) {
    shape.push_back(dim.has_dim_value() ? ToIntDim(dim.dim_value()) : -1);
  }
  return shape;
}

std::vector<int> ToNhwcShape(const std::vector<int>& shape) {
  if (shape.size() != 4) {
    return shape;
  }
  return {shape[0], shape[2], shape[3], shape[1]};
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

std::string TensorProtoDataTypeToString(int32_t elem_type) {
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return "float32";
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return "uint8";
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return "int8";
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      return "uint16";
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return "int16";
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return "int32";
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return "int64";
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      return "bool";
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return "float16";
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return "float64";
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      return "uint32";
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      return "uint64";
    default:
      return "unknown";
  }
}

std::string NodeArgTypeToString(const NodeArg& node_arg) {
  const auto* type_proto = node_arg.TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type()) {
    const auto* type = node_arg.Type();
    return type != nullptr ? *type : "unknown";
  }

  return TensorProtoDataTypeToString(type_proto->tensor_type().elem_type());
}

bool IsFourDimensional(const std::vector<int>& shape) {
  return shape.size() == 4;
}

bool IsQuantizedTensorType(std::string_view tensor_type) {
  return tensor_type == "int8" || tensor_type == "uint8";
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

bool IsOrtInsertedQdqName(std::string_view name) {
  return EndsWith(name, "/duplicated") ||
         EndsWith(name, "_pre_q") ||
         EndsWith(name, "_q_to_dq") ||
         EndsWith(name, "_post_dq");
}

bool IsOrtInsertedQdqHelperNode(const Node& node) {
  if (node.OpType() != "QuantizeLinear" && node.OpType() != "DequantizeLinear") {
    return false;
  }

  for (const NodeArg* input : node.InputDefs()) {
    if (input != nullptr && IsOrtInsertedQdqName(input->Name())) {
      return true;
    }
  }

  for (const NodeArg* output : node.OutputDefs()) {
    if (output != nullptr && IsOrtInsertedQdqName(output->Name())) {
      return true;
    }
  }

  return false;
}

const ONNX_NAMESPACE::AttributeProto* FindAttribute(const Node& node, const std::string& name) {
  const auto& attributes = node.GetAttributes();
  const auto attr = attributes.find(name);
  return attr == attributes.end() ? nullptr : &attr->second;
}

std::vector<int> GetIntsAttribute(const Node& node, const std::string& name, std::vector<int> default_value) {
  const auto* attr = FindAttribute(node, name);
  if (attr == nullptr || attr->type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INTS) {
    return default_value;
  }

  std::vector<int> values;
  values.reserve(static_cast<size_t>(attr->ints_size()));
  for (int i = 0; i < attr->ints_size(); ++i) {
    values.push_back(ToIntDim(attr->ints(i)));
  }
  return values;
}

std::string GetStringAttribute(const Node& node, const std::string& name, const std::string& default_value) {
  const auto* attr = FindAttribute(node, name);
  if (attr == nullptr || attr->type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRING) {
    return default_value;
  }
  return attr->s();
}

std::optional<std::string> TryMapToTfliteOp(std::string_view onnx_op_type) {
  if (onnx_op_type == "Conv" || onnx_op_type == "QLinearConv") return "CONV_2D";
  if (onnx_op_type == "ConvTranspose") return "TRANSPOSE_CONV";
  if (onnx_op_type == "Relu") return "RELU";
  if (onnx_op_type == "Add" || onnx_op_type == "QLinearAdd") return "ADD";
  if (onnx_op_type == "Sub") return "SUB";
  if (onnx_op_type == "Mul" || onnx_op_type == "QLinearMul") return "MUL";
  if (onnx_op_type == "Div") return "DIV";
  if (onnx_op_type == "MaxPool") return "MAX_POOL_2D";
  if (onnx_op_type == "AveragePool") return "AVERAGE_POOL_2D";
  if (onnx_op_type == "GlobalAveragePool" || onnx_op_type == "QLinearGlobalAveragePool" || onnx_op_type == "ReduceMean") return "MEAN";
  if (onnx_op_type == "Concat" || onnx_op_type == "QLinearConcat") return "CONCATENATION";
  if (onnx_op_type == "Reshape" || onnx_op_type == "Flatten") return "RESHAPE";
  if (onnx_op_type == "Resize" || onnx_op_type == "Upsample") return "RESIZE_NEAREST_NEIGHBOR";
  if (onnx_op_type == "Softmax" || onnx_op_type == "QLinearSoftmax") return "SOFTMAX";
  if (onnx_op_type == "Sigmoid" || onnx_op_type == "QLinearSigmoid") return "LOGISTIC";
  if (onnx_op_type == "Tanh") return "TANH";
  if (onnx_op_type == "Transpose") return "TRANSPOSE";
  if (onnx_op_type == "MatMul" || onnx_op_type == "Gemm" || onnx_op_type == "QGemm") return "FULLY_CONNECTED";
  if (onnx_op_type == "Pad") return "PAD";
  if (onnx_op_type == "Slice") return "SLICE";
  if (onnx_op_type == "Split") return "SPLIT";
  if (onnx_op_type == "QuantizeLinear") return "QUANTIZE";
  if (onnx_op_type == "DequantizeLinear") return "DEQUANTIZE";
  return std::nullopt;
}

bool UsesNhwcInternalLayout(std::string_view onnx_op_type) {
  static const std::unordered_set<std::string_view> nhwc_ops = {
      "AveragePool",
      "Conv",
      "ConvTranspose",
      "GlobalAveragePool",
      "GlobalMaxPool",
      "MaxPool",
      "QLinearConv",
      "QLinearGlobalAveragePool",
      "Resize",
      "Sigmoid",
      "QLinearSigmoid",
      "Tanh",
      "Upsample",
  };

  return nhwc_ops.find(onnx_op_type) != nhwc_ops.end();
}

json AttributeToJson(const ONNX_NAMESPACE::AttributeProto& attr) {
  switch (attr.type()) {
    case ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOAT:
      return attr.f();
    case ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT:
      return attr.i();
    case ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRING:
      return attr.s();
    case ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_FLOATS: {
      std::vector<float> values(attr.floats().begin(), attr.floats().end());
      return values;
    }
    case ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INTS: {
      std::vector<int64_t> values(attr.ints().begin(), attr.ints().end());
      return values;
    }
    case ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRINGS: {
      std::vector<std::string> values(attr.strings().begin(), attr.strings().end());
      return values;
    }
    case ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_TENSOR:
      return json{{"tensor_name", attr.t().name()},
                  {"tensor_type", TensorProtoDataTypeToString(attr.t().data_type())},
                  {"tensor_shape", TensorProtoShape(attr.t())}};
    default:
      return json{{"unsupported_attribute_type", attr.type()}};
  }
}

std::string InferTfliteConvPadding(const Node& node, const std::vector<int>& pads) {
  const std::string auto_pad = GetStringAttribute(node, "auto_pad", "NOTSET");
  if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    return "SAME";
  }
  if (auto_pad == "VALID") {
    return "VALID";
  }
  if (pads.size() != 4) {
    return "VALID";
  }
  if (pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
    return "VALID";
  }
  if (pads[0] == pads[2] && pads[1] == pads[3]) {
    return "SAME";
  }
  return "EXPLICIT";
}

json MakeAttributesJson(const Node& node) {
  json attributes = json::object();
  attributes["source_op_type"] = node.OpType();
  attributes["source_domain"] = node.Domain();

  for (const auto& entry : node.GetAttributes()) {
    attributes[entry.first] = AttributeToJson(entry.second);
  }

  if (node.OpType() == "Conv" || node.OpType() == "QLinearConv") {
    const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
    const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});
    const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
    attributes["tflite_filter_layout"] = "OHWI";
    attributes["tflite_input_layout"] = "NHWC";
    attributes["tflite_padding"] = InferTfliteConvPadding(node, pads);
    attributes["stride_h"] = strides.empty() ? 1 : strides[0];
    attributes["stride_w"] = strides.size() > 1 ? strides[1] : 1;
    attributes["dilation_h_factor"] = dilations.empty() ? 1 : dilations[0];
    attributes["dilation_w_factor"] = dilations.size() > 1 ? dilations[1] : 1;
  }

  return attributes;
}

template <typename T>
Status UnpackVector(const ONNX_NAMESPACE::TensorProto& tensor,
                    const std::filesystem::path& model_path,
                    std::vector<T>& values) {
  const auto shape = TensorProtoShape(tensor);
  const size_t count = ElementCount(shape);
  values.resize(count);
  if (count == 0) {
    return Status::OK();
  }
  return utils::UnpackTensor<T>(tensor, model_path, values.data(), values.size());
}

Status ReadFloatInitializerAsJson(const GraphViewer& graph_viewer, const std::string& name, json& value) {
  const ONNX_NAMESPACE::TensorProto* tensor = graph_viewer.GetConstantInitializer(name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing float initializer: ", name);
  std::vector<float> values;
  ORT_RETURN_IF_ERROR(UnpackVector<float>(*tensor, graph_viewer.ModelPath(), values));
  if (values.size() == 1) {
    value = values[0];
  } else {
    value = values;
  }
  return Status::OK();
}

Status ReadZeroPointInitializerAsJson(const GraphViewer& graph_viewer, const std::string& name, json& value) {
  const ONNX_NAMESPACE::TensorProto* tensor = graph_viewer.GetConstantInitializer(name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing zero point initializer: ", name);

  if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    std::vector<uint8_t> values;
    ORT_RETURN_IF_ERROR(UnpackVector<uint8_t>(*tensor, graph_viewer.ModelPath(), values));
    if (values.size() == 1) {
      value = static_cast<int>(values[0]) - 128;
    } else {
      json result = json::array();
      for (uint8_t v : values) {
        result.push_back(static_cast<int>(v) - 128);
      }
      value = std::move(result);
    }
    return Status::OK();
  }

  if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    std::vector<int8_t> values;
    ORT_RETURN_IF_ERROR(UnpackVector<int8_t>(*tensor, graph_viewer.ModelPath(), values));
    if (values.size() == 1) {
      value = static_cast<int>(values[0]);
    } else {
      json result = json::array();
      for (int8_t v : values) {
        result.push_back(static_cast<int>(v));
      }
      value = std::move(result);
    }
    return Status::OK();
  }

  if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    std::vector<int32_t> values;
    ORT_RETURN_IF_ERROR(UnpackVector<int32_t>(*tensor, graph_viewer.ModelPath(), values));
    if (values.size() == 1) {
      value = ToIntDim(values[0]);
    } else {
      json result = json::array();
      for (int32_t v : values) {
        result.push_back(ToIntDim(v));
      }
      value = std::move(result);
    }
    return Status::OK();
  }

  value = 0;
  return Status::OK();
}

void AddQuantSpec(const GraphViewer& graph_viewer,
                  const std::string& tensor_name,
                  const std::string& scale_name,
                  const std::string& zero_point_name,
                  json& parent) {
  json spec{
      {"tensor_name", tensor_name},
      {"scale_initializer", scale_name},
      {"zero_point_initializer", zero_point_name},
  };

  json scale_json;
  if (ReadFloatInitializerAsJson(graph_viewer, scale_name, scale_json).IsOK()) {
    spec["scale"] = std::move(scale_json);
  }

  json zero_point_json;
  if (ReadZeroPointInitializerAsJson(graph_viewer, zero_point_name, zero_point_json).IsOK()) {
    spec["zero_point"] = std::move(zero_point_json);
  }

  parent = std::move(spec);
}

void AppendQuantSpec(const GraphViewer& graph_viewer,
                     const std::string& tensor_name,
                     const std::string& scale_name,
                     const std::string& zero_point_name,
                     json& parent_array) {
  json spec;
  AddQuantSpec(graph_viewer, tensor_name, scale_name, zero_point_name, spec);
  parent_array.push_back(std::move(spec));
}

void MaybeAddQuantParameters(const GraphViewer& graph_viewer, const Node& node, json& record) {
  const auto inputs = node.InputDefs();
  const auto outputs = node.OutputDefs();
  const std::string& op_type = node.OpType();

  json quant = json::object();
  if (op_type == "QuantizeLinear" || op_type == "DequantizeLinear") {
    if (inputs.size() >= 3 && inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr) {
      AddQuantSpec(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(), quant["tensor"]);
    }
  } else if (op_type == "QLinearConv") {
    if (inputs.size() >= 8 && inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
        inputs[3] != nullptr && inputs[4] != nullptr && inputs[5] != nullptr &&
        inputs[6] != nullptr && inputs[7] != nullptr) {
      AddQuantSpec(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(), quant["input"]);
      AddQuantSpec(graph_viewer, inputs[3]->Name(), inputs[4]->Name(), inputs[5]->Name(), quant["weight"]);
      AddQuantSpec(graph_viewer, outputs.empty() || outputs[0] == nullptr ? "" : outputs[0]->Name(),
                   inputs[6]->Name(), inputs[7]->Name(), quant["output"]);
    }
  } else if (op_type == "QLinearAdd" || op_type == "QLinearMul") {
    if (inputs.size() >= 8) {
      AddQuantSpec(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(), quant["input_1"]);
      AddQuantSpec(graph_viewer, inputs[3]->Name(), inputs[4]->Name(), inputs[5]->Name(), quant["input_2"]);
      AddQuantSpec(graph_viewer, outputs.empty() || outputs[0] == nullptr ? "" : outputs[0]->Name(),
                   inputs[6]->Name(), inputs[7]->Name(), quant["output"]);
    }
  } else if (op_type == "QLinearSigmoid" || op_type == "QLinearSoftmax" || op_type == "QLinearGlobalAveragePool") {
    if (inputs.size() >= 5) {
      AddQuantSpec(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(), quant["input"]);
      AddQuantSpec(graph_viewer, outputs.empty() || outputs[0] == nullptr ? "" : outputs[0]->Name(),
                   inputs[3]->Name(), inputs[4]->Name(), quant["output"]);
    }
  } else if (op_type == "QLinearConcat") {
    if (inputs.size() >= 5 && inputs[0] != nullptr && inputs[1] != nullptr) {
      AddQuantSpec(graph_viewer, outputs.empty() || outputs[0] == nullptr ? "" : outputs[0]->Name(),
                   inputs[0]->Name(), inputs[1]->Name(), quant["output"]);
      quant["inputs"] = json::array();
      for (size_t i = 2; i + 2 < inputs.size(); i += 3) {
        if (inputs[i] == nullptr || inputs[i + 1] == nullptr || inputs[i + 2] == nullptr) {
          continue;
        }
        AppendQuantSpec(graph_viewer, inputs[i]->Name(), inputs[i + 1]->Name(), inputs[i + 2]->Name(),
                        quant["inputs"]);
      }
    }
  } else if (op_type == "QGemm") {
    if (inputs.size() >= 9 && inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr &&
        inputs[3] != nullptr && inputs[4] != nullptr && inputs[5] != nullptr &&
        inputs[7] != nullptr && inputs[8] != nullptr) {
      AddQuantSpec(graph_viewer, inputs[0]->Name(), inputs[1]->Name(), inputs[2]->Name(), quant["input"]);
      AddQuantSpec(graph_viewer, inputs[3]->Name(), inputs[4]->Name(), inputs[5]->Name(), quant["weight"]);
      if (inputs[6] != nullptr) {
        quant["bias"] = json{
            {"tensor_name", inputs[6]->Name()},
            {"derivation", "bias_scale = input_scale * weight_scale"},
        };
      }
      AddQuantSpec(graph_viewer,
                   outputs.empty() || outputs[0] == nullptr ? "" : outputs[0]->Name(),
                   inputs[7]->Name(),
                   inputs[8]->Name(),
                   quant["output"]);
    }
  }

  if (!quant.empty()) {
    record["quant_parameters"] = std::move(quant);
  }
}

json MakeTensorSummaryFromNodeArg(const GraphViewer& graph_viewer, const NodeArg& node_arg, bool nhwc_view) {
  json result;
  result["tensor_name"] = node_arg.Name();
  result["tensor_type"] = NodeArgTypeToString(node_arg);
  result["tensor_shape"] = nhwc_view ? ToNhwcShape(NodeArgShape(node_arg)) : NodeArgShape(node_arg);
  result["is_constant"] = graph_viewer.IsConstantInitializer(node_arg.Name(), true);
  return result;
}

json MakeTensorSummaryFromInitializer(const ONNX_NAMESPACE::TensorProto& tensor,
                                      bool nhwc_view,
                                      bool is_conv_weight) {
  json result;
  result["tensor_name"] = tensor.name();
  result["tensor_type"] = TensorProtoDataTypeToString(tensor.data_type());
  result["is_constant"] = true;

  const std::vector<int> source_shape = TensorProtoShape(tensor);
  std::vector<int> target_shape = source_shape;
  if (is_conv_weight) {
    if (source_shape.size() == 4) {
      target_shape = {source_shape[0], source_shape[2], source_shape[3], source_shape[1]};
      result["layout_hint"] = "OHWI";
    }
  } else if (nhwc_view) {
    target_shape = ToNhwcShape(source_shape);
    if (source_shape.size() == 4) {
      result["layout_hint"] = "NHWC";
    }
  }
  result["tensor_shape"] = target_shape;
  result["source_tensor_shape"] = source_shape;
  result["value_count"] = ElementCount(source_shape);
  return result;
}

bool HasQdqNeighbor(const GraphViewer& graph_viewer, const Node& node) {
  for (const NodeArg* input : node.InputDefs()) {
    if (input == nullptr) {
      continue;
    }
    const Node* producer = graph_viewer.GetProducerNode(input->Name());
    if (producer != nullptr &&
        (producer->OpType() == "QuantizeLinear" || producer->OpType() == "DequantizeLinear")) {
      return true;
    }
  }

  for (const NodeArg* output : node.OutputDefs()) {
    if (output == nullptr) {
      continue;
    }
    for (const Node* consumer : graph_viewer.GetConsumerNodes(output->Name())) {
      if (consumer != nullptr &&
          (consumer->OpType() == "QuantizeLinear" || consumer->OpType() == "DequantizeLinear")) {
        return true;
      }
    }
  }

  return false;
}

bool HasQuantizedInputOrOutput(const Node& node) {
  for (const NodeArg* input : node.InputDefs()) {
    if (input != nullptr && IsQuantizedTensorType(NodeArgTypeToString(*input))) {
      return true;
    }
  }
  for (const NodeArg* output : node.OutputDefs()) {
    if (output != nullptr && IsQuantizedTensorType(NodeArgTypeToString(*output))) {
      return true;
    }
  }
  return false;
}

std::string DetermineQuantContext(const GraphViewer& graph_viewer, const Node& node) {
  const std::string& op_type = node.OpType();
  if (op_type.rfind("QLinear", 0) == 0 || op_type == "QGemm") {
    return "qlinear_native";
  }
  if (op_type == "QuantizeLinear" || op_type == "DequantizeLinear") {
    return "qdq_bridge";
  }

  const bool has_qdq_neighbor = HasQdqNeighbor(graph_viewer, node);
  const bool has_quantized_io = HasQuantizedInputOrOutput(node);
  if (has_qdq_neighbor && has_quantized_io) {
    return "mixed";
  }
  if (has_qdq_neighbor) {
    return "qdq_wrapped";
  }
  if (has_quantized_io) {
    return "mixed";
  }
  return "float_native";
}

json MakeInsertedTransposeMetadata(const std::string& tensor_name,
                                   const std::vector<int>& perm,
                                   const std::string& reason) {
  return json{
      {"op_type", "TRANSPOSE"},
      {"tensor_name", tensor_name},
      {"perm", perm},
      {"reason", reason},
  };
}

json MakeInsertedPadMetadata(const std::vector<int>& pads, const std::string& reason) {
  return json{
      {"op_type", "PAD"},
      {"pads", pads},
      {"reason", reason},
  };
}

void PopulateInsertedOps(const GraphViewer& graph_viewer,
                         const Node& node,
                         const std::unordered_set<std::string>& graph_outputs,
                         json& record) {
  json inserted_pre_ops = json::array();
  json inserted_post_ops = json::array();
  const bool uses_nhwc = UsesNhwcInternalLayout(node.OpType());

  if (uses_nhwc) {
    for (const NodeArg* input : node.InputDefs()) {
      if (input == nullptr || graph_viewer.IsConstantInitializer(input->Name(), true)) {
        continue;
      }
      const std::vector<int> input_shape = NodeArgShape(*input);
      if (IsFourDimensional(input_shape)) {
        inserted_pre_ops.push_back(
            MakeInsertedTransposeMetadata(input->Name(), {0, 2, 3, 1}, "NCHW_TO_NHWC activation view"));
      }
    }
  }

  if (node.OpType() == "Conv" || node.OpType() == "QLinearConv") {
    const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
    const bool has_non_zero_pad = pads.size() == 4 &&
                                  std::any_of(pads.begin(), pads.end(), [](int pad) { return pad != 0; });
    if (has_non_zero_pad) {
      inserted_pre_ops.push_back(MakeInsertedPadMetadata(pads, "Explicit SAME/EXPLICIT pad before convolution"));
    }
  }

  if (uses_nhwc) {
    for (const NodeArg* output : node.OutputDefs()) {
      if (output == nullptr) {
        continue;
      }
      if (graph_outputs.find(output->Name()) == graph_outputs.end()) {
        continue;
      }
      const std::vector<int> output_shape = NodeArgShape(*output);
      if (IsFourDimensional(output_shape)) {
        inserted_post_ops.push_back(
            MakeInsertedTransposeMetadata(output->Name(), {0, 3, 1, 2}, "NHWC_TO_NCHW graph output restore"));
      }
    }
  }

  record["inserted_pre_ops"] = std::move(inserted_pre_ops);
  record["inserted_post_ops"] = std::move(inserted_post_ops);
}

json BuildNodeRecord(const GraphViewer& graph_viewer,
                     const Node& node,
                     const std::unordered_set<std::string>& graph_outputs) {
  json record;
  record["source_node_index"] = static_cast<int64_t>(node.Index());
  record["source_op_type"] = node.OpType();
  record["source_op_name"] = node.Name().empty() ? node.OpType() + "_" + std::to_string(node.Index()) : node.Name();
  record["source_domain"] = node.Domain();

  const auto mapped_tflite_op = TryMapToTfliteOp(node.OpType());
  record["mapped_tflite_op"] = mapped_tflite_op.has_value() ? *mapped_tflite_op : "UNMAPPED";
  record["supported"] = mapped_tflite_op.has_value();
  record["reason"] = mapped_tflite_op.has_value() ? "" : "No per-op TFLite mapping rule is implemented yet.";
  record["quant_context"] = DetermineQuantContext(graph_viewer, node);

  const bool uses_nhwc = UsesNhwcInternalLayout(node.OpType());
  record["input_layout"] = uses_nhwc ? "ONNX_NCHW" : "ONNX_NATIVE";
  record["internal_layout"] = uses_nhwc ? "TFLITE_NHWC" : "ONNX_NATIVE";
  record["output_layout"] = uses_nhwc ? "TFLITE_NHWC" : "ONNX_NATIVE";

  json input_tensors = json::array();
  const auto input_defs = node.InputDefs();
  for (size_t i = 0; i < input_defs.size(); ++i) {
    const NodeArg* input_arg = input_defs[i];
    if (input_arg == nullptr || !input_arg->Exists()) {
      continue;
    }

    const auto* initializer = graph_viewer.GetConstantInitializer(input_arg->Name(), true);
    const bool is_conv_weight = (node.OpType() == "Conv" || node.OpType() == "QLinearConv") && i == 1;
    if (initializer != nullptr) {
      input_tensors.push_back(
          MakeTensorSummaryFromInitializer(*initializer, uses_nhwc && !is_conv_weight, is_conv_weight));
    } else {
      input_tensors.push_back(MakeTensorSummaryFromNodeArg(graph_viewer, *input_arg, uses_nhwc));
    }
  }
  record["input_tensors"] = std::move(input_tensors);

  json output_tensors = json::array();
  for (const NodeArg* output_arg : node.OutputDefs()) {
    if (output_arg == nullptr || !output_arg->Exists()) {
      continue;
    }
    output_tensors.push_back(MakeTensorSummaryFromNodeArg(graph_viewer, *output_arg, uses_nhwc));
  }
  record["output_tensors"] = std::move(output_tensors);

  record["attributes"] = MakeAttributesJson(node);
  PopulateInsertedOps(graph_viewer, node, graph_outputs, record);
  record["fused_ops"] = json::array();

  json notes = json::array();
  if (uses_nhwc) {
    notes.push_back("This operator is modeled with an internal NHWC/TFLite-style activation view.");
  }
  if (record["quant_context"] == "mixed") {
    notes.push_back("This operator sits in a mixed float/quantized region and should be reviewed by backend lowering.");
  } else if (record["quant_context"] == "qdq_wrapped") {
    notes.push_back("This operator is surrounded by QuantizeLinear/DequantizeLinear bridge nodes.");
  } else if (record["quant_context"] == "qlinear_native") {
    notes.push_back("This operator is already expressed in ONNX QLinear form.");
  }
  record["notes"] = std::move(notes);

  MaybeAddQuantParameters(graph_viewer, node, record);
  return record;
}

json MakeGraphTensorSummary(const GraphViewer& graph_viewer, const NodeArg& node_arg) {
  json result;
  result["tensor_name"] = node_arg.Name();
  result["tensor_type"] = NodeArgTypeToString(node_arg);
  result["tensor_shape"] = NodeArgShape(node_arg);
  result["amlogic_nhwc_shape"] = ToNhwcShape(NodeArgShape(node_arg));
  result["is_constant"] = graph_viewer.IsConstantInitializer(node_arg.Name(), true);
  return result;
}

}  // namespace

Status DumpGraphInfoPerOpAsJson(const GraphViewer& graph_viewer,
                                const std::string& output_path,
                                const logging::Logger& logger) {
  std::unordered_set<std::string> graph_outputs;
  for (const NodeArg* output : graph_viewer.GetOutputs()) {
    if (output != nullptr && output->Exists()) {
      graph_outputs.insert(output->Name());
    }
  }

  json ops = json::array();
  for (NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const Node* node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }
    if (IsOrtInsertedQdqHelperNode(*node)) {
      continue;
    }
    ops.push_back(BuildNodeRecord(graph_viewer, *node, graph_outputs));
  }

  json graph_inputs = json::array();
  for (const NodeArg* input : graph_viewer.GetInputs()) {
    if (input != nullptr && input->Exists()) {
      graph_inputs.push_back(MakeGraphTensorSummary(graph_viewer, *input));
    }
  }

  json graph_outputs_json = json::array();
  for (const NodeArg* output : graph_viewer.GetOutputs()) {
    if (output != nullptr && output->Exists()) {
      graph_outputs_json.push_back(MakeGraphTensorSummary(graph_viewer, *output));
    }
  }

  json root{
      {"format", "amlogic_onnx_per_op_tflite_mapping"},
      {"version", 1},
      {"export_mode", "per_op"},
      {"model_name", graph_viewer.Name()},
      {"model_path", graph_viewer.ModelPath().string()},
      {"source_layout", "ONNX_NCHW"},
      {"target_layout", "TFLITE_NHWC"},
      {"notes", json::array(
                    {"Each entry in ops corresponds to one original ONNX node.",
                     "Inserted layout bridges and auxiliary pad/requant hints are attached to the source node metadata instead of replacing the source node.",
                     "supported=false means no explicit per-op TFLite mapping rule exists yet; the node is still emitted for full coverage."})},
      {"graph_inputs", std::move(graph_inputs)},
      {"graph_outputs", std::move(graph_outputs_json)},
      {"ops", std::move(ops)},
  };

  std::ofstream output(output_path, std::ios::out | std::ios::trunc);
  ORT_RETURN_IF_NOT(output.good(), "Failed to open Amlogic per-op mapping dump file: ", output_path);
  output << root.dump(2) << '\n';
  output.close();

  LOGS(logger, WARNING) << "Amlogic dumped per-op ONNX->TFLite mapping info to "
                        << output_path << ", ops: " << root["ops"].size();
  return Status::OK();
}

}  // namespace amlogic
}  // namespace onnxruntime
