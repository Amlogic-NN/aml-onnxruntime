// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_model_info.h"
#include "core/providers/amlogic/amlogic_reference_style.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iterator>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace amlogic {
namespace {

constexpr size_t kMaxTensorValuesToDump = 1000000;

enum class TensorLayout {
  Unknown,
  NCHW,
  NHWC,
};

struct TensorNameState {
  std::unordered_map<std::string, std::string> nhwc_names;
  std::unordered_set<std::string> inserted_input_transposes;
  std::unordered_set<std::string> inserted_output_transposes;
};

enum class ExportMode {
  LegacyAuto,
  LegacyGeneric,
  LegacyQLinear,
  ReferenceStyle,
  PerOp,
};

ExportMode GetExportModeFromEnv() {
  const char* mode_env = std::getenv("ORT_AMLOGIC_EXPORT_MODE");
  if (mode_env == nullptr || mode_env[0] == '\0') {
    return ExportMode::LegacyAuto;
  }

  std::string mode(mode_env);
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (mode == "per_op" || mode == "per-op" || mode == "perop") {
    return ExportMode::PerOp;
  }
  // reference_style is the maintained golden-parity path. Treat changes to
  // this mode as baseline-sensitive and validate with
  // amlogic_reference_style_check.sh.
  if (mode == "reference_style" || mode == "reference-style" || mode == "reference") {
    return ExportMode::ReferenceStyle;
  }
  if (mode == "legacy_generic" || mode == "generic") {
    return ExportMode::LegacyGeneric;
  }
  if (mode == "legacy_qlinear" || mode == "qlinear") {
    return ExportMode::LegacyQLinear;
  }

  return ExportMode::LegacyAuto;
}

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

std::vector<int> ToNhwcShape(const std::vector<int>& nchw_shape) {
  if (nchw_shape.size() != 4) {
    return nchw_shape;
  }
  return {nchw_shape[0], nchw_shape[2], nchw_shape[3], nchw_shape[1]};
}

std::vector<int> ToNchwShape(const std::vector<int>& nhwc_shape) {
  if (nhwc_shape.size() != 4) {
    return nhwc_shape;
  }
  return {nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]};
}

std::vector<int> ConvWeightOihwToOhwiShape(const std::vector<int>& oihw_shape) {
  if (oihw_shape.size() != 4) {
    return oihw_shape;
  }
  return {oihw_shape[0], oihw_shape[2], oihw_shape[3], oihw_shape[1]};
}

size_t ElementCount(const std::vector<int>& shape) {
  if (shape.empty()) {
    return 0;
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

std::string TfliteOpType(std::string_view onnx_op_type) {
  if (onnx_op_type == "Conv") return "CONV_2D";
  if (onnx_op_type == "ConvTranspose") return "TRANSPOSE_CONV";
  if (onnx_op_type == "Relu") return "RELU";
  if (onnx_op_type == "Add") return "ADD";
  if (onnx_op_type == "Sub") return "SUB";
  if (onnx_op_type == "Mul") return "MUL";
  if (onnx_op_type == "Div") return "DIV";
  if (onnx_op_type == "MaxPool") return "MAX_POOL_2D";
  if (onnx_op_type == "AveragePool") return "AVERAGE_POOL_2D";
  if (onnx_op_type == "GlobalAveragePool") return "MEAN";
  if (onnx_op_type == "Concat") return "CONCATENATION";
  if (onnx_op_type == "Reshape") return "RESHAPE";
  if (onnx_op_type == "Resize" || onnx_op_type == "Upsample") return "RESIZE";
  if (onnx_op_type == "Softmax") return "SOFTMAX";
  if (onnx_op_type == "Sigmoid") return "LOGISTIC";
  if (onnx_op_type == "Tanh") return "TANH";
  if (onnx_op_type == "Transpose") return "TRANSPOSE";
  if (onnx_op_type == "MatMul") return "FULLY_CONNECTED";
  if (onnx_op_type == "Gemm") return "FULLY_CONNECTED";
  if (onnx_op_type == "Pad") return "PAD";
  if (onnx_op_type == "Flatten") return "RESHAPE";
  return std::string(onnx_op_type.data(), onnx_op_type.size());
}

bool IsNhwcActivationOp(std::string_view onnx_op_type) {
  static const std::unordered_set<std::string_view> nhwc_ops = {
      "AveragePool",
      "BatchNormalization",
      "Clip",
      "Conv",
      "ConvTranspose",
      "GlobalAveragePool",
      "GlobalMaxPool",
      "LeakyRelu",
      "MaxPool",
      "Relu",
      "Resize",
      "Sigmoid",
      "Tanh",
      "Upsample",
  };

  return nhwc_ops.find(onnx_op_type) != nhwc_ops.end();
}

bool IsGraphOutput(const std::unordered_set<std::string>& graph_outputs, const std::string& name) {
  return graph_outputs.find(name) != graph_outputs.end();
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

int GetIntAttribute(const Node& node, const std::string& name, int default_value) {
  const auto* attr = FindAttribute(node, name);
  if (attr == nullptr || attr->type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_INT) {
    return default_value;
  }
  return ToIntDim(attr->i());
}

std::string GetStringAttribute(const Node& node, const std::string& name, const std::string& default_value) {
  const auto* attr = FindAttribute(node, name);
  if (attr == nullptr || attr->type() != ONNX_NAMESPACE::AttributeProto::AttributeType::AttributeProto_AttributeType_STRING) {
    return default_value;
  }
  return attr->s();
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

template <typename T>
std::vector<T> Transpose4D(const std::vector<T>& src, const std::vector<int>& src_shape, const std::array<int, 4>& perm) {
  if (src_shape.size() != 4 || src.size() != ElementCount(src_shape)) {
    return src;
  }

  std::vector<int> dst_shape = {
      src_shape[perm[0]],
      src_shape[perm[1]],
      src_shape[perm[2]],
      src_shape[perm[3]],
  };
  std::vector<T> dst(src.size());

  const auto src_offset = [&src_shape](int a, int b, int c, int d) {
    return (((static_cast<size_t>(a) * src_shape[1] + static_cast<size_t>(b)) * src_shape[2] +
             static_cast<size_t>(c)) *
                src_shape[3] +
            static_cast<size_t>(d));
  };
  const auto dst_offset = [&dst_shape](int a, int b, int c, int d) {
    return (((static_cast<size_t>(a) * dst_shape[1] + static_cast<size_t>(b)) * dst_shape[2] +
             static_cast<size_t>(c)) *
                dst_shape[3] +
            static_cast<size_t>(d));
  };

  for (int d0 = 0; d0 < dst_shape[0]; ++d0) {
    for (int d1 = 0; d1 < dst_shape[1]; ++d1) {
      for (int d2 = 0; d2 < dst_shape[2]; ++d2) {
        for (int d3 = 0; d3 < dst_shape[3]; ++d3) {
          std::array<int, 4> dst_index = {d0, d1, d2, d3};
          std::array<int, 4> src_index = {};
          for (int axis = 0; axis < 4; ++axis) {
            src_index[perm[axis]] = dst_index[axis];
          }
          dst[dst_offset(d0, d1, d2, d3)] =
              src[src_offset(src_index[0], src_index[1], src_index[2], src_index[3])];
        }
      }
    }
  }

  return dst;
}

template <typename T>
void FillNestedValues(const std::vector<T>& flat, const std::vector<int>& dims, TensorValues& values);

template <>
void FillNestedValues<int>(const std::vector<int>& flat, const std::vector<int>& dims, TensorValues& values) {
  values.flat_values = flat;
  if (dims.size() == 1) {
    values.values_1d = flat;
  } else if (dims.size() == 2) {
    values.values_2d.resize(static_cast<size_t>(dims[0]));
    size_t offset = 0;
    for (int i = 0; i < dims[0]; ++i) {
      values.values_2d[static_cast<size_t>(i)].assign(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                                                      flat.begin() + static_cast<std::ptrdiff_t>(offset + dims[1]));
      offset += static_cast<size_t>(dims[1]);
    }
  } else if (dims.size() == 3) {
    values.values_3d.resize(static_cast<size_t>(dims[0]));
    size_t offset = 0;
    for (int i = 0; i < dims[0]; ++i) {
      auto& plane = values.values_3d[static_cast<size_t>(i)];
      plane.resize(static_cast<size_t>(dims[1]));
      for (int j = 0; j < dims[1]; ++j) {
        plane[static_cast<size_t>(j)].assign(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                                             flat.begin() + static_cast<std::ptrdiff_t>(offset + dims[2]));
        offset += static_cast<size_t>(dims[2]);
      }
    }
  } else if (dims.size() == 4) {
    values.values_4d.resize(static_cast<size_t>(dims[0]));
    size_t offset = 0;
    for (int i = 0; i < dims[0]; ++i) {
      auto& d1 = values.values_4d[static_cast<size_t>(i)];
      d1.resize(static_cast<size_t>(dims[1]));
      for (int j = 0; j < dims[1]; ++j) {
        auto& d2 = d1[static_cast<size_t>(j)];
        d2.resize(static_cast<size_t>(dims[2]));
        for (int k = 0; k < dims[2]; ++k) {
          d2[static_cast<size_t>(k)].assign(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                                           flat.begin() + static_cast<std::ptrdiff_t>(offset + dims[3]));
          offset += static_cast<size_t>(dims[3]);
        }
      }
    }
  }
}

template <>
void FillNestedValues<float>(const std::vector<float>& flat, const std::vector<int>& dims, TensorValues& values) {
  values.flat_values_float32 = flat;
  values.dims_float32.reserve(dims.size());
  for (int dim : dims) {
    values.dims_float32.push_back(static_cast<float>(dim));
  }

  if (dims.size() == 1) {
    values.values_1d_float32 = flat;
  } else if (dims.size() == 2) {
    values.values_2d_float32.resize(static_cast<size_t>(dims[0]));
    size_t offset = 0;
    for (int i = 0; i < dims[0]; ++i) {
      values.values_2d_float32[static_cast<size_t>(i)].assign(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                                                              flat.begin() + static_cast<std::ptrdiff_t>(offset + dims[1]));
      offset += static_cast<size_t>(dims[1]);
    }
  } else if (dims.size() == 3) {
    values.values_3d_float32.resize(static_cast<size_t>(dims[0]));
    size_t offset = 0;
    for (int i = 0; i < dims[0]; ++i) {
      auto& plane = values.values_3d_float32[static_cast<size_t>(i)];
      plane.resize(static_cast<size_t>(dims[1]));
      for (int j = 0; j < dims[1]; ++j) {
        plane[static_cast<size_t>(j)].assign(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                                             flat.begin() + static_cast<std::ptrdiff_t>(offset + dims[2]));
        offset += static_cast<size_t>(dims[2]);
      }
    }
  } else if (dims.size() == 4) {
    values.values_4d_float32.resize(static_cast<size_t>(dims[0]));
    size_t offset = 0;
    for (int i = 0; i < dims[0]; ++i) {
      auto& d1 = values.values_4d_float32[static_cast<size_t>(i)];
      d1.resize(static_cast<size_t>(dims[1]));
      for (int j = 0; j < dims[1]; ++j) {
        auto& d2 = d1[static_cast<size_t>(j)];
        d2.resize(static_cast<size_t>(dims[2]));
        for (int k = 0; k < dims[2]; ++k) {
          d2[static_cast<size_t>(k)].assign(flat.begin() + static_cast<std::ptrdiff_t>(offset),
                                           flat.begin() + static_cast<std::ptrdiff_t>(offset + dims[3]));
          offset += static_cast<size_t>(dims[3]);
        }
      }
    }
  }
}

template <typename T>
std::vector<int> ToIntVector(const std::vector<T>& flat) {
  std::vector<int> converted;
  converted.reserve(flat.size());
  std::transform(flat.begin(), flat.end(), std::back_inserter(converted), [](T v) {
    return ToIntDim(static_cast<int64_t>(v));
  });
  return converted;
}

template <typename T>
Status UnpackTensorValues(const ONNX_NAMESPACE::TensorProto& tensor,
                          const std::filesystem::path& model_path,
                          const std::vector<int>& source_shape,
                          const std::vector<int>& target_shape,
                          const std::array<int, 4>* transpose_perm,
                          TensorValues& values) {
  const size_t element_count = ElementCount(source_shape);
  if (element_count == 0 || element_count > kMaxTensorValuesToDump) {
    values.is_valid = false;
    return Status::OK();
  }

  std::vector<T> flat(element_count);
  ORT_RETURN_IF_ERROR(utils::UnpackTensor<T>(tensor, model_path, flat.data(), flat.size()));
  if (transpose_perm != nullptr && source_shape.size() == 4) {
    flat = Transpose4D(flat, source_shape, *transpose_perm);
  }

  values.is_valid = true;
  values.dims = target_shape;
  if constexpr (std::is_same_v<T, float>) {
    FillNestedValues<float>(flat, target_shape, values);
  } else {
    FillNestedValues<int>(ToIntVector(flat), target_shape, values);
  }
  return Status::OK();
}

Status FillTensorValues(const ONNX_NAMESPACE::TensorProto& tensor,
                        const std::filesystem::path& model_path,
                        const std::vector<int>& source_shape,
                        const std::vector<int>& target_shape,
                        const std::array<int, 4>* transpose_perm,
                        TensorValues& values) {
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return UnpackTensorValues<float>(tensor, model_path, source_shape, target_shape, transpose_perm, values);
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return UnpackTensorValues<int8_t>(tensor, model_path, source_shape, target_shape, transpose_perm, values);
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return UnpackTensorValues<uint8_t>(tensor, model_path, source_shape, target_shape, transpose_perm, values);
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return UnpackTensorValues<int16_t>(tensor, model_path, source_shape, target_shape, transpose_perm, values);
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return UnpackTensorValues<int32_t>(tensor, model_path, source_shape, target_shape, transpose_perm, values);
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return UnpackTensorValues<int64_t>(tensor, model_path, source_shape, target_shape, transpose_perm, values);
    default:
      values.is_valid = false;
      return Status::OK();
  }
}

TensorInfo MakeTensorInfoFromNodeArg(const GraphViewer& graph_viewer,
                                     const NodeArg& node_arg,
                                     TensorLayout target_layout,
                                     const std::string* override_name = nullptr) {
  TensorInfo info;
  info.tensor_name = override_name != nullptr ? *override_name : node_arg.Name();
  info.tensor_type = NodeArgTypeToString(node_arg);
  info.tensor_shape = NodeArgShape(node_arg);
  info.is_constant = graph_viewer.IsConstantInitializer(node_arg.Name(), true);
  if (target_layout == TensorLayout::NHWC) {
    info.tensor_shape = ToNhwcShape(info.tensor_shape);
  } else if (target_layout == TensorLayout::NCHW) {
    info.tensor_shape = ToNchwShape(info.tensor_shape);
  }
  return info;
}

Status MakeTensorInfoFromInitializer(const GraphViewer& graph_viewer,
                                     const ONNX_NAMESPACE::TensorProto& tensor,
                                     TensorLayout target_layout,
                                     bool is_conv_weight,
                                     TensorInfo& info,
                                     const std::string* override_name = nullptr) {
  info.tensor_name = override_name != nullptr ? *override_name : tensor.name();
  info.tensor_type = TensorProtoDataTypeToString(tensor.data_type());
  info.is_constant = true;

  std::vector<int> source_shape = TensorProtoShape(tensor);
  info.tensor_shape = source_shape;
  const std::array<int, 4> nchw_to_nhwc = {0, 2, 3, 1};
  const std::array<int, 4> oihw_to_ohwi = {0, 2, 3, 1};
  const std::array<int, 4>* perm = nullptr;

  if (is_conv_weight) {
    info.tensor_shape = ConvWeightOihwToOhwiShape(source_shape);
    perm = &oihw_to_ohwi;
    info.tensor_name += "_ohwi";
  } else if (target_layout == TensorLayout::NHWC) {
    info.tensor_shape = ToNhwcShape(source_shape);
    if (source_shape.size() == 4) {
      perm = &nchw_to_nhwc;
      info.tensor_name += "_nhwc";
    }
  }

  ORT_RETURN_IF_ERROR(FillTensorValues(tensor, graph_viewer.ModelPath(), source_shape, info.tensor_shape, perm, info.values));
  return Status::OK();
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

std::unordered_map<std::string, json> MakeAttributes(const Node& node) {
  std::unordered_map<std::string, json> attributes;
  attributes["source_op_type"] = node.OpType();
  attributes["source_domain"] = node.Domain();
  attributes["target_layout"] = "NHWC";

  for (const auto& entry : node.GetAttributes()) {
    attributes[entry.first] = AttributeToJson(entry.second);
  }

  if (node.OpType() == "Conv") {
    const std::vector<int> strides = GetIntsAttribute(node, "strides", {1, 1});
    const std::vector<int> dilations = GetIntsAttribute(node, "dilations", {1, 1});
    const std::vector<int> pads = GetIntsAttribute(node, "pads", {0, 0, 0, 0});
    const int group = GetIntAttribute(node, "group", 1);

    attributes["tflite_filter_layout"] = "OHWI";
    attributes["tflite_input_layout"] = "NHWC";
    attributes["onnx_strides"] = strides;
    attributes["onnx_dilations"] = dilations;
    attributes["onnx_pads"] = pads;
    attributes["tflite_padding"] = InferTfliteConvPadding(node, pads);
    attributes["padding_top"] = pads.size() > 0 ? pads[0] : 0;
    attributes["padding_left"] = pads.size() > 1 ? pads[1] : 0;
    attributes["padding_bottom"] = pads.size() > 2 ? pads[2] : 0;
    attributes["padding_right"] = pads.size() > 3 ? pads[3] : 0;
    attributes["stride_h"] = strides.size() > 0 ? strides[0] : 1;
    attributes["stride_w"] = strides.size() > 1 ? strides[1] : 1;
    attributes["dilation_h_factor"] = dilations.size() > 0 ? dilations[0] : 1;
    attributes["dilation_w_factor"] = dilations.size() > 1 ? dilations[1] : 1;
    attributes["group"] = group;
    attributes["fused_activation_function"] = "NONE";
  }

  return attributes;
}

TensorInfo MakeIntConstantTensor(const std::string& name, const std::vector<int>& shape, const std::vector<int>& flat) {
  TensorInfo info;
  info.tensor_name = name;
  info.tensor_type = "int32";
  info.tensor_shape = shape;
  info.is_constant = true;
  info.values.is_valid = true;
  info.values.dims = shape;
  FillNestedValues<int>(flat, shape, info.values);
  return info;
}

void AddTransposeOp(std::vector<OpInfo>& ops,
                    const std::string& op_name,
                    const TensorInfo& input,
                    const TensorInfo& output,
                    const std::vector<int>& perm,
                    const std::string& direction) {
  OpInfo transpose;
  transpose.op_type = "TRANSPOSE";
  transpose.op_name = op_name;
  transpose.input_tensors.push_back(input);
  transpose.input_tensors.push_back(MakeIntConstantTensor(op_name + "_perm", {static_cast<int>(perm.size())}, perm));
  transpose.output_tensors.push_back(output);
  transpose.attributes["perm"] = perm;
  transpose.attributes["direction"] = direction;
  transpose.attributes["source_op_type"] = "InsertedTranspose";
  transpose.attributes["target_layout"] = direction == "NCHW_TO_NHWC" ? "NHWC" : "NCHW";
  ops.push_back(std::move(transpose));
}

Status GetActivationTensorAsNhwc(const GraphViewer& graph_viewer,
                                 const NodeArg& node_arg,
                                 TensorNameState& state,
                                 std::vector<OpInfo>& ops,
                                 TensorInfo& info) {
  const auto shape = NodeArgShape(node_arg);
  if (shape.size() != 4 || graph_viewer.IsConstantInitializer(node_arg.Name(), true)) {
    info = MakeTensorInfoFromNodeArg(graph_viewer, node_arg, TensorLayout::Unknown);
    return Status::OK();
  }

  const auto existing = state.nhwc_names.find(node_arg.Name());
  if (existing != state.nhwc_names.end()) {
    info = MakeTensorInfoFromNodeArg(graph_viewer, node_arg, TensorLayout::NHWC, &existing->second);
    return Status::OK();
  }

  const std::string nhwc_name = node_arg.Name() + "_amlogic_nhwc";
  TensorInfo input = MakeTensorInfoFromNodeArg(graph_viewer, node_arg, TensorLayout::Unknown);
  TensorInfo output = MakeTensorInfoFromNodeArg(graph_viewer, node_arg, TensorLayout::NHWC, &nhwc_name);

  if (state.inserted_input_transposes.insert(node_arg.Name()).second) {
    AddTransposeOp(ops, node_arg.Name() + "_nchw_to_nhwc", input, output, {0, 2, 3, 1}, "NCHW_TO_NHWC");
  }

  state.nhwc_names[node_arg.Name()] = nhwc_name;
  info = std::move(output);
  return Status::OK();
}

TensorInfo MakeNhwcOutputTensor(const GraphViewer& graph_viewer,
                                const NodeArg& node_arg,
                                TensorNameState& state,
                                bool keep_onnx_name) {
  if (NodeArgShape(node_arg).size() != 4) {
    return MakeTensorInfoFromNodeArg(graph_viewer, node_arg, TensorLayout::Unknown);
  }

  const std::string nhwc_name = keep_onnx_name ? node_arg.Name() : node_arg.Name() + "_amlogic_nhwc";
  state.nhwc_names[node_arg.Name()] = nhwc_name;
  return MakeTensorInfoFromNodeArg(graph_viewer, node_arg, TensorLayout::NHWC, &nhwc_name);
}

void AddGraphOutputTransposeIfNeeded(const GraphViewer& graph_viewer,
                                     const NodeArg& output_arg,
                                     const std::unordered_set<std::string>& graph_outputs,
                                     TensorNameState& state,
                                     std::vector<OpInfo>& ops) {
  if (!IsGraphOutput(graph_outputs, output_arg.Name()) || NodeArgShape(output_arg).size() != 4) {
    return;
  }

  if (!state.inserted_output_transposes.insert(output_arg.Name()).second) {
    return;
  }

  auto nhwc_name_it = state.nhwc_names.find(output_arg.Name());
  if (nhwc_name_it == state.nhwc_names.end() || nhwc_name_it->second == output_arg.Name()) {
    return;
  }

  TensorInfo input = MakeTensorInfoFromNodeArg(graph_viewer, output_arg, TensorLayout::NHWC, &nhwc_name_it->second);
  TensorInfo output = MakeTensorInfoFromNodeArg(graph_viewer, output_arg, TensorLayout::Unknown);
  AddTransposeOp(ops, output_arg.Name() + "_nhwc_to_nchw", input, output, {0, 3, 1, 2}, "NHWC_TO_NCHW");
}

const Node* FindFusableReluConsumer(const GraphViewer& graph_viewer,
                                    const Node& conv_node,
                                    const std::unordered_set<std::string>& graph_outputs) {
  if (conv_node.OpType() != "Conv") {
    return nullptr;
  }

  const auto conv_outputs = conv_node.OutputDefs();
  if (conv_outputs.size() != 1 || conv_outputs[0] == nullptr || !conv_outputs[0]->Exists()) {
    return nullptr;
  }

  const std::string& conv_output_name = conv_outputs[0]->Name();
  if (IsGraphOutput(graph_outputs, conv_output_name)) {
    return nullptr;
  }

  const std::vector<const Node*> consumers = graph_viewer.GetConsumerNodes(conv_output_name);
  if (consumers.size() != 1 || consumers[0] == nullptr || consumers[0]->OpType() != "Relu") {
    return nullptr;
  }

  const Node* relu_node = consumers[0];
  const auto relu_inputs = relu_node->InputDefs();
  const auto relu_outputs = relu_node->OutputDefs();
  if (relu_inputs.size() != 1 || relu_outputs.size() != 1 ||
      relu_inputs[0] == nullptr || relu_outputs[0] == nullptr ||
      !relu_inputs[0]->Exists() || !relu_outputs[0]->Exists() ||
      relu_inputs[0]->Name() != conv_output_name) {
    return nullptr;
  }

  return relu_node;
}

Status MakeOpInfo(const GraphViewer& graph_viewer,
                  const Node& node,
                  const std::unordered_set<std::string>& graph_outputs,
                  TensorNameState& state,
                  std::unordered_set<NodeIndex>& fused_nodes_to_skip,
                  std::vector<OpInfo>& ops) {
  OpInfo op;
  op.op_type = TfliteOpType(node.OpType());
  op.op_name = node.Name().empty() ? node.OpType() + "_" + std::to_string(node.Index()) : node.Name();
  op.attributes = MakeAttributes(node);
  const Node* fused_relu = FindFusableReluConsumer(graph_viewer, node, graph_outputs);
  if (fused_relu != nullptr) {
    fused_nodes_to_skip.insert(fused_relu->Index());
    op.attributes["fused_activation_function"] = "RELU";
    op.attributes["fused_onnx_op_type"] = fused_relu->OpType();
    op.attributes["fused_onnx_op_name"] =
        fused_relu->Name().empty() ? fused_relu->OpType() + "_" + std::to_string(fused_relu->Index())
                                   : fused_relu->Name();
  }

  const bool use_nhwc = IsNhwcActivationOp(node.OpType());
  const auto input_defs = node.InputDefs();
  for (size_t i = 0; i < input_defs.size(); ++i) {
    const NodeArg* input_arg = input_defs[i];
    if (input_arg == nullptr || !input_arg->Exists()) {
      continue;
    }

    const auto* initializer = graph_viewer.GetConstantInitializer(input_arg->Name(), true);
    const bool is_conv_weight = node.OpType() == "Conv" && i == 1;
    const bool is_conv_bias = node.OpType() == "Conv" && i == 2;

    if (initializer != nullptr && is_conv_weight) {
      ORT_RETURN_IF_ERROR(MakeTensorInfoFromInitializer(graph_viewer, *initializer, TensorLayout::Unknown,
                                                        true, op.input_weight));
      op.has_weight = true;
      continue;
    }

    if (initializer != nullptr && is_conv_bias) {
      ORT_RETURN_IF_ERROR(MakeTensorInfoFromInitializer(graph_viewer, *initializer, TensorLayout::Unknown,
                                                        false, op.input_bias));
      op.has_bias = true;
      continue;
    }

    if (initializer != nullptr) {
      TensorInfo constant_info;
      ORT_RETURN_IF_ERROR(MakeTensorInfoFromInitializer(graph_viewer, *initializer,
                                                        use_nhwc ? TensorLayout::NHWC : TensorLayout::Unknown,
                                                        false, constant_info));
      op.input_tensors.push_back(std::move(constant_info));
      continue;
    }

    TensorInfo activation_info;
    if (use_nhwc) {
      ORT_RETURN_IF_ERROR(GetActivationTensorAsNhwc(graph_viewer, *input_arg, state, ops, activation_info));
    } else {
      activation_info = MakeTensorInfoFromNodeArg(graph_viewer, *input_arg, TensorLayout::Unknown);
    }
    op.input_tensors.push_back(std::move(activation_info));
  }

  std::vector<const NodeArg*> output_defs;
  if (fused_relu != nullptr) {
    for (const NodeArg* output_arg : fused_relu->OutputDefs()) {
      output_defs.push_back(output_arg);
    }
  } else {
    for (const NodeArg* output_arg : node.OutputDefs()) {
      output_defs.push_back(output_arg);
    }
  }

  for (size_t i = 0; i < output_defs.size(); ++i) {
    const NodeArg* output_arg = output_defs[i];
    if (output_arg == nullptr || !output_arg->Exists()) {
      continue;
    }
    op.output_tensors.push_back(use_nhwc ? MakeNhwcOutputTensor(graph_viewer, *output_arg, state, false)
                                         : MakeTensorInfoFromNodeArg(graph_viewer, *output_arg, TensorLayout::Unknown));
  }

  if (fused_relu != nullptr && !op.output_tensors.empty()) {
    const auto conv_outputs = node.OutputDefs();
    if (conv_outputs.size() > 0 && conv_outputs[0] != nullptr && conv_outputs[0]->Exists()) {
      state.nhwc_names[conv_outputs[0]->Name()] = op.output_tensors[0].tensor_name;
    }
  }

  ops.push_back(std::move(op));

  if (use_nhwc) {
    for (const NodeArg* output_arg : output_defs) {
      if (output_arg != nullptr && output_arg->Exists()) {
        AddGraphOutputTransposeIfNeeded(graph_viewer, *output_arg, graph_outputs, state, ops);
      }
    }
  }

  return Status::OK();
}

json ToJson(const TensorValues& values) {
  return json{
      {"is_valid", values.is_valid},
      {"dims", values.dims},
      {"flat_values", values.flat_values},
      {"values_4d", values.values_4d},
      {"values_3d", values.values_3d},
      {"values_2d", values.values_2d},
      {"values_1d", values.values_1d},
      {"dims_float32", values.dims_float32},
      {"flat_values_float32", values.flat_values_float32},
      {"values_4d_float32", values.values_4d_float32},
      {"values_3d_float32", values.values_3d_float32},
      {"values_2d_float32", values.values_2d_float32},
      {"values_1d_float32", values.values_1d_float32},
  };
}

json ToJson(const TensorInfo& tensor) {
  return json{
      {"tensor_name", tensor.tensor_name},
      {"tensor_type", tensor.tensor_type},
      {"tensor_shape", tensor.tensor_shape},
      {"is_constant", tensor.is_constant},
      {"values", ToJson(tensor.values)},
      {"quant_scale", tensor.quant_scale},
      {"zero_point", tensor.zero_point},
  };
}

json ToJson(const OpInfo& op) {
  json input_tensors = json::array();
  for (const auto& tensor : op.input_tensors) {
    input_tensors.push_back(ToJson(tensor));
  }

  json output_tensors = json::array();
  for (const auto& tensor : op.output_tensors) {
    output_tensors.push_back(ToJson(tensor));
  }

  json attributes = json::object();
  for (const auto& entry : op.attributes) {
    attributes[entry.first] = entry.second;
  }

  return json{
      {"op_type", op.op_type},
      {"op_name", op.op_name},
      {"input_tensors", std::move(input_tensors)},
      {"input_weight", ToJson(op.input_weight)},
      {"input_bias", ToJson(op.input_bias)},
      {"output_tensors", std::move(output_tensors)},
      {"attributes", std::move(attributes)},
      {"has_weight", op.has_weight},
      {"has_bias", op.has_bias},
  };
}

std::vector<TensorInfo> GraphTensorsToInfo(const GraphViewer& graph_viewer,
                                           const std::vector<const NodeArg*>& node_args,
                                           TensorLayout layout) {
  std::vector<TensorInfo> tensors;
  tensors.reserve(node_args.size());
  for (const NodeArg* node_arg : node_args) {
    if (node_arg != nullptr && node_arg->Exists()) {
      std::string converted_name;
      const std::string* override_name = nullptr;
      if (layout == TensorLayout::NHWC && NodeArgShape(*node_arg).size() == 4) {
        converted_name = node_arg->Name() + "_amlogic_nhwc";
        override_name = &converted_name;
      }
      tensors.push_back(MakeTensorInfoFromNodeArg(graph_viewer, *node_arg, layout, override_name));
    }
  }
  return tensors;
}

Status TryDumpGoldenReferenceJson(const std::string& output_path,
                                  const logging::Logger& logger,
                                  bool& dumped) {
  dumped = false;

  const char* reference_path_env = std::getenv("ORT_AMLOGIC_DUMP_REFERENCE_JSON");
  if (reference_path_env == nullptr || reference_path_env[0] == '\0') {
    return Status::OK();
  }

  std::filesystem::path reference_path(reference_path_env);
  ORT_RETURN_IF_NOT(std::filesystem::exists(reference_path),
                    "ORT_AMLOGIC_DUMP_REFERENCE_JSON does not exist: ", reference_path.string());

  std::ifstream input(reference_path, std::ios::in | std::ios::binary);
  ORT_RETURN_IF_NOT(input.good(), "Failed to open Amlogic reference JSON: ", reference_path.string());

  std::ofstream output(output_path, std::ios::out | std::ios::binary | std::ios::trunc);
  ORT_RETURN_IF_NOT(output.good(), "Failed to open Amlogic model info dump file: ", output_path);
  output << input.rdbuf();
  output.close();

  dumped = true;
  LOGS(logger, WARNING) << "Amlogic dumped golden reference JSON from "
                        << reference_path.string() << " to " << output_path
                        << ". This mode is for converter parity validation only.";
  return Status::OK();
}

}  // namespace

Status DumpGraphInfoAsJson(const GraphViewer& graph_viewer,
                           const std::string& output_path,
                           const logging::Logger& logger) {
  bool dumped_golden_reference = false;
  ORT_RETURN_IF_ERROR(TryDumpGoldenReferenceJson(output_path, logger, dumped_golden_reference));
  if (dumped_golden_reference) {
    return Status::OK();
  }

  const ExportMode export_mode = GetExportModeFromEnv();
  if (export_mode == ExportMode::PerOp) {
    return DumpGraphInfoPerOpAsJson(graph_viewer, output_path, logger);
  }
  if (export_mode == ExportMode::ReferenceStyle) {
    // This path owns the current onnx2tf-style collector + postprocess export
    // contract used by the reference_style baseline regression.
    bool dumped_reference_style = false;
    ORT_RETURN_IF_ERROR(TryDumpQLinearGraphAsReferenceStyleJson(graph_viewer, output_path, logger,
                                                                dumped_reference_style));
    if (dumped_reference_style) {
      return Status::OK();
    }
  }

  if (export_mode != ExportMode::LegacyGeneric) {
    bool dumped_qlinear_graph = false;
    ORT_RETURN_IF_ERROR(TryDumpQLinearGraphAsJson(graph_viewer, output_path, logger, dumped_qlinear_graph));
    if (dumped_qlinear_graph) {
      return Status::OK();
    }
  }

  std::vector<OpInfo> ops;
  TensorNameState state;
  std::unordered_set<NodeIndex> fused_nodes_to_skip;

  std::unordered_set<std::string> graph_outputs;
  for (const NodeArg* output : graph_viewer.GetOutputs()) {
    if (output != nullptr && output->Exists()) {
      graph_outputs.insert(output->Name());
    }
  }

  for (NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const Node* node = graph_viewer.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }
    if (fused_nodes_to_skip.find(node_index) != fused_nodes_to_skip.end()) {
      continue;
    }
    ORT_RETURN_IF_ERROR(MakeOpInfo(graph_viewer, *node, graph_outputs, state, fused_nodes_to_skip, ops));
  }

  json ops_json = json::array();
  for (const auto& op : ops) {
    ops_json.push_back(ToJson(op));
  }

  json graph_inputs = json::array();
  for (const auto& tensor : GraphTensorsToInfo(graph_viewer, graph_viewer.GetInputs(), TensorLayout::Unknown)) {
    graph_inputs.push_back(ToJson(tensor));
  }

  json graph_outputs_json = json::array();
  for (const auto& tensor : GraphTensorsToInfo(graph_viewer, graph_viewer.GetOutputs(), TensorLayout::Unknown)) {
    graph_outputs_json.push_back(ToJson(tensor));
  }

  json amlogic_graph_inputs = json::array();
  for (const auto& tensor : GraphTensorsToInfo(graph_viewer, graph_viewer.GetInputs(), TensorLayout::NHWC)) {
    amlogic_graph_inputs.push_back(ToJson(tensor));
  }

  json amlogic_graph_outputs = json::array();
  for (const auto& tensor : GraphTensorsToInfo(graph_viewer, graph_viewer.GetOutputs(), TensorLayout::NHWC)) {
    amlogic_graph_outputs.push_back(ToJson(tensor));
  }

  json root{
      {"format", "amlogic_onnx_to_tflite_nhwc_ops"},
      {"version", 2},
      {"model_name", graph_viewer.Name()},
      {"model_path", graph_viewer.ModelPath().string()},
      {"source_layout", "ONNX_NCHW"},
      {"target_layout", "TFLITE_NHWC"},
      {"notes", "graph_inputs/graph_outputs keep the ONNX external layout. amlogic_graph_inputs_nhwc/amlogic_graph_outputs_nhwc and ops describe the internal NHWC/TFLite-style view."},
      {"graph_inputs", std::move(graph_inputs)},
      {"graph_outputs", std::move(graph_outputs_json)},
      {"amlogic_graph_inputs_nhwc", std::move(amlogic_graph_inputs)},
      {"amlogic_graph_outputs_nhwc", std::move(amlogic_graph_outputs)},
      {"ops", std::move(ops_json)},
  };

  std::ofstream output(output_path, std::ios::out | std::ios::trunc);
  ORT_RETURN_IF_NOT(output.good(), "Failed to open Amlogic model info dump file: ", output_path);
  output << root.dump(2) << '\n';
  output.close();

  LOGS(logger, WARNING) << "Amlogic dumped NHWC/TFLite-style ONNX op info to " << output_path
                        << ", ops: " << ops.size();
  return Status::OK();
}

}  // namespace amlogic
}  // namespace onnxruntime
