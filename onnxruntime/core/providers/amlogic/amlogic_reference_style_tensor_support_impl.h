// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside the anonymous namespace. It keeps the low-level tensor/value unpack
// helpers shared by the legacy qlinear exporter and reference_style support
// logic in one place without changing behavior.

template <typename T>
std::vector<T> Transpose4D(const std::vector<T>& src, const std::vector<int>& src_shape,
                           const std::array<int, 4>& perm) {
  if (src_shape.size() != 4 || src.size() != ElementCount(src_shape)) {
    return src;
  }

  std::vector<int> dst_shape = {src_shape[perm[0]], src_shape[perm[1]], src_shape[perm[2]], src_shape[perm[3]]};
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
ordered_json NestedJsonImpl(const std::vector<T>& flat, const std::vector<int>& dims, size_t axis, size_t& offset) {
  ordered_json result = ordered_json::array();
  if (axis + 1 == dims.size()) {
    for (int i = 0; i < dims[axis]; ++i) {
      result.push_back(flat[offset++]);
    }
    return result;
  }

  for (int i = 0; i < dims[axis]; ++i) {
    result.push_back(NestedJsonImpl(flat, dims, axis + 1, offset));
  }
  return result;
}

template <typename T>
ordered_json ValuesToJson(const std::vector<T>& flat, const std::vector<int>& dims) {
  if (dims.empty() || dims.size() == 1) {
    ordered_json result = ordered_json::array();
    for (const auto& value : flat) {
      result.push_back(value);
    }
    return result;
  }

  if (flat.size() != ElementCount(dims)) {
    ordered_json result = ordered_json::array();
    for (const auto& value : flat) {
      result.push_back(value);
    }
    return result;
  }

  size_t offset = 0;
  return NestedJsonImpl(flat, dims, 0, offset);
}

template <typename T>
std::vector<int> ToIntVector(const std::vector<T>& values, bool uint8_to_int8 = false) {
  std::vector<int> result;
  result.reserve(values.size());
  for (T value : values) {
    int v = ToIntDim(static_cast<int64_t>(value));
    if (uint8_to_int8) {
      v -= 128;
    }
    result.push_back(v);
  }
  return result;
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

Status TensorValuesJson(const ONNX_NAMESPACE::TensorProto& tensor,
                        const std::filesystem::path& model_path,
                        const std::vector<int>& source_shape,
                        const std::vector<int>& target_shape,
                        const std::array<int, 4>* transpose_perm,
                        bool quantized_uint8_to_int8,
                        ordered_json& values) {
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      std::vector<float> flat;
      ORT_RETURN_IF_ERROR(UnpackVector<float>(tensor, model_path, flat));
      if (transpose_perm != nullptr) {
        flat = Transpose4D(flat, source_shape, *transpose_perm);
      }
      values = ValuesToJson(flat, target_shape);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      std::vector<int8_t> flat;
      ORT_RETURN_IF_ERROR(UnpackVector<int8_t>(tensor, model_path, flat));
      if (transpose_perm != nullptr) {
        flat = Transpose4D(flat, source_shape, *transpose_perm);
      }
      values = ValuesToJson(ToIntVector(flat), target_shape);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      std::vector<uint8_t> flat;
      ORT_RETURN_IF_ERROR(UnpackVector<uint8_t>(tensor, model_path, flat));
      if (transpose_perm != nullptr) {
        flat = Transpose4D(flat, source_shape, *transpose_perm);
      }
      values = ValuesToJson(ToIntVector(flat, quantized_uint8_to_int8), target_shape);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      std::vector<int32_t> flat;
      ORT_RETURN_IF_ERROR(UnpackVector<int32_t>(tensor, model_path, flat));
      values = ValuesToJson(ToIntVector(flat), target_shape);
      return Status::OK();
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      std::vector<int64_t> flat;
      ORT_RETURN_IF_ERROR(UnpackVector<int64_t>(tensor, model_path, flat));
      values = ValuesToJson(ToIntVector(flat), target_shape);
      return Status::OK();
    }
    default:
      values = ordered_json::array();
      return Status::OK();
  }
}

Status FloatInitializerValues(const GraphViewer& graph_viewer, const std::string& name, std::vector<float>& values) {
  const ONNX_NAMESPACE::TensorProto* tensor = graph_viewer.GetConstantInitializer(name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing float initializer: ", name);
  ORT_RETURN_IF_ERROR(UnpackVector<float>(*tensor, graph_viewer.ModelPath(), values));
  return Status::OK();
}

Status InitializerAsFloatJson(const GraphViewer& graph_viewer, const std::string& name, ordered_json& value) {
  std::vector<float> values;
  ORT_RETURN_IF_ERROR(FloatInitializerValues(graph_viewer, name, values));
  if (values.size() == 1) {
    value = values[0];
  } else {
    value = ordered_json::array();
    for (float v : values) {
      value.push_back(v);
    }
  }
  return Status::OK();
}

Status InitializerAsZeroPointJson(const GraphViewer& graph_viewer, const std::string& name, ordered_json& value) {
  const ONNX_NAMESPACE::TensorProto* tensor = graph_viewer.GetConstantInitializer(name, true);
  ORT_RETURN_IF_NOT(tensor != nullptr, "Missing zero point initializer: ", name);

  if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    std::vector<uint8_t> values;
    ORT_RETURN_IF_ERROR(UnpackVector<uint8_t>(*tensor, graph_viewer.ModelPath(), values));
    if (values.size() == 1) {
      value = static_cast<int>(values[0]) - 128;
    } else {
      value = ordered_json::array();
      for (uint8_t v : values) {
        value.push_back(static_cast<int>(v) - 128);
      }
    }
    return Status::OK();
  }

  if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    std::vector<int8_t> values;
    ORT_RETURN_IF_ERROR(UnpackVector<int8_t>(*tensor, graph_viewer.ModelPath(), values));
    if (values.size() == 1) {
      value = static_cast<int>(values[0]);
    } else {
      value = ordered_json::array();
      for (int8_t v : values) {
        value.push_back(static_cast<int>(v));
      }
    }
    return Status::OK();
  }

  if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    std::vector<int32_t> values;
    ORT_RETURN_IF_ERROR(UnpackVector<int32_t>(*tensor, graph_viewer.ModelPath(), values));
    if (values.size() == 1) {
      value = ToIntDim(values[0]);
    } else {
      value = ordered_json::array();
      for (int32_t v : values) {
        value.push_back(ToIntDim(v));
      }
    }
    return Status::OK();
  }

  value = 0;
  return Status::OK();
}
