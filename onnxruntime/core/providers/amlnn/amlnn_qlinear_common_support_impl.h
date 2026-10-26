int ToIntDim(int64_t dim) {
  if (dim > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  if (dim < static_cast<int64_t>(std::numeric_limits<int>::min())) {
    return std::numeric_limits<int>::min();
  }
  return static_cast<int>(dim);
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

bool IsOrtInsertedQdqName(std::string_view name) {
  std::string_view onnx_name = name;
  if (onnx_name.rfind("wa/", 0) == 0) {
    onnx_name.remove_prefix(2);
  }
  return EndsWith(onnx_name, "/duplicated") ||
         EndsWith(onnx_name, "_pre_q") ||
         EndsWith(onnx_name, "_q_to_dq") ||
         EndsWith(onnx_name, "_post_dq");
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

int MapAxisToNhwc(int axis, size_t rank) {
  if (axis < 0) {
    axis += static_cast<int>(rank);
  }
  if (rank != 4) {
    return axis;
  }
  if (axis == 1) return 3;
  if (axis == 2) return 1;
  if (axis == 3) return 2;
  return axis;
}

int NormalizeSliceIndex(int value, int dim) {
  if (dim <= 0) {
    return value;
  }
  if (value < 0) {
    value += dim;
  }
  return std::max(0, std::min(value, dim));
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}
