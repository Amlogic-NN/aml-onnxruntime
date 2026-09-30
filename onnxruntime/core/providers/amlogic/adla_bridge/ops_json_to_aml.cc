#include "ops_json_to_aml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <iostream>

namespace amlogic::adla_bridge {
namespace {
using json = nlohmann::json;

const std::unordered_map<std::string, AML_OperationType> kOpMap = {
    {"Add", AML_OperationType::kAdd}, {"QLinearAdd", AML_OperationType::kAdd},
    {"Conv", AML_OperationType::kConv2D}, {"QLinearConv", AML_OperationType::kConv2D},
    {"DepthwiseConv", AML_OperationType::kDepthwiseConv2D},
    {"Relu", AML_OperationType::kRelu}, {"MaxPool", AML_OperationType::kMaxPool2D},
    // The reference-style exporter accepts only ONNX Clip(0, 6), which is
    // exactly TFLite RELU6.
    {"Clip", AML_OperationType::kRelu6},
    {"AveragePool", AML_OperationType::kAveragePool2D},
    // ONNX GlobalAveragePool is emitted by reference_style with a rank-1
    // constant axis input.  ADLA's Mean lowering expects all inputs to be 4D,
    // so lower it to a normal NHWC AveragePool2D instead.
    {"GlobalAveragePool", AML_OperationType::kAveragePool2D},
    {"Mul", AML_OperationType::kMul}, {"QLinearMul", AML_OperationType::kMul},
    {"Concat", AML_OperationType::kConcatenation}, {"QLinearConcat", AML_OperationType::kConcatenation},
    {"Reshape", AML_OperationType::kReshape}, {"reshape", AML_OperationType::kReshape},
    {"Transpose", AML_OperationType::kTranspose},
    {"ConvTranspose", AML_OperationType::kTransposeConv},
    {"Softmax", AML_OperationType::kSoftmax}, {"QLinearSoftmax", AML_OperationType::kSoftmax},
    {"QuantizeLinear", AML_OperationType::kQuantize},
    {"DequantizeLinear", AML_OperationType::kDequantize},
    {"Pad", AML_OperationType::kPad},
    {"MirrorPad", AML_OperationType::kMirrorPad},
    {"Sigmoid", AML_OperationType::kLogistic}, {"QLinearSigmoid", AML_OperationType::kLogistic},
    {"QLinearGlobalAveragePool", AML_OperationType::kAveragePool2D},
    {"ReduceMean", AML_OperationType::kMean},
    // These map one-to-one to implemented Compiler3 operators.  The
    // reference-style exporter includes the reduction-axis tensor explicitly,
    // which is the same operand convention as TFLite SUM/REDUCE_MAX.
    {"ReduceSum", AML_OperationType::kSum},
    {"ReduceMax", AML_OperationType::kReduceMax},
    {"Sub", AML_OperationType::kSub}, {"Div", AML_OperationType::kDiv},
    {"Rsqrt", AML_OperationType::kRsqrt},
    {"Tanh", AML_OperationType::kTanh}, {"Exp", AML_OperationType::kExp},
    {"Sqrt", AML_OperationType::kSqrt}, {"LRN", AML_OperationType::kLocalResponseNormalization},
    {"Pow", AML_OperationType::kPow}, {"BroadcastTo", AML_OperationType::kBroadcastTo},
    // The reference exporter represents ONNX Split with data, split sizes and
    // axis operands, i.e. TFLite SPLIT_V semantics rather than SPLIT.
    {"Split", AML_OperationType::kSplitV},
    {"Slice", AML_OperationType::kSlice},
    {"Reshape_nearest", AML_OperationType::kResizeNearestNeighbor},
    {"Reshape_bilinear", AML_OperationType::kResizeBilinear},
    {"Resize_nearest", AML_OperationType::kResizeNearestNeighbor},
    {"LeakyRelu", AML_OperationType::kLeakyRelu},
    {"HardSwish", AML_OperationType::kHardSwish},
    {"Gemm", AML_OperationType::kFullyConnected}, {"QGemm", AML_OperationType::kFullyConnected},
    {"MatMul", AML_OperationType::kFullyConnected},
};

bool IsLayoutFoldsDisabled() {
  const char* value = std::getenv("AML_ADLA_BRIDGE_DISABLE_LAYOUT_FOLDS");
  // Scheme 3's supported regression and release baseline preserves layout
  // nodes. An opt-in false value is required for a layout-fold experiment so
  // direct C ABI callers cannot silently diverge from batch compilation.
  if (value == nullptr || value[0] == '\0') return true;
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0 && std::strcmp(value, "off") != 0 &&
         std::strcmp(value, "OFF") != 0;
}

int TensorType(const std::string& type) {
  static const std::unordered_map<std::string, AML_TensorType> map = {
      {"float32", AML_TensorType::kFloat32}, {"float16", AML_TensorType::kFloat16},
      {"int8", AML_TensorType::kInt8}, {"uint8", AML_TensorType::kUInt8},
      {"int16", AML_TensorType::kInt16}, {"int32", AML_TensorType::kInt32},
      {"int64", AML_TensorType::kInt64}, {"bool", AML_TensorType::kBool}};
  const auto it = map.find(type);
  return static_cast<int>(it == map.end() ? AML_TensorType::kNoType : it->second);
}

void FlattenBytes(const json& values, const std::string& type, std::vector<uint8_t>* out) {
  if (values.is_array()) {
    for (const auto& value : values) FlattenBytes(value, type, out);
    return;
  }
  if (type == "float32") { const float v = values.get<float>(); const auto* p = reinterpret_cast<const uint8_t*>(&v); out->insert(out->end(), p, p + sizeof(v)); }
  else if (type == "float16") { const uint16_t v = values.get<uint16_t>(); const auto* p = reinterpret_cast<const uint8_t*>(&v); out->insert(out->end(), p, p + sizeof(v)); }
  else if (type == "int32") { const int32_t v = values.get<int32_t>(); const auto* p = reinterpret_cast<const uint8_t*>(&v); out->insert(out->end(), p, p + sizeof(v)); }
  else if (type == "int64") { const int64_t v = values.get<int64_t>(); const auto* p = reinterpret_cast<const uint8_t*>(&v); out->insert(out->end(), p, p + sizeof(v)); }
  else if (type == "uint8") out->push_back(values.get<uint8_t>());
  else out->push_back(static_cast<uint8_t>(values.get<int>()));  // int8 / bool
}

size_t ValueCount(const json& values) {
  if (!values.is_array()) return 1;
  size_t count = 0;
  for (const auto& value : values) count += ValueCount(value);
  return count;
}

size_t TensorTypeByteWidth(const std::string& type) {
  if (type == "float16" || type == "int16") return 2;
  if (type == "float32" || type == "int32") return 4;
  if (type == "int64") return 8;
  return 1;
}

// Converts one [output_channels, C, H, W] fully-connected weight matrix to
// [output_channels, H, W, C] order. The source activation stays NHWC when an
// immediately preceding NHWC -> NCHW Transpose is folded away.
void ReorderNchwFcWeightsToNhwc(AML_Tensor* weight, int height, int width, int channels) {
  if (weight == nullptr || weight->dims.size() != 2 || height <= 0 || width <= 0 || channels <= 0) return;
  const size_t rows = static_cast<size_t>(weight->dims[0]);
  const size_t columns = static_cast<size_t>(height) * width * channels;
  if (weight->dims[1] != static_cast<int>(columns) || columns == 0 ||
      weight->data.size() % (rows * columns) != 0) return;
  const size_t element_size = weight->data.size() / (rows * columns);
  std::vector<uint8_t> reordered(weight->data.size());
  for (size_t row = 0; row < rows; ++row) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          const size_t nchw = (static_cast<size_t>(c) * height * width + h * width + w);
          const size_t nhwc = ((static_cast<size_t>(h) * width + w) * channels + c);
          std::memcpy(reordered.data() + (row * columns + nhwc) * element_size,
                      weight->data.data() + (row * columns + nchw) * element_size,
                      element_size);
        }
      }
    }
  }
  weight->data.swap(reordered);
}

AML_Tensor ParseTensor(const json& value, int index) {
  AML_Tensor tensor;
  tensor.name = value.value("tensor_name", "");
  const std::string type = value.value("tensor_type", "");
  tensor.dtype = TensorType(type);
  tensor.dims = value.value("tensor_shape", std::vector<int>{});
  tensor.is_const = value.value("is_constant", false);
  tensor.tensor_index = index;
  // Reference-style JSON uses all of null, "None", scalar and array forms
  // for optional quantization fields.  Missing quantization must remain empty;
  // it is not an error and is common for float graph inputs.
  const auto parse_ints = [](const json& field) {
    std::vector<int> result;
    if (field.is_number_integer()) result.push_back(field.get<int>());
    else if (field.is_array()) for (const auto& item : field) if (item.is_number_integer()) result.push_back(item.get<int>());
    return result;
  };
  const auto parse_floats = [](const json& field) {
    std::vector<float> result;
    if (field.is_number()) result.push_back(field.get<float>());
    else if (field.is_array()) for (const auto& item : field) if (item.is_number()) result.push_back(item.get<float>());
    return result;
  };
  if (value.contains("zero_point")) tensor.zp = parse_ints(value["zero_point"]);
  if (value.contains("quant_scale")) tensor.scale = parse_floats(value["quant_scale"]);
  if (tensor.is_const && value.contains("values")) {
    const json& values = value["values"];
    FlattenBytes(values, type, &tensor.data);
    // Reference-style JSON may describe a scalar constant with the logical
    // broadcast shape of its consumer. TFLite requires the constant buffer
    // to contain every element implied by that shape; a four-byte scalar with
    // shape [1,1,1,C] is rejected as an invalid FlatBuffer. Materialize only
    // the unambiguous one-value case and preserve all other constants.
    size_t element_count = 1;
    for (const int dim : tensor.dims) {
      if (dim < 0) { element_count = 0; break; }
      element_count *= static_cast<size_t>(dim);
    }
    const size_t scalar_width = TensorTypeByteWidth(type);
    if (ValueCount(values) == 1 && element_count > 1 &&
        tensor.data.size() == scalar_width) {
      const std::vector<uint8_t> scalar = tensor.data;
      tensor.data.reserve(scalar.size() * element_count);
      for (size_t i = 1; i < element_count; ++i) {
        tensor.data.insert(tensor.data.end(), scalar.begin(), scalar.end());
      }
    }
  }
  tensor.size = tensor.data.size();
  return tensor;
}

void AddParams(const json& detail, AML_Node* node) {
  const auto add_one = [node](const std::string& raw_key, const json& value) {
    std::string key = raw_key;
    // Names emitted by reference_style and expected by aml_compiler_core differ
    // for a few TFLite option fields.
    if (key == "dilation_h_factor") key = "dilation_h";
    else if (key == "dilation_w_factor") key = "dilation_w";
    else if (key == "filter_height") key = "filter_h";
    else if (key == "filter_width") key = "filter_w";
    else if (key == "fused_activation_function") key = "activation";
    else if (key == "keepdims") key = "keep_dims";
    else if (key == "depth_multiplier") key = "multiplier";
    if (key == "beta" && value.is_number()) {
      node->fparams[key] = value.get<float>();
      node->params.erase(key);
    }
    else if (value.is_boolean()) node->params[key] = value.get<bool>() ? 1 : 0;
    else if (value.is_number_integer()) node->params[key] = value.get<int>();
    else if (value.is_number_float()) node->fparams[key] = value.get<float>();
    else if (value.is_string()) {
      const std::string v = value.get<std::string>();
      if (v == "SAME" || v == "Same") node->params[key] = static_cast<int>(AML_Padding::kSame);
      else if (v == "VALID" || v == "Valid") node->params[key] = static_cast<int>(AML_Padding::kValid);
      else if (v == "RELU" || v == "Relu") node->params[key] = static_cast<int>(AML_FusedActivation::kRelu);
      else if (v == "RELU6" || v == "Relu6") node->params[key] = static_cast<int>(AML_FusedActivation::kRelu6);
      else if (v == "NONE" || v == "None") node->params[key] = static_cast<int>(AML_FusedActivation::kNone);
      else if (key == "mode" && (v == "REFLECT" || v == "Reflect")) node->params[key] = 0;
      else if (key == "mode" && (v == "SYMMETRIC" || v == "Symmetric")) node->params[key] = 1;
    }
  };
  for (auto it = detail.begin(); it != detail.end(); ++it) {
    if (it.key().find("input_") == 0 || it.key().find("output_") == 0 || it.key() == "op_type_info") continue;
    if (it.value().is_object()) {
      for (auto child = it.value().begin(); child != it.value().end(); ++child) add_one(child.key(), child.value());
    } else {
      add_one(it.key(), it.value());
    }
  }
}
}  // namespace

bool BuildAmlModelFromOpsJson(const std::string& text, const std::string& target,
                              AML_Model* model, std::string* error) {
  if (!model) return false;
  try {
    const json root = json::parse(text);
    const json& ops = root.is_object() && root.contains("ops") ? root.at("ops") : root;
    if (!ops.is_array() || ops.empty()) { if (error) *error = "ops_json must contain a non-empty ops array"; return false; }
    const bool disable_layout_folds = IsLayoutFoldsDisabled();
    *model = AML_Model{};
    model->target = target;
    model->graph_names.push_back("reference_style_graph");
    // reference_style materializes ONNX explicit convolution padding as
    // PAD(data, [N,H,W,C]) -> QLinearConv(..., VALID). Compiler3 can create
    // an IPadOperator but cannot group that operator with the quantized Conv
    // engine. Recognize the canonical zero batch/channel-pad form and fold it
    // back to a native SAME Conv before building AML_Model.
    std::map<std::string, AML_Tensor> fused_same_pad_inputs;
    std::set<size_t> fused_pad_ops;
    for (size_t i = 0; i < ops.size(); ++i) {
      if (disable_layout_folds) continue;
      const auto pad_it = ops[i].begin();
      if (pad_it.key() != "Pad") continue;
      const json& pad = pad_it.value();
      const auto data_it = pad.find("input_tensor_info_1");
      const auto pads_it = pad.find("input_tensor_info_2");
      const auto output_it = pad.find("output_tensor_info");
      if (data_it == pad.end() || pads_it == pad.end() || output_it == pad.end() ||
          !pads_it->value("is_constant", false) ||
          !pads_it->contains("values")) continue;
      const std::vector<int> values = pads_it->at("values").get<std::vector<int>>();
      if (values.size() != 8 || values[0] != 0 || values[1] != 0 ||
          values[6] != 0 || values[7] != 0) continue;
      const std::string padded_name = output_it->value("tensor_name", "");
      bool feeds_conv = false;
      for (size_t j = i + 1; j < ops.size(); ++j) {
        const auto consumer_it = ops[j].begin();
        if (consumer_it.key() != "Conv" && consumer_it.key() != "QLinearConv" &&
            consumer_it.key() != "DepthwiseConv") continue;
        const auto conv_input_it = consumer_it.value().find("input_tensor_info");
        if (conv_input_it != consumer_it.value().end() &&
            conv_input_it->value("tensor_name", "") == padded_name) {
          feeds_conv = true;
          break;
        }
      }
      if (!feeds_conv) continue;
      const auto source = ParseTensor(*data_it, 0);
      const auto padded = ParseTensor(*output_it, 0);
      if (source.dims.size() != 4 || padded.dims.size() != 4) continue;
      fused_same_pad_inputs.emplace(padded.name, source);
      fused_pad_ops.insert(i);
    }
    // ORT's NHWC reference export sometimes emits the terminal
    // GlobalAveragePool -> Transpose([0,3,1,2]) -> Reshape sequence with an
    // unchanged declared 4-D shape.  With H=W=1 the transpose has no data
    // effect, while the extra layout node can make Compiler3's duplicate-edge
    // reduction reject the graph.  Fold only this exact, local pattern; all
    // ordinary Transpose nodes remain observable to the bridge.
    std::map<std::string, AML_Tensor> folded_terminal_transpose_inputs;
    std::set<size_t> folded_terminal_transpose_ops;
    for (size_t i = 0; i + 1 < ops.size(); ++i) {
      if (disable_layout_folds) continue;
      const auto transpose_it = ops[i].begin();
      const auto reshape_it = ops[i + 1].begin();
      if (transpose_it.key() != "Transpose" ||
          (reshape_it.key() != "Reshape" && reshape_it.key() != "reshape")) continue;
      const json& transpose = transpose_it.value();
      const json& reshape = reshape_it.value();
      const auto data_it = transpose.find("input_tensor_info_1");
      const auto perm_it = transpose.find("input_tensor_info_2");
      const auto output_it = transpose.find("output_tensor_info");
      const auto reshape_input_it = reshape.find("input_tensor_info_1");
      if (data_it == transpose.end() || perm_it == transpose.end() ||
          output_it == transpose.end() || reshape_input_it == reshape.end() ||
          !perm_it->value("is_constant", false) || !perm_it->contains("values")) continue;
      const std::vector<int> permutation = perm_it->at("values").get<std::vector<int>>();
      if (permutation != std::vector<int>{0, 3, 1, 2} ||
          output_it->value("tensor_name", "") != reshape_input_it->value("tensor_name", "")) continue;
      const auto source = ParseTensor(*data_it, 0);
      const auto output = ParseTensor(*output_it, 0);
      if (source.dims.size() != 4 || source.dims != output.dims ||
          source.dims[2] != 1 || source.dims[3] != 1) continue;
      folded_terminal_transpose_inputs.emplace(output.name, source);
      folded_terminal_transpose_ops.insert(i);
    }
    // VGG's classifier tail is emitted as NHWC MaxPool -> Transpose(NCHW) ->
    // Reshape -> Gemm. Compiler3 rejects the Transpose/Reshape pair in its
    // duplicate-edge pass. Fold only this exact three-node pattern and
    // reorder the following FC weight matrix from NCHW columns to NHWC
    // columns, preserving the original ONNX result.
    std::map<std::string, AML_Tensor> folded_nchw_transpose_inputs;
    std::map<std::string, std::vector<int>> folded_nchw_fc_inputs;
    std::set<size_t> folded_nchw_transpose_ops;
    for (size_t i = 0; i + 2 < ops.size(); ++i) {
      if (disable_layout_folds) continue;
      const auto transpose_it = ops[i].begin();
      const auto reshape_it = ops[i + 1].begin();
      const auto gemm_it = ops[i + 2].begin();
      if (transpose_it.key() != "Transpose" ||
          (reshape_it.key() != "Reshape" && reshape_it.key() != "reshape") ||
          gemm_it.key() != "Gemm") continue;
      const json& transpose = transpose_it.value();
      const json& reshape = reshape_it.value();
      const auto data_it = transpose.find("input_tensor_info_1");
      const auto perm_it = transpose.find("input_tensor_info_2");
      const auto output_it = transpose.find("output_tensor_info");
      const auto reshape_input_it = reshape.find("input_tensor_info_1");
      const auto reshape_output_it = reshape.find("output_tensor_info");
      if (data_it == transpose.end() || perm_it == transpose.end() || output_it == transpose.end() ||
          reshape_input_it == reshape.end() || reshape_output_it == reshape.end() ||
          !perm_it->value("is_constant", false) || !perm_it->contains("values")) continue;
      const std::vector<int> permutation = perm_it->at("values").get<std::vector<int>>();
      const auto source = ParseTensor(*data_it, 0);
      if (permutation != std::vector<int>{0, 3, 1, 2} || source.dims.size() != 4 ||
          output_it->value("tensor_name", "") != reshape_input_it->value("tensor_name", "")) continue;
      bool reshape_feeds_gemm = false;
      for (auto it = gemm_it.value().begin(); it != gemm_it.value().end(); ++it) {
        if (it.key().find("input_tensor_info") == 0 && it.value().is_object() &&
            !it.value().value("is_constant", false) &&
            it.value().value("tensor_name", "") == reshape_output_it->value("tensor_name", "")) {
          reshape_feeds_gemm = true;
          break;
        }
      }
      if (!reshape_feeds_gemm) continue;
      folded_nchw_transpose_inputs.emplace(output_it->value("tensor_name", ""), source);
      folded_nchw_fc_inputs.emplace(reshape_output_it->value("tensor_name", ""), source.dims);
      folded_nchw_transpose_ops.insert(i);
    }
    // Classifier tails use either an identity reshape or a Flatten reshape
    // immediately before Gemm. Compiler3 requires the FC path to remain 4-D,
    // so bypass these reshapes when the element count is unchanged.
    std::map<std::string, AML_Tensor> folded_identity_reshape_inputs;
    std::set<size_t> folded_identity_reshape_ops;
    for (size_t i = 0; i + 1 < ops.size(); ++i) {
      if (disable_layout_folds) continue;
      const auto reshape_it = ops[i].begin();
      const auto gemm_it = ops[i + 1].begin();
      if ((reshape_it.key() != "Reshape" && reshape_it.key() != "reshape") ||
          gemm_it.key() != "Gemm") continue;
      const json& reshape = reshape_it.value();
      const auto output_it = reshape.find("output_tensor_info");
      const json* data = nullptr;
      for (auto it = reshape.begin(); it != reshape.end(); ++it) {
        if (it.key().find("input_tensor_info") == 0 && it.value().is_object() &&
            !it.value().value("is_constant", false)) {
          data = &it.value();
          break;
        }
      }
      if (data == nullptr || output_it == reshape.end()) continue;
      bool consumed_by_gemm = false;
      for (auto it = gemm_it.value().begin(); it != gemm_it.value().end(); ++it) {
        if (it.key().find("input_tensor_info") == 0 && it.value().is_object() &&
            it.value().value("tensor_name", "") == output_it->value("tensor_name", "")) {
          consumed_by_gemm = true;
          break;
        }
      }
      if (!consumed_by_gemm) continue;
      const auto source = ParseTensor(*data, 0);
      const auto output = ParseTensor(*output_it, 0);
      const bool identity = source.dims == output.dims;
      if (source.dims.empty() || !identity) continue;
      folded_identity_reshape_inputs.emplace(output.name, source);
      folded_identity_reshape_ops.insert(i);
    }
    // A current ORT reference-style corner case labels a MaxPool result as
    // int8 even though the immediately following QuantizeLinear consumes it
    // as the pre-quantized float activation. Keep both endpoints consistent;
    // otherwise Compiler3 sees one tensor name with incompatible port types.
    std::set<std::string> maxpool_quantize_sources;
    for (size_t i = 0; i + 1 < ops.size(); ++i) {
      const auto producer = ops[i].begin();
      const auto consumer = ops[i + 1].begin();
      if (producer.key() != "MaxPool" || consumer.key() != "QuantizeLinear") continue;
      const auto out = producer.value().find("output_tensor_info");
      const auto in = consumer.value().find("input_tensor_info");
      if (out != producer.value().end() && in != consumer.value().end() &&
          out->value("tensor_name", "") == in->value("tensor_name", "")) {
        maxpool_quantize_sources.insert(out->value("tensor_name", ""));
      }
    }
    std::map<std::string, AML_Tensor> known;
    std::set<std::string> produced;
    std::set<std::string> declared_graph_inputs;
    if (root.is_object() && root.contains("graph_inputs")) {
      for (const auto& name : root.at("graph_inputs")) declared_graph_inputs.insert(name.get<std::string>());
    }
    std::vector<std::string> unresolved_inputs;
    // AlexNet/CaffeNet encode a two-group convolution as
    // Split -> (Pad -> Conv) x2 -> Concat. Compiler3 rejects the branched
    // form during shape evaluation. Record only this exact consecutive
    // pattern; it is lowered to one block-diagonal Conv in the main loop.
    std::map<size_t, std::array<size_t, 6>> group_conv_patterns;
    std::set<size_t> group_conv_tail_ops;
    for (size_t i = 0; i + 5 < ops.size(); ++i) {
      const std::array<std::string, 6> expected = {
          "Split", "Pad", "Pad", "Conv", "Conv", "Concat"};
      bool matches = true;
      for (size_t offset = 0; offset < expected.size(); ++offset) {
        if (ops[i + offset].begin().key() != expected[offset]) matches = false;
      }
      if (!matches) continue;
      const auto split_op_it = ops[i].begin();
      const json& split = split_op_it.value();
      const auto sizes_it = split.find("input_tensor_info_2");
      const auto axis_it = split.find("input_tensor_info_3");
      if (sizes_it == split.end() || axis_it == split.end() ||
          !sizes_it->contains("values") || !axis_it->contains("values") ||
          sizes_it->at("values").get<std::vector<int>>().size() != 2 ||
          axis_it->at("values").get<std::vector<int>>() != std::vector<int>{3}) continue;
      group_conv_patterns.emplace(i, std::array<size_t, 6>{i, i + 1, i + 2, i + 3, i + 4, i + 5});
      for (size_t offset = 1; offset < 6; ++offset) group_conv_tail_ops.insert(i + offset);
    }
    // Some exporters append shape-restoring Reshape nodes after their first
    // consumer.  Build from a stable topological order instead of trusting
    // JSON array order; otherwise a valid forward edge is misdiagnosed as a
    // graph input with no producer (for example Whisper attention residuals).
    std::unordered_map<std::string, size_t> tensor_producer;
    for (size_t producer_i = 0; producer_i < ops.size(); ++producer_i) {
      const auto producer_op = ops[producer_i].begin();
      const auto& producer_detail = producer_op.value();
      for (auto it = producer_detail.begin(); it != producer_detail.end(); ++it) {
        if (it.key().find("output_tensor_info") != 0) continue;
        const std::string name = it.value().value("tensor_name", "");
        if (!name.empty()) tensor_producer.emplace(name, producer_i);
      }
    }
    std::set<size_t> hard_sigmoid_relu_starts;
    std::set<size_t> hard_sigmoid_tail_ops;
    std::set<size_t> channel_shuffle_starts;
    std::set<size_t> channel_shuffle_tail_ops;
    const auto is_scalar_value = [](const json& tensor, float expected) {
      const auto values = tensor.find("values");
      return tensor.value("is_constant", false) && values != tensor.end() &&
          values->is_array() && values->size() == 1 && (*values)[0].is_number() &&
          std::fabs((*values)[0].get<float>() - expected) < 1.0e-5f;
    };
    for (size_t i = 0; i + 2 < ops.size(); ++i) {
      const auto mul6 = ops[i].begin();
      const auto clip = ops[i + 1].begin();
      const auto scale = ops[i + 2].begin();
      if (mul6.key() != "Mul" || clip.key() != "Clip" || scale.key() != "Mul") continue;
      const json& mul6_detail = mul6.value();
      const json& clip_detail = clip.value();
      const json& scale_detail = scale.value();
      const auto mul6_data = mul6_detail.find("input_tensor_info_1");
      const auto mul6_scalar = mul6_detail.find("input_tensor_info_2");
      const auto mul6_output = mul6_detail.find("output_tensor_info");
      const auto clip_input = clip_detail.find("input_tensor_info");
      const auto clip_output = clip_detail.find("output_tensor_info");
      const auto scale_input = scale_detail.find("input_tensor_info_1");
      const auto scale_scalar = scale_detail.find("input_tensor_info_2");
      if (mul6_data == mul6_detail.end() || mul6_scalar == mul6_detail.end() ||
          mul6_output == mul6_detail.end() || clip_input == clip_detail.end() ||
          clip_output == clip_detail.end() || scale_input == scale_detail.end() ||
          scale_scalar == scale_detail.end() ||
          !is_scalar_value(*mul6_scalar, 6.0f) ||
          !is_scalar_value(*scale_scalar, 1.0f / 6.0f) ||
          mul6_output->value("tensor_name", "") != clip_input->value("tensor_name", "") ||
          clip_output->value("tensor_name", "") != scale_input->value("tensor_name", "")) continue;
      const auto affine_producer = tensor_producer.find(mul6_data->value("tensor_name", ""));
      if (affine_producer == tensor_producer.end() ||
          ops[affine_producer->second].begin().key() != "Add") continue;
      hard_sigmoid_relu_starts.insert(i);
      hard_sigmoid_tail_ops.insert(i + 1);
      hard_sigmoid_tail_ops.insert(i + 2);
    }
    for (size_t i = 0; i + 4 < ops.size(); ++i) {
      const auto t0 = ops[i].begin();
      const auto r0 = ops[i + 1].begin();
      const auto t1 = ops[i + 2].begin();
      const auto r1 = ops[i + 3].begin();
      const auto t2 = ops[i + 4].begin();
      if (t0.key() != "Transpose" ||
          (r0.key() != "Reshape" && r0.key() != "reshape") ||
          t1.key() != "Transpose" ||
          (r1.key() != "Reshape" && r1.key() != "reshape") ||
          t2.key() != "Transpose") continue;
      const json& t0_detail = t0.value();
      const json& r0_detail = r0.value();
      const json& t1_detail = t1.value();
      const json& t2_detail = t2.value();
      const auto input_it = t0_detail.find("input_tensor_info_1");
      const auto rank5_it = r0_detail.find("output_tensor_info");
      const auto middle_perm_it = t1_detail.find("input_tensor_info_2");
      const auto output_it = t2_detail.find("output_tensor_info");
      if (input_it == t0_detail.end() || rank5_it == r0_detail.end() ||
          middle_perm_it == t1_detail.end() || output_it == t2_detail.end() ||
          !middle_perm_it->contains("values")) continue;
      const std::vector<int> input_shape = input_it->value("tensor_shape", std::vector<int>{});
      const std::vector<int> rank5_shape = rank5_it->value("tensor_shape", std::vector<int>{});
      const std::vector<int> output_shape = output_it->value("tensor_shape", std::vector<int>{});
      const std::vector<int> middle_perm =
          middle_perm_it->at("values").get<std::vector<int>>();
      if (input_shape.size() != 4 || output_shape != input_shape ||
          rank5_shape.size() != 5 || rank5_shape[0] != input_shape[0] ||
          rank5_shape[1] != 2 || rank5_shape[2] * 2 != input_shape[3] ||
          rank5_shape[3] != input_shape[1] || rank5_shape[4] != input_shape[2] ||
          middle_perm != std::vector<int>({0, 2, 1, 3, 4})) continue;
      channel_shuffle_starts.insert(i);
      for (size_t offset = 1; offset < 5; ++offset) {
        channel_shuffle_tail_ops.insert(i + offset);
      }
    }
    std::vector<std::vector<size_t>> consumers(ops.size());
    std::vector<int> pending_dependencies(ops.size(), 0);
    for (size_t consumer_i = 0; consumer_i < ops.size(); ++consumer_i) {
      const auto consumer_op = ops[consumer_i].begin();
      const auto& consumer_detail = consumer_op.value();
      std::set<size_t> unique_dependencies;
      for (auto it = consumer_detail.begin(); it != consumer_detail.end(); ++it) {
        if (it.key().find("input_tensor_info") != 0 ||
            it.value().value("is_constant", false)) continue;
        const std::string name = it.value().value("tensor_name", "");
        const auto producer = tensor_producer.find(name);
        if (producer != tensor_producer.end() && producer->second != consumer_i) {
          unique_dependencies.insert(producer->second);
        }
      }
      pending_dependencies[consumer_i] = static_cast<int>(unique_dependencies.size());
      for (const size_t producer_i : unique_dependencies) consumers[producer_i].push_back(consumer_i);
    }
    std::map<size_t, AML_Tensor> fused_relu6_outputs;
    std::set<size_t> fused_relu6_clip_ops;
    for (size_t i = 0; i + 1 < ops.size(); ++i) {
      const auto producer = ops[i].begin();
      const auto clip = ops[i + 1].begin();
      if ((producer.key() != "Conv" && producer.key() != "QLinearConv" &&
           producer.key() != "DepthwiseConv") || clip.key() != "Clip" ||
          consumers[i].size() != 1 || consumers[i][0] != i + 1) continue;
      const json& producer_detail = producer.value();
      const json& clip_detail = clip.value();
      const auto producer_output = producer_detail.find("output_tensor_info");
      const auto clip_input = clip_detail.find("input_tensor_info");
      const auto clip_output = clip_detail.find("output_tensor_info");
      const auto attributes = clip_detail.find("attributes_info");
      const bool clip_is_relu6 = attributes == clip_detail.end() ||
          (std::fabs(attributes->value("min", 0.0f)) <= 1.0e-6f &&
           std::fabs(attributes->value("max", 6.0f) - 6.0f) <= 1.0e-6f);
      if (producer_output == producer_detail.end() || clip_input == clip_detail.end() ||
          clip_output == clip_detail.end() || !clip_is_relu6 ||
          producer_output->value("tensor_name", "") != clip_input->value("tensor_name", "")) continue;
      fused_relu6_outputs.emplace(i, ParseTensor(*clip_output, 0));
      fused_relu6_clip_ops.insert(i + 1);
    }
    std::set<size_t> ready;
    for (size_t op_i = 0; op_i < ops.size(); ++op_i) {
      if (pending_dependencies[op_i] == 0) ready.insert(op_i);
    }
    std::vector<size_t> op_order;
    while (!ready.empty()) {
      const size_t op_i = *ready.begin();
      ready.erase(ready.begin());
      op_order.push_back(op_i);
      for (const size_t consumer_i : consumers[op_i]) {
        if (--pending_dependencies[consumer_i] == 0) ready.insert(consumer_i);
      }
    }
    // Preserve source order for cyclic/ambiguous remnants. The normal bridge
    // validation below will still reject any real missing input deterministically.
    if (op_order.size() != ops.size()) {
      std::set<size_t> emitted(op_order.begin(), op_order.end());
      for (size_t op_i = 0; op_i < ops.size(); ++op_i) {
        if (emitted.count(op_i) == 0) op_order.push_back(op_i);
      }
    }
    for (const size_t i : op_order) {
      const auto first = ops[i].begin();
      const std::string op_name = first.key(); const json& detail = first.value();
      if (channel_shuffle_tail_ops.count(i) != 0) continue;
      if (channel_shuffle_starts.count(i) != 0) {
        AML_Tensor input = ParseTensor(detail.at("input_tensor_info_1"), 0);
        const auto known_input = known.find(input.name);
        if (known_input != known.end()) input = known_input->second;
        const auto final_op_it = ops[i + 4].begin();
        const json& final_detail = final_op_it.value();
        AML_Tensor output = ParseTensor(final_detail.at("output_tensor_info"), 1);
        const int n = input.dims[0];
        const int h = input.dims[1];
        const int w = input.dims[2];
        const int channels = input.dims[3];

        AML_Tensor reshape_output = output;
        reshape_output.name = output.name + "_shuffle_reshape";
        reshape_output.dims = {n, h * w, 2, channels / 2};
        AML_Node reshape{};
        reshape.Node_type = static_cast<int>(AML_OperationType::kReshape);
        reshape.Node_index = static_cast<int>(model->Node_List.size());
        reshape.inputs = {input};
        reshape.outputs = {reshape_output};
        model->op_codes.push_back("Reshape");
        model->Node_List.push_back(std::move(reshape));

        AML_Tensor perm;
        perm.name = output.name + "_shuffle_perm";
        perm.dims = {4};
        perm.dtype = static_cast<int>(AML_TensorType::kInt32);
        perm.is_const = true;
        perm.tensor_index = 1;
        const std::array<int32_t, 4> perm_values = {0, 1, 3, 2};
        perm.data.resize(sizeof(perm_values));
        std::memcpy(perm.data.data(), perm_values.data(), sizeof(perm_values));
        perm.size = perm.data.size();
        AML_Tensor transpose_output = reshape_output;
        transpose_output.name = output.name + "_shuffle_transpose";
        transpose_output.dims = {n, h * w, channels / 2, 2};
        AML_Node transpose{};
        transpose.Node_type = static_cast<int>(AML_OperationType::kTranspose);
        transpose.Node_index = static_cast<int>(model->Node_List.size());
        transpose.inputs = {reshape_output, perm};
        transpose.outputs = {transpose_output};
        model->op_codes.push_back("Transpose");
        model->Node_List.push_back(std::move(transpose));

        AML_Node reshape_back{};
        reshape_back.Node_type = static_cast<int>(AML_OperationType::kReshape);
        reshape_back.Node_index = static_cast<int>(model->Node_List.size());
        reshape_back.inputs = {transpose_output};
        reshape_back.outputs = {output};
        model->op_codes.push_back("Reshape");
        model->Node_List.push_back(std::move(reshape_back));
        known[output.name] = output;
        produced.insert(output.name);
        continue;
      }
      if (fused_relu6_clip_ops.count(i) != 0) continue;
      if (hard_sigmoid_tail_ops.count(i) != 0) continue;
      if (hard_sigmoid_relu_starts.count(i) != 0) {
        AML_Tensor input = ParseTensor(detail.at("input_tensor_info_1"), 0);
        const auto known_input = known.find(input.name);
        if (known_input != known.end()) input = known_input->second;
        const auto final_op_it = ops[i + 2].begin();
        const json& final_detail = final_op_it.value();
        AML_Tensor output = ParseTensor(final_detail.at("output_tensor_info"), 1);
        AML_Tensor relu_output = output;
        relu_output.name = output.name + "_relu";

        AML_Node relu{};
        relu.Node_type = static_cast<int>(AML_OperationType::kRelu);
        relu.Node_index = static_cast<int>(model->Node_List.size());
        relu.inputs = {input};
        relu.outputs = {relu_output};
        model->op_codes.push_back("Relu");
        model->Node_List.push_back(std::move(relu));

        AML_Node relu1{};
        relu1.Node_type = static_cast<int>(AML_OperationType::kReluN1To1);
        relu1.Node_index = static_cast<int>(model->Node_List.size());
        relu1.inputs = {relu_output};
        relu1.outputs = {output};
        model->op_codes.push_back("ReluN1To1");
        model->Node_List.push_back(std::move(relu1));
        known[output.name] = output;
        produced.insert(output.name);
        continue;
      }
      if (group_conv_tail_ops.count(i) != 0) continue;
      const auto group_pattern = group_conv_patterns.find(i);
      if (group_pattern != group_conv_patterns.end()) {
        const auto& indices = group_pattern->second;
        const auto split_op_it = ops[indices[0]].begin();
        const auto concat_op_it = ops[indices[5]].begin();
        const json& split = split_op_it.value();
        const json& concat = concat_op_it.value();
        const auto split_input_it = split.find("input_tensor_info_1");
        const auto split_sizes_it = split.find("input_tensor_info_2");
        const auto concat_output_it = concat.find("output_tensor_info");
        if (split_input_it == split.end() || split_sizes_it == split.end() ||
            concat_output_it == concat.end()) {
          if (error) *error = "group Conv pattern is missing Split/Concat tensors";
          return false;
        }
        AML_Tensor input = ParseTensor(*split_input_it, 0);
        const auto input_producer = known.find(input.name);
        if (!input.is_const && input_producer != known.end()) input = input_producer->second;
        const std::vector<int> group_sizes =
            split_sizes_it->at("values").get<std::vector<int>>();
        if (input.dims.size() != 4 || group_sizes.size() != 2 ||
            group_sizes[0] + group_sizes[1] != input.dims[3]) {
          if (error) *error = "group Conv Split sizes do not match input channels";
          return false;
        }

        // Map each padded branch tensor back to its channel offset.
        std::map<std::string, int> padded_input_offsets;
        std::vector<std::string> split_outputs;
        for (auto it = split.begin(); it != split.end(); ++it) {
          if (it.key().find("output_tensor_info") == 0)
            split_outputs.push_back(it.value().value("tensor_name", ""));
        }
        for (size_t pad_pos = 1; pad_pos <= 2; ++pad_pos) {
          const auto pad_op_it = ops[indices[pad_pos]].begin();
          const json& pad = pad_op_it.value();
          const auto data_it = pad.find("input_tensor_info_1");
          const auto output_it = pad.find("output_tensor_info");
          if (data_it == pad.end() || output_it == pad.end()) continue;
          const std::string source = data_it->value("tensor_name", "");
          const auto branch = std::find(split_outputs.begin(), split_outputs.end(), source);
          if (branch == split_outputs.end()) continue;
          const size_t branch_index = static_cast<size_t>(branch - split_outputs.begin());
          padded_input_offsets[output_it->value("tensor_name", "")] =
              branch_index == 0 ? 0 : group_sizes[0];
        }

        std::vector<const json*> ordered_convs;
        for (size_t concat_input = 1; concat_input <= 2; ++concat_input) {
          const auto input_it = concat.find("input_tensor_info_" + std::to_string(concat_input));
          if (input_it == concat.end()) continue;
          const std::string wanted = input_it->value("tensor_name", "");
          for (size_t conv_pos = 3; conv_pos <= 4; ++conv_pos) {
            const auto conv_op_it = ops[indices[conv_pos]].begin();
            const json& conv = conv_op_it.value();
            const auto output_it = conv.find("output_tensor_info");
            if (output_it != conv.end() && output_it->value("tensor_name", "") == wanted)
              ordered_convs.push_back(&conv);
          }
        }
        if (ordered_convs.size() != 2 || padded_input_offsets.size() != 2) {
          if (error) *error = "group Conv branches cannot be matched";
          return false;
        }

        AML_Tensor combined_weight;
        AML_Tensor combined_bias;
        int total_output_channels = 0;
        int kernel_h = 0;
        int kernel_w = 0;
        size_t element_size = 0;
        for (const json* conv : ordered_convs) {
          AML_Tensor weight = ParseTensor(conv->at("input_weight_info"), 1);
          if (weight.dims.size() != 4 || weight.dims[3] <= 0) {
            if (error) *error = "group Conv weight must be OHWI rank-4";
            return false;
          }
          kernel_h = weight.dims[1];
          kernel_w = weight.dims[2];
          const size_t elements = static_cast<size_t>(weight.dims[0]) * kernel_h * kernel_w * weight.dims[3];
          if (elements == 0 || weight.data.size() % elements != 0) {
            if (error) *error = "group Conv weight byte size is invalid";
            return false;
          }
          element_size = weight.data.size() / elements;
          total_output_channels += weight.dims[0];
        }
        combined_weight = ParseTensor(ordered_convs[0]->at("input_weight_info"), 1);
        combined_weight.name += "_group_fused";
        combined_weight.dims = {total_output_channels, kernel_h, kernel_w, input.dims[3]};
        combined_weight.data.assign(static_cast<size_t>(total_output_channels) * kernel_h * kernel_w *
                                        input.dims[3] * element_size, 0);
        int output_offset = 0;
        for (const json* conv : ordered_convs) {
          AML_Tensor weight = ParseTensor(conv->at("input_weight_info"), 1);
          const std::string padded_name = conv->at("input_tensor_info").value("tensor_name", "");
          const int input_offset = padded_input_offsets.at(padded_name);
          for (int o = 0; o < weight.dims[0]; ++o) {
            for (int h = 0; h < kernel_h; ++h) {
              for (int w = 0; w < kernel_w; ++w) {
                for (int c = 0; c < weight.dims[3]; ++c) {
                  const size_t src = (((static_cast<size_t>(o) * kernel_h + h) * kernel_w + w) *
                                      weight.dims[3] + c) * element_size;
                  const size_t dst = (((static_cast<size_t>(output_offset + o) * kernel_h + h) * kernel_w + w) *
                                      input.dims[3] + input_offset + c) * element_size;
                  std::memcpy(combined_weight.data.data() + dst, weight.data.data() + src, element_size);
                }
              }
            }
          }
          output_offset += weight.dims[0];
        }
        combined_weight.size = combined_weight.data.size();
        combined_bias = ParseTensor(ordered_convs[0]->at("input_bias_info"), 2);
        combined_bias.name += "_group_fused";
        combined_bias.data.clear();
        combined_bias.dims = {total_output_channels};
        for (const json* conv : ordered_convs) {
          AML_Tensor bias = ParseTensor(conv->at("input_bias_info"), 2);
          combined_bias.data.insert(combined_bias.data.end(), bias.data.begin(), bias.data.end());
        }
        combined_bias.size = combined_bias.data.size();

        AML_Tensor output = ParseTensor(*concat_output_it, 3);
        AML_Node fused{};
        fused.Node_type = static_cast<int>(AML_OperationType::kConv2D);
        fused.Node_index = static_cast<int>(model->Node_List.size());
        fused.inputs = {input, std::move(combined_weight), std::move(combined_bias)};
        fused.outputs = {output};
        AddParams(*ordered_convs[0], &fused);
        fused.params["padding"] = static_cast<int>(AML_Padding::kSame);
        fused.inputs[2].dims = {1, 1, 1, total_output_channels};
        model->op_codes.push_back("Conv");
        model->Node_List.push_back(std::move(fused));
        known[output.name] = output;
        produced.insert(output.name);
        continue;
      }
      if (fused_pad_ops.count(i) != 0 || folded_terminal_transpose_ops.count(i) != 0 ||
          folded_identity_reshape_ops.count(i) != 0 || folded_nchw_transpose_ops.count(i) != 0) continue;
      const auto op = kOpMap.find(op_name);
      if (op == kOpMap.end()) { if (error) *error = "AML bridge does not support op: " + op_name; return false; }

      // Compiler3 lowers quantized SPLIT_V outputs to NCHW16c and cannot
      // reconnect them to ordinary NHWC consumers. For the static Split form
      // emitted by the reference exporter, emit one TFLite SLICE per output
      // instead. This has the same values and output names without crossing
      // the unsupported internal layout boundary.
      const auto split_data_it = detail.find("input_tensor_info_1");
      const bool is_quantized_split = split_data_it != detail.end() &&
          (split_data_it->value("tensor_type", "") == "int8" ||
           split_data_it->value("tensor_type", "") == "uint8");
      if (op_name == "Split" && is_quantized_split) {
        // Exporters use both the numbered form (input_tensor_info_1) and
        // the legacy unnumbered form (input_tensor_info) for the data input.
        // ConvTranspose must accept both; the generic operand collector below
        // already does so, but this custom lowering used to reject the latter.
        auto data_it = detail.find("input_tensor_info_1");
        if (data_it == detail.end()) data_it = detail.find("input_tensor_info");
        const auto sizes_it = detail.find("input_tensor_info_2");
        const auto axis_it = detail.find("input_tensor_info_3");
        if (data_it == detail.end() || sizes_it == detail.end() || axis_it == detail.end() ||
            !sizes_it->contains("values") || !axis_it->contains("values")) {
          if (error) *error = "Split requires static data, sizes and axis inputs";
          return false;
        }
        auto data = ParseTensor(*data_it, 0);
        const auto producer = known.find(data.name);
        if (!data.is_const && producer != known.end()) data = producer->second;
        const std::vector<int> sizes = sizes_it->at("values").get<std::vector<int>>();
        const std::vector<int> axis_values = axis_it->at("values").get<std::vector<int>>();
        if (data.dims.empty() || axis_values.size() != 1 || sizes.empty()) {
          if (error) *error = "Split has invalid static axis or sizes";
          return false;
        }
        int axis = axis_values[0];
        if (axis < 0) axis += static_cast<int>(data.dims.size());
        if (axis < 0 || axis >= static_cast<int>(data.dims.size())) {
          if (error) *error = "Split axis is out of range";
          return false;
        }
        std::vector<AML_Tensor> outputs;
        for (auto it = detail.begin(); it != detail.end(); ++it) {
          if (it.key().find("output_tensor_info") == 0) outputs.push_back(ParseTensor(it.value(), 0));
        }
        if (outputs.size() != sizes.size()) {
          if (error) *error = "Split output count does not match static sizes";
          return false;
        }
        // SplitV's quantized implementation produces NCHW16c, which cannot
        // feed YOLO's NHWC branches. Keep SplitV but perform it in float NHWC,
        // then restore every original quantized output tensor.
        AML_Tensor float_input = data;
        float_input.name += "_split_float";
        float_input.dtype = static_cast<int>(AML_TensorType::kFloat32);
        float_input.is_const = false;
        float_input.data.clear();
        float_input.size = 0;
        float_input.scale.clear();
        float_input.zp.clear();
        AML_Node dequant{};
        dequant.Node_type = static_cast<int>(AML_OperationType::kDequantize);
        dequant.Node_index = static_cast<int>(model->Node_List.size());
        dequant.inputs.push_back(data);
        dequant.outputs.push_back(float_input);
        model->op_codes.push_back("DequantizeLinear");
        model->Node_List.push_back(std::move(dequant));

        AML_Node split{};
        split.Node_type = static_cast<int>(AML_OperationType::kSplitV);
        split.Node_index = static_cast<int>(model->Node_List.size());
        split.inputs.push_back(float_input);
        split.inputs.push_back(ParseTensor(*sizes_it, 1));
        split.inputs.push_back(ParseTensor(*axis_it, 2));
        std::vector<AML_Tensor> float_outputs;
        for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
          auto float_output = outputs[output_index];
          float_output.name += "_split_float";
          float_output.dtype = static_cast<int>(AML_TensorType::kFloat32);
          float_output.is_const = false;
          float_output.data.clear();
          float_output.size = 0;
          float_output.scale.clear();
          float_output.zp.clear();
          split.outputs.push_back(float_output);
          float_outputs.push_back(std::move(float_output));
        }
        model->op_codes.push_back("SplitV");
        model->Node_List.push_back(std::move(split));
        for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
          AML_Node quant{};
          quant.Node_type = static_cast<int>(AML_OperationType::kQuantize);
          quant.Node_index = static_cast<int>(model->Node_List.size());
          quant.inputs.push_back(float_outputs[output_index]);
          quant.outputs.push_back(outputs[output_index]);
          known[outputs[output_index].name] = outputs[output_index];
          produced.insert(outputs[output_index].name);
          model->op_codes.push_back("QuantizeLinear");
          model->Node_List.push_back(std::move(quant));
        }
        continue;
      }
      // Float SplitV in the direct Compiler3 API crashes in
      // ReduceReshapePairWithEnclosedOperations for rank-3 attention tensors.
      // Its reference-style form is static, so lower it to one STRIDED_SLICE
      // per output. Compiler3 handles this path more reliably than ISlice.
      if (op_name == "Split") {
        const auto data_it = detail.find("input_tensor_info_1");
        const auto sizes_it = detail.find("input_tensor_info_2");
        const auto axis_it = detail.find("input_tensor_info_3");
        if (data_it == detail.end() || sizes_it == detail.end() || axis_it == detail.end() ||
            !sizes_it->contains("values") || !axis_it->contains("values")) {
          if (error) *error = "Split requires static data, sizes and axis inputs";
          return false;
        }
        auto data = ParseTensor(*data_it, 0);
        const auto producer = known.find(data.name);
        if (!data.is_const && producer != known.end()) data = producer->second;
        const std::vector<int> sizes = sizes_it->at("values").get<std::vector<int>>();
        std::vector<int> axis_values = axis_it->at("values").get<std::vector<int>>();
        int axis = axis_values.size() == 1 ? axis_values[0] : -1;
        if (axis < 0) axis += static_cast<int>(data.dims.size());
        if (data.dims.empty() || axis < 0 || axis >= static_cast<int>(data.dims.size())) {
          if (error) *error = "Split axis is out of range";
          return false;
        }
        std::vector<AML_Tensor> outputs;
        for (auto it = detail.begin(); it != detail.end(); ++it) {
          if (it.key().find("output_tensor_info") == 0) outputs.push_back(ParseTensor(it.value(), 0));
        }
        if (outputs.size() != sizes.size()) {
          if (error) *error = "Split output count does not match static sizes";
          return false;
        }
        int offset = 0;
        for (size_t output_i = 0; output_i < outputs.size(); ++output_i) {
          AML_Tensor begin{};
          begin.name = outputs[output_i].name + "_slice_begin";
          begin.dtype = static_cast<int>(AML_TensorType::kInt32);
          begin.dims = {static_cast<int>(data.dims.size())};
          begin.is_const = true;
          std::vector<int> begin_values(data.dims.size(), 0);
          begin_values[axis] = offset;
          for (const int value : begin_values) {
            const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
            begin.data.insert(begin.data.end(), bytes, bytes + sizeof(value));
          }
          begin.size = begin.data.size();
          AML_Tensor end = begin;
          end.name = outputs[output_i].name + "_slice_end";
          end.data.clear();
          std::vector<int> end_values = data.dims;
          end_values[axis] = offset + sizes[output_i];
          for (const int value : end_values) {
            const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
            end.data.insert(end.data.end(), bytes, bytes + sizeof(value));
          }
          end.size = end.data.size();
          AML_Tensor strides = begin;
          strides.name = outputs[output_i].name + "_slice_strides";
          strides.data.clear();
          for (size_t i = 0; i < data.dims.size(); ++i) {
            const int value = 1;
            const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
            strides.data.insert(strides.data.end(), bytes, bytes + sizeof(value));
          }
          strides.size = strides.data.size();
          AML_Node slice{};
          slice.Node_type = static_cast<int>(AML_OperationType::kStridedSlice);
          slice.Node_index = static_cast<int>(model->Node_List.size());
          slice.inputs = {data, std::move(begin), std::move(end), std::move(strides)};
          slice.params["begin_mask"] = 0;
          slice.params["end_mask"] = 0;
          slice.params["ellipsis_mask"] = 0;
          slice.params["new_axis_mask"] = 0;
          slice.params["shrink_axis_mask"] = 0;
          slice.params["offset"] = 0;
          slice.outputs.push_back(outputs[output_i]);
          known[outputs[output_i].name] = outputs[output_i];
          produced.insert(outputs[output_i].name);
          model->op_codes.push_back("StridedSlice");
          model->Node_List.push_back(std::move(slice));
          offset += sizes[output_i];
        }
        continue;
      }
      if (op_name == "ConvTranspose") {
        // Some reference-style exporters omit the `_1` suffix for the
        // primary tensor input.
        auto data_it = detail.find("input_tensor_info_1");
        if (data_it == detail.end()) data_it = detail.find("input_tensor_info");
        const auto weight_it = detail.find("input_weight_info");
        const auto output_it = detail.find("output_tensor_info");
        if (data_it == detail.end() || weight_it == detail.end() || output_it == detail.end()) {
          if (error) *error = "ConvTranspose requires input, weight and output tensors";
          return false;
        }
        AML_Tensor data = ParseTensor(*data_it, 0);
        const auto producer = known.find(data.name);
        if (!data.is_const && producer != known.end()) data = producer->second;
        AML_Tensor weight = ParseTensor(*weight_it, 1);
        AML_Tensor output = ParseTensor(*output_it, 2);
        if (output.dims.size() != 4) {
          if (error) *error = "ConvTranspose requires a rank-4 NHWC output tensor";
          return false;
        }
        AML_Tensor output_shape{};
        output_shape.name = output.name + "_transpose_conv_output_shape";
        output_shape.dtype = static_cast<int>(AML_TensorType::kInt32);
        output_shape.dims = {4};
        output_shape.is_const = true;
        for (const int dim : output.dims) {
          const auto* bytes = reinterpret_cast<const uint8_t*>(&dim);
          output_shape.data.insert(output_shape.data.end(), bytes, bytes + sizeof(dim));
        }
        const auto bias_it = detail.find("input_bias_info");
        const bool has_bias = bias_it != detail.end();
        AML_Tensor transpose_output = output;
        if (has_bias) transpose_output.name += "_transpose_conv_pre_bias";
        AML_Node transpose{};
        transpose.Node_type = static_cast<int>(AML_OperationType::kTransposeConv);
        transpose.Node_index = static_cast<int>(model->Node_List.size());
        // addNode_TRANSPOSE_CONV consumes these directly.  Unlike the
        // generic lowering below, this custom path must transfer attributes
        // explicitly; otherwise the zero default stride causes Compiler3 to
        // divide by zero while evaluating output shapes.
        const auto attributes_it = detail.find("attributes_info");
        if (attributes_it != detail.end()) {
          const auto& attributes = *attributes_it;
          transpose.params["stride_h"] = attributes.value("stride_h", 1);
          transpose.params["stride_w"] = attributes.value("stride_w", 1);
          const std::string padding = attributes.value("padding", "VALID");
          // AML uses 0=SAME, 1=VALID; the delegate maps this to ADLA's enum.
          transpose.params["padding"] = padding == "SAME" ? 0 : 1;
        } else {
          transpose.params["stride_h"] = 1;
          transpose.params["stride_w"] = 1;
          transpose.params["padding"] = 1;
        }
        transpose.params["activation"] = static_cast<int>(AML_FusedActivation::kNone);
        // TFLite TRANSPOSE_CONV ABI: output_shape, weights, input.
        transpose.inputs = {std::move(output_shape), std::move(weight), std::move(data)};
        transpose.outputs.push_back(transpose_output);
        model->op_codes.push_back("ConvTranspose");
        model->Node_List.push_back(std::move(transpose));
        if (has_bias) {
          AML_Tensor bias = ParseTensor(*bias_it, 3);
          if (bias.dims.size() == 1) bias.dims = {1, 1, 1, bias.dims[0]};
          AML_Node add{};
          add.Node_type = static_cast<int>(AML_OperationType::kAdd);
          add.Node_index = static_cast<int>(model->Node_List.size());
          add.inputs = {transpose_output, std::move(bias)};
          add.outputs.push_back(output);
          add.params["activation"] = static_cast<int>(AML_FusedActivation::kNone);
          model->op_codes.push_back("Add");
          model->Node_List.push_back(std::move(add));
        }
        known[output.name] = output;
        produced.insert(output.name);
        continue;
      }
      // Compiler3 v3.4.4 fails shape evaluation for rank-3 concatenation on
      // the last axis (YOLO's [N, boxes, channels] detection head).  Preserve
      // the logical operation by temporarily representing it as NHWC rank-4:
      // [N, boxes, C] -> [N, 1, boxes, C], concat(axis=3), then reshape back.
      if (op_name == "Concat") {
        std::vector<AML_Tensor> concat_inputs;
        int concat_tensor_index = 0;
        for (auto it = detail.begin(); it != detail.end(); ++it) {
          if (it.key().find("input_tensor_info") != 0) continue;
          auto tensor = ParseTensor(it.value(), concat_tensor_index++);
          const auto producer = known.find(tensor.name);
          if (!tensor.is_const && producer != known.end()) tensor = producer->second;
          concat_inputs.push_back(std::move(tensor));
        }
        const auto output_it = detail.find("output_tensor_info");
        const auto attributes_it = detail.find("attributes_info");
        const int axis = attributes_it == detail.end() ? 0 : attributes_it->value("axis", 0);
        if (output_it != detail.end() && concat_inputs.size() >= 2 &&
            output_it->at("tensor_shape").size() == 3 && axis == 2) {
          AML_Tensor output = ParseTensor(*output_it, 0);
          std::vector<AML_Tensor> lifted_inputs;
          for (auto& input : concat_inputs) {
            AML_Tensor lifted = input;
            lifted.name = input.name + "_concat4d_" + output.name;
            lifted.dims = {input.dims[0], 1, input.dims[1], input.dims[2]};
            AML_Node reshape{};
            reshape.Node_type = static_cast<int>(AML_OperationType::kReshape);
            reshape.Node_index = static_cast<int>(model->Node_List.size());
            reshape.inputs = {input};
            reshape.outputs = {lifted};
            model->op_codes.push_back("Reshape");
            model->Node_List.push_back(std::move(reshape));
            lifted_inputs.push_back(std::move(lifted));
          }
          AML_Tensor lifted_output = output;
          lifted_output.name = output.name + "_concat4d";
          lifted_output.dims = {output.dims[0], 1, output.dims[1], output.dims[2]};
          AML_Node concat{};
          concat.Node_type = static_cast<int>(op->second);
          concat.Node_index = static_cast<int>(model->Node_List.size());
          concat.inputs = std::move(lifted_inputs);
          concat.outputs = {lifted_output};
          concat.params["axis"] = 3;
          concat.params["activation"] = static_cast<int>(AML_FusedActivation::kNone);
          model->op_codes.push_back("Concat");
          model->Node_List.push_back(std::move(concat));
          AML_Node reshape_back{};
          reshape_back.Node_type = static_cast<int>(AML_OperationType::kReshape);
          reshape_back.Node_index = static_cast<int>(model->Node_List.size());
          reshape_back.inputs = {lifted_output};
          reshape_back.outputs = {output};
          model->op_codes.push_back("Reshape");
          model->Node_List.push_back(std::move(reshape_back));
          known[output.name] = output;
          produced.insert(output.name);
          continue;
        }
      }
      AML_Node node{};
      std::string emitted_op_name = op_name;
      node.Node_type = static_cast<int>(op->second);
      // AML_Node::Node_index is the index in the emitted AML subgraph, not
      // the source JSON position. Folding PAD removes source positions, so
      // keep the index dense for aml_compiler_core's dispatcher.
      node.Node_index = static_cast<int>(model->Node_List.size());
      // The reference-style JSON keeps Concat's axis inside attributes_info.
      // Populate it before generic attribute handling so rank-3 detection
      // heads concatenate their last dimension (5 + 80 = 85), rather than
      // using the delegate's zero default.
      if (op_name == "Concat") {
        const auto attributes_it = detail.find("attributes_info");
        if (attributes_it != detail.end()) {
          node.params["axis"] = attributes_it->value("axis", 0);
        }
      }
      int tensor_index = 0;
      bool node_fused_same_pad = false;
      std::vector<int> nchw_fc_source_dims;
      // JSON object ordering is not operator input ordering. In particular,
      // input_bias_info sorts before input_tensor_info. Preserve TFLite/AML
      // order: dynamic operands, then weight, then bias.
      for (auto it = detail.begin(); it != detail.end(); ++it) {
        if (it.key().find("input_tensor_info") != 0) continue;
        auto tensor = ParseTensor(it.value(), tensor_index++);
        // Tensor metadata emitted for a consumer can disagree with the
        // producer declaration (VGG16's final MaxPool -> Transpose is one
        // such reference-style export).  Tensor names define the edges in
        // aml_compiler_core, therefore the producer is authoritative.  Using
        // the consumer copy here can otherwise create a malformed duplicate
        // edge during Compiler3's ReduceDuplicates transform.
        if (!tensor.is_const) {
          const auto producer = known.find(tensor.name);
          if (producer != known.end()) tensor = producer->second;
        }
        const auto folded_transpose = folded_terminal_transpose_inputs.find(tensor.name);
        if (!tensor.is_const && folded_transpose != folded_terminal_transpose_inputs.end()) {
          tensor = folded_transpose->second;
        }
        const auto folded_nchw_transpose = folded_nchw_transpose_inputs.find(tensor.name);
        if (!tensor.is_const && folded_nchw_transpose != folded_nchw_transpose_inputs.end()) {
          tensor = folded_nchw_transpose->second;
        }
        const auto folded_fc = folded_nchw_fc_inputs.find(tensor.name);
        if (!tensor.is_const && folded_fc != folded_nchw_fc_inputs.end()) {
          nchw_fc_source_dims = folded_fc->second;
        }
        const auto folded_reshape = folded_identity_reshape_inputs.find(tensor.name);
        if (!tensor.is_const && folded_reshape != folded_identity_reshape_inputs.end()) {
          tensor = folded_reshape->second;
        }
        if (op_name == "QuantizeLinear" &&
            maxpool_quantize_sources.count(tensor.name) != 0) {
          tensor.dtype = static_cast<int>(AML_TensorType::kFloat32);
          tensor.zp.clear();
          tensor.scale.clear();
        }
        // ORT may retain the pre-quantization producer name (foo_pre_q)
        // while the following float operator refers to foo. This is an alias,
        // not a second graph input. Reconnect it before tensor-name based
        // edge construction in aml_compiler_core.
        if (!tensor.is_const && known.find(tensor.name) == known.end()) {
          const auto pre_q = known.find(tensor.name + "_pre_q");
          if (pre_q != known.end()) tensor = pre_q->second;
        }
        const auto fused_input = fused_same_pad_inputs.find(tensor.name);
        if ((op_name == "Conv" || op_name == "QLinearConv" ||
             op_name == "DepthwiseConv") &&
            fused_input != fused_same_pad_inputs.end()) {
          tensor = fused_input->second;
          node_fused_same_pad = true;
        }
        if (!tensor.is_const && known.find(tensor.name) == known.end() &&
            declared_graph_inputs.count(tensor.name) == 0 &&
            tensor_producer.count(tensor.name) == 0) {
          unresolved_inputs.push_back(op_name + " input has no producer: " + tensor.name);
        }
        known[tensor.name] = tensor;
        node.inputs.push_back(std::move(tensor));
      }
      for (const char* key : {"input_weight_info", "input_bias_info"}) {
        const auto it = detail.find(key);
        if (it == detail.end()) continue;
        auto tensor = ParseTensor(*it, tensor_index++);
        if (key == std::string("input_weight_info") && !nchw_fc_source_dims.empty()) {
          ReorderNchwFcWeightsToNhwc(&tensor, nchw_fc_source_dims[1],
                                     nchw_fc_source_dims[2], nchw_fc_source_dims[3]);
        }
        known[tensor.name] = tensor;
        node.inputs.push_back(std::move(tensor));
      }
      for (auto it = detail.begin(); it != detail.end(); ++it) if (it.key().find("output_tensor_info") == 0) {
        auto t = ParseTensor(it.value(), tensor_index++);
        const auto fused_relu6 = fused_relu6_outputs.find(i);
        if (fused_relu6 != fused_relu6_outputs.end()) t = fused_relu6->second;
        if (op_name == "MaxPool" && maxpool_quantize_sources.count(t.name) != 0) {
          t.dtype = static_cast<int>(AML_TensorType::kFloat32);
          t.zp.clear();
          t.scale.clear();
        }
        produced.insert(t.name); known[t.name] = t; node.outputs.push_back(t);
      }
      if (node.inputs.empty() || node.outputs.empty()) { if (error) *error = "op has no input or output tensors: " + op_name; return false; }
      AddParams(detail, &node);
      if (fused_relu6_outputs.count(i) != 0) {
        node.params["activation"] = static_cast<int>(AML_FusedActivation::kRelu6);
      }
      if (op_name == "DepthwiseConv" && node.inputs.size() >= 2 &&
          node.inputs[0].dims.size() == 4 && node.inputs[1].dims.size() == 4 &&
          node.outputs[0].dims.size() == 4 &&
          node.params["multiplier"] == 1) {
        // Compiler3 v3.4.4 exposes IDepthwiseConvOperator but its legacy
        // conversion path is not supported reliably.  For modest channel
        // counts, represent multiplier-1 depthwise convolution exactly as a
        // normal Conv with block-diagonal OHWI weights.
        AML_Tensor& weight = node.inputs[1];
        const int channels = node.inputs[0].dims[3];
        const int filter_h = weight.dims[1];
        const int filter_w = weight.dims[2];
        const size_t source_elements = static_cast<size_t>(filter_h) * filter_w * channels;
        if (channels > 0 && channels <= 128 && weight.dims[0] == 1 &&
            weight.dims[3] == channels && node.outputs[0].dims[3] == channels &&
            weight.data.size() == source_elements * sizeof(float)) {
          std::vector<uint8_t> dense(static_cast<size_t>(channels) * source_elements *
                                     sizeof(float), 0);
          for (int channel = 0; channel < channels; ++channel) {
            for (int h = 0; h < filter_h; ++h) {
              for (int w = 0; w < filter_w; ++w) {
                const size_t source_index =
                    (static_cast<size_t>(h) * filter_w + w) * channels + channel;
                const size_t dense_index =
                    ((static_cast<size_t>(channel) * filter_h + h) * filter_w + w) *
                        channels + channel;
                std::memcpy(dense.data() + dense_index * sizeof(float),
                            weight.data.data() + source_index * sizeof(float),
                            sizeof(float));
              }
            }
          }
          weight.dims = {channels, filter_h, filter_w, channels};
          weight.data.swap(dense);
          weight.size = weight.data.size();
          node.Node_type = static_cast<int>(AML_OperationType::kConv2D);
          emitted_op_name = "Conv";
        }
      }
      if (op_name == "MaxPool" && node.inputs.size() == 1 && node.outputs.size() == 1 &&
          node.inputs[0].dims.size() == 4 && node.outputs[0].dims == node.inputs[0].dims) {
        // Some ONNX reference-style exports retain the pre-pool shape on the
        // final MaxPool output.  Repair only this unmistakable stale-shape
        // case, using the already-exported TFLite pooling attributes.
        const int stride_h = node.params.count("stride_h") ? node.params["stride_h"] : 1;
        const int stride_w = node.params.count("stride_w") ? node.params["stride_w"] : 1;
        const int filter_h = node.params.count("filter_h") ? node.params["filter_h"] : 1;
        const int filter_w = node.params.count("filter_w") ? node.params["filter_w"] : 1;
        const int padding = node.params.count("padding") ? node.params["padding"] :
            static_cast<int>(AML_Padding::kValid);
        if (stride_h > 1 || stride_w > 1 || filter_h > 1 || filter_w > 1) {
          const auto pooled_dim = [padding](int input, int filter, int stride) {
            return padding == static_cast<int>(AML_Padding::kSame)
                       ? (input + stride - 1) / stride
                       : (input - filter) / stride + 1;
          };
          node.outputs[0].dims[1] = pooled_dim(node.inputs[0].dims[1], filter_h, stride_h);
          node.outputs[0].dims[2] = pooled_dim(node.inputs[0].dims[2], filter_w, stride_w);
          known[node.outputs[0].name] = node.outputs[0];
        }
      }
      // AddParams sees the original VALID attribute. Keep the semantically
      // equivalent native SAME padding selected by the fold above.
      if (node_fused_same_pad) {
        node.params["padding"] = static_cast<int>(AML_Padding::kSame);
      }
      if ((op_name == "Conv" || op_name == "QLinearConv" || op_name == "DepthwiseConv") && node.inputs.size() >= 3 &&
          node.inputs[2].dims.size() == 1) {
        // Compiler3's current Conv shape evaluator requires every input tensor
        // to be rank 4. Preserve C-major bias bytes while representing its
        // TFLite [C] shape as NHWC [1, 1, 1, C].
        node.inputs[2].dims = {1, 1, 1, node.inputs[2].dims[0]};
      }
      if ((op_name == "Add" || op_name == "Sub" || op_name == "Mul" || op_name == "Div") &&
          node.inputs.size() == 2) {
        // Compiler3 does not apply ONNX's trailing-dimension broadcast to a
        // rank-1 vector (e.g. attention bias [C] + activation [N,T,C]).
        // Make the broadcast dimensions explicit without moving any bytes.
        for (size_t input_i = 0; input_i < 2; ++input_i) {
          AML_Tensor& vector = node.inputs[input_i];
          const AML_Tensor& other = node.inputs[1 - input_i];
          if (vector.dims.size() != 1 || other.dims.size() < 2) continue;
          if (vector.dims[0] == 1) {
            // Avoid Compiler3's scalar broadcast path entirely. A constant
            // scalar can be materialized to the peer shape exactly, which is
            // semantically identical for elementwise operators and prevents
            // the HardSigmoid expansion from crashing in shape evaluation.
            if (!vector.data.empty()) {
              size_t element_count = 1;
              for (const int dim : other.dims) element_count *= static_cast<size_t>(dim);
              const std::vector<uint8_t> scalar_bytes = vector.data;
              vector.data.clear();
              vector.data.reserve(scalar_bytes.size() * element_count);
              for (size_t element_i = 0; element_i < element_count; ++element_i) {
                vector.data.insert(vector.data.end(), scalar_bytes.begin(), scalar_bytes.end());
              }
              vector.dims = other.dims;
              vector.size = vector.data.size();
            } else {
              vector.dims.assign(other.dims.size(), 1);
            }
            continue;
          }
          if (vector.dims[0] != other.dims.back()) continue;
          vector.dims.assign(other.dims.size(), 1);
          vector.dims.back() = other.dims.back();
        }
      }
      if (op_name == "Reshape" || op_name == "reshape") {
        if (node.inputs.empty()) { if (error) *error = "Reshape has no data input"; return false; }
        node.inputs.resize(1);  // shape is carried by the output AML tensor
      }
      if (op_name == "GlobalAveragePool" || op_name == "QLinearGlobalAveragePool") {
        const AML_Tensor& input = node.inputs.front();
        if (input.dims.size() != 4) { if (error) *error = "GlobalAveragePool requires NHWC rank-4 input: " + input.name; return false; }
        node.inputs.resize(1);  // drop the ONNX reduction-axis constant
        node.params["stride_h"] = input.dims[1];
        node.params["stride_w"] = input.dims[2];
        node.params["filter_h"] = input.dims[1];
        node.params["filter_w"] = input.dims[2];
        node.params["padding"] = static_cast<int>(AML_Padding::kValid);
        node.params["activation"] = static_cast<int>(AML_FusedActivation::kNone);
        if (op_name == "QLinearGlobalAveragePool" && !node.outputs.empty() &&
            !input.scale.empty() && !input.zp.empty()) {
          const AML_Tensor original_output = node.outputs.front();
          if (original_output.scale != input.scale || original_output.zp != input.zp) {
            AML_Tensor pool_output = original_output;
            pool_output.name += "_average_pool_input_domain";
            pool_output.scale = input.scale;
            pool_output.zp = input.zp;
            node.outputs.front() = pool_output;

            model->op_codes.push_back(emitted_op_name);
            model->Node_List.push_back(std::move(node));

            AML_Tensor float_output = original_output;
            float_output.name += "_average_pool_float";
            float_output.dtype = static_cast<int>(AML_TensorType::kFloat32);
            float_output.scale.clear();
            float_output.zp.clear();

            AML_Node dequant{};
            dequant.Node_type = static_cast<int>(AML_OperationType::kDequantize);
            dequant.Node_index = static_cast<int>(model->Node_List.size());
            dequant.inputs.push_back(pool_output);
            dequant.outputs.push_back(float_output);
            model->op_codes.push_back("DequantizeLinear");
            model->Node_List.push_back(std::move(dequant));

            AML_Node quant{};
            quant.Node_type = static_cast<int>(AML_OperationType::kQuantize);
            quant.Node_index = static_cast<int>(model->Node_List.size());
            quant.inputs.push_back(float_output);
            quant.outputs.push_back(original_output);
            model->op_codes.push_back("QuantizeLinear");
            model->Node_List.push_back(std::move(quant));

            std::cerr << "[aml_adla_bridge] QLinearGlobalAveragePool output "
                      << original_output.name
                      << " requantized through DQ/Q after AveragePool2D"
                      << std::endl;
            continue;
          }
        }
      }
      model->op_codes.push_back(emitted_op_name);
      model->Node_List.push_back(std::move(node));
    }
    if (!unresolved_inputs.empty()) {
      if (error) *error = "invalid reference-style graph: " + unresolved_inputs.front();
      return false;
    }
    if (root.is_object() && root.contains("graph_inputs")) for (const auto& name : root["graph_inputs"]) model->inputs.push_back(known.at(name.get<std::string>()));
    if (root.is_object() && root.contains("graph_outputs")) for (const auto& name : root["graph_outputs"]) model->outputs.push_back(known.at(name.get<std::string>()));
    if (model->inputs.empty()) for (const auto& t : model->Node_List.front().inputs) if (!t.is_const) model->inputs.push_back(t);
    if (model->outputs.empty()) model->outputs = model->Node_List.back().outputs;
    return true;
  } catch (const std::exception& ex) { if (error) *error = ex.what(); return false; }
}
}  // namespace amlogic::adla_bridge
