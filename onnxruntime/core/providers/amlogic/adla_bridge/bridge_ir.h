#pragma once

// Scheme 5 owns this small, translation-only graph representation.  It is
// deliberately independent of aml_compiler_core internal headers.  The
// numeric values match the public compiler_core_context.h schema because the
// final emitter passes them to AmlNodeCreate().

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

enum class AML_TensorType : int {
  kNoType = 0,
  kFloat32 = 1,
  kInt32 = 2,
  kUInt8 = 3,
  kInt64 = 4,
  kBool = 6,
  kInt16 = 7,
  kInt8 = 9,
  kFloat16 = 10,
};

enum class AML_FusedActivation : int {
  kNone = 0,
  kRelu = 1,
  kReluN1To1 = 2,
  kRelu6 = 3,
};

enum class AML_Padding : int {
  kSame = 0,
  kValid = 1,
};

enum class AML_OperationType : uint32_t {
  kAdd = 0,
  kAveragePool2D = 1,
  kConcatenation = 2,
  kConv2D = 3,
  kDepthwiseConv2D = 4,
  kDequantize = 6,
  kFullyConnected = 9,
  kLocalResponseNormalization = 13,
  kLogistic = 14,
  kMaxPool2D = 17,
  kMul = 18,
  kRelu = 19,
  kReluN1To1 = 20,
  kRelu6 = 21,
  kReshape = 22,
  kResizeBilinear = 23,
  kSoftmax = 25,
  kPad = 34,
  kTranspose = 39,
  kMean = 40,
  kSub = 41,
  kDiv = 42,
  kStridedSlice = 45,
  kExp = 47,
  kSlice = 65,
  kTransposeConv = 67,
  kLog = 73,
  kSum = 74,
  kSqrt = 75,
  kRsqrt = 76,
  kPow = 78,
  kReduceMax = 82,
  kResizeNearestNeighbor = 97,
  kLeakyRelu = 98,
  kMirrorPad = 100,
  kSplitV = 102,
  kQuantize = 114,
  kHardSwish = 117,
  kBroadcastTo = 130,
  kTanh = 28,
};

struct AML_Tensor {
  std::string name;
  std::vector<int> dims;
  int dtype = 0;
  std::vector<int> zp;
  std::vector<float> scale;
  bool is_const = false;
  int tensor_index = 0;
  std::vector<uint8_t> data;
  size_t size = 0;
  std::shared_ptr<uint8_t> ptr_data;
};

struct AML_Node {
  int Node_type = 0;
  int Node_index = 0;
  std::vector<AML_Tensor> inputs;
  std::vector<AML_Tensor> outputs;
  std::map<std::string, int> params;
  std::map<std::string, float> fparams;
};

struct AML_Model {
  std::string target;
  std::vector<std::string> graph_names;
  std::vector<AML_Tensor> inputs;
  std::vector<AML_Tensor> outputs;
  std::vector<std::string> op_codes;
  std::vector<AML_Node> Node_List;
};
