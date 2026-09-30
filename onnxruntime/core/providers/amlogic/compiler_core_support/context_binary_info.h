/*******************************************************************************
 * Copyright (C) 2023 Amlogic, Inc. All rights reserved.
 *
 * @file    context_binary_info.h
 * @module  aml_compiler_core
 * @brief   Internal compiler IR (AML_Model / AML_Node / AML_Tensor) + serialize.
 * @note    TFLite-aligned enums (BuiltinOperator / TfLiteType). Not a stable
 *          cross-DSO ABI — exchange SerializeModel() bytes across modules.
 ******************************************************************************/

#ifndef CONTEXT_BINARY_INFO_H
#define CONTEXT_BINARY_INFO_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Tensor in the AML compiler IR.
 * @note @c dtype uses an AML_TensorType numeric value.
 *
 * @var name          Tensor name.
 * @var dims          Shape dimensions.
 * @var dtype         AML_TensorType numeric value.
 * @var zp            Zero-point(s); per-channel when size > 1.
 * @var scale         Scale(s); per-channel when size > 1.
 * @var is_const      True if constant / weight tensor.
 * @var tensor_index  Original tensor index in the source graph.
 * @var data          Inline constant payload (optional).
 * @var size          Byte size of constant payload.
 * @var ptr_data      Shared ownership of external constant bytes (optional).
 */
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

/**
 * @brief Operator node in the AML compiler IR.
 * @note @c Node_type uses an AML_OperationType numeric value.
 *       @c params["activation"] uses an AML_FusedActivation numeric value.
 * @note See the complete per-operator params/fparams string table at the
 *       bottom of this header.
 *
 * @var Node_type  BuiltinOperator numeric value.
 * @var Node_index Node index in the partition / subgraph.
 * @var inputs     Input tensors of this node.
 * @var outputs    Output tensors of this node.
 * @var params     Integer attributes (axis, padding, activation, ...).
 * @var fparams    Floating attributes (alpha, beta, ...).
 */
struct AML_Node {
  int Node_type;
  int Node_index;
  std::vector<AML_Tensor> inputs;
  std::vector<AML_Tensor> outputs;
  std::map<std::string, int> params;
  std::map<std::string, float> fparams;
};

/**
 * @brief One AML IR graph / partition ready for convert + compile.
 *
 * @var target      Optional ADLA target hint.
 * @var graph_names Graph / subgraph name list.
 * @var inputs      Graph input tensors.
 * @var outputs     Graph output tensors.
 * @var op_codes    Optional op-code name table.
 * @var Node_List   Topologically ordered operator nodes.
 */
struct AML_Model {
  std::string target;
  std::vector<std::string> graph_names;
  std::vector<AML_Tensor> inputs;
  std::vector<AML_Tensor> outputs;
  std::vector<std::string> op_codes;
  std::vector<AML_Node> Node_List;
};

/**
 * @brief Serialize AML_Model to a portable byte string for cross-module use.
 * @param model IR model to serialize.
 * @return Opaque byte buffer (not human-readable text).
 */
std::string SerializeModel(const AML_Model& model);

/**
 * @brief Deserialize a buffer produced by SerializeModel().
 * @param buffer Serialized bytes.
 * @return Reconstructed AML_Model.
 */
AML_Model DeserializeModel(const std::string& buffer);

/** @brief Log one AML_Tensor (metadata + payload size; no weight bytes). */
void PrintAML_Tensor(const AML_Tensor& tensor, const char* indent = "");

/** @brief Log one AML_Node (op type, index, params, I/O tensors). */
void PrintAML_Node(const AML_Node& node, const char* indent = "");

/** @brief Log a full AML_Model (graph I/O, op_codes, all nodes). */
void PrintAML_Model(const AML_Model& model, int graph_index = 0);

/**
 * @brief AML tensor data types aligned numerically with TFLite TfLiteType.
 * @note AML-prefixed names avoid collisions with TFLite headers.
 */
enum class AML_TensorType : int {
  kNoType = 0,
  kFloat32 = 1,
  kInt32 = 2,
  kUInt8 = 3,
  kInt64 = 4,
  kString = 5,
  kBool = 6,
  kInt16 = 7,
  kComplex64 = 8,
  kInt8 = 9,
  kFloat16 = 10,
  kFloat64 = 11,
  kComplex128 = 12,
  kUInt64 = 13,
  kResource = 14,
  kVariant = 15,
  kUInt32 = 16,
  kUInt16 = 17,
  kInt4 = 18,
  kBFloat16 = 19,
  kInt2 = 20,
  kUInt4 = 21,
};

/** @brief AML fused activations aligned with TFLite TfLiteFusedActivation. */
enum class AML_FusedActivation : int {
  kNone = 0,
  kRelu = 1,
  kReluN1To1 = 2,
  kRelu6 = 3,
  kTanh = 4,
  kSignBit = 5,
  kSigmoid = 6,
};

/**
 * @brief AML padding values aligned with the TFLite schema Padding enum.
 * @note These are schema values used by AML_Node::params["padding"], not the
 *       TfLitePadding C API values.
 */
enum class AML_Padding : int {
  kSame = 0,
  kValid = 1,
};

/**
 * @brief AML operator types aligned numerically with TFLite BuiltinOperator.
 * @note Every value is explicit so later edits cannot shift serialized IDs.
 */
enum class AML_OperationType : uint32_t {
  kAdd = 0,
  kAveragePool2D = 1,
  kConcatenation = 2,
  kConv2D = 3,
  kDepthwiseConv2D = 4,
  kDepthToSpace = 5,
  kDequantize = 6,
  kEmbeddingLookup = 7,
  kFloor = 8,
  kFullyConnected = 9,
  kHashtableLookup = 10,
  kL2Normalization = 11,
  kL2Pool2D = 12,
  kLocalResponseNormalization = 13,
  kLogistic = 14,
  kLshProjection = 15,
  kLstm = 16,
  kMaxPool2D = 17,
  kMul = 18,
  kRelu = 19,
  kReluN1To1 = 20,
  kRelu6 = 21,
  kReshape = 22,
  kResizeBilinear = 23,
  kRnn = 24,
  kSoftmax = 25,
  kSpaceToDepth = 26,
  kSvdf = 27,
  kTanh = 28,
  kConcatEmbeddings = 29,
  kSkipGram = 30,
  kCall = 31,
  kCustom = 32,
  kEmbeddingLookupSparse = 33,
  kPad = 34,
  kUnidirectionalSequenceRnn = 35,
  kGather = 36,
  kBatchToSpaceNd = 37,
  kSpaceToBatchNd = 38,
  kTranspose = 39,
  kMean = 40,
  kSub = 41,
  kDiv = 42,
  kSqueeze = 43,
  kUnidirectionalSequenceLstm = 44,
  kStridedSlice = 45,
  kBidirectionalSequenceRnn = 46,
  kExp = 47,
  kTopkV2 = 48,
  kSplit = 49,
  kLogSoftmax = 50,
  kDelegate = 51,
  kBidirectionalSequenceLstm = 52,
  kCast = 53,
  kPrelu = 54,
  kMaximum = 55,
  kArgMax = 56,
  kMinimum = 57,
  kLess = 58,
  kNeg = 59,
  kPadV2 = 60,
  kGreater = 61,
  kGreaterEqual = 62,
  kLessEqual = 63,
  kSelect = 64,
  kSlice = 65,
  kSin = 66,
  kTransposeConv = 67,
  kSparseToDense = 68,
  kTile = 69,
  kExpandDims = 70,
  kEqual = 71,
  kNotEqual = 72,
  kLog = 73,
  kSum = 74,
  kSqrt = 75,
  kRsqrt = 76,
  kShape = 77,
  kPow = 78,
  kArgMin = 79,
  kFakeQuant = 80,
  kReduceProd = 81,
  kReduceMax = 82,
  kPack = 83,
  kLogicalOr = 84,
  kOneHot = 85,
  kLogicalAnd = 86,
  kLogicalNot = 87,
  kUnpack = 88,
  kReduceMin = 89,
  kFloorDiv = 90,
  kReduceAny = 91,
  kSquare = 92,
  kZerosLike = 93,
  kFill = 94,
  kFloorMod = 95,
  kRange = 96,
  kResizeNearestNeighbor = 97,
  kLeakyRelu = 98,
  kSquaredDifference = 99,
  kMirrorPad = 100,
  kAbs = 101,
  kSplitV = 102,
  kUnique = 103,
  kCeil = 104,
  kReverseV2 = 105,
  kAddN = 106,
  kGatherNd = 107,
  kCos = 108,
  kWhere = 109,
  kRank = 110,
  kElu = 111,
  kReverseSequence = 112,
  kMatrixDiag = 113,
  kQuantize = 114,
  kMatrixSetDiag = 115,
  kRound = 116,
  kHardSwish = 117,
  kIf = 118,
  kWhile = 119,
  kNonMaxSuppressionV4 = 120,
  kNonMaxSuppressionV5 = 121,
  kScatterNd = 122,
  kSelectV2 = 123,
  kDensify = 124,
  kSegmentSum = 125,
  kBatchMatmul = 126,
  kPlaceholderForGreaterOpCodes = 127,
  kCumsum = 128,
  kCallOnce = 129,
  kBroadcastTo = 130,
  kRfft2D = 131,
  kConv3D = 132,
  kImag = 133,
  kReal = 134,
  kComplexAbs = 135,
  kHashtable = 136,
  kHashtableFind = 137,
  kHashtableImport = 138,
  kHashtableSize = 139,
  kReduceAll = 140,
  kConv3DTranspose = 141,
  kVarHandle = 142,
  kReadVariable = 143,
  kAssignVariable = 144,
  kBroadcastArgs = 145,
  kRandomStandardNormal = 146,
  kBucketize = 147,
  kRandomUniform = 148,
  kMultinomial = 149,
  kGelu = 150,
  kDynamicUpdateSlice = 151,
};

/**
 * @brief String attributes currently consumed by addNode_* converters.
 *
 * A dash means that the current converter does not read AML_Node::params or
 * AML_Node::fparams. Keep this table synchronized with tflite_to_convert.cc.
 *
 * Value conventions:
 * - activation: AML_FusedActivation numeric value.
 * - padding: AML_Padding numeric value.
 * - pot_scale_int16, keep_num_dims, keep_dims, align_corners,
 *   half_pixel_centers, offset, adj_x, adj_y, asymmetric_quantize_inputs and
 *   approximate: boolean encoded as 0 or 1.
 * - mode: TFLite MirrorPadMode (REFLECT=0, SYMMETRIC=1).
 * - stride_*, dilation_*, filter_*, axis, batch_dims, multiplier, block_size,
 *   masks, num_split, values_count and num: integer operator attributes.
 * - alpha and beta: floating-point operator attributes in fparams.
 *
 * @code
 * Operator                         params strings                         fparams strings
 * ADD                              activation, pot_scale_int16            -
 * AVERAGE_POOL_2D                  activation, filter_h, filter_w,         -
 *                                  padding, stride_h, stride_w
 * CONCATENATION                    activation, axis                       -
 * CONV_2D                          activation, dilation_h, dilation_w,     -
 *                                  padding, stride_h, stride_w
 * DEPTHWISE_CONV_2D                activation, dilation_h, dilation_w,     -
 *                                  multiplier, padding, stride_h, stride_w
 * DEPTH_TO_SPACE                   block_size                             -
 * DEQUANTIZE                       -                                      -
 * EMBEDDING_LOOKUP                 -                                      -
 * FLOOR                            -                                      -
 * FULLY_CONNECTED                  activation, keep_num_dims              -
 * HASHTABLE_LOOKUP                 -                                      -
 * L2_NORMALIZATION                 -                                      -
 * L2_POOL_2D                       -                                      -
 * LOCAL_RESPONSE_NORMALIZATION     -                                      -
 * LOGISTIC                         -                                      -
 * LSH_PROJECTION                   -                                      -
 * LSTM                             -                                      -
 * MAX_POOL_2D                      activation, filter_h, filter_w,         -
 *                                  padding, stride_h, stride_w
 * MUL                              activation                             -
 * RELU                             -                                      -
 * RELU_N1_TO_1                     -                                      -
 * RELU6                            -                                      -
 * RESHAPE                          -                                      -
 * RESIZE_BILINEAR                  align_corners, half_pixel_centers       -
 * RNN                              -                                      -
 * SOFTMAX                          -                                      beta
 * SPACE_TO_DEPTH                   block_size                             -
 * SVDF                             -                                      -
 * TANH                             -                                      -
 * CONCAT_EMBEDDINGS                -                                      -
 * SKIP_GRAM                        -                                      -
 * CALL                             -                                      -
 * CUSTOM                           -                                      -
 * EMBEDDING_LOOKUP_SPARSE          -                                      -
 * PAD                              -                                      -
 * UNIDIRECTIONAL_SEQUENCE_RNN      -                                      -
 * GATHER                           axis, batch_dims                       -
 * BATCH_TO_SPACE_ND                -                                      -
 * SPACE_TO_BATCH_ND                -                                      -
 * TRANSPOSE                        -                                      -
 * MEAN                             keep_dims                              -
 * SUB                              activation, pot_scale_int16            -
 * DIV                              activation                             -
 * SQUEEZE                          -                                      -
 * UNIDIRECTIONAL_SEQUENCE_LSTM     -                                      -
 * STRIDED_SLICE                    begin_mask, ellipsis_mask, end_mask,    -
 *                                  new_axis_mask, offset, shrink_axis_mask
 * BIDIRECTIONAL_SEQUENCE_RNN       -                                      -
 * EXP                              -                                      -
 * TOPK_V2                          -                                      -
 * SPLIT                            num_split                              -
 * LOG_SOFTMAX                      -                                      -
 * DELEGATE                         -                                      -
 * BIDIRECTIONAL_SEQUENCE_LSTM      -                                      -
 * CAST                             -                                      -
 * PRELU                            -                                      -
 * MAXIMUM                          -                                      -
 * ARGMAX                           -                                      -
 * MINIMUM                          -                                      -
 * LESS                             -                                      -
 * NEG                              -                                      -
 * PADV2                            -                                      -
 * GREATER                          -                                      -
 * GREATER_EQUAL                    -                                      -
 * LESS_EQUAL                       -                                      -
 * SELECT                           -                                      -
 * SLICE                            -                                      -
 * SIN                              -                                      -
 * TRANSPOSE_CONV                   activation, padding, stride_h, stride_w -
 * SPARSE_TO_DENSE                  -                                      -
 * TILE                             -                                      -
 * EXPAND_DIMS                      -                                      -
 * EQUAL                            -                                      -
 * NOT_EQUAL                        -                                      -
 * LOG                              -                                      -
 * SUM                              keep_dims                              -
 * SQRT                             -                                      -
 * RSQRT                            -                                      -
 * SHAPE                            -                                      -
 * POW                              -                                      -
 * ARG_MIN                          -                                      -
 * FAKE_QUANT                       -                                      -
 * REDUCE_PROD                      -                                      -
 * REDUCE_MAX                       keep_dims                              -
 * PACK                             axis, values_count                     -
 * LOGICAL_OR                       -                                      -
 * ONE_HOT                          -                                      -
 * LOGICAL_AND                      -                                      -
 * LOGICAL_NOT                      -                                      -
 * UNPACK                           axis, num                              -
 * REDUCE_MIN                       keep_dims                              -
 * FLOOR_DIV                        -                                      -
 * REDUCE_ANY                       keep_dims                              -
 * SQUARE                           -                                      -
 * ZEROS_LIKE                       -                                      -
 * FILL                             -                                      -
 * FLOOR_MOD                        -                                      -
 * RANGE                            -                                      -
 * RESIZE_NEAREST_NEIGHBOR          align_corners, half_pixel_centers       -
 * LEAKY_RELU                       -                                      alpha
 * SQUARED_DIFFERENCE               -                                      -
 * MIRROR_PAD                       mode                                   -
 * ABS                              -                                      -
 * SPLIT_V                          -                                      -
 * UNIQUE                           -                                      -
 * CEIL                             -                                      -
 * REVERSE_V2                       -                                      -
 * ADD_N                            -                                      -
 * GATHER_ND                        -                                      -
 * COS                              -                                      -
 * WHERE                            -                                      -
 * RANK                             -                                      -
 * ELU                              -                                      -
 * REVERSE_SEQUENCE                 -                                      -
 * MATRIX_DIAG                      -                                      -
 * QUANTIZE                         -                                      -
 * MATRIX_SET_DIAG                  -                                      -
 * ROUND                            -                                      -
 * HARD_SWISH                       -                                      -
 * IF                               -                                      -
 * WHILE                            -                                      -
 * NON_MAX_SUPPRESSION_V4           -                                      -
 * NON_MAX_SUPPRESSION_V5           -                                      -
 * SCATTER_ND                       -                                      -
 * SELECT_V2                        -                                      -
 * DENSIFY                          -                                      -
 * SEGMENT_SUM                      -                                      -
 * BATCH_MATMUL                     adj_x, adj_y, asymmetric_quantize_inputs -
 * PLACEHOLDER_FOR_GREATER_OP_CODES -                                      -
 * CUMSUM                           -                                      -
 * CALL_ONCE                        -                                      -
 * BROADCAST_TO                     -                                      -
 * RFFT2D                           -                                      -
 * CONV_3D                          -                                      -
 * IMAG                             -                                      -
 * REAL                             -                                      -
 * COMPLEX_ABS                      -                                      -
 * HASHTABLE                        -                                      -
 * HASHTABLE_FIND                   -                                      -
 * HASHTABLE_IMPORT                 -                                      -
 * HASHTABLE_SIZE                   -                                      -
 * REDUCE_ALL                       keep_dims                              -
 * CONV_3D_TRANSPOSE                -                                      -
 * VAR_HANDLE                       -                                      -
 * READ_VARIABLE                    -                                      -
 * ASSIGN_VARIABLE                  -                                      -
 * BROADCAST_ARGS                   -                                      -
 * RANDOM_STANDARD_NORMAL           -                                      -
 * BUCKETIZE                        -                                      -
 * RANDOM_UNIFORM                   -                                      -
 * MULTINOMIAL                      -                                      -
 * GELU                             approximate                            -
 * DYNAMIC_UPDATE_SLICE             -                                      -
 * @endcode
 */

#endif  // CONTEXT_BINARY_INFO_H
