// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_execution_provider.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "core/common/common.h"
#include "core/common/make_string.h"
#include "core/framework/allocator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/node_unit.h"
#include "core/graph/constants.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/amlogic/amlogic_model_info.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace {

bool ParseBoolProviderOption(const ProviderOptions& options, const char* key, bool default_value) {
  const auto it = options.find(key);
  if (it == options.end()) {
    return default_value;
  }

  std::string value = it->second;
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    return false;
  }

  ORT_THROW("Invalid boolean value for Amlogic provider option '", key, "': ", it->second);
  return default_value;
}

std::unordered_set<std::string> ParseStopOps(const std::string& value) {
  std::unordered_set<std::string> stop_ops;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
                 return !std::isspace(ch);
               }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
                 return !std::isspace(ch);
               }).base(),
               item.end());
    if (!item.empty()) {
      stop_ops.insert(std::move(item));
    }
  }
  return stop_ops;
}

bool IsSupportedDomain(std::string_view domain) {
  return domain == kOnnxDomain || domain == kOnnxDomainAlias || domain == kMSInternalNHWCDomain;
}

bool IsLayoutSensitiveOp(std::string_view op_type) {
  static const std::unordered_set<std::string_view> layout_sensitive_ops = {
      "AveragePool",
      "BatchNormalization",
      "Conv",
      "ConvTranspose",
      "DepthToSpace",
      "GlobalAveragePool",
      "GlobalMaxPool",
      "InstanceNormalization",
      "LRN",
      "MaxPool",
      "Resize",
      "SpaceToDepth",
      "Upsample",
  };

  return layout_sensitive_ops.find(op_type) != layout_sensitive_ops.end();
}

bool IsSupportedOp(std::string_view domain, std::string_view op_type) {
  if (!IsSupportedDomain(domain)) {
    return false;
  }

  static const std::unordered_set<std::string_view> supported_ops = {
      "Abs",
      "Add",
      "AveragePool",
      "BatchNormalization",
      "Clip",
      "Concat",
      "Conv",
      "ConvTranspose",
      "DequantizeLinear",
      "Div",
      "Flatten",
      "Gather",
      "Gemm",
      "GlobalAveragePool",
      "GlobalMaxPool",
      "HardSigmoid",
      "LeakyRelu",
      "MatMul",
      "Max",
      "MaxPool",
      "Min",
      "Mul",
      "Pad",
      "QuantizeLinear",
      "ReduceMean",
      "Relu",
      "Reshape",
      "Resize",
      "Sigmoid",
      "Softmax",
      "Split",
      "Sub",
      "Tanh",
      "Transpose",
      "Unsqueeze",
      "Upsample",
  };

  return supported_ops.find(op_type) != supported_ops.end();
}

bool HasStaticTensorShape(const NodeArg& node_arg) {
  const auto* shape = node_arg.Shape();
  if (shape == nullptr) {
    return false;
  }

  for (const auto& dim : shape->dim()) {
    if (!dim.has_dim_value()) {
      return false;
    }
  }

  return true;
}

bool IsSupportedTensorType(const NodeArg& node_arg, bool enable_fp16) {
  const auto* type_proto = node_arg.TypeAsProto();
  if (type_proto == nullptr || !type_proto->has_tensor_type()) {
    return false;
  }

  const auto elem_type = type_proto->tensor_type().elem_type();
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return true;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return enable_fp16;
    default:
      return false;
  }
}

bool HasOnlyConstantInputs(const GraphViewer& graph_viewer, const IndexedSubGraph::MetaDef& meta_def) {
  return std::none_of(meta_def.inputs.begin(), meta_def.inputs.end(), [&graph_viewer](const std::string& input) {
    return !graph_viewer.IsConstantInitializer(input, true);
  });
}

}  // namespace

AmlogicExecutionProviderInfo AmlogicExecutionProviderInfo::FromProviderOptions(
    const ProviderOptions& provider_options) {
  AmlogicExecutionProviderInfo info;

  if (auto it = provider_options.find("device_id"); it != provider_options.end()) {
    info.device_id = std::stoi(it->second);
  }
  info.enable_fp16 = ParseBoolProviderOption(provider_options, "enable_fp16", info.enable_fp16);
  info.require_static_shapes =
      ParseBoolProviderOption(provider_options, "require_static_shapes", info.require_static_shapes);
  info.dump_model_info = ParseBoolProviderOption(provider_options, "dump_model_info", info.dump_model_info);

  if (auto it = provider_options.find("min_subgraph_size"); it != provider_options.end()) {
    const int parsed_value = std::stoi(it->second);
    ORT_ENFORCE(parsed_value > 0, "Amlogic provider option 'min_subgraph_size' must be positive.");
    info.min_subgraph_size = static_cast<size_t>(parsed_value);
  }

  if (auto it = provider_options.find("dump_model_info_path"); it != provider_options.end()) {
    info.dump_model_info_path = it->second;
  }

  if (auto it = provider_options.find("partitioning_stop_ops"); it != provider_options.end()) {
    info.partitioning_stop_ops = ParseStopOps(it->second);
  }

  return info;
}

ProviderOptions AmlogicExecutionProviderInfo::ToProviderOptions() const {
  ProviderOptions options;
  options["device_id"] = std::to_string(device_id);
  options["enable_fp16"] = enable_fp16 ? "1" : "0";
  options["require_static_shapes"] = require_static_shapes ? "1" : "0";
  options["min_subgraph_size"] = std::to_string(min_subgraph_size);
  options["dump_model_info"] = dump_model_info ? "1" : "0";
  options["dump_model_info_path"] = dump_model_info_path;
  if (!partitioning_stop_ops.empty()) {
    std::string value;
    for (const auto& op : partitioning_stop_ops) {
      if (!value.empty()) {
        value += ",";
      }
      value += op;
    }
    options["partitioning_stop_ops"] = std::move(value);
  }

  return options;
}

AmlogicExecutionProvider::AmlogicExecutionProvider(const AmlogicExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kAmlogicExecutionProvider,
                         OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT,
                                   DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtDevice::VendorIds::NONE,
                                   kAlloc4KAlignment)},
      info_(info),
      device_id_(info.device_id),
      backend_(std::make_shared<amlogic::Backend>()) {
}

std::vector<std::unique_ptr<ComputeCapability>>
AmlogicExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                        const IKernelLookup& /*kernel_lookup*/,
                                        const GraphOptimizerRegistry& /*graph_optimizer_registry*/,
                                        IResourceAccountant* /*resource_accountant*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  const auto& logger = *GetLogger();

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  if (info_.dump_model_info) {
    std::lock_guard<std::mutex> lock(model_info_dump_mutex_);
    if (!model_info_dumped_) {
      const auto dump_status = amlogic::DumpGraphInfoAsJson(graph_viewer, info_.dump_model_info_path, logger);
      if (!dump_status.IsOK()) {
        LOGS(logger, WARNING) << "Failed to dump Amlogic ONNX op info: " << dump_status.ErrorMessage();
      }
      model_info_dumped_ = true;
    }
  }

  if (!backend_->IsAvailable()) {
    LOGS(logger, WARNING) << "Amlogic NPU backend is not available. All nodes will fall back to lower priority EPs.";
    return result;
  }

  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() &&
        tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS(logger, WARNING) << "Amlogic NPU EP does not currently support external initializers.";
      return result;
    }
  }

  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer, logger);

  std::unordered_map<const NodeUnit*, bool> node_unit_supported_result;
  node_unit_supported_result.reserve(node_unit_holder.size());

  const auto excluded_nodes = utils::CreateExcludedNodeSet(graph_viewer, info_.partitioning_stop_ops);
  const bool check_excluded_nodes = !excluded_nodes.empty();

  const auto is_node_supported = [&](const Node& node) -> bool {
    const NodeUnit* node_unit = node_unit_map.at(&node);

    const auto cached = node_unit_supported_result.find(node_unit);
    if (cached != node_unit_supported_result.end()) {
      return cached->second;
    }

    const bool excluded = check_excluded_nodes && excluded_nodes.find(&node_unit->GetNode()) != excluded_nodes.end();
    const bool supported = !excluded && IsNodeSupported(*node_unit, graph_viewer);
    node_unit_supported_result[node_unit] = supported;

    LOGS(logger, VERBOSE) << "Amlogic NPU node support: " << supported
                          << ", op_type: " << node_unit->OpType()
                          << ", node_index: " << node_unit->Index()
                          << ", node_name: " << node_unit->Name();
    return supported;
  };

  const auto on_group_closed = [this](const std::vector<const Node*>& group) -> bool {
    return group.size() >= info_.min_subgraph_size;
  };

  const auto gen_metadef_name = [&]() {
    HashValue model_hash;
    int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
    return MakeString("Amlogic_", model_hash, "_", metadef_id);
  };

  result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported, on_group_closed,
                                            gen_metadef_name, "Amlogic", kAmlogicExecutionProvider,
                                            &node_unit_map, true);

  std::for_each(result.begin(), result.end(), [&graph_viewer](auto& capability) {
    if (capability && capability->sub_graph && capability->sub_graph->GetMetaDef()) {
      if (HasOnlyConstantInputs(graph_viewer, *capability->sub_graph->GetMetaDef())) {
        capability.reset();
      }
    }
  });

  const auto num_partitions = std::count_if(result.begin(), result.end(), [](const auto& capability) {
    return capability != nullptr;
  });
  const auto num_supported_nodes = std::accumulate(
      result.begin(), result.end(), size_t{0},
      [](const auto& acc, const auto& partition) -> size_t {
        return acc + (partition && partition->sub_graph ? partition->sub_graph->nodes.size() : 0);
      });

  const auto summary_msg = MakeString(
      "AmlogicExecutionProvider::GetCapability, partitions: ", num_partitions,
      "; nodes in graph: ", graph_viewer.NumberOfNodes(),
      "; supported nodes: ", num_supported_nodes);
  if (num_partitions > 1) {
    LOGS(logger, WARNING) << summary_msg;
  } else {
    LOGS(logger, INFO) << summary_msg;
  }

  return result;
}

bool AmlogicExecutionProvider::IsNodeSupported(const NodeUnit& node_unit, const GraphViewer& /*graph_viewer*/) const {
  const Node& node = node_unit.GetNode();
  if (!IsSupportedOp(node.Domain(), node.OpType())) {
    return false;
  }

  const auto validate_arg = [this](const NodeArg* node_arg) {
    if (node_arg == nullptr || !node_arg->Exists()) {
      return true;
    }
    if (!IsSupportedTensorType(*node_arg, info_.enable_fp16)) {
      return false;
    }
    if (info_.require_static_shapes && !HasStaticTensorShape(*node_arg)) {
      return false;
    }
    return true;
  };

  return std::all_of(node.InputDefs().begin(), node.InputDefs().end(), validate_arg) &&
         std::all_of(node.OutputDefs().begin(), node.OutputDefs().end(), validate_arg);
}

Status AmlogicExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                         std::vector<NodeComputeInfo>& node_compute_funcs) {
  node_compute_funcs.reserve(node_compute_funcs.size() + fused_nodes_and_graphs.size());

  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    std::unique_ptr<amlogic::CompiledGraph> compiled_graph;
    ORT_RETURN_IF_ERROR(backend_->Compile(graph_viewer, fused_node.Name(), compiled_graph));
    ORT_RETURN_IF_NOT(compiled_graph != nullptr, "Amlogic backend returned a null compiled graph.");

    std::shared_ptr<amlogic::CompiledGraph> graph_state(std::move(compiled_graph));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [graph_state](ComputeContext* /*context*/, FunctionState* state) {
      *state = graph_state.get();
      return 0;
    };
    compute_info.release_state_func = [](FunctionState /*state*/) {};
    compute_info.compute_func = [graph_state, this](FunctionState /*state*/,
                                                    const OrtApi* /*api*/,
                                                    OrtKernelContext* context) {
      std::lock_guard<std::mutex> lock(compute_mutex_);
      Ort::KernelContext ort_context(context);
      return graph_state->Run(ort_context, *GetLogger());
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> AmlogicExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  return kernel_registry;
}

std::optional<bool> AmlogicExecutionProvider::ShouldConvertDataLayoutForOp(std::string_view domain,
                                                                           std::string_view op_type,
                                                                           DataLayout target_data_layout) const {
  if (target_data_layout != DataLayout::NHWC || !IsSupportedDomain(domain)) {
    return std::nullopt;
  }

  if (IsLayoutSensitiveOp(op_type)) {
    return true;
  }

  return std::nullopt;
}

std::vector<AllocatorPtr> AmlogicExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> result;
  constexpr const bool use_arena = false;
  AllocatorCreationInfo device_info_cpu_aligned_4k{
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(onnxruntime::CPU_ALIGNED_4K, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE,
                                    device_id, kAlloc4KAlignment)));
      },
      device_id_, use_arena};

  result.push_back(CreateAllocator(device_info_cpu_aligned_4k));
  return result;
}

ProviderOptions AmlogicExecutionProvider::GetProviderOptions() const {
  return info_.ToProviderOptions();
}

}  // namespace onnxruntime
