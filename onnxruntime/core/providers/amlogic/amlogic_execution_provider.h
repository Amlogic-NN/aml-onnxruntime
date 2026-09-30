// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/providers/amlogic/amlogic_backend.h"

namespace onnxruntime {

class NodeUnit;

struct AmlogicExecutionProviderInfo {
  int device_id{0};
  bool enable_fp16{false};
  bool require_static_shapes{true};
  size_t min_subgraph_size{2};
  bool dump_model_info{true};
  std::string dump_model_info_path{"amlogic_onnx_nhwc_ops.json"};
  std::unordered_set<std::string> partitioning_stop_ops{};

  static AmlogicExecutionProviderInfo FromProviderOptions(const ProviderOptions& provider_options);
  ProviderOptions ToProviderOptions() const;
};

class AmlogicExecutionProvider : public IExecutionProvider {
 public:
  explicit AmlogicExecutionProvider(const AmlogicExecutionProviderInfo& info);
  ~AmlogicExecutionProvider() override = default;

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const GraphViewer& graph_viewer,
      const IKernelLookup& kernel_lookup,
      const GraphOptimizerRegistry& graph_optimizer_registry,
      IResourceAccountant* resource_accountant) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  DataLayout GetPreferredLayout() const override { return DataLayout::NHWC; }
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }
  bool ConcurrentRunSupported() const override { return false; }

  std::optional<bool> ShouldConvertDataLayoutForOp(std::string_view domain,
                                                   std::string_view op_type,
                                                   DataLayout target_data_layout) const override;

  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
  ProviderOptions GetProviderOptions() const override;

 private:
  bool IsNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph_viewer) const;

  AmlogicExecutionProviderInfo info_;
  OrtDevice::DeviceId device_id_;
  std::shared_ptr<amlogic::Backend> backend_;
  mutable ModelMetadefIdGenerator metadef_id_generator_;
  std::mutex compute_mutex_;
  mutable std::mutex model_info_dump_mutex_;
  mutable bool model_info_dumped_{false};
};

}  // namespace onnxruntime
