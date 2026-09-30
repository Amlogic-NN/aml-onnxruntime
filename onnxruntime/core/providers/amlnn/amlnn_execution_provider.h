#pragma once
#include "core/framework/execution_provider.h"

namespace onnxruntime {
class AmlnnExecutionProvider final : public IExecutionProvider {
 public:
  AmlnnExecutionProvider();
  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const GraphViewer&, const IKernelLookup&, const GraphOptimizerRegistry&,
      IResourceAccountant*) const override;
  common::Status Compile(const std::vector<FusedNodeAndGraph>&,
                         std::vector<NodeComputeInfo>&) override;
};
}  // namespace onnxruntime
