// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_backend.h"

#include <utility>

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace amlogic {

CompiledGraph::CompiledGraph(std::string graph_name) : graph_name_(std::move(graph_name)) {
}

Status CompiledGraph::Run(Ort::KernelContext& /*context*/, const logging::Logger& /*logger*/) const {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Amlogic NPU runtime adapter is not implemented for graph: ", graph_name_);
}

bool Backend::IsAvailable() const {
#ifdef ORT_AMLOGIC_USE_VENDOR_SDK
  // Wire the Amlogic SDK runtime probe here. Return true only when the NPU
  // device and required user-mode libraries are usable by this process.
  return false;
#else
  return false;
#endif
}

Status Backend::Compile(const GraphViewer& graph_viewer,
                        const std::string& graph_name,
                        std::unique_ptr<CompiledGraph>& compiled_graph) const {
  ORT_UNUSED_PARAMETER(graph_viewer);

#ifdef ORT_AMLOGIC_USE_VENDOR_SDK
  // Wire graph lowering and SDK compilation here. The execution provider has
  // already converted supported layout-sensitive ops to NHWC.
  compiled_graph = std::make_unique<CompiledGraph>(graph_name);
  return Status::OK();
#else
  ORT_UNUSED_PARAMETER(graph_name);
  ORT_UNUSED_PARAMETER(compiled_graph);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Amlogic NPU SDK adapter is not linked. Build with "
                         "onnxruntime_AMLOGIC_NPU_STUB=OFF after implementing "
                         "core/providers/amlogic/amlogic_backend.cc.");
#endif
}

}  // namespace amlogic
}  // namespace onnxruntime
