// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>

#include "core/common/status.h"
#include "core/common/logging/logging.h"

namespace Ort {
struct KernelContext;
}  // namespace Ort

namespace onnxruntime {

class GraphViewer;

namespace amlogic {

class CompiledGraph {
 public:
  explicit CompiledGraph(std::string graph_name);

  Status Run(Ort::KernelContext& context, const logging::Logger& logger) const;

 private:
  std::string graph_name_;
};

class Backend {
 public:
  Backend() = default;

  bool IsAvailable() const;

  Status Compile(const GraphViewer& graph_viewer,
                 const std::string& graph_name,
                 std::unique_ptr<CompiledGraph>& compiled_graph) const;
};

}  // namespace amlogic
}  // namespace onnxruntime
