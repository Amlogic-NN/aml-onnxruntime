#include "aml_adla_bridge.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>

#include "compiler_core_api.h"
#include "compiler_core_context.h"
#include "ops_json_to_aml.h"
#include "compiler_core_dynamic.h"

namespace amlnn::adla_bridge {

namespace {

bool SetTensor(const CompilerCoreApi& core, AmlTensor handle, const AML_Tensor& tensor, std::string* error) {
  auto fail = [error](const char* what) {
    if (error) *error = std::string(what) + " failed while emitting Context API graph";
    return false;
  };
  if (core.AmlTensorSetName(handle, tensor.name.c_str()) != kAmlCompilerStatusOk) return fail("AmlTensorSetName");
  if (core.AmlTensorSetDtype(handle, tensor.dtype) != kAmlCompilerStatusOk) return fail("AmlTensorSetDtype");
  if (core.AmlTensorSetIndex(handle, tensor.tensor_index) != kAmlCompilerStatusOk) return fail("AmlTensorSetIndex");
  if (core.AmlTensorSetDims(handle, tensor.dims.data(), static_cast<uint32_t>(tensor.dims.size())) != kAmlCompilerStatusOk) return fail("AmlTensorSetDims");
  if (core.AmlTensorSetQuant(handle, tensor.zp.data(), static_cast<uint32_t>(tensor.zp.size()),
                        tensor.scale.data(), static_cast<uint32_t>(tensor.scale.size())) != kAmlCompilerStatusOk) {
    return fail("AmlTensorSetQuant");
  }
  if (tensor.is_const && core.AmlTensorSetConstData(handle, tensor.data.data(), tensor.data.size()) != kAmlCompilerStatusOk) {
    return fail("AmlTensorSetConstData");
  }
  return true;
}

bool EmitAndCompile(const CompilerCoreApi& core, const AML_Model& model, const std::string& output_dir,
                    const std::string& model_name, const std::string& target,
                    std::string* adla_path, std::string* error) {
  AmlContext context = core.AmlContextCreate();
  if (context == nullptr) {
    if (error) *error = "AmlContextCreate failed";
    return false;
  }
  AmlGraph graph = nullptr;
  auto cleanup = [&]() { core.AmlContextDestroy(context); };
  if (core.AmlGraphCreate(context, &graph) != kAmlCompilerStatusOk || graph == nullptr) {
    cleanup();
    if (error) *error = "AmlGraphCreate failed";
    return false;
  }
  if (core.AmlGraphSetTarget(graph, target.c_str()) != kAmlCompilerStatusOk ||
      core.AmlGraphSetName(graph, model_name.c_str()) != kAmlCompilerStatusOk) {
    cleanup();
    if (error) *error = "failed to initialize AML graph metadata";
    return false;
  }
  for (const auto& op_code : model.op_codes) {
    if (core.AmlGraphAddOpCode(graph, op_code.c_str()) != kAmlCompilerStatusOk) {
      cleanup();
      if (error) *error = "AmlGraphAddOpCode failed";
      return false;
    }
  }

  std::map<std::string, AmlTensor> tensors;
  auto get_tensor = [&](const AML_Tensor& source, AmlTensor* result) -> bool {
    if (source.name.empty()) {
      if (error) *error = "graph tensor has an empty name";
      return false;
    }
    const auto found = tensors.find(source.name);
    if (found != tensors.end()) {
      *result = found->second;
      return true;
    }
    AmlTensor created = nullptr;
    if (core.AmlTensorCreate(graph, &created) != kAmlCompilerStatusOk || created == nullptr ||
        !SetTensor(core, created, source, error)) {
      if (error && error->empty()) *error = "failed to create AML tensor: " + source.name;
      return false;
    }
    tensors.emplace(source.name, created);
    *result = created;
    return true;
  };

  for (const auto& node : model.Node_List) {
    AmlNode emitted = nullptr;
    if (core.AmlNodeCreate(graph, node.Node_type, node.Node_index, &emitted) != kAmlCompilerStatusOk || emitted == nullptr) {
      cleanup();
      if (error) *error = "AmlNodeCreate failed";
      return false;
    }
    for (const auto& input : node.inputs) {
      AmlTensor tensor = nullptr;
      if (!get_tensor(input, &tensor) || core.AmlNodeAddInput(emitted, tensor) != kAmlCompilerStatusOk) {
        cleanup();
        if (error && error->empty()) *error = "failed to add node input";
        return false;
      }
    }
    for (const auto& output : node.outputs) {
      AmlTensor tensor = nullptr;
      if (!get_tensor(output, &tensor) || core.AmlNodeAddOutput(emitted, tensor) != kAmlCompilerStatusOk) {
        cleanup();
        if (error && error->empty()) *error = "failed to add node output";
        return false;
      }
    }
    for (const auto& param : node.params) {
      if (core.AmlNodeSetParamI(emitted, param.first.c_str(), param.second) != kAmlCompilerStatusOk) {
        cleanup();
        if (error) *error = "AmlNodeSetParamI failed for " + param.first;
        return false;
      }
    }
    for (const auto& param : node.fparams) {
      if (core.AmlNodeSetParamF(emitted, param.first.c_str(), param.second) != kAmlCompilerStatusOk) {
        cleanup();
        if (error) *error = "AmlNodeSetParamF failed for " + param.first;
        return false;
      }
    }
  }
  for (const auto& input : model.inputs) {
    AmlTensor tensor = nullptr;
    if (!get_tensor(input, &tensor) || core.AmlGraphAddInput(graph, tensor) != kAmlCompilerStatusOk) {
      cleanup();
      if (error && error->empty()) *error = "failed to add graph input";
      return false;
    }
  }
  for (const auto& output : model.outputs) {
    AmlTensor tensor = nullptr;
    if (!get_tensor(output, &tensor) || core.AmlGraphAddOutput(graph, tensor) != kAmlCompilerStatusOk) {
      cleanup();
      if (error && error->empty()) *error = "failed to add graph output";
      return false;
    }
  }

  AmlCompilerCoreOptions options{};
  core.AmlCompilerCoreOptionsInitDefaults(&options);
  options.compile_target = target.empty() ? nullptr : target.c_str();
  AmlCompilerCoreCompiledArtifact artifact = nullptr;
  const auto status = core.AmlContextCompile(context, model_name.c_str(), output_dir.c_str(), &options, &artifact);
  cleanup();
  if (status != kAmlCompilerStatusOk || artifact == nullptr) {
    if (error) *error = "AmlContextCompile failed, status=" + std::to_string(status);
    core.AmlCompilerCoreDestroyCompiledArtifact(artifact);
    return false;
  }
  const char* path = nullptr;
  const auto path_status = core.AmlCompilerCoreGetCompiledArtifactAdlaPath(artifact, &path);
  if (path_status != kAmlCompilerStatusOk || path == nullptr || *path == '\0') {
    core.AmlCompilerCoreDestroyCompiledArtifact(artifact);
    if (error) *error = "Core returned no ADLA path, status=" + std::to_string(path_status);
    return false;
  }
  const std::string resolved_path = path;
  std::ifstream file(resolved_path, std::ios::binary | std::ios::ate);
  if (!file.good() || file.tellg() <= 0) {
    core.AmlCompilerCoreDestroyCompiledArtifact(artifact);
    if (error) *error = "Core returned without creating a non-empty ADLA artifact: " + resolved_path;
    return false;
  }
  if (adla_path) *adla_path = resolved_path;
  core.AmlCompilerCoreDestroyCompiledArtifact(artifact);
  return true;
}

}  // namespace

bool CompileOpsJsonToAdla(const std::string& ops_json, const std::string& output_dir,
                          const std::string& model_name, const std::string& target,
                          std::string* adla_path, std::string* error) {
  AML_Model model;
  if (!BuildAmlModelFromOpsJson(ops_json, target, &model, error)) return false;
  if (std::getenv("AML_ADLA_BRIDGE_DUMP_IR") != nullptr) {
    std::cerr << "[aml_adla_bridge] model inputs=" << model.inputs.size()
              << " outputs=" << model.outputs.size()
              << " nodes=" << model.Node_List.size() << '\n';
    std::cerr << "[aml_adla_bridge] graph_inputs=";
    for (const auto& tensor : model.inputs) {
      std::cerr << tensor.name << "[" << tensor.tensor_index << "] ";
    }
    std::cerr << "graph_outputs=";
    for (const auto& tensor : model.outputs) {
      std::cerr << tensor.name << "[" << tensor.tensor_index << "] ";
    }
    std::cerr << '\n';
    for (const auto& node : model.Node_List) {
      std::cerr << "[aml_adla_bridge] node=" << node.Node_index
                << " op=" << node.Node_type << " inputs=";
      for (const auto& tensor : node.inputs) {
        std::cerr << tensor.name << "[" << tensor.tensor_index << ";";
        for (size_t i = 0; i < tensor.dims.size(); ++i) {
          if (i) std::cerr << ',';
          std::cerr << tensor.dims[i];
        }
        std::cerr << "] ";
      }
      std::cerr << "outputs=";
      for (const auto& tensor : node.outputs) {
        std::cerr << tensor.name << "[" << tensor.tensor_index << ";";
        for (size_t i = 0; i < tensor.dims.size(); ++i) {
          if (i) std::cerr << ',';
          std::cerr << tensor.dims[i];
        }
        std::cerr << "] ";
      }
      std::cerr << '\n';
    }
  }
  // The documented C Context API is the sole compilation path. Do not use
  // Core's private C++ serialization ABI: it changed between compiler 3.5.0
  // and 3.5.3 and caused a process crash.
  const auto& core = GetCompilerCoreApi();
  if (!core.ready) {
    if (error) *error = core.error;
    return false;
  }
  return EmitAndCompile(core, model, output_dir, model_name, target, adla_path, error);
}
}  // namespace amlnn::adla_bridge
