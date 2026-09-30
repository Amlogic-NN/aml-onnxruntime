#include "aml_adla_bridge.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

#include "compiler_core_api.h"
#include "ops_json_to_aml.h"

namespace amlogic::adla_bridge {
namespace {

struct CompilerCoreApi {
  void* handle = nullptr;
  decltype(&AmlCompilerCoreOptionsInitDefaults) options_init_defaults = nullptr;
  decltype(&AmlCompilerCoreCompile) compile = nullptr;
  decltype(&AmlCompilerCoreGetCompiledArtifactAdlaPath) get_adla_path = nullptr;
  decltype(&AmlCompilerCoreDestroyCompiledArtifact) destroy_artifact = nullptr;
};

CompilerCoreApi& GetCompilerCoreApi() {
  static CompilerCoreApi api;
  return api;
}

std::mutex& GetCompilerCoreApiMutex() {
  static std::mutex mutex;
  return mutex;
}

template <typename T>
bool BindCompilerCoreSymbol(void* handle, const char* name, T* symbol, std::string* error) {
  dlerror();
  *symbol = reinterpret_cast<T>(dlsym(handle, name));
  const char* dl_error = dlerror();
  if (*symbol == nullptr || dl_error != nullptr) {
    if (error) {
      *error = std::string("missing symbol ") + name;
      if (dl_error != nullptr) {
        *error += ": ";
        *error += dl_error;
      }
    }
    return false;
  }
  return true;
}

bool LoadCompilerCore(std::string* error) {
  auto& api = GetCompilerCoreApi();
  std::lock_guard<std::mutex> lock(GetCompilerCoreApiMutex());
  if (api.handle != nullptr) return true;

  api.handle = dlopen("libaml_compiler_core.so", RTLD_NOW | RTLD_LOCAL);
  if (api.handle == nullptr) {
    const char* dl_error = dlerror();
    if (error) *error = dl_error != nullptr ? dl_error : "unable to load libaml_compiler_core.so";
    return false;
  }

  if (!BindCompilerCoreSymbol(api.handle, "AmlCompilerCoreOptionsInitDefaults",
                              &api.options_init_defaults, error) ||
      !BindCompilerCoreSymbol(api.handle, "AmlCompilerCoreCompile", &api.compile, error) ||
      !BindCompilerCoreSymbol(api.handle, "AmlCompilerCoreGetCompiledArtifactAdlaPath",
                              &api.get_adla_path, error) ||
      !BindCompilerCoreSymbol(api.handle, "AmlCompilerCoreDestroyCompiledArtifact",
                              &api.destroy_artifact, error)) {
    dlclose(api.handle);
    api = CompilerCoreApi{};
    return false;
  }
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
  const std::string serialized = SerializeModel(model);
  AmlSerializedModelBuffer buffer{serialized.data(), serialized.size()};
  AmlCompilerCoreRequest request{&buffer, 1, model_name.c_str(), output_dir.c_str()};
  AmlCompilerCoreOptions options{};
  if (!LoadCompilerCore(error)) return false;
  const auto& compiler_core = GetCompilerCoreApi();
  compiler_core.options_init_defaults(&options);
  options.compile_target = target.empty() ? nullptr : target.c_str();
  AmlCompilerCoreCompiledArtifact artifact = nullptr;
  const auto status = compiler_core.compile(&request, &options, &artifact);
  if (status != kAmlCompilerStatusOk || !artifact) {
    if (error) *error = "AmlCompilerCoreCompile failed, status=" + std::to_string(status);
    return false;
  }
  const char* path = nullptr;
  const auto path_status = compiler_core.get_adla_path(artifact, &path);
  if (path_status != kAmlCompilerStatusOk || path == nullptr || *path == '\0') {
    compiler_core.destroy_artifact(artifact);
    if (error) *error = "Compiler3 returned no ADLA path, status=" + std::to_string(path_status);
    return false;
  }
  const std::string resolved_path = path ? path : "";
  // aml_compiler_core currently cannot propagate several Compiler3 failures:
  // it may return OK and write a fingerprint even when no .adla was emitted.
  // The bridge is the public boundary for ORT, so make artifact existence part
  // of the success contract.
  std::ifstream file(resolved_path, std::ios::binary | std::ios::ate);
  if (!file.good() || file.tellg() <= 0) {
    compiler_core.destroy_artifact(artifact);
    if (error) *error = "Compiler3 returned without creating a non-empty ADLA artifact: " + resolved_path;
    return false;
  }
  if (adla_path) *adla_path = resolved_path;
  compiler_core.destroy_artifact(artifact);
  return true;
}
}  // namespace amlogic::adla_bridge
