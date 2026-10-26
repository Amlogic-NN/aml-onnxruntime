#pragma once

// Loading adapter only. Graph conversion stays in Linghua's bridge sources.
#include <dlfcn.h>
#include <string>

#include "compiler_core_api.h"
#include "compiler_core_context.h"

namespace amlnn::adla_bridge {

struct CompilerCoreApi {
#define AML_CORE_FUNCTION(name) decltype(&::name) name = nullptr
  AML_CORE_FUNCTION(AmlCompilerCoreOptionsInitDefaults);
  AML_CORE_FUNCTION(AmlCompilerCoreGetCompiledArtifactAdlaPath);
  AML_CORE_FUNCTION(AmlCompilerCoreDestroyCompiledArtifact);
  AML_CORE_FUNCTION(AmlContextCreate);
  AML_CORE_FUNCTION(AmlContextDestroy);
  AML_CORE_FUNCTION(AmlContextCompile);
  AML_CORE_FUNCTION(AmlGraphCreate);
  AML_CORE_FUNCTION(AmlGraphSetTarget);
  AML_CORE_FUNCTION(AmlGraphSetName);
  AML_CORE_FUNCTION(AmlGraphAddOpCode);
  AML_CORE_FUNCTION(AmlGraphAddInput);
  AML_CORE_FUNCTION(AmlGraphAddOutput);
  AML_CORE_FUNCTION(AmlTensorCreate);
  AML_CORE_FUNCTION(AmlTensorSetName);
  AML_CORE_FUNCTION(AmlTensorSetDtype);
  AML_CORE_FUNCTION(AmlTensorSetIndex);
  AML_CORE_FUNCTION(AmlTensorSetDims);
  AML_CORE_FUNCTION(AmlTensorSetQuant);
  AML_CORE_FUNCTION(AmlTensorSetConstData);
  AML_CORE_FUNCTION(AmlNodeCreate);
  AML_CORE_FUNCTION(AmlNodeAddInput);
  AML_CORE_FUNCTION(AmlNodeAddOutput);
  AML_CORE_FUNCTION(AmlNodeSetParamI);
  AML_CORE_FUNCTION(AmlNodeSetParamF);
#undef AML_CORE_FUNCTION
  bool ready = false;
  std::string error;

  CompilerCoreApi() {
    handle_ = dlopen("libaml_compiler_core.so", RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char* detail = dlerror();
      error = detail != nullptr ? detail : "unable to load libaml_compiler_core.so";
      return;
    }

#define AML_CORE_BIND(name) if (!Bind(#name, name)) return
    AML_CORE_BIND(AmlCompilerCoreOptionsInitDefaults);
    AML_CORE_BIND(AmlCompilerCoreGetCompiledArtifactAdlaPath);
    AML_CORE_BIND(AmlCompilerCoreDestroyCompiledArtifact);
    // Use only the documented C Context API. The Core SDK's private C++
    // SerializeModel ABI is not stable across compiler releases.
    AML_CORE_BIND(AmlContextCreate);
    AML_CORE_BIND(AmlContextDestroy);
    AML_CORE_BIND(AmlContextCompile);
    AML_CORE_BIND(AmlGraphCreate);
    AML_CORE_BIND(AmlGraphSetTarget);
    AML_CORE_BIND(AmlGraphSetName);
    AML_CORE_BIND(AmlGraphAddOpCode);
    AML_CORE_BIND(AmlGraphAddInput);
    AML_CORE_BIND(AmlGraphAddOutput);
    AML_CORE_BIND(AmlTensorCreate);
    AML_CORE_BIND(AmlTensorSetName);
    AML_CORE_BIND(AmlTensorSetDtype);
    AML_CORE_BIND(AmlTensorSetIndex);
    AML_CORE_BIND(AmlTensorSetDims);
    AML_CORE_BIND(AmlTensorSetQuant);
    AML_CORE_BIND(AmlTensorSetConstData);
    AML_CORE_BIND(AmlNodeCreate);
    AML_CORE_BIND(AmlNodeAddInput);
    AML_CORE_BIND(AmlNodeAddOutput);
    AML_CORE_BIND(AmlNodeSetParamI);
    AML_CORE_BIND(AmlNodeSetParamF);
#undef AML_CORE_BIND
    ready = true;
  }

 private:
  template <typename T>
  bool Bind(const char* name, T& function) {
    dlerror();
    function = reinterpret_cast<T>(dlsym(handle_, name));
    const char* detail = dlerror();
    if (function == nullptr || detail != nullptr) {
      error = std::string("missing Core symbol ") + name;
      if (detail != nullptr) error += std::string(": ") + detail;
      return false;
    }
    return true;
  }

  // Keep Core and its vendor compiler dependency mapped for process lifetime.
  // No dlclose: the compiler can own worker threads and process-global state.
  void* handle_ = nullptr;
};

inline const CompilerCoreApi& GetCompilerCoreApi() {
  // C++ local-static initialization publishes a complete, immutable table.
  static const CompilerCoreApi context_api;
  return context_api;
}

}  // namespace amlnn::adla_bridge
