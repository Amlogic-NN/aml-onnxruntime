#pragma once

// Loading adapter only. Graph conversion stays in Linghua's bridge sources.
#include <dlfcn.h>
#include <string>

#include "bridge_ir.h"
#include "compiler_core_api.h"
#include "compiler_core_context.h"

namespace amlogic::adla_bridge {

struct CompilerCoreApi {
#define AML_CORE_FUNCTION(name) decltype(&::name) name = nullptr
  AML_CORE_FUNCTION(AmlCompilerCoreOptionsInitDefaults);
  AML_CORE_FUNCTION(AmlCompilerCoreGetCompiledArtifactAdlaPath);
  AML_CORE_FUNCTION(AmlCompilerCoreDestroyCompiledArtifact);
  AML_CORE_FUNCTION(AmlCompilerCoreCompile);
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
  // Same signature as Linghua's local declaration; implementation is in Core.
  std::string (*SerializeModel)(const AML_Model&) = nullptr;
  bool ready = false;
  std::string error;

  explicit CompilerCoreApi(bool use_context) {
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
    // Bind only the selected path. The default serialized path does not need
    // Context symbols, and the explicit Context path does not need SerializeModel.
    if (use_context) {
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
    } else {
      AML_CORE_BIND(AmlCompilerCoreCompile);
      // SerializeModel returns std::string and accepts Linghua's AML_Model.
      // Resolve the C++ symbol for the caller's ABI; do not fall back to a
      // different STL ABI (Android libc++ and Linux libstdc++ are incompatible).
#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI
      const char* symbol = "_Z14SerializeModelB5cxx11RK9AML_Model";
#else
      const char* symbol = "_Z14SerializeModelRK9AML_Model";
#endif
      if (!Bind(symbol, SerializeModel)) {
        error += "; SerializeModel requires a matching Core SDK, AML_Model layout and C++ STL ABI";
        return;
      }
    }
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

inline const CompilerCoreApi& GetCompilerCoreApi(bool use_context) {
  // C++ local-static initialization publishes a complete, immutable table.
  if (use_context) {
    static const CompilerCoreApi context_api(true);
    return context_api;
  }
  static const CompilerCoreApi serialized_api(false);
  return serialized_api;
}

}  // namespace amlogic::adla_bridge
