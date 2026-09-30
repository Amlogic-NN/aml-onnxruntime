// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_provider_factory.h"

#include "core/providers/amlogic/amlogic_execution_provider.h"
#include "core/providers/amlogic/amlogic_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

namespace {

struct AmlogicProviderFactory : IExecutionProviderFactory {
  explicit AmlogicProviderFactory(const ProviderOptions& provider_options)
      : info_(AmlogicExecutionProviderInfo::FromProviderOptions(provider_options)) {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<AmlogicExecutionProvider>(info_);
  }

 private:
  AmlogicExecutionProviderInfo info_;
};

}  // namespace

std::shared_ptr<IExecutionProviderFactory> AmlogicProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  return std::make_shared<AmlogicProviderFactory>(provider_options);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Amlogic,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::AmlogicProviderFactoryCreator::Create());
  return nullptr;
}
