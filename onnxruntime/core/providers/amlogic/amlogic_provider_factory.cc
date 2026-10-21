// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/amlogic/amlogic_provider_factory.h"
#include "core/common/common.h"

#include "core/providers/amlogic/amlogic_execution_provider.h"
#include "core/providers/amlogic/amlogic_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

namespace {

struct AmlogicProviderFactory : IExecutionProviderFactory {
  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<AmlogicExecutionProvider>();
  }
};

}  // namespace

std::shared_ptr<IExecutionProviderFactory> AmlogicProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  ORT_ENFORCE(provider_options.empty(), "The unified Amlogic EP does not accept provider options");
  return std::make_shared<AmlogicProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Amlogic,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::AmlogicProviderFactoryCreator::Create());
  return nullptr;
}
