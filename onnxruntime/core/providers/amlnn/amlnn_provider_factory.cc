#include "core/providers/amlnn/amlnn_provider_factory.h"

#include "core/providers/amlnn/amlnn_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct AmlnnProviderFactory : IExecutionProviderFactory {
  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<AmlnnExecutionProvider>();
  }
};

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Amlnn() {
  return std::make_shared<AmlnnProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Amlnn,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Amlnn());
  return nullptr;
}
