#pragma once
#include "core/framework/execution_provider.h"

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Amlnn();
}
