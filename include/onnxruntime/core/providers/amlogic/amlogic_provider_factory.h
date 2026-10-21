// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the Amlogic EP backed by AMLNN/NNSDK. Android builds also perform
// ONNX-to-ADLA conversion during Session initialization.
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Amlogic, _In_ OrtSessionOptions* options);

#ifdef __cplusplus
}
#endif
