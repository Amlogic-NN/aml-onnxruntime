// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside the anonymous namespace. It collects the split shared and legacy
// qlinear helper layers behind a single include entrypoint so the legacy
// exporter mirrors the same "thin entry .cc + layered impl headers" shape
// used by reference_style.

#include "core/providers/amlogic/amlogic_qlinear_common_support_impl.h"
#include "core/providers/amlogic/amlogic_reference_style_support_impl.h"
#include "core/providers/amlogic/amlogic_reference_style_tensor_support_impl.h"
#include "core/providers/amlogic/amlogic_reference_style_state_support_impl.h"

bool IsPaddedQLinearConvNode(const Node* node);

#include "core/providers/amlogic/amlogic_qlinear_legacy_impl.h"
