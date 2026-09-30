// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is intentionally included from amlogic_qlinear_model_info.cc
// inside namespace reference_style_internal. It provides a single entrypoint
// for the reference_style internal implementation layers:
// - shared helpers
// - compatibility helpers
// - per-op collectors
// - postprocess passes

#include "core/providers/amlogic/amlogic_reference_style_shared_impl.h"
#include "core/providers/amlogic/amlogic_reference_style_compat_impl.h"
#include "core/providers/amlogic/amlogic_reference_style_collectors_impl.h"
#include "core/providers/amlogic/amlogic_reference_style_postprocess_impl.h"
