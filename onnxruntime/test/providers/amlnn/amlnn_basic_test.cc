// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Basic AMLNN Provider construction coverage.
//
// This source is intentionally not registered in the CMake test targets yet.
// It provides a no-hardware baseline test without changing existing build or
// runtime behavior. When AMLNN unit-test integration is enabled later, this
// test must remain independent of Session initialization, Compile(), NNSDK,
// and the external Core/Compiler shared libraries.

#include "core/graph/constants.h"
#include "core/providers/amlnn/amlnn_provider_factory_creator.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(AmlnnProviderFactoryTest, CreatesAmlnnExecutionProvider) {
  const ProviderOptions provider_options{};
  const auto factory = AmlnnProviderFactoryCreator::Create(provider_options);

  ASSERT_NE(factory, nullptr);
  const auto provider = factory->CreateProvider();
  ASSERT_NE(provider, nullptr);
  EXPECT_EQ(provider->Type(), kAmlnnExecutionProvider);
}

}  // namespace test
}  // namespace onnxruntime
