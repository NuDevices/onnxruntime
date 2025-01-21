#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"

namespace onnxruntime {

// Forward declare the factory function
IExecutionProviderFactory* CreateExecutionProviderFactory_Nudgev(const std::string& device_id = "");

struct NudgevProviderFactoryCreator {
  static IExecutionProviderFactory* Create() {
    return CreateExecutionProviderFactory_Nudgev();
  }
};

}  // namespace onnxruntime
