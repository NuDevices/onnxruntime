#pragma once

#include <memory>
#include <string>
#include "core/providers/providers.h"

namespace onnxruntime {

struct NudgevProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(const std::string& device_id);
};

}  // namespace onnxruntime
