#pragma once

#include <memory>
#include <string>
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/providers.h"

namespace onnxruntime {

struct SessionOptions;

struct NudgevProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
      const ProviderOptions& provider_options_map,
      const SessionOptions* session_options = nullptr);
};

}  // namespace onnxruntime