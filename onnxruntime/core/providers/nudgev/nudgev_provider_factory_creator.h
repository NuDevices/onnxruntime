#pragma once

#include <memory>
#include "core/providers/providers.h"
#include "core/framework/provider_options.h"

struct OrtSessionOptions;

namespace onnxruntime {

class SessionOptions;

struct NudgevProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(
      const ProviderOptions& provider_options_map,
      const SessionOptions* session_options = nullptr);
};

}  // namespace onnxruntime