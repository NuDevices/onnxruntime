#include "core/providers/nudgev/nudgev_provider_factory_creator.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/provider_options.h"
#include "core/session/inference_session.h"

namespace onnxruntime {

struct NudgevProviderFactory : IExecutionProviderFactory {
  NudgevProviderFactory(const ProviderOptions& provider_options_map,
                        const SessionOptions* session_options)
      : provider_options_map_(provider_options_map),
        session_options_(session_options) {}

  ~NudgevProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<NudgevExecutionProvider>(provider_options_map_,
                                                     session_options_);
  }

 private:
  ProviderOptions provider_options_map_;
  const SessionOptions* session_options_;
};

std::shared_ptr<IExecutionProviderFactory>
NudgevProviderFactoryCreator::Create(const ProviderOptions& provider_options_map,
                                     const SessionOptions* session_options) {
  return std::make_shared<NudgevProviderFactory>(provider_options_map,
                                                 session_options);
}

}  // namespace onnxruntime

/*
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nudgev,
                    _In_ OrtSessionOptions* options,
                    _In_opt_ const char* device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  auto factory = onnxruntime::NudgevProviderFactoryCreator::Create(
      onnxruntime::ProviderOptions{{"device_id", ""}},
      reinterpret_cast<const onnxruntime::SessionOptions*>(options));

  options->provider_factories.push_back(factory);

  return nullptr;
}
*/