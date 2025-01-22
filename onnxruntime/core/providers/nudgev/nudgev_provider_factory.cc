#include "core/providers/nudgev/nudgev_provider_factory_creator.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

namespace {
struct NudgevProviderFactory : IExecutionProviderFactory {
  explicit NudgevProviderFactory(const std::string& device_id)
      : device_id_(device_id) {}

  ~NudgevProviderFactory() override = default;
  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<NudgevExecutionProvider>(device_id_);
  }

 private:
  std::string device_id_;
};
}  // namespace

std::shared_ptr<IExecutionProviderFactory> NudgevProviderFactoryCreator::Create(const std::string& device_id) {
  return std::make_shared<NudgevProviderFactory>(device_id);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nudgev, _In_ OrtSessionOptions* options, const char* device_id) {
  std::string device_id_str = (device_id ? device_id : "");
  auto factory = onnxruntime::NudgevProviderFactoryCreator::Create(device_id_str);
  options->provider_factories.push_back(factory);
  return nullptr;
}
