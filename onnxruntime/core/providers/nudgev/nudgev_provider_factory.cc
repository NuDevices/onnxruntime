#include "core/providers/nudgev/nudgev_provider_factory.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"

namespace onnxruntime {

class NudgevProviderFactory : public IExecutionProviderFactory {
 public:
  explicit NudgevProviderFactory(const std::string& device_id = "")
    : device_id_(device_id) {}

  ~NudgevProviderFactory() override = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<NudgevExecutionProvider>(device_id_);
  }

 private:
  std::string device_id_;
};

IExecutionProviderFactory* CreateExecutionProviderFactory_Nudgev(
    const std::string& device_id) {
  return new NudgevProviderFactory(device_id);
}

}  // namespace onnxruntime
