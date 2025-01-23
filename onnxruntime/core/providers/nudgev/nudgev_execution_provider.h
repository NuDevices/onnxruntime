#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {

class NudgevExecutionProvider : public IExecutionProvider {
 public:
  explicit NudgevExecutionProvider(const ProviderOptions& provider_options_map,
                                   const SessionOptions* session_options = nullptr);

  ~NudgevExecutionProvider() override = default;

  FusionStyle GetFusionStyle() const override {
    return FusionStyle::FilteredGraphViewer;
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  DataLayout GetPreferredLayout() const override;

 private:
  Status ParseProviderOptions(const ProviderOptions& provider_options_map);

  bool disable_cpu_ep_fallback_{false};
  bool context_cache_enabled_{false};

  int32_t device_id_{0};
  std::string device_type_{"DEFAULT"};

  bool support_uint8_{true};
  bool force_int8_{false};
  bool offload_quantization_{false};

  mutable bool kernel_registry_initialized_{false};
  ModelMetadefIdGenerator metadef_id_generator_;
};

}  // namespace onnxruntime