#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/framework/provider_options.h"
#include "core/framework/data_transfer_manager.h"

namespace onnxruntime {

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");

    auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(
        X->Location().device, Y->Location().device);
    if (!gpu_data_transfer)
      return Status(common::ONNXRUNTIME, common::EP_FAIL, "gpu data transfer is missing in Nudgev EP.");

    if (!ctx->GetComputeStream())
      return Status(common::ONNXRUNTIME, common::EP_FAIL, "Compute Stream is missing in MemCpy kernel's context.");

    return gpu_data_transfer->CopyTensorAsync(*X, *Y, *(ctx->GetComputeStream()));
  }
};

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
  std::vector<std::unique_ptr<OpKernel>> kernels_;
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
