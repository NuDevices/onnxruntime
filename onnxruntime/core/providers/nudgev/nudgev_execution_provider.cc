#include "core/providers/nudgev/nudgev_execution_provider.h"
#include "core/providers/nudgev/nudgev_matmul.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

static constexpr const char* kNudgevExecutionProvider = "Nudgev";

static void RegisterNudgevKernels(KernelRegistry& kernel_registry) {
  KernelDefBuilder def_builder;
  auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    ORT_UNUSED_PARAMETER(func_mgr);
    out = std::make_unique<nudgev::NudgevMatMul>(info);
    return Status::OK();
  };

  Status status = kernel_registry.Register(
      def_builder
          .SetName("MatMul")
          .SetDomain(kOnnxDomain)
          .SinceVersion(1)
          .Provider(kNudgevExecutionProvider)
          .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
      create_fn);

  ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for MatMul");
}

NudgevExecutionProvider::NudgevExecutionProvider(const std::string& device_id)
    : IExecutionProvider(kNudgevExecutionProvider) {
  ORT_UNUSED_PARAMETER(device_id);
}

std::shared_ptr<KernelRegistry> NudgevExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  if (!kernel_registry_initialized_) {
    kernel_registry_initialized_ = true;
    RegisterNudgevKernels(*kernel_registry);
  }
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
NudgevExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const IKernelLookup& kernel_lookup) const {
  ORT_UNUSED_PARAMETER(kernel_lookup);

  std::vector<std::unique_ptr<ComputeCapability>> result;

  for (auto& node : graph_viewer.Nodes()) {
    if (node.OpType() == "MatMul") {
      // Check if all inputs are int8
      bool all_inputs_int8 = true;
      for (const auto* input : node.InputDefs()) {
        if (input->Type() == nullptr ||
            input->Type()->find("int8") == std::string::npos) {
          all_inputs_int8 = false;
          break;
        }
      }

      if (all_inputs_int8) {
        auto sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node.Index());
        result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
      }
    }
  }

  return result;
}

Status NudgevExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {

  for (auto& fused_node : fused_nodes_and_graphs) {
    ORT_UNUSED_PARAMETER(fused_node);

    NodeComputeInfo compute_info;
    compute_info.compute_func = [](FunctionState state,
                                 const OrtApi* api,
                                 OrtKernelContext* context) -> Status {
      ORT_UNUSED_PARAMETER(state);
      ORT_UNUSED_PARAMETER(api);
      ORT_UNUSED_PARAMETER(context);
      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }
  return Status::OK();
}

}  // namespace onnxruntime
