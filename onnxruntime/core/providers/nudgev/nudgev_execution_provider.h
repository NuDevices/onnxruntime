#pragma once

#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

class NudgevExecutionProvider : public IExecutionProvider {
 public:
  explicit NudgevExecutionProvider(const std::string& device_id = "");
  ~NudgevExecutionProvider() = default;

  // Returns a vector of supported node capabilities
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  // Identifies which nodes we can execute
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const IKernelLookup& kernel_lookup) const override;

  // Sets up the actual execution of nodes
  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  mutable bool kernel_registry_initialized_{false};
};

}  // namespace onnxruntime
