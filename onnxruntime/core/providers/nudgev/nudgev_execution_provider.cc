#include "core/providers/nudgev/nudgev_execution_provider.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/common/logging/logging.h"
#include "core/providers/partitioning_utils.h"
#include "core/providers/nudgev/nudgev_conv.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <iostream>

namespace onnxruntime {

static void RegisterNudgevKernels(KernelRegistry& kernel_registry) {
  std::cout << "[NudgevEP] RegisterNudgevKernels started" << std::endl;

  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      out = std::make_unique<onnxruntime::nudgev::NudgevConv>(info);
      return Status::OK();
    };

    std::cout << "[NudgevEP] Registering Conv kernel" << std::endl;
    Status status = kernel_registry.Register(
        def_builder
            .SetName("Conv")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
        create_fn);

    std::cout << "[NudgevEP] Conv kernel registration status: " << status.IsOK() << std::endl;
    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Conv");
  }

  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      out = std::make_unique<Memcpy>(info);
      return Status::OK();
    };

    std::cout << "[NudgevEP] Registering MemcpyFromHost kernel" << std::endl;
    Status status = kernel_registry.Register(
        def_builder
            .SetName("MemcpyFromHost")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .InputMemoryType(OrtMemTypeCPUInput, 0)
            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
        create_fn);

    std::cout << "[NudgevEP] MemcpyFromHost kernel registration status: " << status.IsOK() << std::endl;
    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for MemcpyFromHost");
  }

  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      out = std::make_unique<Memcpy>(info);
      return Status::OK();
    };

    std::cout << "[NudgevEP] Registering MemcpyToHost kernel" << std::endl;
    Status status = kernel_registry.Register(
        def_builder
            .SetName("MemcpyToHost")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .OutputMemoryType(OrtMemTypeCPUOutput, 0)
            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
        create_fn);

    std::cout << "[NudgevEP] MemcpyToHost kernel registration status: " << status.IsOK() << std::endl;
    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for MemcpyToHost");
  }
}

Status NudgevExecutionProvider::ParseProviderOptions(const ProviderOptions& provider_options_map) {
  auto device_id_it = provider_options_map.find("device_id");
  if (device_id_it != provider_options_map.end()) {
    device_id_ = std::stoi(device_id_it->second);
    LOGS_DEFAULT(INFO) << "Setting device_id to " << device_id_;
  }

  auto device_type_it = provider_options_map.find("device_type");
  if (device_type_it != provider_options_map.end()) {
    device_type_ = device_type_it->second;
    LOGS_DEFAULT(INFO) << "Setting device_type to " << device_type_;
  }

  auto support_uint8_it = provider_options_map.find("support_uint8");
  if (support_uint8_it != provider_options_map.end()) {
    support_uint8_ = support_uint8_it->second == "1";
    LOGS_DEFAULT(INFO) << "uint8 support " << (support_uint8_ ? "enabled" : "disabled");
  }

  auto force_int8_it = provider_options_map.find("force_int8");
  if (force_int8_it != provider_options_map.end()) {
    force_int8_ = force_int8_it->second == "1";
    LOGS_DEFAULT(INFO) << "Force int8 " << (force_int8_ ? "enabled" : "disabled");
  }

  return Status::OK();
}

NudgevExecutionProvider::NudgevExecutionProvider(
    const ProviderOptions& provider_options_map,
    const SessionOptions* session_options)
    : IExecutionProvider(kNudgevExecutionProvider) {

  std::cout << "[NudgevEP] Constructor called - EP initialization started" << std::endl;

  if (session_options) {
    disable_cpu_ep_fallback_ = session_options->config_options.GetConfigOrDefault(
                                   kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";

    context_cache_enabled_ = session_options->config_options.GetConfigOrDefault(
                                 kOrtSessionOptionEpContextEnable, "0") == "1";
  }

  ORT_ENFORCE(ParseProviderOptions(provider_options_map).IsOK());
  std::cout << "[NudgevEP] Constructor completed" << std::endl;

}

std::shared_ptr<KernelRegistry> NudgevExecutionProvider::GetKernelRegistry() const {
  std::cout << "[NudgevEP] GetKernelRegistry called" << std::endl;
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  if (!kernel_registry_initialized_) {
    std::cout << "[NudgevEP] Initializing kernel registry" << std::endl;
    kernel_registry_initialized_ = true;
    RegisterNudgevKernels(*kernel_registry);
    std::cout << "[NudgevEP] Kernel registry initialization completed" << std::endl;
  }
  return kernel_registry;
}

DataLayout NudgevExecutionProvider::GetPreferredLayout() const {
  std::cout << "[NudgevEP] GetPreferredLayout called - returning NHWC" << std::endl;
  return DataLayout::NHWC;
}

std::vector<std::unique_ptr<ComputeCapability>>
NudgevExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                     const IKernelLookup& kernel_lookup) const {
    std::cout << "\n[NudgevEP] ============= GetCapability Start =============" << std::endl;
    ORT_UNUSED_PARAMETER(kernel_lookup);

    if (graph_viewer.IsSubgraph()) {
        std::cout << "[NudgevEP] Is subgraph, returning empty result" << std::endl;
        return {};
    }

    std::vector<std::unique_ptr<ComputeCapability>> result;
    std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
    std::unordered_map<const Node*, const NodeUnit*> node_unit_map;

    std::cout << "[NudgevEP] Getting node units..." << std::endl;
    std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer);
    std::cout << "[NudgevEP] Number of node units: " << node_unit_holder.size() << std::endl;

    std::unordered_set<const Node*> supported_nodes;

    for (auto& nu : node_unit_holder) {
        std::cout << "[NudgevEP] Checking NodeUnit with type: " << nu->OpType() << std::endl;

        if (nu->OpType() == "Conv") {
            std::cout << "[NudgevEP] Found Conv NodeUnit: " << nu->GetNode().Name() << std::endl;

            bool all_inputs_supported = true;
            bool has_qdq = false;

            for (const auto& qdq : nu->Inputs()) {
                if (qdq.quant_param) {
                    has_qdq = true;
                    if (!qdq.node_arg.Type()) {
                        std::cout << "[NudgevEP] Input has no type info: " << qdq.node_arg.Name() << std::endl;
                        all_inputs_supported = false;
                        break;
                    }

                    const auto& type_str = *qdq.node_arg.Type();
                    bool type_supported = false;
                    if (type_str.find("int8") != std::string::npos) {
                        type_supported = true;
                        std::cout << "[NudgevEP] Found int8 quantized input: " << qdq.node_arg.Name() << std::endl;
                    } else if (type_str.find("int32") != std::string::npos) {
                        type_supported = true;
                        std::cout << "[NudgevEP] Found int32 input (bias): " << qdq.node_arg.Name() << std::endl;
                    } else if (support_uint8_ && type_str.find("uint8") != std::string::npos) {
                        if (!force_int8_) {
                            type_supported = true;
                            std::cout << "[NudgevEP] Found uint8 quantized input: " << qdq.node_arg.Name() << std::endl;
                        }
                    }

                    if (!type_supported) {
                        std::cout << "[NudgevEP] Unsupported quantized type for input: " << qdq.node_arg.Name() << std::endl;
                        all_inputs_supported = false;
                        break;
                    }
                }
            }

            if (all_inputs_supported && has_qdq) {
                supported_nodes.insert(&nu->GetNode());

                for (const auto& node : nu->GetAllNodesInGroup()) {
                    supported_nodes.insert(node);
                }
                std::cout << "[NudgevEP] Added complete Conv NodeUnit to supported_nodes: " << nu->GetNode().Name() << std::endl;
            } else {
                std::cout << "[NudgevEP] NodeUnit not supported: " << nu->GetNode().Name() << std::endl;
            }
        }
    }

    if (!supported_nodes.empty()) {
        auto generate_metadef_name = [this, &graph_viewer]() {
            uint64_t model_hash;
            int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);
            return MakeString("Nudgev_", model_hash, "_", metadef_id);
        };

        for (const auto* node : supported_nodes) {
            std::vector<const Node*> node_group{node};
            result.push_back(utils::MakeComputeCapability(
                graph_viewer,
                node_group,
                generate_metadef_name,
                "NudgevExecutionProvider",
                false));
            std::cout << "[NudgevEP] Added compute capability for node: " << node->Name() << std::endl;
        }
    }

    std::cout << "[NudgevEP] Final result size: " << result.size() << std::endl;
    std::cout << "[NudgevEP] ============= GetCapability End =============" << std::endl;

    return result;
}

Status NudgevExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  std::cout << "[NudgevEP] ============= Compile Start =============" << std::endl;
  std::cout << "[NudgevEP] Number of fused nodes: " << fused_nodes_and_graphs.size() << std::endl;

  for (const auto& fused_item : fused_nodes_and_graphs) {
    const Node& fused_node = fused_item.fused_node;
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [](ComputeContext*, FunctionState* state) {
      *state = nullptr;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState) {
    };

    compute_info.compute_func = [fused_node_name = fused_node.Name()](
        FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      ORT_UNUSED_PARAMETER(state);
      ORT_UNUSED_PARAMETER(api);

      Ort::KernelContext ctx(context);
      std::cout << "[NudgevEP] Computing node: " << fused_node_name << std::endl;

      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  std::cout << "[NudgevEP] ============= Compile End =============" << std::endl;
  return Status::OK();
}

}  // namespace onnxruntime
