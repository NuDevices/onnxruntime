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
#include <iostream>

namespace onnxruntime {

static void RegisterNudgevKernels(KernelRegistry& kernel_registry) {
  KernelDefBuilder def_builder;
  auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    ORT_UNUSED_PARAMETER(func_mgr);
    out = std::make_unique<onnxruntime::nudgev::NudgevConv>(info);
    return Status::OK();
  };

  Status status = kernel_registry.Register(
      def_builder
          .SetName("Conv")
          .SetDomain(kOnnxDomain)
          .SinceVersion(1)
          .Provider(kNudgevExecutionProvider)
          .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>()),
      create_fn);

  ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Conv");
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
  if (session_options) {
    disable_cpu_ep_fallback_ = session_options->config_options.GetConfigOrDefault(
                                   kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";

    context_cache_enabled_ = session_options->config_options.GetConfigOrDefault(
                                 kOrtSessionOptionEpContextEnable, "0") == "1";
  }

  ORT_ENFORCE(ParseProviderOptions(provider_options_map).IsOK());

  LOGS_DEFAULT(INFO) << "NudgevExecutionProvider initialized with:"
                     << " device_id=" << device_id_
                     << " device_type=" << device_type_
                     << " support_uint8=" << support_uint8_
                     << " force_int8=" << force_int8_;
}

std::shared_ptr<KernelRegistry> NudgevExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  if (!kernel_registry_initialized_) {
    kernel_registry_initialized_ = true;
    RegisterNudgevKernels(*kernel_registry);
  }
  return kernel_registry;
}

DataLayout NudgevExecutionProvider::GetPreferredLayout() const {
  return DataLayout::NHWC;
}

std::vector<std::unique_ptr<ComputeCapability>>
NudgevExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                       const IKernelLookup& kernel_lookup) const {
  LOGS_DEFAULT(INFO) << "[NudgevEP] Entered GetCapability";
  ORT_UNUSED_PARAMETER(kernel_lookup);
  if (graph_viewer.IsSubgraph()) {
    return {};
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;

  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer);

  std::unordered_set<const Node*> supported_nodes;

  for (auto& nu : node_unit_holder) {
    if (nu->OpType() == "Conv") {
      bool all_inputs_quantized = true;
      for (auto* input_arg : nu->GetNode().InputDefs()) {
        if (!input_arg->Type()) {
          all_inputs_quantized = false;
          break;
        }
        const auto& type_str = *input_arg->Type();
        bool type_supported = false;
        if (type_str.find("int8") != std::string::npos) {
          type_supported = true;
        } else if (support_uint8_ && type_str.find("uint8") != std::string::npos) {
          if (!force_int8_) {
            type_supported = true;
          }
        }

        if (!type_supported) {
          all_inputs_quantized = false;
          break;
        }
      }

      if (all_inputs_quantized) {
        supported_nodes.insert(&nu->GetNode());
        LOGS_DEFAULT(INFO) << "[NudgevEP] Found supported Q/DQ Conv node: "
                           << nu->GetNode().Name()
                           << " with quantized inputs";
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
    }
  }

  return result;
}

Status NudgevExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_item : fused_nodes_and_graphs) {
    const Node& fused_node = fused_item.fused_node;
    ORT_UNUSED_PARAMETER(fused_item.filtered_graph);

    LOGS_DEFAULT(INFO) << "[NudgevEP] Compiling fused node: " << fused_node.Name()
                       << " (Type: " << fused_node.OpType() << ")";

    NodeComputeInfo compute_info;

    compute_info.compute_func = [this, fused_node_name = fused_node.Name()](
                                    FunctionState state, const OrtApi* api, OrtKernelContext* context) -> Status {
      ORT_UNUSED_PARAMETER(state);
      ORT_UNUSED_PARAMETER(api);
      ORT_UNUSED_PARAMETER(context);

      LOGS_DEFAULT(INFO) << "[NudgevEP] Executing compute for node: " << fused_node_name;
      return Status::OK();
    };

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

}  // namespace onnxruntime