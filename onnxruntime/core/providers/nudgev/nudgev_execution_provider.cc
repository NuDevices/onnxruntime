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
// #include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/tensor.h"
#include "core/framework/data_types.h"
#include <chrono>
#include <iostream>

namespace onnxruntime {

static void RegisterNudgevKernels(KernelRegistry& kernel_registry) {
  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      out = std::make_unique<nudgev::NudgevConv>(info);
      return Status::OK();
    };

    Status status = kernel_registry.Register(
        def_builder
            .SetName("Conv")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .TypeConstraint("X", DataTypeImpl::GetTensorType<int8_t>())
            .TypeConstraint("W", DataTypeImpl::GetTensorType<int8_t>())
            .TypeConstraint("B", DataTypeImpl::GetTensorType<int32_t>())
            .TypeConstraint("Y", DataTypeImpl::GetTensorType<int8_t>())
            .InputMemoryType(OrtMemTypeCPUInput, 0)
            .OutputMemoryType(OrtMemTypeCPUOutput, 0),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Conv");
  }
  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      out = std::make_unique<Memcpy>(info);
      return Status::OK();
    };

    Status status = kernel_registry.Register(
        def_builder
            .SetName("MemcpyFromHost")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .InputMemoryType(OrtMemTypeCPUInput, 0)
            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for MemcpyFromHost");
  }

  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      out = std::make_unique<Memcpy>(info);
      return Status::OK();
    };

    Status status = kernel_registry.Register(
        def_builder
            .SetName("MemcpyToHost")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .OutputMemoryType(OrtMemTypeCPUOutput, 0)
            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
        create_fn);
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
  if (session_options) {
    disable_cpu_ep_fallback_ = session_options->config_options.GetConfigOrDefault(
                                   kOrtSessionOptionsDisableCPUEPFallback, "0") == "1";
    // disable_cpu_ep_fallback_ = true;
    context_cache_enabled_ = session_options->config_options.GetConfigOrDefault(
                                 kOrtSessionOptionEpContextEnable, "0") == "1";
  }

  ORT_ENFORCE(ParseProviderOptions(provider_options_map).IsOK());
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
  // return DataLayout::NHWC;
  return DataLayout::NCHW;
}

std::vector<std::unique_ptr<ComputeCapability>>
NudgevExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/) const {
  std::cout << "[NudgevEP] ============= GetCapability Start =============" << std::endl;
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::unordered_set<const Node*> handled_nodes;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  for (const NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const Node* node = graph_viewer.GetNode(node_index);
    if (node->OpType() != "Conv") continue;

    if (handled_nodes.find(node) != handled_nodes.end()) {
      continue;
    }

    std::cout << "\n[NudgevEP] Found Conv node: " << node->Name() << std::endl;

    ConvQuantParams params{};

    // Get Conv attributes
    const auto& attributes = node->GetAttributes();

    // Get strides
    if (attributes.find("strides") != attributes.end()) {
      const auto& strides_attr = attributes.at("strides").ints();
      params.strides = std::vector<int64_t>(strides_attr.begin(), strides_attr.end());
    } else {
      params.strides = {1, 1};
    }
    std::cout << "Strides: [";
    for (size_t i = 0; i < params.strides.size(); ++i) {
      std::cout << params.strides[i];
      if (i < params.strides.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Get pads
    if (attributes.find("pads") != attributes.end()) {
      const auto& pads_attr = attributes.at("pads").ints();
      params.pads = std::vector<int64_t>(pads_attr.begin(), pads_attr.end());
    } else {
      params.pads = {0, 0, 0, 0};
    }
    std::cout << "Pads: [";
    for (size_t i = 0; i < params.pads.size(); ++i) {
      std::cout << params.pads[i];
      if (i < params.pads.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Get dilations
    if (attributes.find("dilations") != attributes.end()) {
      const auto& dilations_attr = attributes.at("dilations").ints();
      params.dilations = std::vector<int64_t>(dilations_attr.begin(), dilations_attr.end());
    } else {
      params.dilations = {1, 1};
    }
    std::cout << "Dilations: [";
    for (size_t i = 0; i < params.dilations.size(); ++i) {
      std::cout << params.dilations[i];
      if (i < params.dilations.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Get group
    if (attributes.find("group") != attributes.end()) {
      params.group = attributes.at("group").i();
    } else {
      params.group = 1;
    }
    std::cout << "Group: " << params.group << std::endl;

    // Get auto_pad
    if (attributes.find("auto_pad") != attributes.end()) {
      params.auto_pad = attributes.at("auto_pad").s();
    } else {
      params.auto_pad = "NOTSET";
    }
    std::cout << "Auto pad: " << params.auto_pad << std::endl;

    // Input parameters (DequantizeLinear)
    const Node* input_dq = graph_viewer.GetProducerNode(node->InputDefs()[0]->Name());
    if (!input_dq || input_dq->OpType() != "DequantizeLinear" ||
        handled_nodes.find(input_dq) != handled_nodes.end()) {
      continue;
    }

    const auto& input_qparams = input_dq->InputDefs();
    const auto* input_scale_init = initializers.at(input_qparams[1]->Name());
    const auto* input_zp_init = initializers.at(input_qparams[2]->Name());

    // Input scale
    if (input_scale_init->has_raw_data()) {
      params.input_scale = *reinterpret_cast<const float*>(input_scale_init->raw_data().data());
    } else if (input_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      params.input_scale = input_scale_init->float_data().empty() ? 0.0f : input_scale_init->float_data(0);
    }
    std::cout << "Input scale: " << params.input_scale << std::endl;

    // Input zero point
    if (input_zp_init->has_raw_data()) {
      params.input_zp = *reinterpret_cast<const int8_t*>(input_zp_init->raw_data().data());
    } else if (input_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      params.input_zp = static_cast<int8_t>(input_zp_init->int32_data().empty() ? 0 : input_zp_init->int32_data(0));
    }
    std::cout << "Input zero point: " << static_cast<int>(params.input_zp) << std::endl;

    // Weight parameters (DequantizeLinear)
    const Node* weight_dq = graph_viewer.GetProducerNode(node->InputDefs()[1]->Name());
    if (!weight_dq || weight_dq->OpType() != "DequantizeLinear" ||
        handled_nodes.find(weight_dq) != handled_nodes.end()) {
      continue;
    }

    const auto& weight_qparams = weight_dq->InputDefs();
    const auto* weight_scale_init = initializers.at(weight_qparams[1]->Name());
    const auto* weight_zp_init = initializers.at(weight_qparams[2]->Name());

    // Weight scale
    if (weight_scale_init->has_raw_data()) {
      params.weight_scale = *reinterpret_cast<const float*>(weight_scale_init->raw_data().data());
    } else if (weight_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      params.weight_scale = weight_scale_init->float_data().empty() ? 0.0f : weight_scale_init->float_data(0);
    }
    std::cout << "Weight scale: " << params.weight_scale << std::endl;

    // Weight zero point
    if (weight_zp_init->has_raw_data()) {
      params.weight_zp = *reinterpret_cast<const int8_t*>(weight_zp_init->raw_data().data());
    } else if (weight_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      params.weight_zp = static_cast<int8_t>(weight_zp_init->int32_data().empty() ? 0 : weight_zp_init->int32_data(0));
    }
    std::cout << "Weight zero point: " << static_cast<int>(params.weight_zp) << std::endl;

    // Get quantized weights
    const auto* weight_tensor = initializers.at(weight_dq->InputDefs()[0]->Name());
    params.weight_shape.assign(weight_tensor->dims().begin(), weight_tensor->dims().end());

    if (weight_tensor->has_raw_data()) {
      const auto& raw_data = weight_tensor->raw_data();
      params.weights.assign(
          reinterpret_cast<const int8_t*>(raw_data.data()),
          reinterpret_cast<const int8_t*>(raw_data.data() + raw_data.size()));
    } else {
      const auto& int8_data = weight_tensor->int32_data();
      params.weights.reserve(int8_data.size());
      for (int32_t val : int8_data) {
        params.weights.push_back(static_cast<int8_t>(val));
      }
    }

    // Bias parameters (optional DequantizeLinear)
    params.has_bias = false;
    const Node* bias_dq = nullptr;
    if (node->InputDefs().size() > 2) {
      bias_dq = graph_viewer.GetProducerNode(node->InputDefs()[2]->Name());
      if (bias_dq && bias_dq->OpType() == "DequantizeLinear") {
        params.has_bias = true;
        const auto& bias_qparams = bias_dq->InputDefs();
        const auto* bias_scale_init = initializers.at(bias_qparams[1]->Name());
        const auto* bias_zp_init = initializers.at(bias_qparams[2]->Name());

        // Bias scale
        if (bias_scale_init->has_raw_data()) {
          params.bias_scale = *reinterpret_cast<const float*>(bias_scale_init->raw_data().data());
        } else if (bias_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          params.bias_scale = bias_scale_init->float_data().empty() ? 0.0f : bias_scale_init->float_data(0);
        }

        // Bias zero point
        if (bias_zp_init->has_raw_data()) {
          params.bias_zp = *reinterpret_cast<const int8_t*>(bias_zp_init->raw_data().data());
        } else if (bias_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
          params.bias_zp = static_cast<int8_t>(bias_zp_init->int32_data().empty() ? 0 : bias_zp_init->int32_data(0));
        }

        // Get quantized bias
        const auto* bias_tensor = initializers.at(bias_dq->InputDefs()[0]->Name());
        params.bias_shape.assign(bias_tensor->dims().begin(), bias_tensor->dims().end());

        if (bias_tensor->has_raw_data()) {
          const auto& raw_data = bias_tensor->raw_data();
          params.bias.assign(
              reinterpret_cast<const int32_t*>(raw_data.data()),
              reinterpret_cast<const int32_t*>(raw_data.data() + raw_data.size()));
        } else {
          params.bias = std::vector<int32_t>(
              bias_tensor->int32_data().begin(),
              bias_tensor->int32_data().end());
        }
      }
    }

    // Get output quantization parameters from the next QuantizeLinear node
    const Node* output_q = nullptr;
    auto conv_consumers = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name());
    if (!conv_consumers.empty()) {
      output_q = conv_consumers[0];
      if (output_q && output_q->OpType() == "QuantizeLinear") {
        const auto& output_qparams = output_q->InputDefs();
        const auto* output_scale_init = initializers.at(output_qparams[1]->Name());
        const auto* output_zp_init = initializers.at(output_qparams[2]->Name());

        // Output scale
        if (output_scale_init->has_raw_data()) {
          params.output_scale = *reinterpret_cast<const float*>(output_scale_init->raw_data().data());
        } else if (output_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          params.output_scale = output_scale_init->float_data().empty() ? 0.0f : output_scale_init->float_data(0);
        }
        std::cout << "Output scale: " << params.output_scale << std::endl;

        // Output zero point
        if (output_zp_init->has_raw_data()) {
          params.output_zp = *reinterpret_cast<const int8_t*>(output_zp_init->raw_data().data());
        } else if (output_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
          params.output_zp = static_cast<int8_t>(output_zp_init->int32_data().empty() ? 0 : output_zp_init->int32_data(0));
        }
        std::cout << "Output zero point: " << static_cast<int>(params.output_zp) << std::endl;
      }
    }

    // Create fused nodes vector - include only Conv and its input DequantizeLinear nodes
    std::vector<const Node*> fused_nodes{input_dq, weight_dq, node};
    if (bias_dq) {
      fused_nodes.push_back(bias_dq);
    }
    if (output_q && output_q->OpType() == "QuantizeLinear") {
      fused_nodes.push_back(output_q);
    }
    uint64_t model_hash;
    int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);
    // This is a temporary (and awful) way to get right NudgevExecutionProvider node names into quant_params
    auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);
    std::cout << "Node name: " << node_name << std::endl;
    quant_params_map_[node_name] = params;

    result.push_back(utils::MakeComputeCapability(
        graph_viewer,
        fused_nodes,
        [model_hash, metadef_id]() {
          return MakeString(model_hash, "_", metadef_id);
        },
        kNudgevExecutionProvider,
        false));

    handled_nodes.insert(fused_nodes.begin(), fused_nodes.end());
  }

  std::cout << "[NudgevEP] Found " << result.size() << " Conv patterns" << std::endl;
  std::cout << "[NudgevEP] ============= GetCapability End ===============" << std::endl;
  return result;
}

Status NudgevExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    const Node& fused_node = fused_node_and_graph.fused_node;
    auto it = quant_params_map_.find(fused_node.Name());
    if (it == quant_params_map_.end()) {
      std::cout << "Error: no parameters found for node " << fused_node.Name() << std::endl;
      return Status(common::ONNXRUNTIME, common::FAIL, "parameters not found");
    }

    const auto& params = it->second;
    auto kernel_params = std::make_unique<ConvQuantParams>(params);
    auto params_copy = kernel_params.get();
    NodeComputeInfo compute_info;

    compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) {
      *state = params_copy;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState) {
    };

    compute_info.compute_func = [](FunctionState state,
                                   const OrtApi*,
                                   OrtKernelContext* context) -> Status {
      if (!state) {
        return Status(common::ONNXRUNTIME, common::FAIL, "state is null");
      }

      auto* params = static_cast<ConvQuantParams*>(state);
      if (!params) {
        return Status(common::ONNXRUNTIME, common::FAIL, "params cast failed");
      }
      if (params->weight_shape.empty()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "weight_shape is empty");
      }

      if (params->strides.empty()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "strides is empty");
      }

      if (params->pads.empty()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "pads is empty");
      }
      auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);

      // Get input tensor
      const Tensor* input = ctx_internal->Input<Tensor>(0);
      if (!input) return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");

      // Get input shapes
      const auto& input_shape = input->Shape();
      const int64_t batch_size = input_shape[0];
      const int64_t input_channels = input_shape[1];
      const int64_t input_height = input_shape[2];
      const int64_t input_width = input_shape[3];

      // Get weight shapes
      const int64_t output_channels = params->weight_shape[0];
      const int64_t kernel_height = params->weight_shape[2];
      const int64_t kernel_width = params->weight_shape[3];

      const int64_t output_height = (input_height + params->pads[0] + params->pads[2] - kernel_height) / params->strides[0] + 1;
      const int64_t output_width = (input_width + params->pads[1] + params->pads[3] - kernel_width) / params->strides[1] + 1;
      std::vector<int64_t> output_shape{batch_size, output_channels, output_height, output_width};

      Tensor* Y = ctx_internal->Output(0, output_shape);
      if (!Y) {
        return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
      }

      if (Y->DataType() != DataTypeImpl::GetType<int8_t>()) {
        std::cout << "Output tensor type mismatch - expected int8_t" << std::endl;
        return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be int8");
      }

      // Get data pointers
      const auto* input_data = input->Data<int8_t>();
      auto* output_data = Y->MutableData<int8_t>();

      // Get parameters
      const int8_t* weights_data = params->weights.data();
      const int32_t* bias_data = params->has_bias ? params->bias.data() : nullptr;

      // Calculate M = Xscale * Wscale / Yscale
      const float M = params->input_scale * params->weight_scale / params->output_scale;

      // 1. Im2row
      auto start = std::chrono::high_resolution_clock::now();
      const int64_t N = batch_size * output_height * output_width;
      const int64_t K = input_channels * kernel_height * kernel_width;

      std::vector<int32_t> im2row_data(N * K);

      // Im2row implementation
      for (int64_t n = 0; n < N; n++) {
        const int64_t b = n / (output_height * output_width);
        const int64_t oh = (n / output_width) % output_height;
        const int64_t ow = n % output_width;

        for (int64_t ic = 0; ic < input_channels; ic++) {
          for (int64_t kh = 0; kh < kernel_height; kh++) {
            for (int64_t kw = 0; kw < kernel_width; kw++) {
              const int64_t ih = oh * params->strides[0] - params->pads[0] + kh;
              const int64_t iw = ow * params->strides[1] - params->pads[1] + kw;

              const int64_t col_idx = n * K + (ic * kernel_height + kh) * kernel_width + kw;

              if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                const int64_t input_idx = ((b * input_channels + ic) * input_height + ih) * input_width + iw;
                im2row_data[col_idx] = static_cast<int32_t>(input_data[input_idx]) - params->input_zp;
              } else {
                im2row_data[col_idx] = -params->input_zp;
              }
            }
          }
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      std::cout << "Conv Op - Im2row execution time: " << duration.count() << " microseconds" << std::endl;
      std::cout << "Conv Op - Starting GEMM..." << std::endl;

      // 2. GEMM computation
      for (int64_t n = 0; n < N; n++) {
        const int64_t b = n / (output_height * output_width);
        const int64_t oh = (n / output_width) % output_height;
        const int64_t ow = n % output_width;

        for (int64_t oc = 0; oc < output_channels; oc++) {
          int32_t acc = params->has_bias ? bias_data[oc] : 0;

          for (int64_t k = 0; k < K; k++) {
            acc += (static_cast<int32_t>(weights_data[k * output_channels + oc]) - params->weight_zp) *
                   im2row_data[n * K + k];
          }

          float float_output = (acc * M) + params->output_zp;
          int32_t int_output = static_cast<int32_t>(std::round(float_output));
          if (params->fused_relu && int_output < 0) {
            int_output = 0;
          }
          int_output = std::min(127, std::max(-128, int_output));
          const int64_t out_idx = ((b * output_channels + oc) * output_height + oh) * output_width + ow;
          output_data[out_idx] = static_cast<int8_t>(int_output);
        }
      }

      std::cout << "Conv Op - Computation completed" << std::endl;
      return Status::OK();
    };

    saved_params_.push_back(std::move(kernel_params));
    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

}  // namespace onnxruntime
