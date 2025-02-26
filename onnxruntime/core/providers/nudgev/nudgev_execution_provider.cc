#include "core/providers/nudgev/nudgev_execution_provider.h"
#include "core/providers/nudgev/operators/nudgev_conv.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/common/logging/logging.h"
#include "core/providers/partitioning_utils.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/tensor.h"
#include "core/framework/data_types.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <immintrin.h>
#include "core/platform/threadpool.h"
#include "core/platform/ort_mutex.h"
#include <algorithm>
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/quantization/quantize_linear.cc"

#include "core/providers/cpu/nn/pool.h"
#include "core/providers/cpu/nn/pool_functors.h"

namespace onnxruntime {

static void RegisterNudgevKernels(KernelRegistry& kernel_registry) {
  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      ORT_UNUSED_PARAMETER(info);
      ORT_UNUSED_PARAMETER(out);
      // out = std::make_unique<nudgev::NudgevConv>(info);
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
            .InputMemoryType(OrtMemTypeDefault, 0)
            .OutputMemoryType(OrtMemTypeDefault, 0),
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

std::unique_ptr<IDataTransfer> NudgevExecutionProvider::GetDataTransfer() const {
  return std::make_unique<NudgevDataTransfer>();
}

OrtDevice NudgevExecutionProvider::GetOrtDeviceByMemType(OrtMemType mem_type) const {
  // For CPU input operations, use regular CPU device
  if (mem_type == OrtMemTypeCPUInput) {
    return OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, device_id_);
  }

  // For CPU output operations, use regular CPU device
  if (mem_type == OrtMemTypeCPUOutput) {
    return OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, device_id_);
  }

  // For default operations, use the NudgeV device
  return OrtDevice(kNudgevDeviceType, OrtDevice::MemType::DEFAULT, device_id_);
}

std::vector<AllocatorPtr> NudgevExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;

  auto device_allocator = std::make_unique<NudgevAllocator>(device_id_, NUDGEV);
  allocators.push_back(std::move(device_allocator));

  return allocators;
}

// cpu allocator
/*
std::vector<AllocatorPtr> NudgevExecutionProvider::CreatePreferredAllocators() {
  std::vector<AllocatorPtr> allocators;

  allocators.push_back(CreateCPUAllocator(device_id_));

  return allocators;
}

AllocatorPtr NudgevExecutionProvider::CreateCPUAllocator(OrtDevice::DeviceId device_id) {
  return std::make_unique<CPUAllocator>(
      OrtMemoryInfo("CPU", OrtAllocatorType::OrtDeviceAllocator,
                    OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, device_id),
                    device_id, OrtMemTypeDefault));
}
*/
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
  return DataLayout::NCHW;
}

std::vector<std::unique_ptr<ComputeCapability>>
NudgevExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                       const IKernelLookup& /*kernel_lookup*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::unordered_set<const Node*> handled_nodes;

  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  const auto& initializers = graph_viewer.GetAllInitializedTensors();
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (const NodeIndex node_index : order) {
    const Node* node = graph_viewer.GetNode(node_index);

    if (handled_nodes.find(node) != handled_nodes.end()) {
      continue;
    }
    if (node->OpType() == "Conv") {
      ConvQuantParams params{};
      const auto& attributes = node->GetAttributes();

      // Check if Conv is fused with Relu
      std::string conv_output_name = node->OutputDefs()[0]->Name();
      bool has_relu = conv_output_name.find("Relu") != std::string::npos;
      params.fused_relu = has_relu;

      // Get strides
      if (attributes.find("strides") != attributes.end()) {
        const auto& strides_attr = attributes.at("strides").ints();
        params.strides = std::vector<int64_t>(strides_attr.begin(), strides_attr.end());
      } else {
        params.strides = {1, 1};
      }

      // Get pads
      if (attributes.find("pads") != attributes.end()) {
        const auto& pads_attr = attributes.at("pads").ints();
        params.pads = std::vector<int64_t>(pads_attr.begin(), pads_attr.end());
      } else {
        params.pads = {0, 0, 0, 0};
      }

      // Get dilations
      if (attributes.find("dilations") != attributes.end()) {
        const auto& dilations_attr = attributes.at("dilations").ints();
        params.dilations = std::vector<int64_t>(dilations_attr.begin(), dilations_attr.end());
      } else {
        params.dilations = {1, 1};
      }

      // Get group
      if (attributes.find("group") != attributes.end()) {
        params.group = attributes.at("group").i();
      } else {
        params.group = 1;
      }

      // Get auto_pad
      if (attributes.find("auto_pad") != attributes.end()) {
        params.auto_pad = attributes.at("auto_pad").s();
      } else {
        params.auto_pad = "NOTSET";
      }

      // Input parameters (DequantizeLinear)
      const Node* input_dq = graph_viewer.GetProducerNode(node->InputDefs()[0]->Name());
      if (!input_dq || input_dq->OpType() != "DequantizeLinear" ||
          handled_nodes.find(input_dq) != handled_nodes.end()) {
        continue;
      }

      // Weight parameters (DequantizeLinear)
      const Node* weight_dq = graph_viewer.GetProducerNode(node->InputDefs()[1]->Name());
      if (!weight_dq || weight_dq->OpType() != "DequantizeLinear" ||
          handled_nodes.find(weight_dq) != handled_nodes.end()) {
        continue;
      }

      // Get weights dimensions first
      const auto* weight_tensor = initializers.at(weight_dq->InputDefs()[0]->Name());
      const auto& weight_dims = weight_tensor->dims();
      const int64_t OC = weight_dims[0];
      const int64_t IC = weight_dims[1];
      const int64_t KH = weight_dims[2];
      const int64_t KW = weight_dims[3];

      // Get input shape
      const auto* shape_proto = input_dq->InputDefs()[0]->Shape();
      const int64_t initial_batch_size = shape_proto->dim(0).dim_value();
      const int64_t input_channels = shape_proto->dim(1).dim_value();
      const int64_t input_height = shape_proto->dim(2).dim_value();
      const int64_t input_width = shape_proto->dim(3).dim_value();

      // Calculate oh and ow shapes
      params.output_height = static_cast<int64_t>(std::floor(
          (input_height + params.pads[0] + params.pads[2] - KH) / static_cast<double>(params.strides[0]) + 1));

      params.output_width = static_cast<int64_t>(std::floor(
          (input_width + params.pads[1] + params.pads[3] - KW) / static_cast<double>(params.strides[1]) + 1));

      int64_t effective_batch_size = initial_batch_size;
      if (initial_batch_size == 0) {
        effective_batch_size = 16;
        params.dynamic_batch = true;
      }

      const int64_t max_padded_height = input_height + params.pads[0] + params.pads[2];
      const int64_t max_padded_width = input_width + params.pads[1] + params.pads[3];
      params.N = effective_batch_size * params.output_height * params.output_width;
      params.K = input_channels * KH * KW;
      const size_t temp_buffer_size = params.N * OC;
      const size_t input_size = effective_batch_size * input_channels * input_height * input_width;
      const size_t padded_input = effective_batch_size * input_channels * max_padded_height * max_padded_width;
      params.padded_buffer.resize(padded_input);
      params.input_centered_buffer.resize(input_size);
      params.temp_buffer.resize(temp_buffer_size);

      const auto& input_qparams = input_dq->InputDefs();
      const auto* input_scale_init = initializers.at(input_qparams[1]->Name());
      const auto* input_zp_init = initializers.at(input_qparams[2]->Name());
      Status status = params.initialize_buffers(
          {OC, IC, KH, KW},
          params.has_bias ? std::vector<int64_t>{OC} : std::vector<int64_t>{},
          effective_batch_size,
          params.output_height,
          params.output_width);

      if (!status.IsOK()) {
        return result;
      }

      // Input scale
      if (input_scale_init->has_raw_data()) {
        params.input_scale = *reinterpret_cast<const float*>(input_scale_init->raw_data().data());
      } else if (input_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        params.input_scale = input_scale_init->float_data().empty() ? 0.0f : input_scale_init->float_data(0);
      }

      // Input zero point
      if (input_zp_init->has_raw_data()) {
        params.input_zp = *reinterpret_cast<const int8_t*>(input_zp_init->raw_data().data());
      } else if (input_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        params.input_zp = static_cast<int8_t>(input_zp_init->int32_data().empty() ? 0 : input_zp_init->int32_data(0));
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

      // Weight zero point
      if (weight_zp_init->has_raw_data()) {
        params.weight_zp = *reinterpret_cast<const int8_t*>(weight_zp_init->raw_data().data());
      } else if (weight_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        params.weight_zp = static_cast<int8_t>(weight_zp_init->int32_data().empty() ? 0 : weight_zp_init->int32_data(0));
      }

      // Get quantized weights and reshape them
      params.weight_shape = std::vector<int64_t>{OC, IC, KH, KW};
      std::vector<int8_t> original_weights;

      if (weight_tensor->has_raw_data()) {
        const auto& raw_data = weight_tensor->raw_data();
        original_weights.assign(
            reinterpret_cast<const int8_t*>(raw_data.data()),
            reinterpret_cast<const int8_t*>(raw_data.data() + raw_data.size()));
      } else {
        const auto& int8_data = weight_tensor->int32_data();
        original_weights.reserve(int8_data.size());
        for (int32_t val : int8_data) {
          original_weights.push_back(static_cast<int8_t>(val));
        }
      }

      params.weights.resize(OC * IC * KH * KW);
      for (int64_t i = 0; i < OC * IC * KH * KW; ++i) {
        params.weights[i] = static_cast<int8_t>(original_weights[i] - params.weight_zp);
      }

      // Bias parameters (optional DequantizeLinear)
      params.has_bias = false;
      const Node* bias_dq = nullptr;
      if (node->InputDefs().size() > 2) {
        bias_dq = graph_viewer.GetProducerNode(node->InputDefs()[2]->Name());
        if (bias_dq && bias_dq->OpType() == "DequantizeLinear") {
          params.has_bias = true;
          const auto* bias_tensor = initializers.at(bias_dq->InputDefs()[0]->Name());
          params.bias_shape.assign(bias_tensor->dims().begin(), bias_tensor->dims().end());
          const size_t bias_size = bias_tensor->dims().empty() ? 0 : bias_tensor->dims()[0];
          params.bias.resize(bias_size);
          if (bias_tensor->has_raw_data()) {
            const auto& raw_data = bias_tensor->raw_data();
            std::memcpy(params.bias.data(), raw_data.data(), bias_size * sizeof(int32_t));
          } else {
            const auto& int32_data = bias_tensor->int32_data();
            std::memcpy(params.bias.data(), int32_data.data(), bias_size * sizeof(int32_t));
          }
          const auto& bias_qparams = bias_dq->InputDefs();
          const auto* bias_scale_init = initializers.at(bias_qparams[1]->Name());
          const auto* bias_zp_init = initializers.at(bias_qparams[2]->Name());

          if (bias_scale_init->has_raw_data()) {
            params.bias_scale = *reinterpret_cast<const float*>(bias_scale_init->raw_data().data());
          } else if (bias_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            params.bias_scale = bias_scale_init->float_data().empty() ? 0.0f : bias_scale_init->float_data(0);
          }

          if (bias_zp_init->has_raw_data()) {
            params.bias_zp = *reinterpret_cast<const int8_t*>(bias_zp_init->raw_data().data());
          } else if (bias_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            params.bias_zp = static_cast<int8_t>(bias_zp_init->int32_data().empty() ? 0 : bias_zp_init->int32_data(0));
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
          // Output zero point
          if (output_zp_init->has_raw_data()) {
            params.output_zp = *reinterpret_cast<const int8_t*>(output_zp_init->raw_data().data());
          } else if (output_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            params.output_zp = static_cast<int8_t>(output_zp_init->int32_data().empty() ? 0 : output_zp_init->int32_data(0));
          }
        }
      }

      float M = params.input_scale * params.weight_scale / params.output_scale;
      params.M_fixed = static_cast<int32_t>(M * (1 << 15));
      params.M = params.input_scale * params.weight_scale / params.output_scale;
      std::fill(params.padded_buffer.begin(), params.padded_buffer.end(), params.input_zp);
      params.node_name = node->Name();

      std::vector<const Node*> fused_nodes{input_dq, weight_dq, node};
      if (bias_dq) {
        fused_nodes.push_back(bias_dq);
      }
      if (output_q && output_q->OpType() == "QuantizeLinear") {
        fused_nodes.push_back(output_q);
      }

      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);

      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);
      quant_params_map_[node_name] = std::move(params);

      result.push_back(utils::MakeComputeCapability(
          graph_viewer,
          fused_nodes,
          [model_hash, metadef_id]() {
            return MakeString(model_hash, "_", metadef_id);
          },
          kNudgevExecutionProvider,
          false));

      handled_nodes.insert(fused_nodes.begin(), fused_nodes.end());
      continue;
    }
  }

  return result;
}

Status NudgevExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    const Node& fused_node = fused_node_and_graph.fused_node;
    NodeComputeInfo compute_info;

    if (quant_params_map_.find(fused_node.Name()) != quant_params_map_.end()) {
      auto it = quant_params_map_.find(fused_node.Name());
      if (it == quant_params_map_.end()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Conv parameters not found");
      }
      auto kernel_params = std::make_unique<ConvQuantParams>(std::move(it->second));
      auto params_copy = kernel_params.get();

      compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) {
        *state = params_copy;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState) {};

      compute_info.compute_func = [](FunctionState state,
                                     const OrtApi*,
                                     OrtKernelContext* context) -> Status {
        return onnxruntime::nudgev::ComputeNudgeVConv(static_cast<ConvQuantParams*>(state), context);
      };

      saved_conv_params_.push_back(std::move(kernel_params));
    } else {
      // Operatore non supportato
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "Unsupported operator: " + fused_node.OpType());
    }

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>
NudgevExecutionProvider::GetDeviceCopyMap() const {
  // This tells ONNX Runtime that the CPU and NudgeV devices can directly access
  // each other's memory, enabling zero-copy transfers

  // Map structure: {src_device_type, {dst_device_type, {memory_types}}}
  // The memory_types set {0} means "default memory type only"
  return {
    {OrtDevice::CPU, {{kNudgevDeviceType, {0}}}},  // CPU can access NudgeV memory
    {kNudgevDeviceType, {{OrtDevice::CPU, {0}}}}   // NudgeV can access CPU memory
  };
}
}  // namespace onnxruntime
