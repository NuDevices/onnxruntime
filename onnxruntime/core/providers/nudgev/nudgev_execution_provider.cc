#include "core/providers/nudgev/nudgev_execution_provider.h"
#include "core/providers/nudgev/operators/nudgev_add.h"
#include "core/providers/nudgev/operators/nudgev_conv_accelerated.h"
#include "core/providers/nudgev/operators/supported_ops.h"
#include "core/providers/nudgev/operators/nudgev_gemm.h"
#include "core/providers/nudgev/operators/nudgev_dequantize.h"
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
#include <unistd.h>    
#include <fcntl.h>    
#include <sys/stat.h> 
#include <errno.h>     

#include "core/providers/nudgev/mock/mock_accelerator_memory.h"

namespace onnxruntime {

static void RegisterNudgevKernels(KernelRegistry& kernel_registry) {
  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      ORT_UNUSED_PARAMETER(info);
      ORT_UNUSED_PARAMETER(out);
      return Status::OK();
    };

    Status status = kernel_registry.Register(
        def_builder
            .SetName("Add")
            .SetDomain(kOnnxDomain)
            .SinceVersion(7)
            .Provider(kNudgevExecutionProvider)
            .TypeConstraint("T", DataTypeImpl::GetTensorType<int8_t>())
            .MayInplace(0, 0)  // Permette l'esecuzione in-place per il primo input
            .MayInplace(1, 0)  // Permette l'esecuzione in-place anche per il secondo input
            .InputMemoryType(OrtMemTypeCPUOutput, 0)
            .InputMemoryType(OrtMemTypeCPUOutput, 1)
            .OutputMemoryType(OrtMemTypeCPUInput, 0),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Add");
  }

  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      ORT_UNUSED_PARAMETER(info);
      ORT_UNUSED_PARAMETER(out);
      return Status::OK();
    };

    Status status = kernel_registry.Register(
        def_builder
            .SetName("DequantizeLinear")
            .SetDomain(kOnnxDomain)
            .SinceVersion(10)
            .Provider(kNudgevExecutionProvider)
            .TypeConstraint("x", {DataTypeImpl::GetTensorType<int8_t>()})
            .TypeConstraint("x_scale", DataTypeImpl::GetTensorType<float>())
            .TypeConstraint("x_zero_point", DataTypeImpl::GetTensorType<int8_t>())
            .TypeConstraint("y", DataTypeImpl::GetTensorType<float>())
            .MayInplace(0, 0)
            .InputMemoryType(OrtMemTypeCPUOutput, 0)
            .InputMemoryType(OrtMemTypeCPUOutput, 1)
            .InputMemoryType(OrtMemTypeCPUOutput, 2)
            .OutputMemoryType(OrtMemTypeCPUInput, 0),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for DequantizeLinear");
  }
  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      ORT_UNUSED_PARAMETER(info);
      ORT_UNUSED_PARAMETER(out);
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
            .MayInplace(0, 0)
            .InputMemoryType(OrtMemTypeCPUOutput, 0)
            .InputMemoryType(OrtMemTypeCPUOutput, 1)
            .InputMemoryType(OrtMemTypeCPUOutput, 2)
            .OutputMemoryType(OrtMemTypeCPUInput, 0),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Conv");
  }
  {
    KernelDefBuilder def_builder;
    auto create_fn = [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
      ORT_UNUSED_PARAMETER(func_mgr);
      ORT_UNUSED_PARAMETER(info);
      ORT_UNUSED_PARAMETER(out);
      return Status::OK();
    };

    Status status = kernel_registry.Register(
        def_builder
            .SetName("Gemm")
            .SetDomain(kOnnxDomain)
            .SinceVersion(9)
            .Provider(kNudgevExecutionProvider)
            .TypeConstraint("A", DataTypeImpl::GetTensorType<int8_t>())
            .TypeConstraint("B", DataTypeImpl::GetTensorType<int8_t>())
            .TypeConstraint("C", DataTypeImpl::GetTensorType<int32_t>())
            .TypeConstraint("Y", DataTypeImpl::GetTensorType<int8_t>())
            .MayInplace(0, 0)
            .InputMemoryType(OrtMemTypeCPUOutput, 0)
            .InputMemoryType(OrtMemTypeCPUOutput, 1)
            .InputMemoryType(OrtMemTypeCPUOutput, 2)
            .OutputMemoryType(OrtMemTypeCPUInput, 0),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Gemm");
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
            .InputMemoryType(OrtMemTypeCPUOutput, 0)
            .MayInplace(0, 0)
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
            .OutputMemoryType(OrtMemTypeCPUInput, 0)
            .MayInplace(0, 0)
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

NudgevExecutionProvider::~NudgevExecutionProvider() {
  for (const auto& param : saved_conv_params_) {
    if (param->weights_bias_fd >= 0) {
      close(param->weights_bias_fd);
    }
  }
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
        effective_batch_size = 8;
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
    
      const int64_t K = IC * KH * KW;
      const int64_t block_size = 32;
    
      int64_t k_blocks = (K + block_size - 1) / block_size;
      int64_t oc_blocks = (OC + block_size - 1) / block_size;
    
      int weights_bias_fd = open("/dev/xdma0_bypass", O_RDWR);
      if (weights_bias_fd == -1) {
          std::cerr << "Error: failed to open /dev/xdma0_bypass, errno=" << errno << std::endl;
          return result;
      }
    
      size_t weights_size = k_blocks * oc_blocks * block_size * block_size;
      std::vector<int8_t> fully_transposed(K * OC, 0);
      for (int64_t oc = 0; oc < OC; oc++) {
        for (int64_t k = 0; k < K; k++) {
          int64_t orig_idx = oc * K + k;
          int64_t trans_idx = k * OC + oc;
          fully_transposed[trans_idx] = params.weights[orig_idx];
        }
      }
      std::vector<int8_t> blocked_weights(weights_size, 0);
      for (int64_t kb = 0; kb < k_blocks; kb++) {
        for (int64_t ocb = 0; ocb < oc_blocks; ocb++) {
          int64_t block_idx = kb * oc_blocks + ocb;
          size_t block_offset = block_idx * block_size * block_size;
          int64_t actual_k_size = std::min(block_size, K - kb * block_size);
          int64_t actual_oc_size = std::min(block_size, OC - ocb * block_size);
          for (int64_t k_offset = 0; k_offset < actual_k_size; k_offset++) {
            for (int64_t oc_offset = 0; oc_offset < actual_oc_size; oc_offset++) {
              int64_t k_idx = kb * block_size + k_offset;
              int64_t oc_idx = ocb * block_size + oc_offset;
              int64_t trans_idx = k_idx * OC + oc_idx;
              size_t dest_idx = block_offset + k_offset * block_size + oc_offset;
              blocked_weights[dest_idx] = fully_transposed[trans_idx];
            }
          }
        }
      }

      void* weights_mapped_memory = mmap(nullptr, weights_size, PROT_READ | PROT_WRITE, MAP_SHARED, weights_bias_fd, 0);
      if (weights_mapped_memory == MAP_FAILED) {
      std::cerr << "Error: failed to mmap weights, errno=" << errno << std::endl;
      close(weights_bias_fd);
      return result;
      }
      std::memcpy(weights_mapped_memory, blocked_weights.data(), weights_size);
      if (msync(weights_mapped_memory, weights_size, MS_SYNC) != 0) {
      std::cerr << "Error: msync failed for weights, errno=" << errno << std::endl;
      munmap(weights_mapped_memory, weights_size);
      close(weights_bias_fd);
      return result;
      }

      params.weights_mapped_memory = weights_mapped_memory;
      params.weights_mapped_size = weights_size;


      if (params.has_bias) {
        std::vector<int32_t> bias_blocks(oc_blocks * block_size, 0);
        for (int64_t ocb = 0; ocb < oc_blocks; ocb++) {
            for (int64_t oc_offset = 0; oc_offset < block_size; oc_offset++) {
                int64_t oc_idx = ocb * block_size + oc_offset;
                if (oc_idx < OC) {
                    bias_blocks[ocb * block_size + oc_offset] = params.bias[oc_idx];
                }
            }
        }
        
        void* bias_mapped_memory = mmap(nullptr, bias_size, PROT_READ | PROT_WRITE, 
                                        MAP_SHARED, weights_bias_fd, params.BIAS_OFFSET);
        if (bias_mapped_memory == MAP_FAILED) {
            std::cerr << "Error: failed to mmap bias, errno=" << errno << std::endl;
            munmap(weights_mapped_memory, weights_size);
            close(weights_bias_fd);
            return result;
        }
        
        std::memcpy(bias_mapped_memory, bias_blocks.data(), bias_size);
        
        if (msync(bias_mapped_memory, bias_size, MS_SYNC) != 0) {
            std::cerr << "Error: msync failed for bias, errno=" << errno << std::endl;
            munmap(bias_mapped_memory, bias_size);
            munmap(weights_mapped_memory, weights_size);
            close(weights_bias_fd);
            return result;
        }
        
        params.bias_mapped_memory = bias_mapped_memory;
        params.bias_mapped_size = bias_size;
      }
      params.weights_bias_fd = weights_bias_fd;
      params.k_blocks = k_blocks;
      params.oc_blocks = oc_blocks;
    
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
    } else if (node->OpType() == "Gemm") {
      GemmParams params{};
      const auto& attributes = node->GetAttributes();

      // Get alpha attribute
      if (attributes.find("alpha") != attributes.end()) {
        params.alpha = attributes.at("alpha").f();
      } else {
        params.alpha = 1.0f;
      }

      // Get beta attribute
      if (attributes.find("beta") != attributes.end()) {
        params.beta = attributes.at("beta").f();
      } else {
        params.beta = 1.0f;
      }

      // Get transA attribute
      if (attributes.find("transA") != attributes.end()) {
        params.transA = attributes.at("transA").i() != 0;
      } else {
        params.transA = false;
      }

      // Get transB attribute
      if (attributes.find("transB") != attributes.end()) {
        params.transB = attributes.at("transB").i() != 0;
      } else {
        params.transB = false;
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

      // Get matrix shapes
      const auto* shape_proto_a = input_dq->InputDefs()[0]->Shape();
      const auto* shape_proto_b = weight_dq->InputDefs()[0]->Shape();

      // For A matrix
      const int64_t initial_batch_size = shape_proto_a->dim(0).dim_value();
      params.dynamic_batch = initial_batch_size == 0;
      params.batch_size = params.dynamic_batch ? 8 : initial_batch_size;

      // Calculate M, N, K dimensions based on input shapes and transpose flags
      if (!params.transA) {
        params.M = shape_proto_a->dim(0).dim_value();
        params.K = shape_proto_a->dim(1).dim_value();
      } else {
        params.M = shape_proto_a->dim(1).dim_value();
        params.K = shape_proto_a->dim(0).dim_value();
      }

      if (!params.transB) {
        params.N = shape_proto_b->dim(1).dim_value();
        // Verify that K dimensions match
        if (params.K != shape_proto_b->dim(0).dim_value()) {
          continue;
        }
      } else {
        params.N = shape_proto_b->dim(0).dim_value();
        // Verify that K dimensions match
        if (params.K != shape_proto_b->dim(1).dim_value()) {
          continue;
        }
      }

      // Get input quantization parameters
      const auto& input_qparams = input_dq->InputDefs();
      const auto* input_scale_init = initializers.at(input_qparams[1]->Name());
      const auto* input_zp_init = initializers.at(input_qparams[2]->Name());

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

      // Get weight quantization parameters
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

      // Get weights and transform to the expected format
      const auto* weight_tensor = initializers.at(weight_dq->InputDefs()[0]->Name());
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

      const size_t weight_size = params.N * params.K;
      params.weights_buffer.resize(weight_size);
      for (size_t i = 0; i < weight_size; ++i) {
        params.weights_buffer[i] = static_cast<int8_t>(original_weights[i] - params.weight_zp);
      }

      // Bias parameters (optional)
      if (node->InputDefs().size() > 2) {
        const Node* bias_dq = graph_viewer.GetProducerNode(node->InputDefs()[2]->Name());
        if (bias_dq && bias_dq->OpType() == "DequantizeLinear") {
          const auto* bias_tensor = initializers.at(bias_dq->InputDefs()[0]->Name());
          const size_t bias_size = bias_tensor->dims().empty() ? 0 : bias_tensor->dims()[0];
          params.bias_buffer.resize(bias_size);

          if (bias_tensor->has_raw_data()) {
            const auto& raw_data = bias_tensor->raw_data();
            std::memcpy(params.bias_buffer.data(), raw_data.data(), bias_size * sizeof(int32_t));
          } else {
            const auto& int32_data = bias_tensor->int32_data();
            std::memcpy(params.bias_buffer.data(), int32_data.data(), bias_size * sizeof(int32_t));
          }
        }
      }

      // Get output quantization parameters from the next QuantizeLinear node
      const Node* output_q = nullptr;
      auto gemm_consumers = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name());
      if (!gemm_consumers.empty()) {
        output_q = gemm_consumers[0];
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

      // Calculate scaling factor (M) for quantized gemm
      float M = params.input_scale * params.weight_scale / params.output_scale;
      params.M_fixed = static_cast<int32_t>(M * (1 << 15));

      // Create a unique node name for this GEMM node
      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);
      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);
      gemm_params_map_[node_name] = std::move(params);

      // Collect all the nodes that will be fused
      std::vector<const Node*> fused_nodes{input_dq, weight_dq, node};
      if (output_q && output_q->OpType() == "QuantizeLinear") {
        fused_nodes.push_back(output_q);
      }

      // Add compute capability
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
  for (const NodeIndex node_index : order) {
    const Node* node = graph_viewer.GetNode(node_index);

    if (handled_nodes.find(node) != handled_nodes.end()) {
      continue;
    }

    if (nudgev::ShouldSkipBeforeQuantized(node->OpType())) {
      bool skip_node = false;
      const std::string& output_name = node->OutputDefs()[0]->Name();
      auto consumer_nodes = graph_viewer.GetConsumerNodes(output_name);

      for (const auto* consumer : consumer_nodes) {
        if (nudgev::IsQuantizedOperatorSupported(consumer->OpType()) &&
            handled_nodes.find(consumer) == handled_nodes.end()) {
          LOGS_DEFAULT(INFO) << "Skipping " << node->OpType() << " node " << node->Name()
                             << " since it feeds into quantized operator " << consumer->Name();
          skip_node = true;
          break;
        }
      }

      if (skip_node) {
        handled_nodes.insert(node);
        continue;
      }
    }

    if (node->OpType() == "DequantizeLinear") {
      DequantizeLinearParams params{};

      const auto* input_shape = node->InputDefs()[0]->Shape();
      if (!input_shape) {
        continue;
      }

      const auto& dims = input_shape->dim();
      if (dims.empty()) {
        continue;
      }

      const auto* scale_tensor = initializers.at(node->InputDefs()[1]->Name());
      const auto* zp_tensor = initializers.at(node->InputDefs()[2]->Name());

      // Scale
      if (scale_tensor->has_raw_data()) {
        params.scale = *reinterpret_cast<const float*>(scale_tensor->raw_data().data());
      } else if (scale_tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        params.scale = scale_tensor->float_data().empty() ? 0.0f : scale_tensor->float_data(0);
      }

      // Zero point
      if (zp_tensor->has_raw_data()) {
        params.zero_point = *reinterpret_cast<const int8_t*>(zp_tensor->raw_data().data());
      } else if (zp_tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        params.zero_point = static_cast<int8_t>(zp_tensor->int32_data().empty() ? 0 : zp_tensor->int32_data(0));
      }

      for (size_t i = 0; i < static_cast<size_t>(dims.size()); i++) {
        params.input_shape.push_back(dims[i].dim_value());
      }

      if (params.input_shape.size() >= 4) {
        params.batch_size = params.input_shape[0];
        params.channels = params.input_shape[1];
        params.height = params.input_shape[2];
        params.width = params.input_shape[3];
      }

      if (params.batch_size == 0) {
        params.batch_size = 8;
        params.dynamic_batch = true;
      }

      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);
      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);

      dequantize_params_map_[node_name] = std::move(params);
      result.push_back(utils::MakeComputeCapability(
          graph_viewer,
          {node},
          [model_hash, metadef_id]() {
            return MakeString(model_hash, "_", metadef_id);
          },
          kNudgevExecutionProvider,
          false));

      handled_nodes.insert(node);
    } else if (node->OpType() == "Add23") {
      const Node* input1_dq = graph_viewer.GetProducerNode(node->InputDefs()[0]->Name());
      const Node* input2_dq = graph_viewer.GetProducerNode(node->InputDefs()[1]->Name());
      if (!input1_dq || input1_dq->OpType() != "DequantizeLinear" ||
          !input2_dq || input2_dq->OpType() != "DequantizeLinear") {
        continue;
      }
      AddParams params{};
      const auto* shape_proto_1 = input1_dq->InputDefs()[0]->Shape();
      const auto* shape_proto_2 = input2_dq->InputDefs()[0]->Shape();
      const auto& input1_qparams = input1_dq->InputDefs();
      const auto* input1_scale_init = initializers.at(input1_qparams[1]->Name());
      const auto* input1_zp_init = initializers.at(input1_qparams[2]->Name());

      if (input1_scale_init->has_raw_data()) {
        params.input1_scale = *reinterpret_cast<const float*>(input1_scale_init->raw_data().data());
      } else if (input1_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        params.input1_scale = input1_scale_init->float_data().empty() ? 0.0f : input1_scale_init->float_data(0);
      }

      if (input1_zp_init->has_raw_data()) {
        params.input1_zp = *reinterpret_cast<const int8_t*>(input1_zp_init->raw_data().data());
      } else if (input1_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        params.input1_zp = static_cast<int8_t>(input1_zp_init->int32_data().empty() ? 0 : input1_zp_init->int32_data(0));
      }
      const auto& input2_qparams = input2_dq->InputDefs();
      const auto* input2_scale_init = initializers.at(input2_qparams[1]->Name());
      const auto* input2_zp_init = initializers.at(input2_qparams[2]->Name());

      if (input2_scale_init->has_raw_data()) {
        params.input2_scale = *reinterpret_cast<const float*>(input2_scale_init->raw_data().data());
      } else if (input2_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        params.input2_scale = input2_scale_init->float_data().empty() ? 0.0f : input2_scale_init->float_data(0);
      }

      if (input2_zp_init->has_raw_data()) {
        params.input2_zp = *reinterpret_cast<const int8_t*>(input2_zp_init->raw_data().data());
      } else if (input2_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
        params.input2_zp = static_cast<int8_t>(input2_zp_init->int32_data().empty() ? 0 : input2_zp_init->int32_data(0));
      }

      const Node* output_q = nullptr;
      auto add_consumers = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name());
      if (!add_consumers.empty()) {
        output_q = add_consumers[0];
        if (output_q && output_q->OpType() == "QuantizeLinear") {
          const auto& output_qparams = output_q->InputDefs();
          const auto* output_scale_init = initializers.at(output_qparams[1]->Name());
          const auto* output_zp_init = initializers.at(output_qparams[2]->Name());

          if (output_scale_init->has_raw_data()) {
            params.output_scale = *reinterpret_cast<const float*>(output_scale_init->raw_data().data());
          } else if (output_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            params.output_scale = output_scale_init->float_data().empty() ? 0.0f : output_scale_init->float_data(0);
          }

          if (output_zp_init->has_raw_data()) {
            params.output_zp = *reinterpret_cast<const int8_t*>(output_zp_init->raw_data().data());
          } else if (output_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            params.output_zp = static_cast<int8_t>(output_zp_init->int32_data().empty() ? 0 : output_zp_init->int32_data(0));
          }
        }
      }
      std::cout << "BBB" << std::endl;
      params.M1_fixed = static_cast<int32_t>((params.input1_scale / params.output_scale) * (1 << 15));
      params.M2_fixed = static_cast<int32_t>((params.input2_scale / params.output_scale) * (1 << 15));

      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);
      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);
      std::cout << "GetCapability - Add node: " << node->Name()
                << ", generating unique name: " << node_name
                << std::endl;
      add_params_map_[node_name] = std::move(params);
      std::vector<const Node*> fused_nodes{input1_dq, input2_dq, node};
      if (output_q && output_q->OpType() == "QuantizeLinear") {
        fused_nodes.push_back(output_q);
      }
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
  // std::cout << "Compile - Starting with " << fused_nodes_and_graphs.size() << " fused nodes." << std::endl;
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    const Node& fused_node = fused_node_and_graph.fused_node;
    NodeComputeInfo compute_info;
    /*
    std::cout << "Compile - Fused node: " << fused_node.Name()
              << ", OpType: " << fused_node.OpType()
              << std::endl;
    */
    if (quant_params_map_.find(fused_node.Name()) != quant_params_map_.end()) {
      auto it = quant_params_map_.find(fused_node.Name());
      if (it == quant_params_map_.end()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Conv parameters not found");
      }

      auto kernel_params = std::make_unique<ConvQuantParams>(it->second);
      auto params_copy = kernel_params.get();

      compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) {
        *state = params_copy;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState) {};

      compute_info.compute_func = [](FunctionState state,
                                     const OrtApi*,
                                     OrtKernelContext* context) -> Status {
        return onnxruntime::nudgev::ComputeNudgeVConvWithAccelerator(
            static_cast<ConvQuantParams*>(state), context);
      };

      saved_conv_params_.push_back(std::move(kernel_params));
    } else if (gemm_params_map_.find(fused_node.Name()) != gemm_params_map_.end()) {
      auto it = gemm_params_map_.find(fused_node.Name());
      if (it == gemm_params_map_.end()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Gemm parameters not found");
      }
      auto kernel_params = std::make_unique<GemmParams>(std::move(it->second));
      auto params_copy = kernel_params.get();

      compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) {
        *state = params_copy;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState) {};

      compute_info.compute_func = [](FunctionState state,
                                     const OrtApi*,
                                     OrtKernelContext* context) -> Status {
        return onnxruntime::nudgev::ComputeNudgeVGemm(static_cast<GemmParams*>(state), context);
      };

      saved_gemm_params_.push_back(std::move(kernel_params));
    } else if (dequantize_params_map_.find(fused_node.Name()) != dequantize_params_map_.end()) {
      auto it = dequantize_params_map_.find(fused_node.Name());
      if (it == dequantize_params_map_.end()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "DequantizeLinear parameters not found");
      }
      auto kernel_params = std::make_unique<DequantizeLinearParams>(std::move(it->second));
      auto params_copy = kernel_params.get();

      compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) -> int {
        *state = params_copy;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState) {};
      compute_info.compute_func = [](FunctionState state,
                                     const OrtApi*,
                                     OrtKernelContext* context) -> Status {
        return onnxruntime::nudgev::ComputeNudgeVDequantizeLinear(
            static_cast<DequantizeLinearParams*>(state), context);
      };

      saved_dequantize_params_.push_back(std::move(kernel_params));
    } else if (add_params_map_.find(fused_node.Name()) != add_params_map_.end()) {
      auto it = add_params_map_.find(fused_node.Name());
      if (it == add_params_map_.end()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Add parameters not found");
      }
      auto kernel_params = std::make_unique<AddParams>(std::move(it->second));
      auto params_copy = kernel_params.get();

      compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) {
        *state = params_copy;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState) {};

      compute_info.compute_func = [](FunctionState state,
                                     const OrtApi*,
                                     OrtKernelContext* context) -> Status {
        return onnxruntime::nudgev::ComputeNudgeVAdd(
            static_cast<AddParams*>(state), context);
      };

      saved_add_params_.push_back(std::move(kernel_params));
    } else {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "Unsupported operator: " + fused_node.OpType());
    }

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

}  // namespace onnxruntime
