#include "core/providers/nudgev/nudgev_execution_provider.h"
#include "core/providers/nudgev/operators/nudgev_conv.h"
#include "core/providers/nudgev/operators/nudgev_gemm.h"
#include "core/providers/nudgev/operators/nudgev_maxpool.h"
#include "core/providers/nudgev/operators/nudgev_add.h"
#include "core/providers/nudgev/operators/nudgev_sigmoid.h"
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
            .SetName("Sigmoid")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .InputMemoryType(OrtMemTypeCPUInput, 0)
            .OutputMemoryType(OrtMemTypeCPUOutput, 0),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Sigmoid");
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
            .SetName("Add")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .InputMemoryType(OrtMemTypeCPUInput, 0)
            .InputMemoryType(OrtMemTypeCPUInput, 1)
            .OutputMemoryType(OrtMemTypeCPUOutput, 0),
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
            .SetName("MaxPool")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .InputMemoryType(OrtMemTypeCPUInput, 0)
            .OutputMemoryType(OrtMemTypeCPUOutput, 0),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for MaxPool");
  }
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
            .SetName("Gemm")
            .SetDomain(kOnnxDomain)
            .SinceVersion(1)
            .Provider(kNudgevExecutionProvider)
            .TypeConstraint("X", DataTypeImpl::GetTensorType<int8_t>())
            .TypeConstraint("W", DataTypeImpl::GetTensorType<int8_t>())
            .TypeConstraint("B", DataTypeImpl::GetTensorType<int32_t>())
            .TypeConstraint("Y", DataTypeImpl::GetTensorType<int8_t>()),
        create_fn);

    ORT_ENFORCE(status.IsOK(), "Failed to register NUDGEV kernel for Gemm");
  }

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
  for (const NodeIndex node_index : order) {
    const Node* node = graph_viewer.GetNode(node_index);
    if (handled_nodes.find(node) != handled_nodes.end()) {
      continue;
    }

    if (node->OpType() == "Gemm") {
      if (handled_nodes.find(node) != handled_nodes.end()) {
        continue;
      }
      GemmParams params{};

      // Get attributes
      const auto& attributes = node->GetAttributes();
      if (attributes.find("alpha") != attributes.end()) {
        params.alpha = attributes.at("alpha").f();
      }
      if (attributes.find("beta") != attributes.end()) {
        params.beta = attributes.at("beta").f();
      }
      if (attributes.find("transA") != attributes.end()) {
        params.transA = attributes.at("transA").i() != 0;
      }
      if (attributes.find("transB") != attributes.end()) {
        params.transB = attributes.at("transB").i() != 0;
      }

      // Input parameters (DequantizeLinear)
      const Node* input_dq = graph_viewer.GetProducerNode(node->InputDefs()[0]->Name());
      if (!input_dq || input_dq->OpType() != "DequantizeLinear" ||
          handled_nodes.find(input_dq) != handled_nodes.end()) {
        continue;
      }

      const auto* shape_proto = input_dq->InputDefs()[0]->Shape();
      const int64_t initial_batch_size = shape_proto->dim(0).dim_value();
      int64_t effective_batch_size = initial_batch_size;
      if (initial_batch_size == 0) {
        effective_batch_size = 8;
        params.dynamic_batch = true;
      }
      params.batch_size = effective_batch_size;

      // Weight parameters (DequantizeLinear)
      const Node* weight_dq = graph_viewer.GetProducerNode(node->InputDefs()[1]->Name());
      if (!weight_dq || weight_dq->OpType() != "DequantizeLinear" ||
          handled_nodes.find(weight_dq) != handled_nodes.end()) {
        continue;
      }

      // Get weight dimensions
      const auto* weight_tensor = initializers.at(weight_dq->InputDefs()[0]->Name());
      if (!weight_tensor) {
        continue;
      }

      const auto& weight_dims = weight_tensor->dims();
      params.M = params.transA ? weight_dims[1] : weight_dims[0];
      params.K = params.transA ? weight_dims[0] : weight_dims[1];
      params.N = params.transB ? weight_dims[0] : weight_dims[1];

      // Get input quantization parameters
      params.input_centered_buffer.resize(params.M * params.K);
      const auto& input_qparams = input_dq->InputDefs();
      const auto* input_scale_init = initializers.at(input_qparams[1]->Name());
      const auto* input_zp_init = initializers.at(input_qparams[2]->Name());

      if (input_scale_init->has_raw_data()) {
        params.input_scale = *reinterpret_cast<const float*>(input_scale_init->raw_data().data());
      } else {
        params.input_scale = input_scale_init->float_data().empty() ? 0.0f : input_scale_init->float_data(0);
      }

      if (input_zp_init->has_raw_data()) {
        params.input_zp = *reinterpret_cast<const int8_t*>(input_zp_init->raw_data().data());
      } else {
        params.input_zp = static_cast<int8_t>(input_zp_init->int32_data().empty() ? 0 : input_zp_init->int32_data(0));
      }

      // Get weight quantization parameters
      const auto& weight_qparams = weight_dq->InputDefs();
      const auto* weight_scale_init = initializers.at(weight_qparams[1]->Name());
      const auto* weight_zp_init = initializers.at(weight_qparams[2]->Name());

      if (weight_scale_init->has_raw_data()) {
        params.weight_scale = *reinterpret_cast<const float*>(weight_scale_init->raw_data().data());
      } else {
        params.weight_scale = weight_scale_init->float_data().empty() ? 0.0f : weight_scale_init->float_data(0);
      }

      if (weight_zp_init->has_raw_data()) {
        params.weight_zp = *reinterpret_cast<const int8_t*>(weight_zp_init->raw_data().data());
      } else {
        params.weight_zp = static_cast<int8_t>(weight_zp_init->int32_data().empty() ? 0 : weight_zp_init->int32_data(0));
      }

      // Get quantized weights and reshape them
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

      // Initialize weights_buffer with centered weights
      params.weights_buffer.resize(params.K * params.N);
      for (int64_t i = 0; i < params.K * params.N; ++i) {
        params.weights_buffer[i] = static_cast<int8_t>(original_weights[i] - params.weight_zp);
      }

      // Get bias if present
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

      // Get output quantization parameters
      const Node* output_q = nullptr;
      auto gemm_consumers = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name());
      if (!gemm_consumers.empty()) {
        output_q = gemm_consumers[0];
        if (output_q && output_q->OpType() == "QuantizeLinear") {
          const auto& output_qparams = output_q->InputDefs();
          const auto* output_scale_init = initializers.at(output_qparams[1]->Name());
          const auto* output_zp_init = initializers.at(output_qparams[2]->Name());

          if (output_scale_init->has_raw_data()) {
            params.output_scale = *reinterpret_cast<const float*>(output_scale_init->raw_data().data());
          } else {
            params.output_scale = output_scale_init->float_data().empty() ? 0.0f : output_scale_init->float_data(0);
          }

          if (output_zp_init->has_raw_data()) {
            params.output_zp = *reinterpret_cast<const int8_t*>(output_zp_init->raw_data().data());
          } else {
            params.output_zp = static_cast<int8_t>(output_zp_init->int32_data().empty() ? 0 : output_zp_init->int32_data(0));
          }

        } else {
          continue;
        }
      }

      // Calculate quantization multiplier
      float M = params.input_scale * params.weight_scale / params.output_scale;
      params.M_fixed = static_cast<int32_t>(M * (1 << 15));
      // Create fused nodes vector
      std::vector<const Node*> fused_nodes{input_dq, weight_dq, node};
      if (output_q && output_q->OpType() == "QuantizeLinear") {
        fused_nodes.push_back(output_q);
      }

      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);

      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);
      gemm_params_map_[node_name] = std::move(params);

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
    } else if (node->OpType() == "Sigmoid") {
      if (handled_nodes.find(node) != handled_nodes.end()) {
        continue;
      }
      SigmoidParams params{};

      const auto* shape_proto = node->InputDefs()[0]->Shape();
      const int64_t dims = shape_proto->dim_size();

      if (dims != 3 && dims != 4) {
        continue;
      }

      const int64_t initial_batch_size = shape_proto->dim(0).dim_value();
      params.batch_size = initial_batch_size;
      if (initial_batch_size == 0) {
        params.batch_size = 8;
        params.dynamic_batch = true;
      }

      if (dims == 4) {
        params.channels = shape_proto->dim(1).dim_value();
        params.height = shape_proto->dim(2).dim_value();
        params.width = shape_proto->dim(3).dim_value();
      } else {
        params.channels = 1;
        params.height = shape_proto->dim(1).dim_value();
        params.width = shape_proto->dim(2).dim_value();
      }

      std::vector<const Node*> fused_nodes{node};

      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);

      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);
      sigmoid_params_map_[node_name] = std::move(params);

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
    } else if (node->OpType() == "MaxPool") {
      if (handled_nodes.find(node) != handled_nodes.end()) {
        continue;
      }
      MaxPoolParams params{};
      params.needs_quantization = false;  // Inizializziamo il flag a false

      // Get attributes
      const auto& attributes = node->GetAttributes();

      // Get kernel_shape
      if (attributes.find("kernel_shape") != attributes.end()) {
        const auto& kernel_shape_attr = attributes.at("kernel_shape").ints();
        params.kernel_shape = std::vector<int64_t>(kernel_shape_attr.begin(), kernel_shape_attr.end());
      } else {
        continue;
      }

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

      // Get other attributes
      if (attributes.find("ceil_mode") != attributes.end()) {
        params.ceil_mode = attributes.at("ceil_mode").i() != 0;
      }
      if (attributes.find("count_include_pad") != attributes.end()) {
        params.count_include_pad = attributes.at("count_include_pad").i() != 0;
      }
      if (attributes.find("storage_order") != attributes.end()) {
        params.storage_order = attributes.at("storage_order").i();
      }

      // Input parameters check for quantization
      const Node* input_dq = graph_viewer.GetProducerNode(node->InputDefs()[0]->Name());
      if (!input_dq || input_dq->OpType() != "DequantizeLinear" ||
          handled_nodes.find(input_dq) != handled_nodes.end()) {
        // Input is not quantized (float) or DequantizeLinear was already handled
        // Skip this node to let it fall back to CPU execution provider
        continue;
      }
      params.needs_quantization = true;

      // Get input shape
      const auto* shape_proto = params.needs_quantization ? input_dq->InputDefs()[0]->Shape() : node->InputDefs()[0]->Shape();

      const int64_t initial_batch_size = shape_proto->dim(0).dim_value();
      int64_t effective_batch_size = initial_batch_size;
      if (initial_batch_size == 0) {
        effective_batch_size = 8;
        params.dynamic_batch = true;
      }
      params.batch_size = effective_batch_size;

      params.channels = shape_proto->dim(1).dim_value();
      params.input_height = shape_proto->dim(2).dim_value();
      params.input_width = shape_proto->dim(3).dim_value();

      // Calculate output dimensions
      params.output_height = static_cast<int64_t>(std::floor(
          (params.input_height + params.pads[0] + params.pads[2] - params.kernel_shape[0]) /
              static_cast<double>(params.strides[0]) +
          1));

      params.output_width = static_cast<int64_t>(std::floor(
          (params.input_width + params.pads[1] + params.pads[3] - params.kernel_shape[1]) /
              static_cast<double>(params.strides[1]) +
          1));

      if (params.needs_quantization) {
        // Get input quantization parameters
        const auto& input_qparams = input_dq->InputDefs();
        const auto* input_scale_init = initializers.at(input_qparams[1]->Name());
        const auto* input_zp_init = initializers.at(input_qparams[2]->Name());

        if (input_scale_init->has_raw_data()) {
          params.input_scale = *reinterpret_cast<const float*>(input_scale_init->raw_data().data());
        } else {
          params.input_scale = input_scale_init->float_data().empty() ? 0.0f : input_scale_init->float_data(0);
        }

        if (input_zp_init->has_raw_data()) {
          params.input_zp = *reinterpret_cast<const int8_t*>(input_zp_init->raw_data().data());
        } else {
          params.input_zp = static_cast<int8_t>(input_zp_init->int32_data().empty() ? 0 : input_zp_init->int32_data(0));
        }

        // Check output quantization
        const Node* output_q = nullptr;
        auto maxpool_consumers = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name());
        if (!maxpool_consumers.empty()) {
          output_q = maxpool_consumers[0];
          if (output_q && output_q->OpType() == "QuantizeLinear") {
            const auto& output_qparams = output_q->InputDefs();
            const auto* output_scale_init = initializers.at(output_qparams[1]->Name());
            const auto* output_zp_init = initializers.at(output_qparams[2]->Name());

            if (output_scale_init->has_raw_data()) {
              params.output_scale = *reinterpret_cast<const float*>(output_scale_init->raw_data().data());
            } else {
              params.output_scale = output_scale_init->float_data().empty() ? 0.0f : output_scale_init->float_data(0);
            }

            if (output_zp_init->has_raw_data()) {
              params.output_zp = *reinterpret_cast<const int8_t*>(output_zp_init->raw_data().data());
            } else {
              params.output_zp = static_cast<int8_t>(output_zp_init->int32_data().empty() ? 0 : output_zp_init->int32_data(0));
            }
          } else {
            continue;
          }
        }
      }

      // Initialize output buffer
      const size_t output_size = effective_batch_size * params.channels * params.output_height * params.output_width;
      params.output_buffer.resize(output_size);

      // Create and save the node info
      std::vector<const Node*> fused_nodes;
      if (params.needs_quantization) {
        fused_nodes.push_back(input_dq);
      }
      fused_nodes.push_back(node);

      if (params.needs_quantization) {
        const Node* output_q = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name())[0];
        if (output_q && output_q->OpType() == "QuantizeLinear") {
          fused_nodes.push_back(output_q);
        }
      }

      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);

      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);
      maxpool_params_map_[node_name] = std::move(params);

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

    } else if (node->OpType() == "Add") {
      if (handled_nodes.find(node) != handled_nodes.end()) {
        continue;
      }

      AddParams params{};
      params.needs_quantization = false;
      params.has_constant = false;

      // Check if Add is fused with Relu
      std::string add_output_name = node->OutputDefs()[0]->Name();
      bool has_relu = add_output_name.find("Relu") != std::string::npos;
      params.fused_relu = has_relu;

      // Controllo sugli input e sulla quantizzazione
      const Node* input1_dq = graph_viewer.GetProducerNode(node->InputDefs()[0]->Name());
      const Node* input2_dq = graph_viewer.GetProducerNode(node->InputDefs()[1]->Name());
      const auto* input1_shape = node->InputDefs()[0]->Shape();
      const auto* input2_shape = node->InputDefs()[1]->Shape();
      // Verifica se uno degli input ha una singola dimensione (quindi è costante)
      bool input1_is_constant = input1_shape && input1_shape->dim_size() == 1;
      bool input2_is_constant = input2_shape && input2_shape->dim_size() == 1;

      if (input1_is_constant || input2_is_constant) {
        params.has_constant = true;
        params.constant_is_first_input = input1_is_constant;

        const auto* constant_tensor = initializers.at(
            params.constant_is_first_input ? node->InputDefs()[0]->Name() : node->InputDefs()[1]->Name());

        if (constant_tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
          params.constant_value = constant_tensor->int64_data(0);
        } else {
          continue;  // Non supportiamo altri tipi di costanti
        }

        // Verifica se l'input non costante è quantizzato
        const Node* input_dq = params.constant_is_first_input ? input2_dq : input1_dq;
        if (input_dq && input_dq->OpType() == "DequantizeLinear") {
          params.needs_quantization = true;
        }
      } else {
        // Verifica se entrambi gli input sono quantizzati
        if (input1_dq && input2_dq &&
            input1_dq->OpType() == "DequantizeLinear" &&
            input2_dq->OpType() == "DequantizeLinear") {
          params.needs_quantization = true;
        }
      }

      // Get input shape based on the scenario
      const ONNX_NAMESPACE::TensorShapeProto* shape_proto = nullptr;
      if (params.has_constant) {
        // Se c'è una costante, prendiamo lo shape dell'altro input
        const Node* non_constant_node = params.constant_is_first_input ? (params.needs_quantization ? input2_dq : nullptr) : (params.needs_quantization ? input1_dq : nullptr);

        shape_proto = params.needs_quantization ? non_constant_node->InputDefs()[0]->Shape() : node->InputDefs()[params.constant_is_first_input ? 1 : 0]->Shape();
      } else {
        // Caso normale senza costanti
        shape_proto = params.needs_quantization ? input1_dq->InputDefs()[0]->Shape() : node->InputDefs()[0]->Shape();
      }

      if (!shape_proto) {
        continue;
      }

      int64_t initial_batch_size;  // Dichiarato fuori dagli scope

      if (shape_proto->dim_size() == 4) {
        initial_batch_size = shape_proto->dim(0).dim_value();
        params.channels = shape_proto->dim(1).dim_value();
        params.height = shape_proto->dim(2).dim_value();
        params.width = shape_proto->dim(3).dim_value();
      } else if (shape_proto->dim_size() == 3) {
        initial_batch_size = shape_proto->dim(0).dim_value();
        params.channels = 1;  // Assumiamo 1 canale
        params.height = shape_proto->dim(1).dim_value();
        params.width = shape_proto->dim(2).dim_value();
      } else {
        continue;
      }

      if (initial_batch_size == 0) {
        params.batch_size = 8;
        params.dynamic_batch = true;
      } else {
        params.batch_size = initial_batch_size;
      }

      if (initial_batch_size == 0) {
        params.batch_size = 8;
        params.dynamic_batch = true;
      } else {
        params.batch_size = initial_batch_size;
      }

      if (params.needs_quantization) {
        if (params.has_constant) {
          // Get input quantization parameters
          const Node* input_dq = params.constant_is_first_input ? input2_dq : input1_dq;
          const auto* input_scale_init = initializers.at(input_dq->InputDefs()[1]->Name());
          const auto* input_zp_init = initializers.at(input_dq->InputDefs()[2]->Name());

          if (!input_scale_init || !input_zp_init) {
            continue;
          }

          if (input_scale_init->has_raw_data()) {
            params.input1_scale = *reinterpret_cast<const float*>(input_scale_init->raw_data().data());
          } else if (input_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            params.input1_scale = input_scale_init->float_data().empty() ? 0.0f : input_scale_init->float_data(0);
          } else {
            params.input1_scale = 0.0f;
          }

          if (input_zp_init->has_raw_data()) {
            params.input1_zp = *reinterpret_cast<const int8_t*>(input_zp_init->raw_data().data());
          } else if (input_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            params.input1_zp = input_zp_init->int32_data().empty() ? 0 : static_cast<int8_t>(input_zp_init->int32_data(0));
          } else {
            params.input1_zp = 0;
          }
        } else {
          // Get input1 quantization parameters
          const auto* input1_scale_init = initializers.at(input1_dq->InputDefs()[1]->Name());
          const auto* input1_zp_init = initializers.at(input1_dq->InputDefs()[2]->Name());

          if (!input1_scale_init || !input1_zp_init) {
            continue;
          }

          if (input1_scale_init->has_raw_data()) {
            params.input1_scale = *reinterpret_cast<const float*>(input1_scale_init->raw_data().data());
          } else if (input1_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            params.input1_scale = input1_scale_init->float_data().empty() ? 0.0f : input1_scale_init->float_data(0);
          } else {
            params.input1_scale = 0.0f;
          }

          if (input1_zp_init->has_raw_data()) {
            params.input1_zp = *reinterpret_cast<const int8_t*>(input1_zp_init->raw_data().data());
          } else if (input1_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            params.input1_zp = input1_zp_init->int32_data().empty() ? 0 : static_cast<int8_t>(input1_zp_init->int32_data(0));
          } else {
            params.input1_zp = 0;
          }

          // Get input2 quantization parameters
          const auto* input2_scale_init = initializers.at(input2_dq->InputDefs()[1]->Name());
          const auto* input2_zp_init = initializers.at(input2_dq->InputDefs()[2]->Name());

          if (!input2_scale_init || !input2_zp_init) {
            continue;
          }

          if (input2_scale_init->has_raw_data()) {
            params.input2_scale = *reinterpret_cast<const float*>(input2_scale_init->raw_data().data());
          } else if (input2_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            params.input2_scale = input2_scale_init->float_data().empty() ? 0.0f : input2_scale_init->float_data(0);
          } else {
            params.input2_scale = 0.0f;
          }

          if (input2_zp_init->has_raw_data()) {
            params.input2_zp = *reinterpret_cast<const int8_t*>(input2_zp_init->raw_data().data());
          } else if (input2_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
            params.input2_zp = input2_zp_init->int32_data().empty() ? 0 : static_cast<int8_t>(input2_zp_init->int32_data(0));
          } else {
            params.input2_zp = 0;
          }
        }

        // Get output quantization parameters
        const Node* output_q = nullptr;
        auto add_consumers = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name());

        if (!add_consumers.empty()) {
          output_q = add_consumers[0];
          if (output_q && output_q->OpType() == "QuantizeLinear") {
            const auto* output_scale_init = initializers.at(output_q->InputDefs()[1]->Name());
            const auto* output_zp_init = initializers.at(output_q->InputDefs()[2]->Name());

            if (!output_scale_init || !output_zp_init) {
              continue;
            }

            if (output_scale_init->has_raw_data()) {
              params.output_scale = *reinterpret_cast<const float*>(output_scale_init->raw_data().data());
            } else if (output_scale_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
              params.output_scale = output_scale_init->float_data().empty() ? 0.0f : output_scale_init->float_data(0);
            } else {
              params.output_scale = 0.0f;
            }

            if (output_zp_init->has_raw_data()) {
              params.output_zp = *reinterpret_cast<const int8_t*>(output_zp_init->raw_data().data());
            } else if (output_zp_init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
              params.output_zp = output_zp_init->int32_data().empty() ? 0 : static_cast<int8_t>(output_zp_init->int32_data(0));
            } else {
              params.output_zp = 0;
            }
          }
        }

        // Calcola i moltiplicatori fissi
        if (params.has_constant) {
          double M = params.input1_scale / params.output_scale;
          params.M1_fixed = static_cast<int32_t>(M * (1 << 24));
        } else {
          double M1 = params.input1_scale / params.output_scale;
          double M2 = params.input2_scale / params.output_scale;
          params.M1_fixed = static_cast<int32_t>(M1 * (1 << 24));
          params.M2_fixed = static_cast<int32_t>(M2 * (1 << 24));
        }
      }

      // Create and save the node info
      std::vector<const Node*> fused_nodes;
      if (params.needs_quantization) {
        if (params.has_constant) {
          // Solo un input è DequantizeLinear
          const Node* input_dq = graph_viewer.GetProducerNode(
              node->InputDefs()[params.constant_is_first_input ? 1 : 0]->Name());
          fused_nodes.push_back(input_dq);
        } else {
          // Entrambi gli input sono DequantizeLinear
          fused_nodes.push_back(input1_dq);
          fused_nodes.push_back(input2_dq);
        }
      }
      fused_nodes.push_back(node);

      if (params.needs_quantization) {
        const Node* output_q = graph_viewer.GetConsumerNodes(node->OutputDefs()[0]->Name())[0];
        if (output_q && output_q->OpType() == "QuantizeLinear") {
          fused_nodes.push_back(output_q);
        }
      }

      uint64_t model_hash;
      int metadef_id = this->metadef_id_generator_.GenerateId(graph_viewer, model_hash);
      auto node_name = MakeString("NudgevExecutionProvider_", model_hash, "_", metadef_id, "_", metadef_id);

      add_params_map_[node_name] = std::move(params);

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
        auto start = std::chrono::high_resolution_clock::now();
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
        concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

        // Get input tensor
        const Tensor* input = ctx_internal->Input<Tensor>(0);
        if (!input) return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");

        // Get input shapes
        const auto& input_shape = input->Shape();
        const int64_t actual_batch_size = input_shape[0];
        const int64_t input_channels = input_shape[1];
        const int64_t input_height = input_shape[2];
        const int64_t input_width = input_shape[3];

        if (params->dynamic_batch && actual_batch_size != params->batch_size) {
          const int64_t new_N = actual_batch_size * params->output_height * params->output_width;
          params->N = new_N;

          const int64_t max_padded_height = input_height + params->pads[0] + params->pads[2];
          const int64_t max_padded_width = input_width + params->pads[1] + params->pads[3];

          const size_t padded_input = actual_batch_size * input_channels * max_padded_height * max_padded_width;

          params->padded_buffer.resize(padded_input);
          std::fill(params->padded_buffer.begin(), params->padded_buffer.end(), params->input_zp);

          params->im2row_buffer.resize(params->N * params->K);
          params->batch_size = actual_batch_size;
        }

        // Get weight shapes
        const int64_t output_channels = params->weight_shape[0];
        const int64_t kernel_height = params->weight_shape[2];
        const int64_t kernel_width = params->weight_shape[3];
        std::vector<int64_t> output_shape{actual_batch_size, output_channels, params->output_height, params->output_width};

        Tensor* Y = ctx_internal->Output(0, output_shape);
        if (!Y) {
          return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
        }

        if (Y->DataType() != DataTypeImpl::GetType<int8_t>()) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be int8");
        }

        // Get data pointers
        const auto* input_data = input->Data<int8_t>();
        auto* output_data = Y->MutableData<int8_t>();
        int8_t* im2row_ptr = params->im2row_buffer.data();

        onnxruntime::nudgev::im2col(input_data,
                                    &im2row_ptr,
                                    actual_batch_size,
                                    input_channels,
                                    input_height,
                                    input_width,
                                    kernel_height,
                                    kernel_width,
                                    params->strides[0],
                                    params->pads,
                                    params->padded_buffer.data(),
                                    tp);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        const int64_t patches_per_image = params->output_height * params->output_width;
        auto start_gemm = std::chrono::high_resolution_clock::now();
        onnxruntime::nudgev::gemm_i8_after_im2col_xzp(
            params->weights.data(),
            im2row_ptr,
            output_data,
            output_channels,
            params->K,
            patches_per_image,
            actual_batch_size,
            params->has_bias ? params->bias.data() : nullptr,
            params->has_bias,
            params->M_fixed,
            params->output_zp,
            params->input_zp,
            params->fused_relu,
            tp);
        auto end_gemm = std::chrono::high_resolution_clock::now();
        auto duration_gemm = std::chrono::duration_cast<std::chrono::microseconds>(end_gemm - start_gemm);

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - start);

        std::cout << "=== Conv Op - Im2col Debug Info ===\n";
        std::cout << "Input Shape: (" << actual_batch_size << ", " << input_channels << ", "
                  << input_height << ", " << input_width << ")\n";
        std::cout << "Im2col Output Shape: [" << params->K << " x " << params->N << "]" << std::endl;
        std::cout << "Weights Shape (2D): [" << output_channels << " x " << params->K << "]" << std::endl;
        std::cout << "Kernel Size: (" << kernel_height << "x" << kernel_width << ")\n";
        std::cout << "Stride: (" << params->strides[0] << ", " << params->strides[1] << ")\n";
        std::cout << "Padding: (" << params->pads[0] << ", " << params->pads[1] << ")\n";
        std::cout << "Output Shape: (" << actual_batch_size << ", " << output_channels << ", "
                  << params->output_height << ", " << params->output_width << ")\n";
        std::cout << "Quantization Params:\n";
        std::cout << "  Input Scale: " << params->input_scale << "\n";
        std::cout << "  Output Scale: " << params->output_scale << "\n";
        std::cout << "Conv Op - Im2col execution time: " << duration.count() << " microseconds" << std::endl;
        std::cout << "Conv Op - Total Gemm execution time: " << duration_gemm.count() << " microseconds" << std::endl;
        std::cout << "Conv Op - Total execution time: " << total_duration.count() << " microseconds" << std::endl;
        std::cout << "====================================\n";
        return Status::OK();
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
        auto start_gemm = std::chrono::high_resolution_clock::now();
        if (!state) {
          return Status(common::ONNXRUNTIME, common::FAIL, "state is null");
        }

        auto* params = static_cast<GemmParams*>(state);
        if (!params) {
          return Status(common::ONNXRUNTIME, common::FAIL, "params cast failed");
        }

        auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
        concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

        // Get input tensor
        const Tensor* input = ctx_internal->Input<Tensor>(0);
        if (!input) {
          return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");
        }

        const auto& input_shape = input->Shape();
        const int64_t actual_batch_size = input_shape[0];

        if (params->dynamic_batch && actual_batch_size != params->batch_size) {
          params->batch_size = actual_batch_size;
        }

        std::vector<int64_t> output_shape{params->batch_size, params->N};
        Tensor* Y = ctx_internal->Output(0, output_shape);
        if (!Y) {
          return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
        }

        if (Y->DataType() != DataTypeImpl::GetType<int8_t>()) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be int8");
        }

        // Get data pointers
        const auto* input_data = input->Data<int8_t>();
        auto* output_data = Y->MutableData<int8_t>();

        if (!params->weights_buffer.data()) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Weights buffer is null");
        }
        if (!input_data) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
        }
        if (!output_data) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Output data is null");
        }

        onnxruntime::nudgev::nudgev_gemm(
            input_data,
            params->weights_buffer.data(),
            output_data,
            params->batch_size,
            params->K,
            params->N,
            params->bias_buffer.data(),
            !params->bias_buffer.empty(),
            params->M_fixed,
            params->input_zp,
            params->output_zp,
            false,
            tp);

        auto end_gemm = std::chrono::high_resolution_clock::now();
        auto duration_gemm = std::chrono::duration_cast<std::chrono::microseconds>(end_gemm - start_gemm);
        std::cout << "Conv Op - Total Gemm execution time: " << duration_gemm.count() << " microseconds" << std::endl;

        return Status::OK();
      };

      saved_gemm_params_.push_back(std::move(kernel_params));
    } else if (maxpool_params_map_.find(fused_node.Name()) != maxpool_params_map_.end()) {
      auto it = maxpool_params_map_.find(fused_node.Name());
      if (it == maxpool_params_map_.end()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "MaxPool parameters not found");
      }
      auto kernel_params = std::make_unique<MaxPoolParams>(std::move(it->second));
      auto params_copy = kernel_params.get();

      compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) {
        *state = params_copy;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState) {};

      compute_info.compute_func = [](FunctionState state,
                                     const OrtApi*,
                                     OrtKernelContext* context) -> Status {
        auto start = std::chrono::high_resolution_clock::now();
        if (!state) {
          return Status(common::ONNXRUNTIME, common::FAIL, "state is null");
        }

        auto* params = static_cast<MaxPoolParams*>(state);
        if (!params) {
          return Status(common::ONNXRUNTIME, common::FAIL, "params cast failed");
        }

        auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
        concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

        // Get input tensor
        const Tensor* input = ctx_internal->Input<Tensor>(0);
        if (!input) {
          return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");
        }

        const auto& input_shape = input->Shape();
        const int64_t actual_batch_size = input_shape[0];

        if (params->dynamic_batch && actual_batch_size != params->batch_size) {
          params->batch_size = actual_batch_size;
        }

        std::vector<int64_t> output_shape{actual_batch_size, params->channels,
                                          params->output_height, params->output_width};
        Tensor* Y = ctx_internal->Output(0, output_shape);
        if (!Y) {
          return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
        }

        if (params->needs_quantization) {
          // Caso quantizzato (int8)
          if (Y->DataType() != DataTypeImpl::GetType<int8_t>()) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be int8 for quantized version");
          }

          const auto* input_data = input->Data<int8_t>();
          auto* output_data = Y->MutableData<int8_t>();

          if (!input_data) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
          }
          if (!output_data) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Output data is null");
          }

          onnxruntime::nudgev::nudgev_maxpool_requantized(
              input_data,
              output_data,
              actual_batch_size,
              params->channels,
              params->input_height,
              params->input_width,
              params->kernel_shape[0],
              params->kernel_shape[1],
              params->strides[0],
              params->strides[1],
              params->pads,
              params->ceil_mode,
              tp);
        } else {
          // Caso non quantizzato (float)
          if (Y->DataType() != DataTypeImpl::GetType<float>()) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be float for non-quantized version");
          }

          const auto* input_data = input->Data<float>();
          auto* output_data = Y->MutableData<float>();

          if (!input_data) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
          }
          if (!output_data) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Output data is null");
          }

          onnxruntime::nudgev::nudgev_maxpool(
              input_data,
              output_data,
              actual_batch_size,
              params->channels,
              params->input_height,
              params->input_width,
              params->kernel_shape[0],
              params->kernel_shape[1],
              params->strides[0],
              params->strides[1],
              params->pads,
              params->ceil_mode,
              tp);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "[DEBUG] MaxPool Op (" << (params->needs_quantization ? "quantized" : "float")
                  << ") - Total execution time: " << duration.count()
                  << " microseconds" << std::endl;

        return Status::OK();
      };

      saved_maxpool_params_.push_back(std::move(kernel_params));
    } else if (sigmoid_params_map_.find(fused_node.Name()) != sigmoid_params_map_.end()) {
      auto it = sigmoid_params_map_.find(fused_node.Name());
      if (it == sigmoid_params_map_.end()) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Sigmoid parameters not found");
      }

      auto kernel_params = std::make_unique<SigmoidParams>(std::move(it->second));
      auto params_copy = kernel_params.get();

      compute_info.create_state_func = [params_copy](ComputeContext*, FunctionState* state) {
        *state = params_copy;
        return 0;
      };

      compute_info.release_state_func = [](FunctionState) {};

      compute_info.compute_func = [](FunctionState state,
                                     const OrtApi*,
                                     OrtKernelContext* context) -> Status {
        auto start = std::chrono::high_resolution_clock::now();
        auto* params = static_cast<SigmoidParams*>(state);
        auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
        const Tensor* input = ctx_internal->Input<Tensor>(0);
        if (!input) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Input tensor is null");
        }

        const auto& input_shape = input->Shape();
        const int64_t actual_batch_size = input_shape[0];

        if (params->dynamic_batch) {
          params->batch_size = actual_batch_size;
        }

        Tensor* output = ctx_internal->Output(0, input_shape);
        if (!output) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Failed to create output tensor");
        }

        const auto* input_data = input->Data<float>();
        auto* output_data = output->MutableData<float>();

        if (!input_data || !output_data) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Input/Output data is null");
        }

        concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();
        onnxruntime::nudgev::nudgev_sigmoid(
            input_data,
            output_data,
            actual_batch_size,
            params->channels,
            params->height,
            params->width,
            tp);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Sigmoid Op - Total execution time: " << duration.count() << " microseconds" << std::endl;

        return Status::OK();
      };

      saved_sigmoid_params_.push_back(std::move(kernel_params));

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
        auto start = std::chrono::high_resolution_clock::now();
        auto* params = static_cast<AddParams*>(state);
        auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
        const Tensor* input1 = ctx_internal->Input<Tensor>(0);
        const Tensor* input2 = ctx_internal->Input<Tensor>(params->needs_quantization ? 3 : 1);

        if (!input1 || !input2) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Input tensors cannot be null");
        }

        const auto& input_shape = input1->Shape();
        int64_t actual_batch_size = input_shape[0];

        if (params->dynamic_batch) {
          params->batch_size = actual_batch_size;
        }

        Tensor* output = ctx_internal->Output(0, input_shape);
        if (!output) {
          return Status(common::ONNXRUNTIME, common::FAIL, "Failed to create output tensor");
        }

        concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

        if (params->needs_quantization) {
          // Versione quantizzata
          const auto* input1_data = input1->Data<int8_t>();
          const auto* input2_data = input2->Data<int8_t>();
          auto* output_data = output->MutableData<int8_t>();

          if (!input1_data || !input2_data || !output_data) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Tensor data pointers cannot be null");
          }

          if (params->fused_relu) {
            onnxruntime::nudgev::nudgev_add_relu_requantized(
                input1_data,
                input2_data,
                output_data,
                params->batch_size,
                params->channels,
                params->height,
                params->width,
                params->M1_fixed,
                params->M2_fixed,
                params->input1_zp,
                params->input2_zp,
                params->output_zp,
                tp);
          } else {
            onnxruntime::nudgev::nudgev_add_requantized(
                input1_data,
                input2_data,
                output_data,
                params->batch_size,
                params->channels,
                params->height,
                params->width,
                params->M1_fixed,
                params->M2_fixed,
                params->input1_zp,
                params->input2_zp,
                params->output_zp,
                tp);
          }
        } else {
          // Versione non quantizzata
          const auto* input1_data = input1->Data<float>();
          const auto* input2_data = input2->Data<float>();
          auto* output_data = output->MutableData<float>();

          if (!input1_data || !input2_data || !output_data) {
            return Status(common::ONNXRUNTIME, common::FAIL, "Tensor data pointers cannot be null");
          }

          if (params->fused_relu) {
            onnxruntime::nudgev::nudgev_add_relu(
                input1_data,
                input2_data,
                output_data,
                params->batch_size,
                params->channels,
                params->height,
                params->width,
                tp);
          } else {
            onnxruntime::nudgev::nudgev_add(
                input1_data,
                input2_data,
                output_data,
                params->batch_size,
                params->channels,
                params->height,
                params->width,
                tp);
          }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Add Op - Total execution time: " << duration.count() << " microseconds" << std::endl;
        return Status::OK();
      };
      saved_add_params_.push_back(std::move(kernel_params));
    }

    node_compute_funcs.push_back(std::move(compute_info));
  }

  return Status::OK();
}

}  // namespace onnxruntime