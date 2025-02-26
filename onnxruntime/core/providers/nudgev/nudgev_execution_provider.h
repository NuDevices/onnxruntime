#pragma once
#include "core/providers/nudgev/nudgev_device_type.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/framework/provider_options.h"
#include "core/framework/data_transfer_manager.h"
#include "core/providers/nudgev/nudgev_data_transfer.h"
#include "core/providers/nudgev/nudgev_allocator.h"
#include <unordered_map>
#include <immintrin.h>
#include <vector>
#include <string>

namespace onnxruntime {

struct alignas(32) DequantizeLinearParams {
  int64_t batch_size{};
  int64_t channels{};
  int64_t height{};
  int64_t width{};
  bool dynamic_batch{false};
  float scale{};
  int8_t zero_point{};
  std::vector<int64_t> input_shape;
};

struct alignas(32) SigmoidParams {
  int64_t batch_size{};
  int64_t channels{};
  int64_t height{};
  int64_t width{};
  bool dynamic_batch{false};
};
struct alignas(32) AddParams {
  int64_t batch_size{};
  int64_t channels{};
  int64_t height{};
  int64_t width{};
  bool dynamic_batch{false};
  bool fused_relu{};
  bool needs_quantization;
  bool has_constant;
  bool constant_is_first_input;
  int64_t constant_value;
  double input1_scale{};
  int8_t input1_zp{};
  double input2_scale{};
  int8_t input2_zp{};
  double output_scale{};
  int8_t output_zp{};
  int32_t M1_fixed{};
  int32_t M2_fixed{};
};

struct alignas(32) MaxPoolParams {
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
  bool ceil_mode{false};
  bool count_include_pad{false};
  bool needs_quantization{false};
  int64_t storage_order{0};
  int64_t batch_size{};
  bool dynamic_batch{false};
  int64_t channels{};
  int64_t input_height{};
  int64_t input_width{};
  int64_t output_height{};
  int64_t output_width{};
  alignas(32) std::vector<int8_t> output_buffer;
  double input_scale{};
  int8_t input_zp{};
  double output_scale{};
  int8_t output_zp{};
};

struct alignas(32) GemmParams {
  double alpha{1.0f};
  double beta{1.0f};
  bool transA{false};
  bool transB{false};
  int64_t M{};
  int64_t N{};
  int64_t K{};
  int64_t batch_size{};
  bool dynamic_batch{false};
  double input_scale{};
  int8_t input_zp{};
  double weight_scale{};
  int8_t weight_zp{};
  double output_scale{};
  int8_t output_zp{};
  int32_t M_fixed{};
  alignas(32) std::vector<int8_t> weights_buffer;
  alignas(32) std::vector<int32_t> bias_buffer;
  alignas(32) std::vector<int8_t> input_centered_buffer;
};

struct alignas(32) ConvQuantParams {
  double input_scale{};
  int8_t input_zp{};
  double weight_scale{};
  int8_t weight_zp{};
  double bias_scale{};
  int8_t bias_zp{};
  double output_scale{};
  int8_t output_zp{};

  int64_t group{};
  bool has_bias{};
  bool fused_relu{};
  bool dynamic_batch{};
  double M{};
  int32_t M_fixed{};
  int64_t N{};
  int64_t K{};
  int64_t output_height{};
  int64_t output_width{};
  int64_t batch_size{};

  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> bias_shape;
  std::string auto_pad;
  std::string node_name;

  alignas(32) std::vector<int8_t> weights;
  alignas(32) std::vector<int32_t> bias;
  alignas(32) std::vector<int8_t> im2row_buffer;
  alignas(32) std::vector<int8_t> temp_buffer;
  alignas(32) std::vector<int8_t> input_centered_buffer;
  alignas(32) std::vector<int8_t> padded_buffer;

  Status initialize_buffers(const std::vector<int64_t>& weight_shape_,
                            const std::vector<int64_t>& bias_shape_,
                            int64_t batch_size_, int64_t output_height_,
                            int64_t output_width_) {
    weight_shape = weight_shape_;
    bias_shape = bias_shape_;
    batch_size = batch_size_;
    output_height = output_height_;
    output_width = output_width_;
    const int64_t OC = weight_shape[0];
    const int64_t IC = weight_shape[1];
    const int64_t KH = weight_shape[2];
    const int64_t KW = weight_shape[3];

    const size_t K = IC * KH * KW;
    const size_t N = batch_size * output_height * output_width;

    weights.resize(K * OC);
    im2row_buffer.resize(N * K);

    if (!bias_shape.empty()) {
      bias.resize(bias_shape[0]);
    }

    return Status::OK();
  }
};

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(X != nullptr, "Memcpy: Input tensor is nullptr.");
    /*
        std::cout << "Memcpy: Input tensor details:" << std::endl;
        std::cout << "  Name: " << X->Location().name << std::endl;
        std::cout << "  Address: " << X->DataRaw() << std::endl;
        std::cout << "  Size: " << X->SizeInBytes() << std::endl;
        std::cout << "  Device Type: " << static_cast<int>(X->Location().device.Type()) << std::endl;
    */
    Tensor* Y = ctx->Output(0, X->Shape());
    ORT_ENFORCE(Y != nullptr, "Memcpy: Failed to allocate output tensor.");
    /*
        std::cout << "Memcpy: Output tensor details:" << std::endl;
        std::cout << "  Name: " << Y->Location().name << std::endl;
        std::cout << "  Address: " << Y->MutableDataRaw() << std::endl;
        std::cout << "  Size: " << Y->SizeInBytes() << std::endl;
        std::cout << "  Device Type: " << static_cast<int>(Y->Location().device.Type()) << std::endl;
    */
    auto* data_transfer = Info().GetDataTransferManager().GetDataTransfer(
        X->Location().device, Y->Location().device);

    ORT_RETURN_IF_ERROR(data_transfer->CopyTensorAsync(*X, *Y, *ctx->GetComputeStream()));

    return Status::OK();
  }
};

class NudgevExecutionProvider : public IExecutionProvider {
 public:
  AllocatorPtr CreateCPUAllocator(OrtDevice::DeviceId device_id);
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

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;
  OrtDevice GetOrtDeviceByMemType(OrtMemType mem_type) const override;

  std::unordered_map<int, std::unordered_map<int, std::unordered_set<int>>>
  GetDeviceCopyMap() const override;

 private:
  mutable std::unordered_map<std::string, SigmoidParams> sigmoid_params_map_;
  mutable std::unordered_map<std::string, ConvQuantParams> quant_params_map_;
  mutable std::unordered_map<std::string, GemmParams> gemm_params_map_;
  mutable std::unordered_map<std::string, MaxPoolParams> maxpool_params_map_;
  mutable std::unordered_map<std::string, AddParams> add_params_map_;
  std::vector<std::unique_ptr<SigmoidParams>> saved_sigmoid_params_;
  std::vector<std::unique_ptr<ConvQuantParams>> saved_conv_params_;
  std::vector<std::unique_ptr<GemmParams>> saved_gemm_params_;
  std::vector<std::unique_ptr<AddParams>> saved_add_params_;
  std::vector<std::unique_ptr<MaxPoolParams>> saved_maxpool_params_;
  std::unordered_map<std::string, std::string> node_name_mapping_;

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
