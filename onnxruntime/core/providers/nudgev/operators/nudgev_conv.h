#pragma once
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"
#include <vector>

namespace onnxruntime {
namespace nudgev {

void im2col(const int8_t* input, int8_t** output,
            int64_t batch_size, int64_t channels,
            int64_t height, int64_t width,
            int64_t kernel_h, int64_t kernel_w,
            int64_t stride_h, const std::vector<int64_t>& pads,
            int8_t* padded_buffer,
            concurrency::ThreadPool* tp);

void gemm_i8_after_im2col_xzp(
    const int8_t* weights,
    const int8_t* im2col_output,
    int8_t* output,
    int64_t OC,
    int64_t K,
    int64_t patches_per_image,
    int64_t batch_size,
    const int32_t* bias,
    bool has_bias,
    int32_t M_fixed,
    int8_t output_zero_point,
    int8_t input_zero_point,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp);

Status ComputeNudgeVConv(ConvQuantParams* conv_params, OrtKernelContext* context);

}  // namespace nudgev
}  // namespace onnxruntime