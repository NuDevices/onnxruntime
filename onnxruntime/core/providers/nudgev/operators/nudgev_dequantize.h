#pragma once
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"
#include <vector>

namespace onnxruntime {
namespace nudgev {

void nudgev_dequantize_linear(
    const int8_t* input,
    float* output,
    int64_t total_elements,
    float scale,
    int8_t zero_point,
    onnxruntime::concurrency::ThreadPool* tp);

void nudgev_dequantize_linear_int32(
    const int32_t* input,
    float* output,
    int64_t total_elements,
    float scale,
    int32_t zero_point,
    onnxruntime::concurrency::ThreadPool* tp);

Status ComputeNudgeVDequantizeLinear(DequantizeLinearParams* dequantize_params, OrtKernelContext* context);

}  // namespace nudgev
}  // namespace onnxruntime
