#pragma once
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"
#include <vector>

namespace onnxruntime {
namespace nudgev {

void compute_broadcast_indices(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2,
    const std::vector<int64_t>& output_shape,
    int64_t flat_idx,
    int64_t& idx1,
    int64_t& idx2);

void add_quantized_int8(
    const int8_t* input1_data,
    const int8_t* input2_data,
    int8_t* output_data,
    const std::vector<int64_t>& input1_shape,
    const std::vector<int64_t>& input2_shape,
    const std::vector<int64_t>& output_shape,
    int8_t input1_zp,
    int8_t input2_zp,
    int8_t output_zp,
    int32_t M1_fixed,
    int32_t M2_fixed,
    int64_t total_elements,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp);

std::vector<int64_t> compute_broadcast_output_shape(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2);

Status ComputeNudgeVAdd(AddParams* add_params, OrtKernelContext* context);

}  // namespace nudgev
}  // namespace onnxruntime