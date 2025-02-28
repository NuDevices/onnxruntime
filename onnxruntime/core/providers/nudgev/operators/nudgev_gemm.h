#pragma once
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/platform/threadpool.h"
#include "core/providers/nudgev/nudgev_execution_provider.h"
#include <vector>

namespace onnxruntime {
namespace nudgev {

Status ComputeNudgeVGemm(GemmParams* gemm_params, OrtKernelContext* context);

}  // namespace nudgev
}  // namespace onnxruntime