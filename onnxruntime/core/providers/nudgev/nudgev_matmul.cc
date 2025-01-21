#include "core/providers/nudgev/nudgev_matmul.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace nudgev {

Status NudgevMatMul::Compute(OpKernelContext* context) const {
  // Get the input tensors
  const Tensor* A = context->Input<Tensor>(0);
  const Tensor* B = context->Input<Tensor>(1);
  ORT_ENFORCE(A != nullptr && B != nullptr);

  // Verify input data types are int8
  if (A->DataType() != DataTypeImpl::GetType<int8_t>() ||
      B->DataType() != DataTypeImpl::GetType<int8_t>()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                 "Nudgev MatMul only supports INT8 inputs");
  }

  // Use MatMulHelper to handle broadcasting and shape inference
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(A->Shape(), B->Shape()));

  // Create output tensor
  Tensor* Y = context->Output(0, helper.OutputShape());

  // Get raw pointers to the tensor data
  const auto* a_data = A->Data<int8_t>();
  const auto* b_data = B->Data<int8_t>();
  auto* y_data = Y->MutableData<int8_t>();

  // Get dimensions
  const auto& a_shape = A->Shape();
  const auto& b_shape = B->Shape();

  // For simplicity, let's handle the 2D case first
  // M x K * K x N = M x N
  int64_t M = a_shape[0];
  int64_t K = a_shape[1];
  int64_t N = b_shape[1];

  // Simple iterative matmul implementation
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      int32_t sum = 0;  // Use int32 to prevent overflow during accumulation
      for (int64_t k = 0; k < K; k++) {
        sum += static_cast<int32_t>(a_data[m * K + k]) *
               static_cast<int32_t>(b_data[k * N + n]);
      }
      // Saturate to int8 range
      sum = std::min(std::max(sum, static_cast<int32_t>(INT8_MIN)),
                     static_cast<int32_t>(INT8_MAX));
      y_data[m * N + n] = static_cast<int8_t>(sum);
    }
  }

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime
