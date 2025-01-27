
#include "core/providers/nudgev/nudgev_conv.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace nudgev {

Status NudgevConv::Compute(OpKernelContext* context) const {
  std::cout << "[NudgevConv] Compute starting" << std::endl;

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);

  if (!X || !W || !B) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Missing input tensor");
  }

  TensorShape output_shape = X->Shape();
  Tensor* Y = context->Output(0, output_shape);

  if (!Y) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Failed to create output tensor");
  }

  memcpy(Y->MutableDataRaw(), X->DataRaw(), X->SizeInBytes());

  std::cout << "[NudgevConv] Compute completed successfully" << std::endl;
  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime