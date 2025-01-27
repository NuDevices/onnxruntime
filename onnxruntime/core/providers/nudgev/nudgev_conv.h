
#pragma once

#include "core/framework/op_kernel.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace nudgev {

class NudgevConv final : public OpKernel {
 public:
  explicit NudgevConv(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace nudgev
}  // namespace onnxruntime