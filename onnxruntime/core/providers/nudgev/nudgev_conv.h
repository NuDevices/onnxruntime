#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace nudgev {

class NudgevConv final : public OpKernel {
 public:
  explicit NudgevConv(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override;

 private:
  // add conv params
};

}  // namespace nudgev
}  // namespace onnxruntime
