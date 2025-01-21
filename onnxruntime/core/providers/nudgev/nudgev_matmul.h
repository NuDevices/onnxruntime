#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/cpu_provider_factory.h"

namespace onnxruntime {
namespace nudgev {

class NudgevMatMul final : public OpKernel {
 public:
  explicit NudgevMatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;

 private:
  // Add any member variables needed for optimization/caching
};

}  // namespace nudgev
}  // namespace onnxruntime
