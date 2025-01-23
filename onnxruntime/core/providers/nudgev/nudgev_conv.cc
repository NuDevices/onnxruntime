#include "core/providers/nudgev/nudgev_conv.h"
#include "core/common/logging/logging.h"
#include <iostream>
namespace onnxruntime {
namespace nudgev {

Status NudgevConv::Compute(OpKernelContext* context) const {
  std::cout << "[NudgevConv] Compute method called for Conv" << std::endl;

  std::cout << "[NudgevConv] Input count: " << context->InputCount() << std::endl;
  std::cout << "[NudgevConv] Output count: " << context->OutputCount() << std::endl;

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime
