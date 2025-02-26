#pragma once

#include "core/framework/data_transfer.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class NudgevDataTransfer : public IDataTransfer {
 public:
  NudgevDataTransfer();
  ~NudgevDataTransfer();

  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;
};

}  // namespace onnxruntime