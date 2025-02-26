#pragma once

#include "core/framework/data_transfer.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

  class NudgevDataTransfer : public IDataTransfer {
   public:
    NudgevDataTransfer();
    ~NudgevDataTransfer();

    // Check if direct memory copying is possible between devices
    bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override;

    // Copy tensor data (synchronous version)
    common::Status CopyTensor(const Tensor& src, Tensor& dst) const override;

    // Copy tensor data asynchronously using a stream
    common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override;

   private:
    // Helper method to implement copying, with or without a stream
    common::Status CopyTensorImpl(const Tensor& src, Tensor& dst, Stream* stream) const;
  };

  }  // namespace onnxruntime
