#include "core/providers/nudgev/nudgev_data_transfer.h"
#include "core/providers/nudgev/nudgev_allocator.h"
#include "core/providers/nudgev/nudgev_device_type.h"
#include <iostream>

namespace onnxruntime {

NudgevDataTransfer::NudgevDataTransfer() {}

NudgevDataTransfer::~NudgevDataTransfer() {}

common::Status NudgevDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return CopyTensorImpl(src, dst, nullptr);
}

common::Status NudgevDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const {
  return CopyTensorImpl(src, dst, &stream);
}

bool NudgevDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  // Same device type and ID - no copy needed
  if (src_device.Type() == dst_device.Type() && src_device.Id() == dst_device.Id()) {
    std::cout << "CanCopy: Same device type and ID - copy allowed" << std::endl;
    return true;
  }

  // Allow CPU <-> NudgeV transfers (this is key for zero-copy to work)
  if ((src_device.Type() == OrtDevice::CPU && dst_device.Type() == kNudgevDeviceType) ||
      (src_device.Type() == kNudgevDeviceType && dst_device.Type() == OrtDevice::CPU)) {
    std::cout << "CanCopy: Transfer between CPU and NudgeV - copy allowed" << std::endl;
    return true;
  }

  std::cout << "CanCopy: Copy not allowed between different devices" << std::endl;
  return false;
}

common::Status NudgevDataTransfer::CopyTensorImpl(const Tensor& src, Tensor& dst, Stream* stream) const {
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  size_t bytes = src.SizeInBytes();

  // Check for zero-copy opportunity: if source and destination addresses are the same
  if (dst_data == src_data) {
    std::cout << "✅ Zero-copy: Same memory address detected at " << src_data << std::endl;
    return Status::OK();
  }

  // Log detailed information about the transfer for debugging
  std::cout << "📋 Copy needed from "
            << (src.Location().device.Type() == OrtDevice::CPU ? "CPU" : "NudgeV")
            << " to "
            << (dst.Location().device.Type() == OrtDevice::CPU ? "CPU" : "NudgeV")
            << ": " << src_data << " -> " << dst_data
            << ", size=" << bytes << " bytes" << std::endl;

  // Perform the actual memory copy
  std::memcpy(dst_data, src_data, bytes);

  return Status::OK();
}

}  // namespace onnxruntime
