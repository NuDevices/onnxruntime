
#include "core/providers/nudgev/nudgev_data_transfer.h"
#include "core/providers/nudgev/nudgev_allocator.h"
#include "core/providers/nudgev/nudgev_device_type.h"
namespace onnxruntime {

NudgevDataTransfer::NudgevDataTransfer() {}

NudgevDataTransfer::~NudgevDataTransfer() {}

common::Status NudgevDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  if (dst_data != src_data) {
    std::memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

bool NudgevDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  if (src_device.Type() == dst_device.Type() &&
      src_device.Id() == dst_device.Id()) {
    std::cout << "NudgevDataTransfer::CanCopy - Source device: "
              << static_cast<int>(src_device.Type())
              << ", Destination device: " << static_cast<int>(dst_device.Type()) << std::endl;
    return true;
  }
  return true;
}

Status NudgevDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const {
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  if (dst_data == src_data) {
    std::cout << "✅ Zero copy - stesso indirizzo di memoria" << std::endl;
    return Status::OK();
  }
  std::cout << "SRC device type: " << static_cast<int>(src.Location().device.Type()) << std::endl;
  std::cout << "SRC device type: " << static_cast<int>(src.Location().device.Type()) << std::endl;
  if (src.Location().device.Type() == OrtDevice::CPU &&
      dst.Location().device.Type() == OrtDevice::CPU) {
    std::cout << "📋 Copia CPU->CPU necessaria: "
              << src_data << " -> " << dst_data
              << ", dimensione=" << src.SizeInBytes() << " bytes" << std::endl;
  } else {
    std::cout << "📋 Copia cross-device: sorgente=" << src.Location().name
              << ", destinazione=" << dst.Location().name
              << ", dimensione=" << src.SizeInBytes() << " bytes" << std::endl;
  }
  size_t bytes = src.SizeInBytes();
  std::memcpy(dst_data, src_data, bytes);

  return Status::OK();
}
}  // namespace onnxruntime