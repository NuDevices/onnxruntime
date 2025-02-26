
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
  // Same device type - no copy needed if they're the same device
  if (src_device.Type() == dst_device.Type() && src_device.Id() == dst_device.Id()) {
    std::cout << "CanCopy: Stesso tipo di device e ID - copy consentita" << std::endl;
    return true;
  }

  // Nudgev device can directly access CPU memory if it's pinned
  if ((src_device.Type() == OrtDevice::CPU &&
       dst_device.Type() == kNudgevDeviceType) ||
      (src_device.Type() == kNudgevDeviceType &&
       dst_device.Type() == OrtDevice::CPU)) {
    std::cout << "CanCopy: Trasferimento tra CPU e Nudgev" << std::endl;
    std::cout << "  Src MemType: " << static_cast<int>(src_device.MemType()) << std::endl;
    std::cout << "  Dst MemType: " << static_cast<int>(dst_device.MemType()) << std::endl;

    // Check if it's pinned memory
    if (src_device.MemType() == NUDGEV_PINNED_MEMORY_TYPE ||
        dst_device.MemType() == NUDGEV_PINNED_MEMORY_TYPE) {
      std::cout << "CanCopy: Memoria pinnata rilevata - copy consentita" << std::endl;
      return true;
    }
  }

  // Otherwise, a copy is required
  std::cout << "CanCopy: Copy necessaria tra device diversi" << std::endl;
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
  std::cout << "DST device type: " << static_cast<int>(dst.Location().device.Type()) << std::endl;
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