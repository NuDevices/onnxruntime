#pragma once
#include "core/framework/ortdevice.h"

namespace onnxruntime {

// constexpr OrtDevice::DeviceType kNudgevDeviceType = OrtDevice::CPU;
constexpr OrtDevice::DeviceType kNudgevDeviceType = static_cast<OrtDevice::DeviceType>(10);

// Name for the allocator
static const char* NUDGEV = "Nudgev";
}  // namespace onnxruntime
