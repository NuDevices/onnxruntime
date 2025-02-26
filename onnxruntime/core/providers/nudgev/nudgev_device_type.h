#pragma once
#include "core/framework/ortdevice.h"

namespace onnxruntime {

// constexpr OrtDevice::DeviceType kNudgevDeviceType = OrtDevice::CPU;
constexpr OrtDevice::DeviceType kNudgevDeviceType = static_cast<OrtDevice::DeviceType>(5);

static const char* NUDGEV = "Nudgev";
static const char* NUDGEV_PINNED = "NudgevPinned";

// Use an integer constant instead of trying to use OrtDevice::MemType directly
constexpr int NUDGEV_PINNED_MEMORY_TYPE = 2;
}  // namespace onnxruntime