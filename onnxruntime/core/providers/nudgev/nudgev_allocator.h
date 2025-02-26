// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/providers/nudgev/nudgev_device_type.h"

namespace onnxruntime {

class NudgevAllocator : public IAllocator {
 public:
  NudgevAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(kNudgevDeviceType, OrtDevice::MemType::DEFAULT, device_id),
                          device_id, OrtMemTypeDefault)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

class NudgevPinnedAllocator : public IAllocator {
 public:
  NudgevPinnedAllocator(OrtDevice::DeviceId device_id, const char* name)
      : IAllocator(
            OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                          OrtDevice(OrtDevice::CPU, NUDGEV_PINNED_MEMORY_TYPE, device_id),
                          device_id, OrtMemTypeCPUOutput)) {}

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};

}  // namespace onnxruntime