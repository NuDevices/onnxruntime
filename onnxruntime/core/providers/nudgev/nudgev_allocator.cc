#include "core/providers/nudgev/nudgev_allocator.h"
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

namespace onnxruntime {

void* NudgevAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = _mm_malloc(size, 32);
    if (p) {
      std::cout << "NudgevAllocator: Allocated " << size
                << " bytes at address " << p << std::endl;
    } else {
      std::cerr << "NudgevAllocator: Failed to allocate "
                << size << " bytes" << std::endl;
    }
  }
  return p;
}

void NudgevAllocator::Free(void* p) {
  if (p) {
    // std::cout << "NudgevAllocator: Liberati bytes a " << p << std::endl;
    _mm_free(p);
  }
}

void* NudgevPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    size_t aligned_size = (size + 31) & ~31;
    p = aligned_alloc(32, aligned_size);

    if (p) {
      std::cout << "NudgevPinnedAllocator: Allocating " << size
                << " bytes at address " << p << std::endl;
    } else {
      std::cerr << "NudgevPinnedAllocator: Failed to allocate "
                << size << " bytes" << std::endl;
    }
  }
  return p;
}

void NudgevPinnedAllocator::Free(void* p) {
  if (p) {
    // std::cout << "NudgevPinnedAllocator: Liberata memoria pinnata a " << p << std::endl;
    free(p);
  }
}

}  // namespace onnxruntime