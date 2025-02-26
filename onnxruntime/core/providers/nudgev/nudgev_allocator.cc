#include "core/providers/nudgev/nudgev_allocator.h"
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

namespace onnxruntime {

void* NudgevAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    // Use _mm_malloc for 32-byte alignment which is optimal for SIMD operations
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
    _mm_free(p);
  }
}

}  // namespace onnxruntime
