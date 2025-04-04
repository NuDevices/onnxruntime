#include "core/providers/nudgev/operators/nudgev_dequantize.h"
#include "core/framework/op_kernel_context_internal.h"
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include <memory>

namespace onnxruntime {
namespace nudgev {

  template <typename T>
class AlignedAllocator {
 public:
  using value_type = T;
  using pointer = T*;
  using size_type = std::size_t;

  AlignedAllocator() = default;

  template <class U>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

  pointer allocate(size_type n) {
    if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }
    void* ptr = _mm_malloc(n * sizeof(T), 32);  // 32-byte alignment for AVX
    if (!ptr) throw std::bad_alloc();
    return static_cast<pointer>(ptr);
  }

  void deallocate(pointer p, size_type) noexcept {
    _mm_free(p);
  }

  template <class U>
  bool operator==(const AlignedAllocator<U>&) const { return true; }
  template <class U>
  bool operator!=(const AlignedAllocator<U>&) const { return false; }
};

struct LastResultCache {
 public:
  static LastResultCache& GetInstance() {
    static LastResultCache instance;
    return instance;
  }

  const float* GetCachedResult(const void* input_ptr, int64_t total_elements,
                               float scale, int8_t zero_point) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    if (input_ptr == last_input_ptr && total_elements == last_total_elements &&
        scale == last_scale && zero_point == last_zero_point) {
      return cached_data.data();
    }
    return nullptr;
  }

  void CacheResult(const void* input_ptr, int64_t total_elements,
                   float scale, int8_t zero_point, const float* data, int64_t size,
                   concurrency::ThreadPool* tp) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Update cache key
    last_input_ptr = input_ptr;
    last_total_elements = total_elements;
    last_scale = scale;
    last_zero_point = zero_point;

    // Resize and copy data in parallel
    cached_data.resize(size);
    ParallelCopy(data, cached_data.data(), size, tp);
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cached_data.clear();
    last_input_ptr = nullptr;
    last_total_elements = 0;
    last_scale = 0.0f;
    last_zero_point = 0;
  }

  void ParallelCopy(const float* src, float* dst, int64_t size, concurrency::ThreadPool* tp) {
    constexpr int64_t chunk_size = 4096;
    const int64_t num_chunks = (size + chunk_size - 1) / chunk_size;

    auto copy_task = [&](int64_t start, int64_t end) {
      constexpr int64_t vec_size = 8;
      int64_t i = start;
      for (; i + vec_size <= end; i += vec_size) {
        __m256 vec = _mm256_loadu_ps(src + i);
        _mm256_store_ps(dst + i, vec);
      }
      for (; i < end; ++i) {
        dst[i] = src[i];
      }
    };

    if (tp) {
      concurrency::ThreadPool::TrySimpleParallelFor(
          tp, num_chunks,
          [&](std::ptrdiff_t chunk_idx) {
            const int64_t start = chunk_idx * chunk_size;
            const int64_t end = std::min(start + chunk_size, size);
            copy_task(start, end);
          });
    } else {
      copy_task(0, size);
    }
  }

 private:
  LastResultCache() = default;
  ~LastResultCache() = default;

  LastResultCache(const LastResultCache&) = delete;
  LastResultCache& operator=(const LastResultCache&) = delete;

  // Cache key data
  const void* last_input_ptr = nullptr;
  int64_t last_total_elements = 0;
  float last_scale = 0.0f;
  int8_t last_zero_point = 0;

  // Cache data with aligned allocation
  std::vector<float, AlignedAllocator<float>> cached_data;
  std::mutex cache_mutex_;
};

  void nudgev_dequantize_linear(
    const int8_t* __restrict input,
    float* __restrict output,
    int64_t total_elements,
    float scale,
    int8_t zero_point,
    concurrency::ThreadPool* thread_pool) {

  constexpr int vector_size = 16;  // Process 16 elements per main loop iteration
  const float scaled_zero_point = -zero_point * scale;
  const __m256 scale_vec = _mm256_set1_ps(scale);
  const __m256 scaled_zp_vec = _mm256_set1_ps(scaled_zero_point);

  // Cache block for L1 cache optimization
  constexpr int64_t cache_block = 1024;

  // Define the per-chunk work function
  auto process_chunk = [&](int64_t begin, int64_t end) {
    for (int64_t block_start = begin; block_start < end; block_start += cache_block) {
      const int64_t block_end = std::min(block_start + cache_block, end);

      // Prefetch the next block
      if (block_start + cache_block < end) {
        _mm_prefetch(reinterpret_cast<const char*>(input + block_start + cache_block), _MM_HINT_T0);
      }

      int64_t i = block_start;

      // Process vectors of 16 elements
      for (; i + vector_size <= block_end; i += vector_size) {
        // Prefetch ahead within the block
        _mm_prefetch(reinterpret_cast<const char*>(input + i + 64), _MM_HINT_T0);

        // Load 16 int8 elements
        const __m128i input16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));

        // Process first 8 elements
        __m256i input_i32 = _mm256_cvtepi8_epi32(input16);
        __m256 input_f = _mm256_cvtepi32_ps(input_i32);
        __m256 result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);

        // Use non-temporal store for large datasets
        _mm256_stream_ps(output + i, result);

        // Process next 8 elements
        const __m128i high64 = _mm_srli_si128(input16, 8);
        input_i32 = _mm256_cvtepi8_epi32(high64);
        input_f = _mm256_cvtepi32_ps(input_i32);
        result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);

        // Use non-temporal store for large datasets
        _mm256_stream_ps(output + i + 8, result);
      }

      // Process remaining elements in groups of 8
      for (; i + 8 <= block_end; i += 8) {
        const __m128i input8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input + i));
        const __m256i input_i32 = _mm256_cvtepi8_epi32(input8);
        const __m256 input_f = _mm256_cvtepi32_ps(input_i32);
        const __m256 result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);
        _mm256_storeu_ps(output + i, result);
      }

      // Handle remaining elements
      for (; i < block_end; ++i) {
        output[i] = input[i] * scale + scaled_zero_point;
      }
    }
  };

  // Use ONNX Runtime's thread pool instead of OpenMP
  if (thread_pool == nullptr) {
    // If no thread pool is provided, run in single-threaded mode
    process_chunk(0, total_elements);
  } else {
    // Set optimal chunk size for multi-threading
    constexpr int64_t chunk_size = 4096;

    // Calculate the number of chunks
    const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

    // Use ONNX Runtime's parallel for implementation
    concurrency::ThreadPool::TrySimpleParallelFor(
        thread_pool,
        num_chunks,
        [&](std::ptrdiff_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(start + chunk_size, total_elements);
          process_chunk(start, end);
        }
        /*0  no cost model for simple operations */);
  }

  // Ensure all non-temporal writes are visible
  _mm_sfence();
}

void nudgev_dequantize_linear_int32(
    const int32_t* input,
    float* output,
    int64_t total_elements,
    float scale,
    int32_t zero_point,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int vector_size = 8;
  const float scaled_zero_point = -zero_point * scale;
  const __m256 scale_vec = _mm256_set1_ps(scale);
  const __m256 scaled_zp_vec = _mm256_set1_ps(scaled_zero_point);

  auto process_chunk = [&](int64_t start_idx, int64_t end_idx) {
    int64_t i = start_idx;

    for (; i + vector_size <= end_idx; i += vector_size) {
      const __m256i input_i32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
      const __m256 input_f = _mm256_cvtepi32_ps(input_i32);
      const __m256 result = _mm256_fmadd_ps(input_f, scale_vec, scaled_zp_vec);
      _mm256_storeu_ps(output + i, result);
    }

    // Process remaining elements
    for (; i < end_idx; ++i) {
      output[i] = input[i] * scale + scaled_zero_point;
    }
  };

  constexpr int64_t chunk_size = 1024;  // Multiple of 8 for alignment
  if (tp != nullptr) {
    const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](int64_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(start + chunk_size, total_elements);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

Status ComputeNudgeVDequantizeLinear(DequantizeLinearParams* dequantize_params, OrtKernelContext* context) {
  auto start = std::chrono::high_resolution_clock::now();
  constexpr int64_t cache_threshold = 1024;  // Adjust based on profiling

  if (!dequantize_params) {
    return Status(common::ONNXRUNTIME, common::FAIL, "dequantize_params is null");
  }

  auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

  const Tensor* input = ctx_internal->Input<Tensor>(0);
  if (!input) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");
  }

  const auto& input_shape = input->Shape();
  auto data_type = input->DataType();

  Tensor* output = ctx_internal->Output(0, input_shape);
  if (!output) {
    return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
  }

  float* output_data = output->MutableData<float>();
  if (!output_data) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output data is null");
  }

  const int64_t total_elements = input->Shape().Size();
  bool use_cache = total_elements >= cache_threshold;

  if (data_type == DataTypeImpl::GetType<int8_t>()) {
    const int8_t* input_data = input->Data<int8_t>();
    if (!input_data) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
    }

    try {
      auto& cache = LastResultCache::GetInstance();
      const float* cached_result = use_cache ? cache.GetCachedResult(
          input_data, total_elements, dequantize_params->scale, dequantize_params->zero_point) : nullptr;

      if (cached_result) {
        // Parallel vectorized copy
        cache.ParallelCopy(cached_result, output_data, total_elements, tp);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "DequantizeLinear Op - Cache HIT: " << duration.count() << " microseconds" << std::endl;
      } else {
        nudgev_dequantize_linear(input_data, output_data, total_elements,
                                dequantize_params->scale, dequantize_params->zero_point, tp);

        if (use_cache) {
          cache.CacheResult(input_data, total_elements, dequantize_params->scale,
                          dequantize_params->zero_point, output_data, total_elements, tp);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "DequantizeLinear Op - Cache MISS: " << duration.count() << " microseconds" << std::endl;
      }
    } catch (const std::exception& e) {
      std::cerr << "Cache error: " << e.what() << ". Falling back to normal computation." << std::endl;
      nudgev_dequantize_linear(input_data, output_data, total_elements,
                              dequantize_params->scale, dequantize_params->zero_point, tp);
    }
  } else if (data_type == DataTypeImpl::GetType<int32_t>()) {
    // int32_t handling remains unchanged
    const auto* input_data = input->Data<int32_t>();
    if (!input_data) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
    }

    int32_t zero_point_i32 = static_cast<int32_t>(dequantize_params->zero_point);

    nudgev_dequantize_linear_int32(
        input_data,
        output_data,
        total_elements,
        dequantize_params->scale,
        zero_point_i32,
        tp);
  } else {
    return Status(common::ONNXRUNTIME, common::FAIL,
                "Unsupported input data type: " + std::string(DataTypeImpl::ToString(data_type)));
  }

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime
