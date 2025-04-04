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

struct DequantizeCacheKey {
  const void* input_ptr;
  int64_t total_elements;
  float scale;
  int8_t zero_point;

  bool operator==(const DequantizeCacheKey& other) const {
    return input_ptr == other.input_ptr &&
           total_elements == other.total_elements &&
           scale == other.scale &&
           zero_point == other.zero_point;
  }
};

struct DequantizeCacheKeyHash {
  std::size_t operator()(const DequantizeCacheKey& key) const {
    std::size_t hash = std::hash<const void*>{}(key.input_ptr);
    hash ^= std::hash<int64_t>{}(key.total_elements) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<float>{}(key.scale) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int8_t>{}(key.zero_point) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct DequantizeCacheEntry {
  std::shared_ptr<std::vector<float>> data;
  std::chrono::steady_clock::time_point last_used;
};


class DequantizeCache {
public:
  static DequantizeCache& GetInstance() {
    static DequantizeCache instance;
    return instance;
  }

  const float* GetCachedResult(const DequantizeCacheKey& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second.last_used = std::chrono::steady_clock::now();
      return it->second.data->data();
    }
    return nullptr;
  }

  void CacheResult(const DequantizeCacheKey& key, const float* data, int64_t size) {
    std::lock_guard<std::mutex> lock(cache_mutex_);

     if too large, evict oldest entries
    if (cache_.size() > MAX_CACHE_ENTRIES) {
      EvictOldestEntries();
    }

    auto vec_ptr = std::make_shared<std::vector<float>>(size);
    std::memcpy(vec_ptr->data(), data, size * sizeof(float));

    DequantizeCacheEntry entry;
    entry.data = vec_ptr;
    entry.last_used = std::chrono::steady_clock::now();

    cache_[key] = std::move(entry);
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
  }

private:
  DequantizeCache() = default;
  ~DequantizeCache() = default;

  DequantizeCache(const DequantizeCache&) = delete;
  DequantizeCache& operator=(const DequantizeCache&) = delete;

  void EvictOldestEntries() {
    if (cache_.empty()) return;

    std::vector<std::pair<DequantizeCacheKey, std::chrono::steady_clock::time_point>> entries;
    entries.reserve(cache_.size());

    for (const auto& entry : cache_) {
      entries.emplace_back(entry.first, entry.second.last_used);
    }

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    size_t to_remove = cache_.size() / 4;
    if (to_remove == 0) to_remove = 1;

    for (size_t i = 0; i < to_remove && i < entries.size(); ++i) {
      cache_.erase(entries[i].first);
    }
  }

  static constexpr size_t MAX_CACHE_ENTRIES = 50;
  std::unordered_map<DequantizeCacheKey, DequantizeCacheEntry, DequantizeCacheKeyHash> cache_;
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

  auto* output_data = output->MutableData<float>();
  if (!output_data) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output data is null");
  }

  const int64_t total_elements = input->Shape().Size();

  if (data_type == DataTypeImpl::GetType<int8_t>()) {
    const auto* input_data = input->Data<int8_t>();
    if (!input_data) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Input data is null");
    }

    try {
      DequantizeCacheKey cache_key{
        input_data,
        total_elements,
        dequantize_params->scale,
        dequantize_params->zero_point
      };

      auto& cache = DequantizeCache::GetInstance();
      const float* cached_result = cache.GetCachedResult(cache_key);

      if (cached_result) {
        // Cache hit
        std::memcpy(output_data, cached_result, total_elements * sizeof(float));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "DequantizeLinear Op - Cache HIT: " << duration.count() << " microseconds" << std::endl;
      } else {
        // Cache miss
        nudgev_dequantize_linear(
            input_data,
            output_data,
            total_elements,
            dequantize_params->scale,
            dequantize_params->zero_point,
            tp);

        cache.CacheResult(cache_key, output_data, total_elements);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "DequantizeLinear Op - Cache MISS: " << duration.count() << " microseconds" << std::endl;
      }
    } catch (const std::exception& e) {
      // Fall back to normal computation if caching fails
      std::cerr << "Cache error: " << e.what() << ". Falling back to normal computation." << std::endl;

      nudgev_dequantize_linear(
          input_data,
          output_data,
          total_elements,
          dequantize_params->scale,
          dequantize_params->zero_point,
          tp);
    }
  } else if (data_type == DataTypeImpl::GetType<int32_t>()) {
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
