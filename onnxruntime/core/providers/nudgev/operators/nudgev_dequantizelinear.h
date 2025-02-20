#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include "core/platform/threadpool.h"
#include <immintrin.h>

namespace onnxruntime {
namespace nudgev {

void nudgev_dequantize(
    const int8_t* input,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    float scale,
    int8_t zero_point,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;
  constexpr int64_t block_size = 32;

  const __m256 vscale = _mm256_set1_ps(scale);
  const __m256i vzp = _mm256_set1_epi8(zero_point);

  auto process_chunk = [&](int64_t i) {
    if (i + block_size <= total_elements) {
      __m256i vinput = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
      vinput = _mm256_sub_epi8(vinput, vzp);

      __m128i vlow = _mm256_extractf128_si256(vinput, 0);
      __m256i vlow_int32 = _mm256_cvtepi8_epi32(vlow);
      __m256 vlow_float = _mm256_cvtepi32_ps(vlow_int32);

      __m128i vhigh = _mm256_extractf128_si256(vinput, 1);
      __m256i vhigh_int32 = _mm256_cvtepi8_epi32(vhigh);
      __m256 vhigh_float = _mm256_cvtepi32_ps(vhigh_int32);

      vlow_float = _mm256_mul_ps(vlow_float, vscale);
      vhigh_float = _mm256_mul_ps(vhigh_float, vscale);

      _mm256_storeu_ps(output + i, vlow_float);
      _mm256_storeu_ps(output + i + 8, vhigh_float);
    } else {
      for (int64_t j = i; j < total_elements; ++j) {
        float dequantized = static_cast<float>(input[j] - zero_point) * scale;
        output[j] = dequantized;
      }
    }
  };

  if (tp != nullptr) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp,
        (total_elements + block_size - 1) / block_size,
        [&](std::ptrdiff_t i) {
          process_chunk(i * block_size);
        });
  } else {
    for (int64_t i = 0; i < total_elements; i += block_size) {
      process_chunk(i);
    }
  }
}

}  // namespace nudgev
}  // namespace onnxruntime