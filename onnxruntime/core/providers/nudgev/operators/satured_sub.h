#pragma once
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>

namespace onnxruntime {
namespace nudgev {
inline void satured_sub(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t elements_per_batch,
    int8_t x_zero_point,
    onnxruntime::concurrency::ThreadPool* tp) {
  const __m256i vzp = _mm256_set1_epi8(x_zero_point);
  constexpr int vec_size = 32;

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](long b) {
      const int8_t* batch_input = input + b * elements_per_batch;
      int8_t* batch_output = output + b * elements_per_batch;
      ptrdiff_t i = 0;
      for (; i < static_cast<ptrdiff_t>(elements_per_batch) && (i % vec_size != 0); ++i) {
        __m128i v = _mm_set1_epi8(batch_input[i]);
        __m128i z = _mm_set1_epi8(x_zero_point);
        __m128i r = _mm_subs_epi8(v, z);
        batch_output[i] = _mm_extract_epi8(r, 0);
      }

      for (; i + vec_size <= static_cast<ptrdiff_t>(elements_per_batch); i += vec_size) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(batch_input + i));
        __m256i r = _mm256_subs_epi8(v, vzp);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(batch_output + i), r);
      }

      for (; i < static_cast<ptrdiff_t>(elements_per_batch); ++i) {
        __m128i v = _mm_set1_epi8(batch_input[i]);
        __m128i z = _mm_set1_epi8(x_zero_point);
        __m128i r = _mm_subs_epi8(v, z);
        batch_output[i] = _mm_extract_epi8(r, 0);
      }
    });
  } else {
    ptrdiff_t i = 0;
    for (; i < static_cast<ptrdiff_t>(elements_per_batch) && (i % vec_size != 0); ++i) {
      __m128i v = _mm_set1_epi8(input[i]);
      __m128i z = _mm_set1_epi8(x_zero_point);
      __m128i r = _mm_subs_epi8(v, z);
      output[i] = _mm_extract_epi8(r, 0);
    }

    for (; i + vec_size <= static_cast<ptrdiff_t>(elements_per_batch); i += vec_size) {
      __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
      __m256i r = _mm256_subs_epi8(v, vzp);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), r);
    }

    for (; i < static_cast<ptrdiff_t>(elements_per_batch); ++i) {
      __m128i v = _mm_set1_epi8(input[i]);
      __m128i z = _mm_set1_epi8(x_zero_point);
      __m128i r = _mm_subs_epi8(v, z);
      output[i] = _mm_extract_epi8(r, 0);
    }
  }
}
}  // namespace nudgev
}  // namespace onnxruntime