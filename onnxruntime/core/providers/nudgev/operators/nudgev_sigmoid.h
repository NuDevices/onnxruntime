#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include "core/platform/threadpool.h"
#include <immintrin.h>

namespace onnxruntime {
namespace nudgev {

inline __m256 exp256_ps(__m256 x) {
  const __m256 max_exp = _mm256_set1_ps(88.0f);
  const __m256 min_exp = _mm256_set1_ps(-88.0f);
  x = _mm256_max_ps(_mm256_min_ps(x, max_exp), min_exp);

  const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
  const __m256 inv_ln2 = _mm256_set1_ps(1.442695040888963f);
  const __m256 one = _mm256_set1_ps(1.0f);

  __m256 fx = _mm256_mul_ps(x, inv_ln2);
  fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
  fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  x = _mm256_fnmadd_ps(fx, ln2, x);  // Usiamo FMA

  const __m256 c1 = _mm256_set1_ps(1.0f);
  const __m256 c2 = _mm256_set1_ps(0.5f);
  const __m256 c3 = _mm256_set1_ps(0.166666666666f);
  const __m256 c4 = _mm256_set1_ps(0.041666666666f);
  const __m256 c5 = _mm256_set1_ps(0.008333333333f);
  const __m256 c6 = _mm256_set1_ps(0.001388888888f);

  __m256 r = x;
  __m256 r2 = _mm256_mul_ps(r, r);
  __m256 r3 = _mm256_mul_ps(r2, r);
  __m256 r4 = _mm256_mul_ps(r2, r2);
  __m256 r5 = _mm256_mul_ps(r4, r);
  __m256 r6 = _mm256_mul_ps(r4, r2);

  __m256 px = _mm256_fmadd_ps(r6, c6,
                              _mm256_fmadd_ps(r5, c5,
                                              _mm256_fmadd_ps(r4, c4,
                                                              _mm256_fmadd_ps(r3, c3,
                                                                              _mm256_fmadd_ps(r2, c2,
                                                                                              _mm256_fmadd_ps(r, c1, one))))));

  __m256i pow2n = _mm256_cvtps_epi32(fx);
  pow2n = _mm256_add_epi32(pow2n, _mm256_set1_epi32(127));
  pow2n = _mm256_slli_epi32(pow2n, 23);
  __m256 pow2 = _mm256_castsi256_ps(pow2n);

  return _mm256_mul_ps(px, pow2);
}

inline __m256 sigmoid_avx2(__m256 x) {
  const __m256 one = _mm256_set1_ps(1.0f);
  __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
  __m256 exp_neg_x = exp256_ps(neg_x);
  __m256 denom = _mm256_add_ps(one, exp_neg_x);
  return _mm256_div_ps(one, denom);
}

void nudgev_sigmoid(
    const float* __restrict input,
    float* __restrict output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;
  constexpr int64_t block_size = 32;
  const int64_t num_blocks = (total_elements + block_size - 1) / block_size;

  auto process_block = [&](int64_t block_idx) {
    const int64_t start_idx = block_idx * block_size;
    const int64_t end_idx = std::min(start_idx + block_size, total_elements);

    int64_t i = start_idx;
    for (; i + 31 < end_idx; i += 32) {
      __m256 x1 = _mm256_loadu_ps(input + i);
      __m256 x2 = _mm256_loadu_ps(input + i + 8);
      __m256 x3 = _mm256_loadu_ps(input + i + 16);
      __m256 x4 = _mm256_loadu_ps(input + i + 24);

      __m256 r1 = sigmoid_avx2(x1);
      __m256 r2 = sigmoid_avx2(x2);
      __m256 r3 = sigmoid_avx2(x3);
      __m256 r4 = sigmoid_avx2(x4);

      _mm256_storeu_ps(output + i, r1);
      _mm256_storeu_ps(output + i + 8, r2);
      _mm256_storeu_ps(output + i + 16, r3);
      _mm256_storeu_ps(output + i + 24, r4);
    }

    for (; i < end_idx; ++i) {
      output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
  };

  if (tp != nullptr && num_blocks > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_blocks, [&](std::ptrdiff_t block_idx) {
          process_block(block_idx);
        });
  } else {
    for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
      process_block(block_idx);
    }
  }
}

}  // namespace nudgev
}  // namespace onnxruntime