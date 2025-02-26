#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>

namespace onnxruntime {
namespace nudgev {

void gemm(
    const int8_t* weights,
    const int8_t* input,
    int8_t* output,
    int64_t M,
    int64_t K,
    int64_t N,
    const int32_t* bias,
    bool has_bias,
    int32_t M_fixed,
    int8_t input_zero_point,
    int8_t output_zero_point,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int vector_size = 16;

  std::vector<int32_t> zp_compensation(N);
  for (int n = 0; n < N; ++n) {
    int32_t weight_sum = 0;
    const int8_t* weight_row = weights + n * K;
    for (int k = 0; k < K; ++k) {
      weight_sum += static_cast<int32_t>(weight_row[k]);
    }
    zp_compensation[n] = weight_sum * static_cast<int32_t>(input_zero_point);
  }

  auto process_chunk = [&](int64_t start_idx, int64_t end_idx) {
    for (int64_t idx = start_idx; idx < end_idx; ++idx) {
      int64_t m = idx / N;
      int64_t n = idx % N;
      const int8_t* input_row = input + m * K;
      int8_t* output_ptr = output + m * N;
      const int8_t* weight_row = weights + n * K;

      __m256i acc_lo = _mm256_setzero_si256();
      __m256i acc_hi = _mm256_setzero_si256();

      int64_t k = 0;
      for (; k <= K - vector_size; k += vector_size) {
        __m128i inp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input_row + k));
        __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight_row + k));

        __m256i inp_epi16 = _mm256_cvtepi8_epi16(inp);
        __m256i w_epi16 = _mm256_cvtepi8_epi16(w);

        __m256i prod = _mm256_mullo_epi16(w_epi16, inp_epi16);
        __m128i prod_lo = _mm256_castsi256_si128(prod);
        __m128i prod_hi = _mm256_extracti128_si256(prod, 1);

        acc_lo = _mm256_add_epi32(acc_lo, _mm256_cvtepi16_epi32(prod_lo));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_cvtepi16_epi32(prod_hi));
      }

      alignas(32) int32_t acc_array[8];
      _mm256_store_si256(reinterpret_cast<__m256i*>(acc_array), acc_lo);
      int32_t acc = 0;
      for (int i = 0; i < 8; ++i) {
        acc += acc_array[i];
      }
      _mm256_store_si256(reinterpret_cast<__m256i*>(acc_array), acc_hi);
      for (int i = 0; i < 8; ++i) {
        acc += acc_array[i];
      }

      for (; k < K; ++k) {
        acc += static_cast<int32_t>(weight_row[k]) * static_cast<int32_t>(input_row[k]);
      }
      acc -= zp_compensation[n];
      if (has_bias) {
        acc += bias[n];
      }
      int64_t scaled_acc = static_cast<int64_t>(acc) * M_fixed;
      int32_t sum = static_cast<int32_t>((scaled_acc + 0x4000) >> 15);

      if (fused_relu) {
        sum = std::max(sum, 0);
      }

      sum += output_zero_point;
      sum = std::min(127, std::max(-128, sum));

      output_ptr[n] = static_cast<int8_t>(sum);
    }
  };

  const int64_t total_work = M * N;
  if (tp != nullptr) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_work,
        [&](int64_t work_idx) {
          process_chunk(work_idx, work_idx + 1);
        });
  } else {
    process_chunk(0, total_work);
  }
}

void nudgev_gemm(
    const int8_t* input,
    const int8_t* weights,
    int8_t* output,
    int64_t M,
    int64_t K,
    int64_t N,
    const int32_t* bias,
    bool has_bias,
    int32_t M_fixed,
    int8_t input_zp,
    int8_t output_zp,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  gemm(weights,
       input,
       output,
       M,
       K,
       N,
       bias,
       has_bias,
       M_fixed,
       input_zp,
       output_zp,
       fused_relu,
       tp);
}

}  // namespace nudgev
}  // namespace onnxruntime