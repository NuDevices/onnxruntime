#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include "core/providers/nudgev/operators/satured_sub.h"
#include <immintrin.h>

namespace onnxruntime {
namespace nudgev {

void gemm(
    const int8_t* weights,
    const int8_t* im2col_output,
    int8_t* output,
    int64_t OC,
    int64_t K,
    int64_t patches_per_image,
    int64_t batch_size,
    const int32_t* bias,
    bool has_bias,
    int32_t M_fixed,
    int8_t output_zero_point,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int vector_size = 16;

  auto process_chunk = [&](int64_t start_idx, int64_t end_idx) {
    for (int64_t idx = start_idx; idx < end_idx; ++idx) {
      int64_t b = idx / OC;
      int64_t oc = idx % OC;
      const int8_t* batch_input = im2col_output + b * K * patches_per_image;
      int8_t* batch_output = output + b * OC * patches_per_image;
      const int8_t* weight_row = weights + oc * K;

      int64_t n = 0;
      for (; n <= patches_per_image - vector_size; n += vector_size) {
        __m256i acc_lo = _mm256_setzero_si256();
        __m256i acc_hi = _mm256_setzero_si256();
        for (int64_t k = 0; k < K; ++k) {
          int8_t w_val = weight_row[k];
          __m256i w_vec = _mm256_set1_epi16(static_cast<int16_t>(w_val));
          const int8_t* inp_ptr = batch_input + k * patches_per_image + n;
          __m128i inp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inp_ptr));
          __m256i inp_epi16 = _mm256_cvtepi8_epi16(inp);
          __m256i prod = _mm256_mullo_epi16(w_vec, inp_epi16);
          __m128i prod_lo = _mm256_castsi256_si128(prod);
          __m128i prod_hi = _mm256_extracti128_si256(prod, 1);
          acc_lo = _mm256_add_epi32(acc_lo, _mm256_cvtepi16_epi32(prod_lo));
          acc_hi = _mm256_add_epi32(acc_hi, _mm256_cvtepi16_epi32(prod_hi));
        }
        alignas(32) int32_t acc_array[vector_size];
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc_array), acc_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(acc_array + 8), acc_hi);
        for (int i = 0; i < vector_size; ++i) {
          int32_t acc = acc_array[i];
          if (has_bias)
            acc += bias[oc];
          int64_t scaled_acc = static_cast<int64_t>(acc) * M_fixed;
          int32_t sum = static_cast<int32_t>((scaled_acc + 0x4000) >> 15);
          if (fused_relu)
            sum = std::max(sum, 0);
          sum += output_zero_point;
          sum = std::min(127, std::max(-128, sum));
          batch_output[oc * patches_per_image + n + i] = static_cast<int8_t>(sum);
        }
      }
      for (; n < patches_per_image; ++n) {
        int32_t acc = 0;
        for (int64_t k = 0; k < K; ++k) {
          acc += static_cast<int32_t>(weight_row[k]) *
                 static_cast<int32_t>(batch_input[k * patches_per_image + n]);
        }
        if (has_bias)
          acc += bias[oc];
        int64_t scaled_acc = static_cast<int64_t>(acc) * M_fixed;
        int32_t sum = static_cast<int32_t>((scaled_acc + 0x4000) >> 15);
        if (fused_relu)
          sum = std::max(sum, 0);
        sum += output_zero_point;
        sum = std::min(127, std::max(-128, sum));
        batch_output[oc * patches_per_image + n] = static_cast<int8_t>(sum);
      }
    }
  };

  const int64_t total_work = batch_size * OC;
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
    int8_t* input_centered_buffer,
    onnxruntime::concurrency::ThreadPool* tp) {
  onnxruntime::nudgev::satured_sub(
      input,
      input_centered_buffer,
      M,
      K,
      input_zp,
      tp);

  gemm(weights,
       input_centered_buffer,
       output,
       N,
       K,
       1,
       M,
       bias,
       has_bias,
       M_fixed,
       output_zp,
       fused_relu,
       tp);
}

}  // namespace nudgev
}  // namespace onnxruntime