#pragma once
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace nudgev {

thread_local struct {
  alignas(32) int32_t temp_buffer[32];
  alignas(32) int32_t accumulators[16];
} local_buffers;

void nudgev_add_requantized(
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int32_t M1_fixed,
    int32_t M2_fixed,
    int8_t input1_zp,
    int8_t input2_zp,
    int8_t output_zp,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;

  const __m256i input1_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input1_zp));
  const __m256i input2_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input2_zp));
  const __m256i output_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(output_zp));
  const __m256i min_vec = _mm256_set1_epi32(-128);
  const __m256i max_vec = _mm256_set1_epi32(127);

  auto process_chunk = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i += 256) {
      _mm_prefetch(reinterpret_cast<const char*>(input_a + i + 256), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(input_b + i + 256), _MM_HINT_T0);
    }

    for (int64_t i = start; i < end; i += 16) {
      if (i + 16 <= end) {
        __m128i a_8_low = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_a + i));
        __m128i a_8_high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_a + i + 8));
        __m128i b_8_low = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_b + i));
        __m128i b_8_high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_b + i + 8));

        __m256i a_32_low = _mm256_sub_epi32(_mm256_cvtepi8_epi32(a_8_low), input1_zp_vec);
        __m256i a_32_high = _mm256_sub_epi32(_mm256_cvtepi8_epi32(a_8_high), input1_zp_vec);
        __m256i b_32_low = _mm256_sub_epi32(_mm256_cvtepi8_epi32(b_8_low), input2_zp_vec);
        __m256i b_32_high = _mm256_sub_epi32(_mm256_cvtepi8_epi32(b_8_high), input2_zp_vec);

        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer), a_32_low);
        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer + 8), a_32_high);
        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer + 16), b_32_low);
        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer + 24), b_32_high);

        int64_t a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[0]) * M1_fixed;
        int64_t a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[1]) * M1_fixed;
        int64_t a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[2]) * M1_fixed;
        int64_t a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[3]) * M1_fixed;
        int64_t b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[16]) * M2_fixed;
        int64_t b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[17]) * M2_fixed;
        int64_t b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[18]) * M2_fixed;
        int64_t b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[19]) * M2_fixed;

        local_buffers.accumulators[0] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[1] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[2] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[3] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[4]) * M1_fixed;
        a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[5]) * M1_fixed;
        a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[6]) * M1_fixed;
        a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[7]) * M1_fixed;
        b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[20]) * M2_fixed;
        b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[21]) * M2_fixed;
        b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[22]) * M2_fixed;
        b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[23]) * M2_fixed;

        local_buffers.accumulators[4] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[5] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[6] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[7] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[8]) * M1_fixed;
        a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[9]) * M1_fixed;
        a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[10]) * M1_fixed;
        a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[11]) * M1_fixed;
        b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[24]) * M2_fixed;
        b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[25]) * M2_fixed;
        b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[26]) * M2_fixed;
        b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[27]) * M2_fixed;

        local_buffers.accumulators[8] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[9] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[10] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[11] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[12]) * M1_fixed;
        a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[13]) * M1_fixed;
        a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[14]) * M1_fixed;
        a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[15]) * M1_fixed;
        b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[28]) * M2_fixed;
        b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[29]) * M2_fixed;
        b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[30]) * M2_fixed;
        b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[31]) * M2_fixed;

        local_buffers.accumulators[12] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[13] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[14] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[15] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        __m256i result_low = _mm256_load_si256(reinterpret_cast<const __m256i*>(local_buffers.accumulators));
        __m256i result_high = _mm256_load_si256(reinterpret_cast<const __m256i*>(local_buffers.accumulators + 8));

        result_low = _mm256_add_epi32(result_low, output_zp_vec);
        result_high = _mm256_add_epi32(result_high, output_zp_vec);

        result_low = _mm256_min_epi32(_mm256_max_epi32(result_low, min_vec), max_vec);
        result_high = _mm256_min_epi32(_mm256_max_epi32(result_high, min_vec), max_vec);

        __m128i result_16_low = _mm_packs_epi32(_mm256_castsi256_si128(result_low),
                                                _mm256_extracti128_si256(result_low, 1));
        __m128i result_16_high = _mm_packs_epi32(_mm256_castsi256_si128(result_high),
                                                 _mm256_extracti128_si256(result_high, 1));

        __m128i result_8 = _mm_packs_epi16(result_16_low, result_16_high);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), result_8);
      } else {
        for (int64_t j = i; j < end; ++j) {
          int32_t a = static_cast<int32_t>(input_a[j]) - input1_zp;
          int32_t b = static_cast<int32_t>(input_b[j]) - input2_zp;

          int64_t scaled_a = static_cast<int64_t>(a) * M1_fixed;
          int64_t scaled_b = static_cast<int64_t>(b) * M2_fixed;

          int32_t sum = static_cast<int32_t>((scaled_a + scaled_b + 0x800000LL) >> 24);
          sum = sum + output_zp;
          sum = std::min(127, std::max(-128, sum));
          output[j] = static_cast<int8_t>(sum);
        }
      }
    }
  };

  const int64_t chunk_size = 4096;
  const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

  if (tp != nullptr) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](std::ptrdiff_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(total_elements, start + chunk_size);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

void nudgev_add_relu_requantized(
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int32_t M1_fixed,
    int32_t M2_fixed,
    int8_t input1_zp,
    int8_t input2_zp,
    int8_t output_zp,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;

  const __m256i input1_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input1_zp));
  const __m256i input2_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input2_zp));
  const __m256i output_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(output_zp));
  const __m256i zero_vec = _mm256_setzero_si256();
  const __m256i min_vec = _mm256_set1_epi32(-128);
  const __m256i max_vec = _mm256_set1_epi32(127);

  auto process_chunk = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i += 256) {
      _mm_prefetch(reinterpret_cast<const char*>(input_a + i + 256), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(input_b + i + 256), _MM_HINT_T0);
    }

    for (int64_t i = start; i < end; i += 16) {
      if (i + 16 <= end) {
        __m128i a_8_low = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_a + i));
        __m128i a_8_high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_a + i + 8));
        __m128i b_8_low = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_b + i));
        __m128i b_8_high = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_b + i + 8));

        __m256i a_32_low = _mm256_sub_epi32(_mm256_cvtepi8_epi32(a_8_low), input1_zp_vec);
        __m256i a_32_high = _mm256_sub_epi32(_mm256_cvtepi8_epi32(a_8_high), input1_zp_vec);
        __m256i b_32_low = _mm256_sub_epi32(_mm256_cvtepi8_epi32(b_8_low), input2_zp_vec);
        __m256i b_32_high = _mm256_sub_epi32(_mm256_cvtepi8_epi32(b_8_high), input2_zp_vec);

        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer), a_32_low);
        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer + 8), a_32_high);
        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer + 16), b_32_low);
        _mm256_store_si256(reinterpret_cast<__m256i*>(local_buffers.temp_buffer + 24), b_32_high);

        int64_t a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[0]) * M1_fixed;
        int64_t a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[1]) * M1_fixed;
        int64_t a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[2]) * M1_fixed;
        int64_t a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[3]) * M1_fixed;
        int64_t b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[16]) * M2_fixed;
        int64_t b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[17]) * M2_fixed;
        int64_t b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[18]) * M2_fixed;
        int64_t b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[19]) * M2_fixed;

        local_buffers.accumulators[0] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[1] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[2] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[3] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[4]) * M1_fixed;
        a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[5]) * M1_fixed;
        a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[6]) * M1_fixed;
        a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[7]) * M1_fixed;
        b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[20]) * M2_fixed;
        b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[21]) * M2_fixed;
        b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[22]) * M2_fixed;
        b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[23]) * M2_fixed;

        local_buffers.accumulators[4] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[5] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[6] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[7] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[8]) * M1_fixed;
        a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[9]) * M1_fixed;
        a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[10]) * M1_fixed;
        a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[11]) * M1_fixed;
        b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[24]) * M2_fixed;
        b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[25]) * M2_fixed;
        b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[26]) * M2_fixed;
        b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[27]) * M2_fixed;

        local_buffers.accumulators[8] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[9] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[10] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[11] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(local_buffers.temp_buffer[12]) * M1_fixed;
        a_val1 = static_cast<int64_t>(local_buffers.temp_buffer[13]) * M1_fixed;
        a_val2 = static_cast<int64_t>(local_buffers.temp_buffer[14]) * M1_fixed;
        a_val3 = static_cast<int64_t>(local_buffers.temp_buffer[15]) * M1_fixed;
        b_val0 = static_cast<int64_t>(local_buffers.temp_buffer[28]) * M2_fixed;
        b_val1 = static_cast<int64_t>(local_buffers.temp_buffer[29]) * M2_fixed;
        b_val2 = static_cast<int64_t>(local_buffers.temp_buffer[30]) * M2_fixed;
        b_val3 = static_cast<int64_t>(local_buffers.temp_buffer[31]) * M2_fixed;

        local_buffers.accumulators[12] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        local_buffers.accumulators[13] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        local_buffers.accumulators[14] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        local_buffers.accumulators[15] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        __m256i result_low = _mm256_load_si256(reinterpret_cast<const __m256i*>(local_buffers.accumulators));
        __m256i result_high = _mm256_load_si256(reinterpret_cast<const __m256i*>(local_buffers.accumulators + 8));

        result_low = _mm256_max_epi32(result_low, zero_vec);
        result_high = _mm256_max_epi32(result_high, zero_vec);

        result_low = _mm256_add_epi32(result_low, output_zp_vec);
        result_high = _mm256_add_epi32(result_high, output_zp_vec);

        result_low = _mm256_min_epi32(_mm256_max_epi32(result_low, min_vec), max_vec);
        result_high = _mm256_min_epi32(_mm256_max_epi32(result_high, min_vec), max_vec);

        __m128i result_16_low = _mm_packs_epi32(_mm256_castsi256_si128(result_low),
                                                _mm256_extracti128_si256(result_low, 1));
        __m128i result_16_high = _mm_packs_epi32(_mm256_castsi256_si128(result_high),
                                                 _mm256_extracti128_si256(result_high, 1));

        __m128i result_8 = _mm_packs_epi16(result_16_low, result_16_high);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), result_8);
      } else {
        for (int64_t j = i; j < end; ++j) {
          int32_t a = static_cast<int32_t>(input_a[j]) - input1_zp;
          int32_t b = static_cast<int32_t>(input_b[j]) - input2_zp;

          int64_t scaled_a = static_cast<int64_t>(a) * M1_fixed;
          int64_t scaled_b = static_cast<int64_t>(b) * M2_fixed;

          int32_t sum = static_cast<int32_t>((scaled_a + scaled_b + 0x800000LL) >> 24);
          sum = std::max(sum, 0);
          sum = sum + output_zp;
          sum = std::min(127, std::max(-128, sum));
          output[j] = static_cast<int8_t>(sum);
        }
      }
    }
  };

  const int64_t chunk_size = 4096;
  const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

  if (tp != nullptr) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](std::ptrdiff_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(total_elements, start + chunk_size);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

void nudgev_add(
    const float* input_a,
    const float* input_b,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;

  auto process_chunk = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i += 256) {
      _mm_prefetch(reinterpret_cast<const char*>(input_a + i + 256), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(input_b + i + 256), _MM_HINT_T0);
    }

    for (int64_t i = start; i < end; i += 8) {
      if (i + 8 <= end) {
        __m256 a = _mm256_loadu_ps(input_a + i);
        __m256 b = _mm256_loadu_ps(input_b + i);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_storeu_ps(output + i, result);
      } else {
        for (int64_t j = i; j < end; ++j) {
          output[j] = input_a[j] + input_b[j];
        }
      }
    }
  };

  const int64_t chunk_size = 4096;
  const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

  if (tp != nullptr) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](std::ptrdiff_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(total_elements, start + chunk_size);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

void nudgev_add_relu(
    const float* input_a,
    const float* input_b,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;

  const __m256 zero_vec = _mm256_setzero_ps();

  auto process_chunk = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i += 256) {
      _mm_prefetch(reinterpret_cast<const char*>(input_a + i + 256), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(input_b + i + 256), _MM_HINT_T0);
    }

    for (int64_t i = start; i < end; i += 8) {
      if (i + 8 <= end) {
        __m256 a = _mm256_loadu_ps(input_a + i);
        __m256 b = _mm256_loadu_ps(input_b + i);
        __m256 sum = _mm256_add_ps(a, b);
        __m256 result = _mm256_max_ps(sum, zero_vec);
        _mm256_storeu_ps(output + i, result);
      } else {
        for (int64_t j = i; j < end; ++j) {
          float sum = input_a[j] + input_b[j];
          output[j] = std::max(0.0f, sum);
        }
      }
    }
  };

  const int64_t chunk_size = 4096;
  const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

  if (tp != nullptr) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](std::ptrdiff_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = std::min(total_elements, start + chunk_size);
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

}  // namespace nudgev
}  // namespace onnxruntime