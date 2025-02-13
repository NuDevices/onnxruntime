#pragma once
#include <cstring>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include "core/platform/threadpool.h"
#include "core/providers/nudgev/operators/satured_sub.h"

namespace onnxruntime {
namespace nudgev {

void optimized_nudgev_add(
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
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;

  const __m256i input1_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input1_zp));
  const __m256i input2_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input2_zp));
  const __m256i output_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(output_zp));
  const __m256i zero_vec = _mm256_setzero_si256();
  const __m256i min_vec = _mm256_set1_epi32(-128);
  const __m256i max_vec = _mm256_set1_epi32(127);

  struct alignas(64) ThreadLocalData {
    int32_t temp_buffer[32];
    int32_t accumulators[16];
  };

  thread_local static ThreadLocalData tls;

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

        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.temp_buffer), a_32_low);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.temp_buffer + 8), a_32_high);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.temp_buffer + 16), b_32_low);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.temp_buffer + 24), b_32_high);

        int64_t a_val0 = static_cast<int64_t>(tls.temp_buffer[0]) * M1_fixed;
        int64_t a_val1 = static_cast<int64_t>(tls.temp_buffer[1]) * M1_fixed;
        int64_t a_val2 = static_cast<int64_t>(tls.temp_buffer[2]) * M1_fixed;
        int64_t a_val3 = static_cast<int64_t>(tls.temp_buffer[3]) * M1_fixed;
        int64_t b_val0 = static_cast<int64_t>(tls.temp_buffer[16]) * M2_fixed;
        int64_t b_val1 = static_cast<int64_t>(tls.temp_buffer[17]) * M2_fixed;
        int64_t b_val2 = static_cast<int64_t>(tls.temp_buffer[18]) * M2_fixed;
        int64_t b_val3 = static_cast<int64_t>(tls.temp_buffer[19]) * M2_fixed;

        tls.accumulators[0] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        tls.accumulators[1] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        tls.accumulators[2] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        tls.accumulators[3] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(tls.temp_buffer[4]) * M1_fixed;
        a_val1 = static_cast<int64_t>(tls.temp_buffer[5]) * M1_fixed;
        a_val2 = static_cast<int64_t>(tls.temp_buffer[6]) * M1_fixed;
        a_val3 = static_cast<int64_t>(tls.temp_buffer[7]) * M1_fixed;
        b_val0 = static_cast<int64_t>(tls.temp_buffer[20]) * M2_fixed;
        b_val1 = static_cast<int64_t>(tls.temp_buffer[21]) * M2_fixed;
        b_val2 = static_cast<int64_t>(tls.temp_buffer[22]) * M2_fixed;
        b_val3 = static_cast<int64_t>(tls.temp_buffer[23]) * M2_fixed;

        tls.accumulators[4] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        tls.accumulators[5] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        tls.accumulators[6] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        tls.accumulators[7] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(tls.temp_buffer[8]) * M1_fixed;
        a_val1 = static_cast<int64_t>(tls.temp_buffer[9]) * M1_fixed;
        a_val2 = static_cast<int64_t>(tls.temp_buffer[10]) * M1_fixed;
        a_val3 = static_cast<int64_t>(tls.temp_buffer[11]) * M1_fixed;
        b_val0 = static_cast<int64_t>(tls.temp_buffer[24]) * M2_fixed;
        b_val1 = static_cast<int64_t>(tls.temp_buffer[25]) * M2_fixed;
        b_val2 = static_cast<int64_t>(tls.temp_buffer[26]) * M2_fixed;
        b_val3 = static_cast<int64_t>(tls.temp_buffer[27]) * M2_fixed;

        tls.accumulators[8] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        tls.accumulators[9] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        tls.accumulators[10] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        tls.accumulators[11] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        a_val0 = static_cast<int64_t>(tls.temp_buffer[12]) * M1_fixed;
        a_val1 = static_cast<int64_t>(tls.temp_buffer[13]) * M1_fixed;
        a_val2 = static_cast<int64_t>(tls.temp_buffer[14]) * M1_fixed;
        a_val3 = static_cast<int64_t>(tls.temp_buffer[15]) * M1_fixed;
        b_val0 = static_cast<int64_t>(tls.temp_buffer[28]) * M2_fixed;
        b_val1 = static_cast<int64_t>(tls.temp_buffer[29]) * M2_fixed;
        b_val2 = static_cast<int64_t>(tls.temp_buffer[30]) * M2_fixed;
        b_val3 = static_cast<int64_t>(tls.temp_buffer[31]) * M2_fixed;

        tls.accumulators[12] = static_cast<int32_t>((a_val0 + b_val0 + 0x800000LL) >> 24);
        tls.accumulators[13] = static_cast<int32_t>((a_val1 + b_val1 + 0x800000LL) >> 24);
        tls.accumulators[14] = static_cast<int32_t>((a_val2 + b_val2 + 0x800000LL) >> 24);
        tls.accumulators[15] = static_cast<int32_t>((a_val3 + b_val3 + 0x800000LL) >> 24);

        __m256i result_low = _mm256_load_si256(reinterpret_cast<const __m256i*>(tls.accumulators));
        __m256i result_high = _mm256_load_si256(reinterpret_cast<const __m256i*>(tls.accumulators + 8));

        if (fused_relu) {
          result_low = _mm256_max_epi32(result_low, zero_vec);
          result_high = _mm256_max_epi32(result_high, zero_vec);
        }

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

          if (fused_relu) {
            sum = std::max(sum, 0);
          }

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
/* old versions

void optimized_nudgev_add2(
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
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t total_elements = batch_size * channels * height * width;
  constexpr size_t chunk_size = 4096;

  struct alignas(64) ThreadLocalData {
    int32_t input1_buffer[32];
    int32_t input2_buffer[32];
    int32_t output_buffer[32];
  };
  thread_local static ThreadLocalData tls;
  constexpr int64_t prefetch_distance = 16;

  const __m256i vzp1 = _mm256_set1_epi32(static_cast<int32_t>(input1_zp));
  const __m256i vzp2 = _mm256_set1_epi32(static_cast<int32_t>(input2_zp));
  const __m256i vzp_out = _mm256_set1_epi32(static_cast<int32_t>(output_zp));
  const __m256i zero_vec = _mm256_setzero_si256();
  const __m256i min_vec = _mm256_set1_epi32(-128);
  const __m256i max_vec = _mm256_set1_epi32(127);

  auto process_chunk = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i += prefetch_distance) {
      _mm_prefetch(reinterpret_cast<const char*>(input_a + i + prefetch_distance), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(input_b + i + prefetch_distance), _MM_HINT_T0);
    }

    for (int64_t i = start; i < end; i += 8) {
      if (i + 8 <= end) {
        __m128i a_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_a + i));
        __m128i b_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_b + i));

        __m256i a_32 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(a_8), vzp1);
        __m256i b_32 = _mm256_sub_epi32(_mm256_cvtepi8_epi32(b_8), vzp2);

        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.input1_buffer), a_32);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.input2_buffer), b_32);

        for (int j = 0; j < 8; j++) {
          int64_t scaled_a = static_cast<int64_t>(tls.input1_buffer[j]) * M1_fixed;
          int64_t scaled_b = static_cast<int64_t>(tls.input2_buffer[j]) * M2_fixed;

          int32_t a_rounded = static_cast<int32_t>((scaled_a + 0x800000LL) >> 24);
          int32_t b_rounded = static_cast<int32_t>((scaled_b + 0x800000LL) >> 24);
          tls.output_buffer[j] = a_rounded + b_rounded;
        }

        __m256i result = _mm256_load_si256(reinterpret_cast<const __m256i*>(tls.output_buffer));

        if (fused_relu) {
          result = _mm256_max_epi32(result, zero_vec);
        }

        result = _mm256_add_epi32(result, vzp_out);
        result = _mm256_min_epi32(_mm256_max_epi32(result, min_vec), max_vec);

        __m128i result_16 = _mm_packs_epi32(_mm256_castsi256_si128(result),
                                            _mm256_extracti128_si256(result, 1));
        __m128i result_8 = _mm_packs_epi16(result_16, result_16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), result_8);
      } else {
        for (int64_t j = i; j < end; ++j) {
          int32_t a = static_cast<int32_t>(input_a[j]) - input1_zp;
          int32_t b = static_cast<int32_t>(input_b[j]) - input2_zp;

          int64_t scaled_a = static_cast<int64_t>(a) * M1_fixed;
          int64_t scaled_b = static_cast<int64_t>(b) * M2_fixed;

          int32_t a_rounded = static_cast<int32_t>((scaled_a + 0x800000LL) >> 24);
          int32_t b_rounded = static_cast<int32_t>((scaled_b + 0x800000LL) >> 24);

          int32_t sum = a_rounded + b_rounded;

          if (fused_relu) {
            sum = std::max(sum, 0);
          }

          sum = sum + output_zp;
          sum = std::min(127, std::max(-128, sum));
          output[j] = static_cast<int8_t>(sum);
        }
      }
    }
  };

  const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
  if (tp != nullptr) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](std::ptrdiff_t chunk_idx) {
          const int64_t start = chunk_idx * chunk_size;
          const int64_t end = (start + chunk_size <= total_elements) ? start + chunk_size : total_elements;
          process_chunk(start, end);
        });
  } else {
    process_chunk(0, total_elements);
  }
}

void optimized_nudgev_add(
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
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t elements_per_batch = channels * height * width;
  const int64_t total_elements = batch_size * elements_per_batch;

  const __m256i input1_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input1_zp));
  const __m256i input2_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(input2_zp));
  const __m256i output_zp_vec = _mm256_set1_epi32(static_cast<int32_t>(output_zp));
  const __m256i zero_vec = _mm256_setzero_si256();
  const __m256i min_vec = _mm256_set1_epi32(-128);
  const __m256i max_vec = _mm256_set1_epi32(127);

  const int64_t prefetch_distance = 16;

  struct alignas(32) ThreadLocalData {
    int32_t temp_buffer[16];
  };

  thread_local static ThreadLocalData tls;

  auto process_chunk = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i += 8) {
      if (i + 8 <= end) {
        _mm_prefetch(reinterpret_cast<const char*>(input_a + i + prefetch_distance), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(input_b + i + prefetch_distance), _MM_HINT_T0);

        __m128i a_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_a + i));
        __m128i b_8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input_b + i));

        __m256i a_32 = _mm256_cvtepi8_epi32(a_8);
        __m256i b_32 = _mm256_cvtepi8_epi32(b_8);

        a_32 = _mm256_sub_epi32(a_32, input1_zp_vec);
        b_32 = _mm256_sub_epi32(b_32, input2_zp_vec);

        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.temp_buffer), a_32);
        _mm256_store_si256(reinterpret_cast<__m256i*>(tls.temp_buffer + 8), b_32);

        for (int j = 0; j < 8; j++) {
          int64_t a_val = static_cast<int64_t>(tls.temp_buffer[j]) * M1_fixed;
          int64_t b_val = static_cast<int64_t>(tls.temp_buffer[j + 8]) * M2_fixed;
          tls.temp_buffer[j] = static_cast<int32_t>((a_val + b_val + 0x800000LL) >> 24);
        }

        __m256i result = _mm256_load_si256(reinterpret_cast<const __m256i*>(tls.temp_buffer));

        if (fused_relu) {
          result = _mm256_max_epi32(result, zero_vec);
        }

        result = _mm256_add_epi32(result, output_zp_vec);
        result = _mm256_min_epi32(_mm256_max_epi32(result, min_vec), max_vec);

        __m128i result_16 = _mm_packs_epi32(_mm256_castsi256_si128(result),
                                            _mm256_extracti128_si256(result, 1));
        __m128i result_8 = _mm_packs_epi16(result_16, result_16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), result_8);
      } else {
        for (int64_t j = i; j < end; ++j) {
          int32_t a = static_cast<int32_t>(input_a[j]) - input1_zp;
          int32_t b = static_cast<int32_t>(input_b[j]) - input2_zp;

          int64_t scaled_a = static_cast<int64_t>(a) * M1_fixed;
          int64_t scaled_b = static_cast<int64_t>(b) * M2_fixed;

          int32_t sum = static_cast<int32_t>((scaled_a + scaled_b + 0x800000LL) >> 24);

          if (fused_relu) {
            sum = std::max(sum, 0);
          }

          sum = sum + output_zp;
          sum = std::min(127, std::max(-128, sum));
          output[j] = static_cast<int8_t>(sum);
        }
      }
    }
  };

  const int64_t chunk_size = 1024;
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
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t total_elements = batch_size * channels * height * width;

  auto process_range = [=](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      int32_t a = static_cast<int32_t>(input_a[i]) - static_cast<int32_t>(input1_zp);
      int32_t b = static_cast<int32_t>(input_b[i]) - static_cast<int32_t>(input2_zp);
      int64_t scaled_a = static_cast<int64_t>(a) * static_cast<int64_t>(M1_fixed);
      int64_t scaled_b = static_cast<int64_t>(b) * static_cast<int64_t>(M2_fixed);
      int32_t a_rounded = static_cast<int32_t>((scaled_a + 0x800000LL) >> 24);
      int32_t b_rounded = static_cast<int32_t>((scaled_b + 0x800000LL) >> 24);
      int32_t sum = a_rounded + b_rounded;
      if (fused_relu) {
        sum = (sum < 0) ? 0 : sum;
      }
      sum += static_cast<int32_t>(output_zp);
      if (sum > 127)
        sum = 127;
      else if (sum < -128)
        sum = -128;

      output[i] = static_cast<int8_t>(sum);
    }
  };

  const int64_t chunk_size = 1024;
  if (tp != nullptr) {
    const int64_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_chunks,
        [&](std::ptrdiff_t chunk_idx) {
          int64_t start = chunk_idx * chunk_size;
          int64_t end = std::min(total_elements, start + chunk_size);
          process_range(start, end);
        });
  } else {
    process_range(0, total_elements);
  }
}
*/
}  // namespace nudgev
}  // namespace onnxruntime