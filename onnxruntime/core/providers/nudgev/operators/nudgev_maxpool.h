#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include "core/platform/threadpool.h"
#include <immintrin.h>

namespace onnxruntime {
namespace nudgev {

inline int8_t horizontal_max_epi8_256(__m256i v) {
  __m128i vlow = _mm256_castsi256_si128(v);
  __m128i vhigh = _mm256_extracti128_si256(v, 1);
  __m128i vmax = _mm_max_epi8(vlow, vhigh);

  vmax = _mm_max_epi8(vmax, _mm_srli_si128(vmax, 8));
  vmax = _mm_max_epi8(vmax, _mm_srli_si128(vmax, 4));
  vmax = _mm_max_epi8(vmax, _mm_srli_si128(vmax, 2));
  vmax = _mm_max_epi8(vmax, _mm_srli_si128(vmax, 1));
  return static_cast<int8_t>(_mm_cvtsi128_si32(vmax) & 0xFF);
}

inline int8_t horizontal_max_epi8_128(__m128i v) {
  v = _mm_max_epi8(v, _mm_srli_si128(v, 8));
  v = _mm_max_epi8(v, _mm_srli_si128(v, 4));
  v = _mm_max_epi8(v, _mm_srli_si128(v, 2));
  v = _mm_max_epi8(v, _mm_srli_si128(v, 1));
  return static_cast<int8_t>(_mm_cvtsi128_si32(v) & 0xFF);
}

void nudgev_maxpool_requantized(
    const int8_t* input_centered,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    const std::vector<int64_t>& pads,
    bool /*ceil_mode*/,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_height = static_cast<int64_t>(std::floor(
      (input_height + pads[0] + pads[2] - kernel_h) / static_cast<double>(stride_h) + 1));
  const int64_t output_width = static_cast<int64_t>(std::floor(
      (input_width + pads[1] + pads[3] - kernel_w) / static_cast<double>(stride_w) + 1));

  auto process = [&](int64_t b, int64_t c) {
    const int64_t batch_offset = b * channels * input_height * input_width;
    const int64_t output_batch_offset = b * channels * output_height * output_width;
    const int64_t input_c_offset = batch_offset + c * input_height * input_width;
    const int64_t output_c_offset = output_batch_offset + c * output_height * output_width;

    for (int64_t oh = 0; oh < output_height; ++oh) {
      const int64_t ih_start = oh * stride_h - pads[0];
      const int64_t ih_end = std::min(ih_start + kernel_h, input_height);
      const int64_t ih_start_valid = std::max(ih_start, static_cast<int64_t>(0));

      for (int64_t ow = 0; ow < output_width; ++ow) {
        const int64_t iw_start = ow * stride_w - pads[1];
        const int64_t iw_end = std::min(iw_start + kernel_w, input_width);
        const int64_t iw_start_valid = std::max(iw_start, static_cast<int64_t>(0));

        int8_t max_val = std::numeric_limits<int8_t>::min();
        for (int64_t ih = ih_start_valid; ih < ih_end; ++ih) {
          const int8_t* row_ptr = input_centered + input_c_offset + ih * input_width + iw_start_valid;
          const int64_t window_length = iw_end - iw_start_valid;
          int8_t local_max = std::numeric_limits<int8_t>::min();

          if (window_length >= 32) {
            __m256i v_max = _mm256_set1_epi8(std::numeric_limits<int8_t>::min());
            int64_t i = 0;
            for (; i <= window_length - 32; i += 32) {
              __m256i v_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row_ptr + i));
              v_max = _mm256_max_epi8(v_max, v_data);
            }
            local_max = std::max(local_max, horizontal_max_epi8_256(v_max));
            for (; i < window_length; ++i)
              local_max = std::max(local_max, row_ptr[i]);

          } else if (window_length >= 16) {
            __m128i v_max = _mm_set1_epi8(std::numeric_limits<int8_t>::min());
            int64_t i = 0;
            for (; i <= window_length - 16; i += 16) {
              __m128i v_data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_ptr + i));
              v_max = _mm_max_epi8(v_max, v_data);
            }
            local_max = std::max(local_max, horizontal_max_epi8_128(v_max));
            for (; i < window_length; ++i)
              local_max = std::max(local_max, row_ptr[i]);

          } else {
            for (int64_t i = 0; i < window_length; ++i)
              local_max = std::max(local_max, row_ptr[i]);
          }

          max_val = std::max(max_val, local_max);
        }

        output[output_c_offset + oh * output_width + ow] = max_val;
      }
    }
  };

  const int64_t total_work = batch_size * channels;
  if (tp != nullptr && total_work > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_work, [&](int64_t idx) {
          const int64_t b = idx / channels;
          const int64_t c = idx % channels;
          process(b, c);
        });
  } else {
    for (int64_t b = 0; b < batch_size; ++b)
      for (int64_t c = 0; c < channels; ++c)
        process(b, c);
  }
}

void nudgev_maxpool(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    const std::vector<int64_t>& pads,
    bool /*ceil_mode*/,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_height = static_cast<int64_t>(std::floor(
      (input_height + pads[0] + pads[2] - kernel_h) / static_cast<double>(stride_h) + 1));
  const int64_t output_width = static_cast<int64_t>(std::floor(
      (input_width + pads[1] + pads[3] - kernel_w) / static_cast<double>(stride_w) + 1));

  auto process = [&](int64_t b, int64_t c) {
    const int64_t batch_offset = b * channels * input_height * input_width;
    const int64_t output_batch_offset = b * channels * output_height * output_width;
    const int64_t input_c_offset = batch_offset + c * input_height * input_width;
    const int64_t output_c_offset = output_batch_offset + c * output_height * output_width;

    for (int64_t oh = 0; oh < output_height; ++oh) {
      const int64_t ih_start = oh * stride_h - pads[0];
      const int64_t ih_end = std::min(ih_start + kernel_h, input_height);
      const int64_t ih_start_valid = std::max(ih_start, static_cast<int64_t>(0));

      for (int64_t ow = 0; ow < output_width; ++ow) {
        const int64_t iw_start = ow * stride_w - pads[1];
        const int64_t iw_end = std::min(iw_start + kernel_w, input_width);
        const int64_t iw_start_valid = std::max(iw_start, static_cast<int64_t>(0));

        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t ih = ih_start_valid; ih < ih_end; ++ih) {
          const float* row_ptr = input + input_c_offset + ih * input_width + iw_start_valid;
          const int64_t window_length = iw_end - iw_start_valid;
          float local_max = -std::numeric_limits<float>::infinity();

          if (window_length >= 8) {
            __m256 v_max = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
            int64_t i = 0;
            for (; i <= window_length - 8; i += 8) {
              __m256 v_data = _mm256_loadu_ps(row_ptr + i);
              v_max = _mm256_max_ps(v_max, v_data);
            }
            __m128 high = _mm256_extractf128_ps(v_max, 1);
            __m128 low = _mm256_castps256_ps128(v_max);
            __m128 max4 = _mm_max_ps(high, low);
            max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
            max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
            local_max = std::max(local_max, _mm_cvtss_f32(max4));

            for (; i < window_length; ++i)
              local_max = std::max(local_max, row_ptr[i]);
          } else {
            for (int64_t i = 0; i < window_length; ++i)
              local_max = std::max(local_max, row_ptr[i]);
          }

          max_val = std::max(max_val, local_max);
        }

        output[output_c_offset + oh * output_width + ow] = max_val;
      }
    }
  };

  const int64_t total_work = batch_size * channels;
  if (tp != nullptr && total_work > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_work, [&](int64_t idx) {
          const int64_t b = idx / channels;
          const int64_t c = idx % channels;
          process(b, c);
        });
  } else {
    for (int64_t b = 0; b < batch_size; ++b)
      for (int64_t c = 0; c < channels; ++c)
        process(b, c);
  }
}

}  // namespace nudgev
}  // namespace onnxruntime
