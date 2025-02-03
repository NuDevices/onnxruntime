#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include "core/platform/threadpool.h"
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>

void convert_and_center_saturated(
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
void im2row_1x1_optimized(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t stride,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = (height - 1) / stride + 1;
  const int64_t output_w = (width - 1) / stride + 1;
  const int64_t patches_per_image = output_h * output_w;

  const int64_t hw_size = height * width;
  const int64_t batch_stride = channels * hw_size;
  const int64_t out_batch_stride = channels * patches_per_image;

  // Adjusted tile sizes for better cache utilization
  constexpr int64_t TILE_H = 32;
  constexpr int64_t TILE_W = 32;
  constexpr int64_t TILE_C = 64;  // Increased for better channel locality

  auto process_tile = [=](const int8_t* batch_input, int8_t* batch_output,
                          int64_t c_start, int64_t c_end,
                          int64_t h_start, int64_t h_end,
                          int64_t w_start, int64_t w_end) {
    constexpr int64_t channel_unroll = 8;

    // Process channels in blocks of 8 with packed writes
    for (int64_t c = c_start; c + channel_unroll <= c_end; c += channel_unroll) {
      const int8_t* channel_inputs[channel_unroll];
      for (int i = 0; i < channel_unroll; ++i) {
        channel_inputs[i] = batch_input + (c + i) * hw_size;
      }

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t input_h = h * stride;
        const int64_t h_offset = input_h * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          const int64_t input_w = w * stride;
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;
          const int64_t input_offset = h_offset + input_w;

          uint64_t packed = 0;
          packed |= static_cast<uint64_t>(channel_inputs[0][input_offset]) << 0;
          packed |= static_cast<uint64_t>(channel_inputs[1][input_offset]) << 8;
          packed |= static_cast<uint64_t>(channel_inputs[2][input_offset]) << 16;
          packed |= static_cast<uint64_t>(channel_inputs[3][input_offset]) << 24;
          packed |= static_cast<uint64_t>(channel_inputs[4][input_offset]) << 32;
          packed |= static_cast<uint64_t>(channel_inputs[5][input_offset]) << 40;
          packed |= static_cast<uint64_t>(channel_inputs[6][input_offset]) << 48;
          packed |= static_cast<uint64_t>(channel_inputs[7][input_offset]) << 56;

          std::memcpy(batch_output + out_offset, &packed, sizeof(packed));
        }
      }
    }

    // Handle remaining channels
    for (int64_t c = c_end - (c_end % channel_unroll); c < c_end; ++c) {
      const int8_t* channel_input = batch_input + c * hw_size;

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t input_h = h * stride;
        const int64_t h_offset = input_h * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          const int64_t input_w = w * stride;
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;
          const int64_t input_offset = h_offset + input_w;

          batch_output[out_offset] = channel_input[input_offset];
        }
      }
    }
  };

  auto process_batch = [=](int64_t b) {
    const int8_t* batch_input = input + b * batch_stride;
    int8_t* batch_output = output + b * out_batch_stride;

    // Optimized loop order: channels first for better input reuse
    for (int64_t c_start = 0; c_start < channels; c_start += TILE_C) {
      int64_t c_end = std::min(c_start + TILE_C, channels);
      for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
        int64_t h_end = std::min(h_start + TILE_H, output_h);
        for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
          int64_t w_end = std::min(w_start + TILE_W, output_w);
          process_tile(batch_input, batch_output, c_start, c_end,
                       h_start, h_end, w_start, w_end);
        }
      }
    }
  };

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, batch_size, process_batch);
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}
void im2row_1x1_stride1(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  // For stride=1, output dimensions equal input dimensions
  const int64_t output_h = height;
  const int64_t output_w = width;
  const int64_t patches_per_image = output_h * output_w;

  const int64_t hw_size = height * width;
  const int64_t batch_stride = channels * hw_size;
  const int64_t out_batch_stride = channels * patches_per_image;

  constexpr int64_t TILE_H = 32;
  constexpr int64_t TILE_W = 32;
  constexpr int64_t TILE_C = 64;

  auto process_tile = [=](const int8_t* batch_input, int8_t* batch_output,
                          int64_t c_start, int64_t c_end,
                          int64_t h_start, int64_t h_end,
                          int64_t w_start, int64_t w_end) {
    constexpr int64_t channel_unroll = 8;

    for (int64_t c = c_start; c + channel_unroll <= c_end; c += channel_unroll) {
      const int8_t* channel_inputs[channel_unroll];
      for (int i = 0; i < channel_unroll; ++i) {
        channel_inputs[i] = batch_input + (c + i) * hw_size;
      }

      for (int64_t h = h_start; h < h_end; ++h) {
        // For stride=1, input_h = h
        const int64_t h_offset = h * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          // For stride=1, input_w = w
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;
          const int64_t input_offset = h_offset + w;  // Simplified for stride=1

          uint64_t packed = 0;
          packed |= static_cast<uint64_t>(channel_inputs[0][input_offset]) << 0;
          packed |= static_cast<uint64_t>(channel_inputs[1][input_offset]) << 8;
          packed |= static_cast<uint64_t>(channel_inputs[2][input_offset]) << 16;
          packed |= static_cast<uint64_t>(channel_inputs[3][input_offset]) << 24;
          packed |= static_cast<uint64_t>(channel_inputs[4][input_offset]) << 32;
          packed |= static_cast<uint64_t>(channel_inputs[5][input_offset]) << 40;
          packed |= static_cast<uint64_t>(channel_inputs[6][input_offset]) << 48;
          packed |= static_cast<uint64_t>(channel_inputs[7][input_offset]) << 56;

          std::memcpy(batch_output + out_offset, &packed, sizeof(packed));
        }
      }
    }

    for (int64_t c = c_end - (c_end % channel_unroll); c < c_end; ++c) {
      const int8_t* channel_input = batch_input + c * hw_size;

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t h_offset = h * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;
          const int64_t input_offset = h_offset + w;

          batch_output[out_offset] = channel_input[input_offset];
        }
      }
    }
  };

  auto process_batch = [=](int64_t b) {
    const int8_t* batch_input = input + b * batch_stride;
    int8_t* batch_output = output + b * out_batch_stride;

    for (int64_t c_start = 0; c_start < channels; c_start += TILE_C) {
      int64_t c_end = std::min(c_start + TILE_C, channels);
      for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
        int64_t h_end = std::min(h_start + TILE_H, output_h);
        for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
          int64_t w_end = std::min(w_start + TILE_W, output_w);
          process_tile(batch_input, batch_output, c_start, c_end,
                       h_start, h_end, w_start, w_end);
        }
      }
    }
  };

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, batch_size, process_batch);
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}

// Specialized version for stride=2
void im2row_1x1_stride2(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  // For stride=2, output dimensions are halved (rounded up)
  const int64_t output_h = (height + 1) / 2;
  const int64_t output_w = (width + 1) / 2;
  const int64_t patches_per_image = output_h * output_w;

  const int64_t hw_size = height * width;
  const int64_t batch_stride = channels * hw_size;
  const int64_t out_batch_stride = channels * patches_per_image;

  constexpr int64_t TILE_H = 32;
  constexpr int64_t TILE_W = 32;
  constexpr int64_t TILE_C = 64;

  auto process_tile = [=](const int8_t* batch_input, int8_t* batch_output,
                          int64_t c_start, int64_t c_end,
                          int64_t h_start, int64_t h_end,
                          int64_t w_start, int64_t w_end) {
    constexpr int64_t channel_unroll = 8;

    for (int64_t c = c_start; c + channel_unroll <= c_end; c += channel_unroll) {
      const int8_t* channel_inputs[channel_unroll];
      for (int i = 0; i < channel_unroll; ++i) {
        channel_inputs[i] = batch_input + (c + i) * hw_size;
      }

      for (int64_t h = h_start; h < h_end; ++h) {
        // For stride=2, input_h = h * 2
        const int64_t h_offset = (h << 1) * width;  // Optimized multiplication by 2

        for (int64_t w = w_start; w < w_end; ++w) {
          // For stride=2, input_w = w * 2
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;
          const int64_t input_offset = h_offset + (w << 1);  // Optimized multiplication by 2

          uint64_t packed = 0;
          packed |= static_cast<uint64_t>(channel_inputs[0][input_offset]) << 0;
          packed |= static_cast<uint64_t>(channel_inputs[1][input_offset]) << 8;
          packed |= static_cast<uint64_t>(channel_inputs[2][input_offset]) << 16;
          packed |= static_cast<uint64_t>(channel_inputs[3][input_offset]) << 24;
          packed |= static_cast<uint64_t>(channel_inputs[4][input_offset]) << 32;
          packed |= static_cast<uint64_t>(channel_inputs[5][input_offset]) << 40;
          packed |= static_cast<uint64_t>(channel_inputs[6][input_offset]) << 48;
          packed |= static_cast<uint64_t>(channel_inputs[7][input_offset]) << 56;

          std::memcpy(batch_output + out_offset, &packed, sizeof(packed));
        }
      }
    }

    for (int64_t c = c_end - (c_end % channel_unroll); c < c_end; ++c) {
      const int8_t* channel_input = batch_input + c * hw_size;

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t h_offset = (h << 1) * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;
          const int64_t input_offset = h_offset + (w << 1);

          batch_output[out_offset] = channel_input[input_offset];
        }
      }
    }
  };

  auto process_batch = [=](int64_t b) {
    const int8_t* batch_input = input + b * batch_stride;
    int8_t* batch_output = output + b * out_batch_stride;

    for (int64_t c_start = 0; c_start < channels; c_start += TILE_C) {
      int64_t c_end = std::min(c_start + TILE_C, channels);
      for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
        int64_t h_end = std::min(h_start + TILE_H, output_h);
        for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
          int64_t w_end = std::min(w_start + TILE_W, output_w);
          process_tile(batch_input, batch_output, c_start, c_end,
                       h_start, h_end, w_start, w_end);
        }
      }
    }
  };

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, batch_size, process_batch);
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}

void im2row_1x1_dispatch(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t stride,
    onnxruntime::concurrency::ThreadPool* tp) {
  switch (stride) {
    case 1:
      im2row_1x1_stride1(input, output, batch_size, channels, height, width, tp);
      break;
    case 2:
      im2row_1x1_stride2(input, output, batch_size, channels, height, width, tp);
      break;
    default:
      im2row_1x1_optimized(input, output, batch_size, channels, height, width, stride, tp);
      break;
  }
}

void im2row_3x3_unrolled(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t stride,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = (height - 3) / stride + 1;
  const int64_t output_w = (width - 3) / stride + 1;
  const int64_t patches_per_image = output_h * output_w;
  constexpr int64_t patch_size = 9;
  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](long b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * patches_per_image * channels * patch_size;

      for (int64_t h_idx = 0; h_idx < output_h; ++h_idx) {
        for (int64_t w_idx = 0; w_idx < output_w; ++w_idx) {
          const int64_t h_offset = h_idx * stride;
          const int64_t w_offset = w_idx * stride;
          int64_t patch = h_idx * output_w + w_idx;
          const int64_t row0 = (h_offset + 0) * width + w_offset;
          const int64_t row1 = (h_offset + 1) * width + w_offset;
          const int64_t row2 = (h_offset + 2) * width + w_offset;

          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;

            patch_output[0] = channel_input[row0 + 0];
            patch_output[1] = channel_input[row0 + 1];
            patch_output[2] = channel_input[row0 + 2];

            patch_output[3] = channel_input[row1 + 0];
            patch_output[4] = channel_input[row1 + 1];
            patch_output[5] = channel_input[row1 + 2];

            patch_output[6] = channel_input[row2 + 0];
            patch_output[7] = channel_input[row2 + 1];
            patch_output[8] = channel_input[row2 + 2];
          }
        }
      }
    });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * patches_per_image * channels * patch_size;
      for (int64_t h_idx = 0; h_idx < output_h; ++h_idx) {
        for (int64_t w_idx = 0; w_idx < output_w; ++w_idx) {
          int64_t patch = h_idx * output_w + w_idx;
          const int64_t h_offset = h_idx * stride;
          const int64_t w_offset = w_idx * stride;
          const int64_t row0 = (h_offset + 0) * width + w_offset;
          const int64_t row1 = (h_offset + 1) * width + w_offset;
          const int64_t row2 = (h_offset + 2) * width + w_offset;
          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;

            patch_output[0] = channel_input[row0 + 0];
            patch_output[1] = channel_input[row0 + 1];
            patch_output[2] = channel_input[row0 + 2];

            patch_output[3] = channel_input[row1 + 0];
            patch_output[4] = channel_input[row1 + 1];
            patch_output[5] = channel_input[row1 + 2];

            patch_output[6] = channel_input[row2 + 0];
            patch_output[7] = channel_input[row2 + 1];
            patch_output[8] = channel_input[row2 + 2];
          }
        }
      }
    }
  }
}

void im2row_5x5_unrolled(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t stride,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = (height - 5) / stride + 1;
  const int64_t output_w = (width - 5) / stride + 1;
  const int64_t patches_per_image = output_h * output_w;
  constexpr int64_t patch_size = 25;

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](long b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * patches_per_image * channels * patch_size;

      for (int64_t h_idx = 0; h_idx < output_h; ++h_idx) {
        for (int64_t w_idx = 0; w_idx < output_w; ++w_idx) {
          int64_t patch = h_idx * output_w + w_idx;
          const int64_t h_offset = h_idx * stride;
          const int64_t w_offset = w_idx * stride;

          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;

            for (int64_t kh = 0; kh < 5; ++kh) {
              const int8_t* input_row = channel_input + (h_offset + kh) * width + w_offset;
              int8_t* output_row = patch_output + kh * 5;
              output_row[0] = input_row[0];
              output_row[1] = input_row[1];
              output_row[2] = input_row[2];
              output_row[3] = input_row[3];
              output_row[4] = input_row[4];
            }
          }
        }
      }
    });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * patches_per_image * channels * patch_size;
      for (int64_t h_idx = 0; h_idx < output_h; ++h_idx) {
        for (int64_t w_idx = 0; w_idx < output_w; ++w_idx) {
          int64_t patch = h_idx * output_w + w_idx;
          const int64_t h_offset = h_idx * stride;
          const int64_t w_offset = w_idx * stride;
          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;
            for (int64_t kh = 0; kh < 5; ++kh) {
              const int8_t* input_row = channel_input + (h_offset + kh) * width + w_offset;
              int8_t* output_row = patch_output + kh * 5;
              output_row[0] = input_row[0];
              output_row[1] = input_row[1];
              output_row[2] = input_row[2];
              output_row[3] = input_row[3];
              output_row[4] = input_row[4];
            }
          }
        }
      }
    }
  }
}

void im2row_7x7_unrolled(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t stride,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = (height - 7) / stride + 1;
  const int64_t output_w = (width - 7) / stride + 1;
  const int64_t patches_per_image = output_h * output_w;
  constexpr int64_t patch_size = 49;

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](long b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * patches_per_image * channels * patch_size;

      for (int64_t h_idx = 0; h_idx < output_h; ++h_idx) {
        for (int64_t w_idx = 0; w_idx < output_w; ++w_idx) {
          int64_t patch = h_idx * output_w + w_idx;
          const int64_t h_offset = h_idx * stride;
          const int64_t w_offset = w_idx * stride;

          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;

            for (int64_t kh = 0; kh < 7; ++kh) {
              const int8_t* input_row = channel_input + (h_offset + kh) * width + w_offset;
              int8_t* output_row = patch_output + kh * 7;
              output_row[0] = input_row[0];
              output_row[1] = input_row[1];
              output_row[2] = input_row[2];
              output_row[3] = input_row[3];
              output_row[4] = input_row[4];
              output_row[5] = input_row[5];
              output_row[6] = input_row[6];
            }
          }
        }
      }
    });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * patches_per_image * channels * patch_size;
      for (int64_t h_idx = 0; h_idx < output_h; ++h_idx) {
        for (int64_t w_idx = 0; w_idx < output_w; ++w_idx) {
          int64_t patch = h_idx * output_w + w_idx;
          const int64_t h_offset = h_idx * stride;
          const int64_t w_offset = w_idx * stride;
          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;
            for (int64_t kh = 0; kh < 7; ++kh) {
              const int8_t* input_row = channel_input + (h_offset + kh) * width + w_offset;
              int8_t* output_row = patch_output + kh * 7;
              output_row[0] = input_row[0];
              output_row[1] = input_row[1];
              output_row[2] = input_row[2];
              output_row[3] = input_row[3];
              output_row[4] = input_row[4];
              output_row[5] = input_row[5];
              output_row[6] = input_row[6];
            }
          }
        }
      }
    }
  }
}

void im2row(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    const std::vector<int64_t>& pads,
    int8_t x_zero_point,
    int8_t* input_centered_buffer,
    int8_t* padded_buffer,
    onnxruntime::concurrency::ThreadPool* tp) {
  auto start = std::chrono::high_resolution_clock::now();

  convert_and_center_saturated(input, input_centered_buffer, batch_size, channels * height * width, x_zero_point, tp);

  auto end_conversion = std::chrono::high_resolution_clock::now();
  auto duration_conversion = std::chrono::duration_cast<std::chrono::microseconds>(end_conversion - start);
  std::cout << "conversion time: " << duration_conversion.count() << " microseconds" << std::endl;

  int64_t padded_height = height + pads[0] + pads[2];
  int64_t padded_width = width + pads[1] + pads[3];
  const int8_t* im2row_input = nullptr;

  if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
    auto pad_func = [=](int64_t n) {
      for (int64_t c = 0; c < channels; ++c) {
        const int64_t input_offset = (n * channels + c) * height * width;
        const int64_t output_offset = (n * channels + c) * padded_height * padded_width;

        const int8_t* input_channel = input_centered_buffer + input_offset;
        int8_t* output_channel = padded_buffer + output_offset;

        for (int64_t h = 0; h < height; ++h) {
          const int64_t h_pad = h + pads[0];
          int8_t* output_row = output_channel + h_pad * padded_width + pads[1];
          const int8_t* input_row = input_channel + h * width;
          std::memcpy(output_row, input_row, width * sizeof(int8_t));
        }
      }
    };

    if (tp != nullptr && batch_size > 1) {
      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, batch_size, pad_func);
    } else {
      for (int64_t n = 0; n < batch_size; ++n) {
        pad_func(n);
      }
    }
    im2row_input = padded_buffer;
  } else {
    im2row_input = input_centered_buffer;
  }

  auto end_padding = std::chrono::high_resolution_clock::now();
  auto duration_padding = std::chrono::duration_cast<std::chrono::microseconds>(end_padding - start);
  std::cout << "padding time: " << duration_padding.count() << " microseconds" << std::endl;

  auto im2row_start = std::chrono::high_resolution_clock::now();

  if (kernel_h == 1 && kernel_w == 1) {
    im2row_1x1_dispatch(im2row_input, output,
                        batch_size, channels, padded_height, padded_width,
                        stride_h, tp);
  } else if (kernel_h == 3 && kernel_w == 3) {
    im2row_3x3_unrolled(im2row_input, output,
                        batch_size, channels, padded_height, padded_width,
                        stride_h, tp);
  } else if (kernel_h == 5 && kernel_w == 5) {
    im2row_5x5_unrolled(im2row_input, output,
                        batch_size, channels, padded_height, padded_width,
                        stride_h, tp);
  } else if (kernel_h == 7 && kernel_w == 7) {
    im2row_7x7_unrolled(im2row_input, output,
                        batch_size, channels, padded_height, padded_width,
                        stride_h, tp);
  } else {
    throw std::runtime_error("Unsupported kernel size");
  }

  auto im2row_end = std::chrono::high_resolution_clock::now();
  auto im2row_duration = std::chrono::duration_cast<std::chrono::microseconds>(im2row_end - im2row_start);
  std::cout << "Im2row time: " << im2row_duration.count() << " microseconds" << std::endl;
}