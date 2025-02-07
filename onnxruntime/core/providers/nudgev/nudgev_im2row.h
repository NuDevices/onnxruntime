#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include "core/platform/threadpool.h"
#include <algorithm>

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
  std::cout << "STRIDE VALUE: " << stride << std::endl;
  const int64_t output_h = (height - 1) / stride + 1;
  const int64_t output_w = (width - 1) / stride + 1;
  const int64_t patches_per_image = output_h * output_w;

  const int64_t hw_size = height * width;
  const int64_t batch_stride = channels * hw_size;
  const int64_t out_batch_stride = channels * patches_per_image;

  constexpr int64_t TILE_H = 16;
  constexpr int64_t TILE_W = 16;
  constexpr int64_t CHANNEL_UNROLL = 8;

  auto process_tile = [=](const int8_t* batch_input, int8_t* batch_output,
                          int64_t h_start, int64_t h_end,
                          int64_t w_start, int64_t w_end) {
    for (int64_t c = 0; c < channels; c += CHANNEL_UNROLL) {
      const int64_t c_end = std::min(c + CHANNEL_UNROLL, channels);
      const int actual_unroll = c_end - c;

      const int8_t* channel_inputs[CHANNEL_UNROLL];
      for (int i = 0; i < actual_unroll; ++i) {
        channel_inputs[i] = batch_input + (c + i) * hw_size;
      }

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t input_h = h * stride;
        const int64_t h_offset = input_h * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          const int64_t input_w = w * stride;
          const int64_t input_offset = h_offset + input_w;
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;

          if (actual_unroll == CHANNEL_UNROLL) {
            uint64_t packed = 0;
            for (int i = 0; i < CHANNEL_UNROLL; ++i) {
              packed |= static_cast<uint64_t>(channel_inputs[i][input_offset]) << (i * 8);
            }
            std::memcpy(batch_output + out_offset, &packed, sizeof(packed));
          } else {
            for (int i = 0; i < actual_unroll; ++i) {
              batch_output[out_offset + i] = channel_inputs[i][input_offset];
            }
          }
        }
      }
    }
  };

  auto process_batch = [=](int64_t b) {
    const int8_t* batch_input = input + b * batch_stride;
    int8_t* batch_output = output + b * out_batch_stride;

    for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
      const int64_t h_end = std::min(h_start + TILE_H, output_h);
      for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
        const int64_t w_end = std::min(w_start + TILE_W, output_w);
        process_tile(batch_input, batch_output, h_start, h_end, w_start, w_end);
      }
    }
  };

  if (tp != nullptr && batch_size > 4) {
    auto spatial_process = [&](int64_t work_id) {
      const int64_t b = work_id / output_h;
      const int64_t h = work_id % output_h;
      if (b >= batch_size) return;

      const int8_t* batch_input = input + b * batch_stride;
      int8_t* batch_output = output + b * out_batch_stride;

      const int64_t h_start = h;
      const int64_t h_end = h + 1;
      process_tile(batch_input, batch_output, h_start, h_end, 0, output_w);
    };

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, batch_size * output_h, spatial_process);
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
  const int64_t output_h = (height - 1) / 1 + 1;
  const int64_t output_w = (width - 1) / 1 + 1;
  const int64_t patches_per_image = output_h * output_w;

  const int64_t hw_size = height * width;
  const int64_t batch_stride = channels * hw_size;
  const int64_t out_batch_stride = channels * patches_per_image;

  constexpr int64_t TILE_H = 16;
  constexpr int64_t TILE_W = 16;
  constexpr int64_t CHANNEL_UNROLL = 8;

  auto process_tile = [=](const int8_t* batch_input, int8_t* batch_output,
                          int64_t h_start, int64_t h_end,
                          int64_t w_start, int64_t w_end) {
    for (int64_t c = 0; c < channels; c += CHANNEL_UNROLL) {
      const int64_t c_end = std::min(c + CHANNEL_UNROLL, channels);
      const int actual_unroll = c_end - c;

      const int8_t* channel_inputs[CHANNEL_UNROLL];
      for (int i = 0; i < actual_unroll; ++i) {
        channel_inputs[i] = batch_input + (c + i) * hw_size;
      }

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t input_h = h;
        const int64_t h_offset = input_h * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          const int64_t input_w = w;
          const int64_t input_offset = h_offset + input_w;
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;

          if (actual_unroll == CHANNEL_UNROLL) {
            uint64_t packed = 0;
            for (int i = 0; i < CHANNEL_UNROLL; ++i) {
              packed |= static_cast<uint64_t>(channel_inputs[i][input_offset]) << (i * 8);
            }
            std::memcpy(batch_output + out_offset, &packed, sizeof(packed));
          } else {
            for (int i = 0; i < actual_unroll; ++i) {
              batch_output[out_offset + i] = channel_inputs[i][input_offset];
            }
          }
        }
      }
    }
  };

  auto process_batch = [=](int64_t b) {
    const int8_t* batch_input = input + b * batch_stride;
    int8_t* batch_output = output + b * out_batch_stride;

    for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
      const int64_t h_end = std::min(h_start + TILE_H, output_h);
      for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
        const int64_t w_end = std::min(w_start + TILE_W, output_w);
        process_tile(batch_input, batch_output, h_start, h_end, w_start, w_end);
      }
    }
  };

  if (tp != nullptr && batch_size > 4) {
    auto spatial_process = [&](int64_t work_id) {
      const int64_t b = work_id / output_h;
      const int64_t h = work_id % output_h;
      if (b >= batch_size) return;

      const int8_t* batch_input = input + b * batch_stride;
      int8_t* batch_output = output + b * out_batch_stride;

      const int64_t h_start = h;
      const int64_t h_end = h + 1;
      process_tile(batch_input, batch_output, h_start, h_end, 0, output_w);
    };

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, batch_size * output_h, spatial_process);
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}

void im2row_1x1_stride2(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = (height - 1) / 2 + 1;
  const int64_t output_w = (width - 1) / 2 + 1;
  const int64_t patches_per_image = output_h * output_w;

  const int64_t hw_size = height * width;
  const int64_t batch_stride = channels * hw_size;
  const int64_t out_batch_stride = channels * patches_per_image;

  constexpr int64_t TILE_H = 16;
  constexpr int64_t TILE_W = 16;
  constexpr int64_t CHANNEL_UNROLL = 8;

  auto process_tile = [=](const int8_t* batch_input, int8_t* batch_output,
                          int64_t h_start, int64_t h_end,
                          int64_t w_start, int64_t w_end) {
    for (int64_t c = 0; c < channels; c += CHANNEL_UNROLL) {
      const int64_t c_end = std::min(c + CHANNEL_UNROLL, channels);
      const int actual_unroll = c_end - c;

      const int8_t* channel_inputs[CHANNEL_UNROLL];
      for (int i = 0; i < actual_unroll; ++i) {
        channel_inputs[i] = batch_input + (c + i) * hw_size;
      }

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t input_h = h * 2;
        const int64_t h_offset = input_h * width;

        for (int64_t w = w_start; w < w_end; ++w) {
          const int64_t input_w = w * 2;
          const int64_t input_offset = h_offset + input_w;
          const int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels + c;

          if (actual_unroll == CHANNEL_UNROLL) {
            uint64_t packed = 0;
            for (int i = 0; i < CHANNEL_UNROLL; ++i) {
              packed |= static_cast<uint64_t>(channel_inputs[i][input_offset]) << (i * 8);
            }
            std::memcpy(batch_output + out_offset, &packed, sizeof(packed));
          } else {
            for (int i = 0; i < actual_unroll; ++i) {
              batch_output[out_offset + i] = channel_inputs[i][input_offset];
            }
          }
        }
      }
    }
  };

  auto process_batch = [=](int64_t b) {
    const int8_t* batch_input = input + b * batch_stride;
    int8_t* batch_output = output + b * out_batch_stride;

    for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
      const int64_t h_end = std::min(h_start + TILE_H, output_h);
      for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
        const int64_t w_end = std::min(w_start + TILE_W, output_w);
        process_tile(batch_input, batch_output, h_start, h_end, w_start, w_end);
      }
    }
  };

  if (tp != nullptr && batch_size > 4) {
    auto spatial_process = [&](int64_t work_id) {
      const int64_t b = work_id / output_h;
      const int64_t h = work_id % output_h;
      if (b >= batch_size) return;

      const int8_t* batch_input = input + b * batch_stride;
      int8_t* batch_output = output + b * out_batch_stride;

      const int64_t h_start = h;
      const int64_t h_end = h + 1;
      process_tile(batch_input, batch_output, h_start, h_end, 0, output_w);
    };

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, batch_size * output_h, spatial_process);
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}

void im2row_1x1_transpose(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t hw_size = height * width;
  const int64_t batch_stride = channels * hw_size;
  const int64_t total_pixels = batch_size * hw_size;
  constexpr int64_t CHANNEL_UNROLL = 8;

  auto process_pixel = [&](int64_t pixel_idx) {
    const int64_t n = pixel_idx / hw_size;
    const int64_t hw_offset = pixel_idx % hw_size;
    const int64_t h = hw_offset / width;
    const int64_t w = hw_offset % width;
    int8_t* out_row = output + pixel_idx * channels;
    const int8_t* batch_input = input + n * batch_stride;
    for (int64_t c = 0; c < channels; c += CHANNEL_UNROLL) {
      const int64_t c_end = std::min(c + CHANNEL_UNROLL, channels);
      const int64_t actual_unroll = c_end - c;
      if (actual_unroll == CHANNEL_UNROLL) {
        uint64_t packed = 0;
        for (int i = 0; i < CHANNEL_UNROLL; i++) {
          const int8_t val = batch_input[(c + i) * hw_size + h * width + w];
          packed |= static_cast<uint64_t>(val) << (i * 8);
        }
        std::memcpy(out_row + c, &packed, sizeof(packed));
      } else {
        for (int i = 0; i < actual_unroll; i++) {
          out_row[c + i] = batch_input[(c + i) * hw_size + h * width + w];
        }
      }
    }
  };

  onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, total_pixels, process_pixel);
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
  if (stride == 1) {
    im2row_1x1_transpose(input, output, batch_size, channels, height, width, tp);
  } else if (stride == 2) {
    im2row_1x1_stride2(input, output, batch_size, channels, height, width, tp);
  } else {
    im2row_1x1_optimized(input, output, batch_size, channels, height, width, stride, tp);
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

void im2row_3x3_stride1_optimized(
    const int8_t* __restrict input,
    int8_t* __restrict output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = height - 2;
  const int64_t output_w = width - 2;
  const int64_t patches_per_image = output_h * output_w;
  constexpr int64_t patch_size = 9;
  constexpr int64_t TILE_H = 32;
  constexpr int64_t TILE_W = 32;
  constexpr int64_t channel_unroll = 8;
  constexpr int64_t w_unroll = 4;

  auto process_tile = [=](int64_t b, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);

    const int8_t* batch_input = input + b * channels * height * width;
    int8_t* batch_output = output + b * patches_per_image * channels * patch_size;

    for (int64_t h = h_start; h < h_end; ++h) {
      const int64_t row0 = h * width;
      const int64_t row1 = (h + 1) * width;
      const int64_t row2 = (h + 2) * width;
      int64_t w = w_start;
      for (; w + w_unroll <= w_end; w += w_unroll) {
        int8_t* output_ptr0 = batch_output + (h * output_w + w) * channels * patch_size;
        int8_t* output_ptr1 = output_ptr0 + channels * patch_size;
        int8_t* output_ptr2 = output_ptr1 + channels * patch_size;
        int8_t* output_ptr3 = output_ptr2 + channels * patch_size;
        int64_t c = 0;
        for (; c <= channels - channel_unroll; c += channel_unroll) {
          for (int i = 0; i < channel_unroll; ++i) {
            const int8_t* channel_input = batch_input + (c + i) * height * width;
            {
              int8_t* patch_output = output_ptr0 + (c + i) * patch_size;
              const int8_t* src0 = channel_input + row0 + w;
              const int8_t* src1 = channel_input + row1 + w;
              const int8_t* src2 = channel_input + row2 + w;

              patch_output[0] = src0[0];
              patch_output[1] = src0[1];
              patch_output[2] = src0[2];
              patch_output[3] = src1[0];
              patch_output[4] = src1[1];
              patch_output[5] = src1[2];
              patch_output[6] = src2[0];
              patch_output[7] = src2[1];
              patch_output[8] = src2[2];
            }
            {
              int8_t* patch_output = output_ptr1 + (c + i) * patch_size;
              const int8_t* src0 = channel_input + row0 + w + 1;
              const int8_t* src1 = channel_input + row1 + w + 1;
              const int8_t* src2 = channel_input + row2 + w + 1;

              patch_output[0] = src0[0];
              patch_output[1] = src0[1];
              patch_output[2] = src0[2];
              patch_output[3] = src1[0];
              patch_output[4] = src1[1];
              patch_output[5] = src1[2];
              patch_output[6] = src2[0];
              patch_output[7] = src2[1];
              patch_output[8] = src2[2];
            }
            {
              int8_t* patch_output = output_ptr2 + (c + i) * patch_size;
              const int8_t* src0 = channel_input + row0 + w + 2;
              const int8_t* src1 = channel_input + row1 + w + 2;
              const int8_t* src2 = channel_input + row2 + w + 2;

              patch_output[0] = src0[0];
              patch_output[1] = src0[1];
              patch_output[2] = src0[2];
              patch_output[3] = src1[0];
              patch_output[4] = src1[1];
              patch_output[5] = src1[2];
              patch_output[6] = src2[0];
              patch_output[7] = src2[1];
              patch_output[8] = src2[2];
            }
            {
              int8_t* patch_output = output_ptr3 + (c + i) * patch_size;
              const int8_t* src0 = channel_input + row0 + w + 3;
              const int8_t* src1 = channel_input + row1 + w + 3;
              const int8_t* src2 = channel_input + row2 + w + 3;

              patch_output[0] = src0[0];
              patch_output[1] = src0[1];
              patch_output[2] = src0[2];
              patch_output[3] = src1[0];
              patch_output[4] = src1[1];
              patch_output[5] = src1[2];
              patch_output[6] = src2[0];
              patch_output[7] = src2[1];
              patch_output[8] = src2[2];
            }
          }
        }
        for (; c < channels; ++c) {
          const int8_t* channel_input = batch_input + c * height * width;
          for (int64_t wi = 0; wi < w_unroll; ++wi) {
            int8_t* patch_output = batch_output + (h * output_w + w + wi) * channels * patch_size + c * patch_size;
            const int8_t* src0 = channel_input + row0 + w + wi;
            const int8_t* src1 = channel_input + row1 + w + wi;
            const int8_t* src2 = channel_input + row2 + w + wi;

            patch_output[0] = src0[0];
            patch_output[1] = src0[1];
            patch_output[2] = src0[2];
            patch_output[3] = src1[0];
            patch_output[4] = src1[1];
            patch_output[5] = src1[2];
            patch_output[6] = src2[0];
            patch_output[7] = src2[1];
            patch_output[8] = src2[2];
          }
        }
      }
      for (; w < w_end; ++w) {
        int8_t* output_ptr = batch_output + (h * output_w + w) * channels * patch_size;

        for (int64_t c = 0; c < channels; ++c) {
          const int8_t* channel_input = batch_input + c * height * width;
          int8_t* patch_output = output_ptr + c * patch_size;

          const int8_t* src0 = channel_input + row0 + w;
          const int8_t* src1 = channel_input + row1 + w;
          const int8_t* src2 = channel_input + row2 + w;

          patch_output[0] = src0[0];
          patch_output[1] = src0[1];
          patch_output[2] = src0[2];
          patch_output[3] = src1[0];
          patch_output[4] = src1[1];
          patch_output[5] = src1[2];
          patch_output[6] = src2[0];
          patch_output[7] = src2[1];
          patch_output[8] = src2[2];
        }
      }
    }
  };

  if (tp != nullptr) {
    const int64_t num_h_tiles = (output_h + TILE_H - 1) / TILE_H;
    const int64_t num_w_tiles = (output_w + TILE_W - 1) / TILE_W;
    const int64_t total_tiles = batch_size * num_h_tiles * num_w_tiles;

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_tiles, [=](int64_t work_index) {
          const int64_t b = work_index / (num_h_tiles * num_w_tiles);
          const int64_t tile_h = (work_index / num_w_tiles) % num_h_tiles;
          const int64_t tile_w = work_index % num_w_tiles;

          process_tile(b, tile_h * TILE_H, tile_w * TILE_W);
        });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t h = 0; h < output_h; h += TILE_H) {
        for (int64_t w = 0; w < output_w; w += TILE_W) {
          process_tile(b, h, w);
        }
      }
    }
  }
}

void im2row_3x3_stride1(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = height - 2;
  const int64_t output_w = width - 2;
  const int64_t patches_per_image = output_h * output_w;
  constexpr int64_t patch_size = 9;
  constexpr int64_t TILE_H = 16;
  constexpr int64_t TILE_W = 16;
  constexpr int64_t channel_unroll = 8;

  auto process_tile = [=](int64_t b, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);

    const int8_t* batch_input = input + b * channels * height * width;
    int8_t* batch_output = output + b * patches_per_image * channels * patch_size;

    for (int64_t c = 0; c + channel_unroll <= channels; c += channel_unroll) {
      const int8_t* channel_inputs[channel_unroll];
      for (int i = 0; i < channel_unroll; ++i) {
        channel_inputs[i] = batch_input + (c + i) * height * width;
      }

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t row0 = h * width;
        const int64_t row1 = (h + 1) * width;
        const int64_t row2 = (h + 2) * width;

        int8_t* row_outputs[channel_unroll];
        for (int i = 0; i < channel_unroll; ++i) {
          row_outputs[i] = batch_output + (h * output_w * channels + (c + i)) * patch_size;
        }

        for (int64_t w = w_start; w < w_end; ++w) {
          for (int i = 0; i < channel_unroll; ++i) {
            int8_t* patch_output = row_outputs[i] + w * channels * patch_size;
            const int8_t* channel_input = channel_inputs[i];

            memcpy(patch_output, channel_input + row0 + w, 3);
            memcpy(patch_output + 3, channel_input + row1 + w, 3);
            memcpy(patch_output + 6, channel_input + row2 + w, 3);
          }
        }
      }
    }

    for (int64_t c = channels - (channels % channel_unroll); c < channels; ++c) {
      const int8_t* channel_input = batch_input + c * height * width;

      for (int64_t h = h_start; h < h_end; ++h) {
        const int64_t row0 = h * width;
        const int64_t row1 = (h + 1) * width;
        const int64_t row2 = (h + 2) * width;

        int8_t* row_output = batch_output + (h * output_w * channels + c) * patch_size;

        for (int64_t w = w_start; w < w_end; ++w) {
          int8_t* patch_output = row_output + w * channels * patch_size;

          memcpy(patch_output, channel_input + row0 + w, 3);
          memcpy(patch_output + 3, channel_input + row1 + w, 3);
          memcpy(patch_output + 6, channel_input + row2 + w, 3);
        }
      }
    }
  };

  if (tp != nullptr) {
    const int64_t num_h_tiles = (output_h + TILE_H - 1) / TILE_H;
    const int64_t num_w_tiles = (output_w + TILE_W - 1) / TILE_W;
    const int64_t total_tiles = batch_size * num_h_tiles * num_w_tiles;

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_tiles, [=](int64_t work_index) {
          const int64_t b = work_index / (num_h_tiles * num_w_tiles);
          const int64_t tile_h = (work_index / num_w_tiles) % num_h_tiles;
          const int64_t tile_w = work_index % num_w_tiles;

          const int64_t h_start = tile_h * TILE_H;
          const int64_t w_start = tile_w * TILE_W;

          process_tile(b, h_start, w_start);
        });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t h = 0; h < output_h; h += TILE_H) {
        for (int64_t w = 0; w < output_w; w += TILE_W) {
          process_tile(b, h, w);
        }
      }
    }
  }
}

void im2row_3x3_stride2(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t output_h = (height - 3) / 2 + 1;
  const int64_t output_w = (width - 3) / 2 + 1;
  const int64_t patches_per_image = output_h * output_w;
  constexpr int64_t patch_size = 9;

  auto process_channel = [=](int64_t b, int64_t c) {
    const int8_t* batch_input = input + b * channels * height * width;
    int8_t* batch_output = output + b * patches_per_image * channels * patch_size;
    const int8_t* channel_input = batch_input + c * height * width;

    for (int64_t h_idx = 0; h_idx < output_h; ++h_idx) {
      const int64_t h_offset = h_idx * 2;
      const int64_t row0 = (h_offset + 0) * width;
      const int64_t row1 = (h_offset + 1) * width;
      const int64_t row2 = (h_offset + 2) * width;

      for (int64_t w_idx = 0; w_idx < output_w; ++w_idx) {
        const int64_t w_offset = w_idx * 2;
        const int64_t patch = h_idx * output_w + w_idx;
        int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;

        memcpy(patch_output, channel_input + row0 + w_offset, 3);
        memcpy(patch_output + 3, channel_input + row1 + w_offset, 3);
        memcpy(patch_output + 6, channel_input + row2 + w_offset, 3);
      }
    }
  };

  if (tp != nullptr) {
    const int64_t total_work = batch_size * channels;
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_work, [=](int64_t work_index) {
          const int64_t b = work_index / channels;
          const int64_t c = work_index % channels;
          process_channel(b, c);
        });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t c = 0; c < channels; ++c) {
        process_channel(b, c);
      }
    }
  }
}

void im2row_3x3_dispatch(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t stride,
    onnxruntime::concurrency::ThreadPool* tp) {
  if (stride == 1) {
    im2row_3x3_stride1_optimized(input, output, batch_size, channels, height, width, tp);
  } else if (stride == 2) {
    im2row_3x3_stride2(input, output, batch_size, channels, height, width, tp);
  } else {
    im2row_3x3_unrolled(input, output, batch_size, channels, height, width, stride, tp);
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

  constexpr int64_t TILE_H = 16;
  constexpr int64_t TILE_W = 16;

  auto process_tile = [=](int64_t b, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);

    const int8_t* batch_input = input + b * channels * height * width;
    int8_t* batch_output = output + b * patches_per_image * channels * patch_size;

    for (int64_t h_idx = h_start; h_idx < h_end; ++h_idx) {
      for (int64_t w_idx = w_start; w_idx < w_end; ++w_idx) {
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
  };

  if (tp != nullptr) {
    const int64_t num_h_tiles = (output_h + TILE_H - 1) / TILE_H;
    const int64_t num_w_tiles = (output_w + TILE_W - 1) / TILE_W;
    const int64_t total_tiles = batch_size * num_h_tiles * num_w_tiles;

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_tiles, [=](int64_t work_index) {
          const int64_t b = work_index / (num_h_tiles * num_w_tiles);
          const int64_t tile_h = (work_index / num_w_tiles) % num_h_tiles;
          const int64_t tile_w = work_index % num_w_tiles;

          const int64_t h_start = tile_h * TILE_H;
          const int64_t w_start = tile_w * TILE_W;

          process_tile(b, h_start, w_start);
        });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t h = 0; h < output_h; h += TILE_H) {
        for (int64_t w = 0; w < output_w; w += TILE_W) {
          process_tile(b, h, w);
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

  auto start_padding = std::chrono::high_resolution_clock::now();

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
  auto duration_padding = std::chrono::duration_cast<std::chrono::microseconds>(end_padding - start_padding);
  std::cout << "padding time: " << duration_padding.count() << " microseconds" << std::endl;

  auto im2row_start = std::chrono::high_resolution_clock::now();

  if (kernel_h == 1 && kernel_w == 1) {
    im2row_1x1_dispatch(im2row_input, output,
                        batch_size, channels, padded_height, padded_width,
                        stride_h, tp);
  } else if (kernel_h == 3 && kernel_w == 3) {
    im2row_3x3_dispatch(im2row_input, output,
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

static inline void transpose_8x8_int8_sse(const int8_t* src, int8_t* dst,
                                          int64_t src_stride, int64_t dst_stride) {
  __m128i row0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 0 * src_stride));
  __m128i row1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 1 * src_stride));
  __m128i row2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 2 * src_stride));
  __m128i row3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 3 * src_stride));
  __m128i row4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 4 * src_stride));
  __m128i row5 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 5 * src_stride));
  __m128i row6 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 6 * src_stride));
  __m128i row7 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 7 * src_stride));

  __m128i int01 = _mm_unpacklo_epi8(row0, row1);
  __m128i int23 = _mm_unpacklo_epi8(row2, row3);
  __m128i int45 = _mm_unpacklo_epi8(row4, row5);
  __m128i int67 = _mm_unpacklo_epi8(row6, row7);

  __m128i int0123_lo = _mm_unpacklo_epi16(int01, int23);
  __m128i int0123_hi = _mm_unpackhi_epi16(int01, int23);
  __m128i int4567_lo = _mm_unpacklo_epi16(int45, int67);
  __m128i int4567_hi = _mm_unpackhi_epi16(int45, int67);

  __m128i int01234567_0 = _mm_unpacklo_epi32(int0123_lo, int4567_lo);
  __m128i int01234567_1 = _mm_unpackhi_epi32(int0123_lo, int4567_lo);
  __m128i int01234567_2 = _mm_unpacklo_epi32(int0123_hi, int4567_hi);
  __m128i int01234567_3 = _mm_unpackhi_epi32(int0123_hi, int4567_hi);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 0 * dst_stride),
                   _mm_unpacklo_epi64(int01234567_0, int01234567_0));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 1 * dst_stride),
                   _mm_unpackhi_epi64(int01234567_0, int01234567_0));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 2 * dst_stride),
                   _mm_unpacklo_epi64(int01234567_1, int01234567_1));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 3 * dst_stride),
                   _mm_unpackhi_epi64(int01234567_1, int01234567_1));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 4 * dst_stride),
                   _mm_unpacklo_epi64(int01234567_2, int01234567_2));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 5 * dst_stride),
                   _mm_unpackhi_epi64(int01234567_2, int01234567_2));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 6 * dst_stride),
                   _mm_unpacklo_epi64(int01234567_3, int01234567_3));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 7 * dst_stride),
                   _mm_unpackhi_epi64(int01234567_3, int01234567_3));
}

static inline void transpose_8x8_int8_avx512(
    const int8_t* src, int8_t* dst,
    int64_t src_stride, int64_t dst_stride) {
  __m512i row0 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 0 * src_stride)));
  __m512i row1 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 1 * src_stride)));
  __m512i row2 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 2 * src_stride)));
  __m512i row3 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 3 * src_stride)));
  __m512i row4 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 4 * src_stride)));
  __m512i row5 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 5 * src_stride)));
  __m512i row6 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 6 * src_stride)));
  __m512i row7 = _mm512_zextsi128_si512(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 7 * src_stride)));

  __m512i int01 = _mm512_unpacklo_epi8(row0, row1);
  __m512i int23 = _mm512_unpacklo_epi8(row2, row3);
  __m512i int45 = _mm512_unpacklo_epi8(row4, row5);
  __m512i int67 = _mm512_unpacklo_epi8(row6, row7);

  __m512i int0123_lo = _mm512_unpacklo_epi16(int01, int23);
  __m512i int0123_hi = _mm512_unpackhi_epi16(int01, int23);
  __m512i int4567_lo = _mm512_unpacklo_epi16(int45, int67);
  __m512i int4567_hi = _mm512_unpackhi_epi16(int45, int67);

  __m512i int01234567_0 = _mm512_unpacklo_epi32(int0123_lo, int4567_lo);
  __m512i int01234567_1 = _mm512_unpackhi_epi32(int0123_lo, int4567_lo);
  __m512i int01234567_2 = _mm512_unpacklo_epi32(int0123_hi, int4567_hi);
  __m512i int01234567_3 = _mm512_unpackhi_epi32(int0123_hi, int4567_hi);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 0 * dst_stride),
                   _mm512_castsi512_si128(int01234567_0));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 1 * dst_stride),
                   _mm512_extracti32x4_epi32(int01234567_0, 1));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 2 * dst_stride),
                   _mm512_castsi512_si128(int01234567_1));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 3 * dst_stride),
                   _mm512_extracti32x4_epi32(int01234567_1, 1));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 4 * dst_stride),
                   _mm512_castsi512_si128(int01234567_2));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 5 * dst_stride),
                   _mm512_extracti32x4_epi32(int01234567_2, 1));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 6 * dst_stride),
                   _mm512_castsi512_si128(int01234567_3));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + 7 * dst_stride),
                   _mm512_extracti32x4_epi32(int01234567_3, 1));
}
static inline void transpose_8x8_int8_fallback(const int8_t* src, int8_t* dst,
                                               int64_t src_stride, int64_t dst_stride) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      dst[j * dst_stride + i] = src[i * src_stride + j];
    }
  }
}
void transpose_output_matrix(const int8_t* input,
                             int8_t* output,
                             int64_t batch_size,
                             int64_t hw,
                             int64_t channels,
                             onnxruntime::concurrency::ThreadPool* tp) {
  auto process_batch = [&](int64_t b) {
    const int8_t* batch_input = input + b * hw * channels;
    int8_t* batch_output = output + b * hw * channels;

    for (int64_t i = 0; i < hw; i += 8) {
      for (int64_t j = 0; j < channels; j += 8) {
        if (i + 8 <= hw && j + 8 <= channels) {
#ifdef __AVX512F__
          if (__builtin_cpu_supports("avx512f")) {
            transpose_8x8_int8_avx512(
                &batch_input[i * channels + j],
                &batch_output[j * hw + i],
                channels,
                hw);
            continue;
          }
#endif

#ifdef __SSE2__
          if (__builtin_cpu_supports("sse2")) {
            transpose_8x8_int8_sse(
                &batch_input[i * channels + j],
                &batch_output[j * hw + i],
                channels,
                hw);
            continue;
          }
#endif
          for (int64_t ii = i; ii < i + 8; ++ii) {
            for (int64_t jj = j; jj < j + 8; ++jj) {
              batch_output[jj * hw + ii] = batch_input[ii * channels + jj];
            }
          }
        } else {
          const int64_t i_end = std::min(i + 8, hw);
          const int64_t j_end = std::min(j + 8, channels);
          for (int64_t ii = i; ii < i_end; ++ii) {
            for (int64_t jj = j; jj < j_end; ++jj) {
              batch_output[jj * hw + ii] = batch_input[ii * channels + jj];
            }
          }
        }
      }
    }
  };

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp,
        batch_size,
        process_batch);
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}