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

void im2row_1x1_unrolled(
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

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](long b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * channels * patches_per_image;

      for (int64_t h = 0; h < output_h; ++h) {
        for (int64_t w = 0; w < output_w; ++w) {
          const int64_t input_h = h * stride;
          const int64_t input_w = w * stride;
          int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels;

          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            batch_output[out_offset + c] = channel_input[input_h * width + input_w];
          }
        }
      }
    });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      const int8_t* batch_input = input + b * channels * height * width;
      int8_t* batch_output = output + b * channels * patches_per_image;
      for (int64_t h = 0; h < output_h; ++h) {
        for (int64_t w = 0; w < output_w; ++w) {
          int64_t patch = h * output_w + w;
          const int64_t out_offset = patch * channels;
          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            batch_output[out_offset + c] = channel_input[h * stride * width + w * stride];
          }
        }
      }
    }
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
          int64_t patch = h_idx * output_w + w_idx;
          const int64_t h_offset = h_idx * stride;
          const int64_t w_offset = w_idx * stride;

          for (int64_t c = 0; c < channels; ++c) {
            const int8_t* channel_input = batch_input + c * height * width;
            int8_t* patch_output = batch_output + (patch * channels + c) * patch_size;

            patch_output[0] = channel_input[(h_offset + 0) * width + (w_offset + 0)];
            patch_output[1] = channel_input[(h_offset + 0) * width + (w_offset + 1)];
            patch_output[2] = channel_input[(h_offset + 0) * width + (w_offset + 2)];

            patch_output[3] = channel_input[(h_offset + 1) * width + (w_offset + 0)];
            patch_output[4] = channel_input[(h_offset + 1) * width + (w_offset + 1)];
            patch_output[5] = channel_input[(h_offset + 1) * width + (w_offset + 2)];

            patch_output[6] = channel_input[(h_offset + 2) * width + (w_offset + 0)];
            patch_output[7] = channel_input[(h_offset + 2) * width + (w_offset + 1)];
            patch_output[8] = channel_input[(h_offset + 2) * width + (w_offset + 2)];
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

            patch_output[0] = channel_input[(h_offset + 0) * width + (w_offset + 0)];
            patch_output[1] = channel_input[(h_offset + 0) * width + (w_offset + 1)];
            patch_output[2] = channel_input[(h_offset + 0) * width + (w_offset + 2)];

            patch_output[3] = channel_input[(h_offset + 1) * width + (w_offset + 0)];
            patch_output[4] = channel_input[(h_offset + 1) * width + (w_offset + 1)];
            patch_output[5] = channel_input[(h_offset + 1) * width + (w_offset + 2)];

            patch_output[6] = channel_input[(h_offset + 2) * width + (w_offset + 0)];
            patch_output[7] = channel_input[(h_offset + 2) * width + (w_offset + 1)];
            patch_output[8] = channel_input[(h_offset + 2) * width + (w_offset + 2)];
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
    im2row_1x1_unrolled(im2row_input, output,
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