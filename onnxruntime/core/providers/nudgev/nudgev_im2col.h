#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"

void convert_and_center_saturated_im2col(
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

void im2col_1x1(
    const int8_t* im2col_input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t padded_height,
    int64_t padded_width,
    int64_t stride_h,
    int64_t output_h,
    int64_t output_w,
    onnxruntime::concurrency::ThreadPool* tp) {
  if (stride_h == 1) {
    const int64_t total_size = batch_size * channels * padded_height * padded_width;
    std::memcpy(output, im2col_input, total_size * sizeof(int8_t));
    return;
  }
  constexpr int64_t kernel_size = 1;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  constexpr int64_t TILE_H = 128;
  constexpr int64_t TILE_W = 128;

  auto process_tile = [=](int64_t b, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);
    const int8_t* batch_input = im2col_input + b * channels * padded_height * padded_width;

    for (int64_t c = 0; c < channels; ++c) {
      const int8_t* channel_input = batch_input + c * padded_height * padded_width;
      const int64_t k_index = c;
      const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

      for (int64_t h_idx = h_start; h_idx < h_end; ++h_idx) {
        const int64_t h_input = h_idx * stride_h;
        const int8_t* input_row = channel_input + h_input * padded_width + w_start * stride_h;
        int8_t* output_base = output + output_k_offset + h_idx * output_w + w_start;
        const int64_t w_count = w_end - w_start;
        if (stride_h == 2) {
          for (int64_t w = 0; w < w_count; ++w) {
            output_base[w] = input_row[w << 1];
          }
        } else {
          for (int64_t w = 0; w < w_count; ++w) {
            output_base[w] = input_row[w * stride_h];
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

void im2col_1x1_stride1_nchw(
    const int8_t* input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t hw_size = height * width;
  const int64_t total_spatial = batch_size * hw_size;
  auto process_channel = [=](int64_t c) {
    const int8_t* input_channel = input + c * hw_size;
    int8_t* output_channel = output + c * total_spatial;
    std::memcpy(output_channel, input_channel, total_spatial * sizeof(int8_t));
  };

  if (tp != nullptr && channels > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, channels, [=](int64_t c) { process_channel(c); });
  } else {
    for (int64_t c = 0; c < channels; ++c) {
      process_channel(c);
    }
  }
}

void im2col_3x3(
    const int8_t* im2col_input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t padded_height,
    int64_t padded_width,
    int64_t stride_h,
    int64_t output_h,
    int64_t output_w,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int64_t kernel_size = 3 * 3;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;
  constexpr int64_t TILE_H = 64;
  constexpr int64_t TILE_W = 64;

  auto process_tile = [=](int64_t b, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);
    const int8_t* batch_input = im2col_input + b * channels * padded_height * padded_width;

    for (int64_t c = 0; c < channels; ++c) {
      const int8_t* channel_input = batch_input + c * padded_height * padded_width;

      for (int64_t kh = 0; kh < 3; ++kh) {
        for (int64_t kw = 0; kw < 3; ++kw) {
          const int64_t k_index = c * kernel_size + kh * 3 + kw;
          const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

          for (int64_t h_idx = h_start; h_idx < h_end; ++h_idx) {
            const int64_t h_input = h_idx * stride_h + kh;
            const int8_t* input_row = channel_input + h_input * padded_width + w_start * stride_h + kw;
            int8_t* output_base = output + output_k_offset + h_idx * output_w + w_start;
            const int64_t w_count = w_end - w_start;

            if (stride_h == 1) {
              std::memcpy(output_base, input_row, w_count * sizeof(int8_t));
            } else {
              if (stride_h == 2) {
                for (int64_t w = 0; w < w_count; ++w) {
                  output_base[w] = input_row[w << 1];
                }
              } else {
                for (int64_t w = 0; w < w_count; ++w) {
                  output_base[w] = input_row[w * stride_h];
                }
              }
            }
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

void im2col_7x7(
    const int8_t* im2col_input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t padded_height,
    int64_t padded_width,
    int64_t stride_h,
    int64_t output_h,
    int64_t output_w,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int64_t kernel_size = 7 * 7;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  constexpr int64_t TILE_H = 32;
  constexpr int64_t TILE_W = 32;

  auto process_tile = [=](int64_t b, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);
    const int8_t* batch_input = im2col_input + b * channels * padded_height * padded_width;

    for (int64_t c = 0; c < channels; ++c) {
      const int8_t* channel_input = batch_input + c * padded_height * padded_width;

      for (int64_t kh = 0; kh < 7; ++kh) {
        for (int64_t kw = 0; kw < 7; ++kw) {
          const int64_t k_index = c * kernel_size + kh * 7 + kw;
          const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

          for (int64_t h_idx = h_start; h_idx < h_end; ++h_idx) {
            const int64_t h_input = h_idx * stride_h + kh;
            const int8_t* input_row = channel_input + h_input * padded_width + w_start * stride_h + kw;
            int8_t* output_base = output + output_k_offset + h_idx * output_w + w_start;
            const int64_t w_count = w_end - w_start;

            if (stride_h == 1) {
              std::memcpy(output_base, input_row, w_count * sizeof(int8_t));
            } else {
              if (stride_h == 2) {
                for (int64_t w = 0; w < w_count; ++w) {
                  output_base[w] = input_row[w << 1];
                }
              } else {
                for (int64_t w = 0; w < w_count; ++w) {
                  output_base[w] = input_row[w * stride_h];
                }
              }
            }
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

void im2col_generic(
    const int8_t* im2col_input,
    int8_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t padded_height,
    int64_t padded_width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t output_h,
    int64_t output_w,
    onnxruntime::concurrency::ThreadPool* tp) {
  const int64_t kernel_size = kernel_h * kernel_w;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  auto process_batch = [=](int64_t b) {
    const int8_t* batch_input = im2col_input + b * channels * padded_height * padded_width;

    for (int64_t ic = 0; ic < channels; ++ic) {
      const int8_t* channel_input = batch_input + ic * padded_height * padded_width;

      for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
          const int64_t k_index = ic * kernel_size + kh * kernel_w + kw;

          for (int64_t oh = 0; oh < output_h; ++oh) {
            int64_t ih = oh * stride_h + kh;

            for (int64_t ow = 0; ow < output_w; ++ow) {
              int64_t iw = ow * stride_h + kw;
              const int64_t spatial_index = oh * output_w + ow;
              const int64_t output_index = (b * K * patches_per_image) +
                                           (k_index * patches_per_image) +
                                           spatial_index;

              output[output_index] = channel_input[ih * padded_width + iw];
            }
          }
        }
      }
    }
  };

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, batch_size, [=](int64_t b) { process_batch(b); });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}

void im2col(
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
  convert_and_center_saturated_im2col(input, input_centered_buffer,
                                      batch_size, channels * height * width,
                                      x_zero_point, tp);
  auto end_conversion = std::chrono::high_resolution_clock::now();
  std::cout << "conversion time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_conversion - start).count()
            << " microseconds" << std::endl;
  int64_t padded_height = height + pads[0] + pads[2];
  int64_t padded_width = width + pads[1] + pads[3];
  const int8_t* im2col_input = nullptr;

  auto start_padding = std::chrono::high_resolution_clock::now();
  if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
    auto pad_func = [=](int64_t n) {
      for (int64_t c = 0; c < channels; ++c) {
        const int64_t input_offset = (n * channels + c) * height * width;
        const int64_t output_offset = (n * channels + c) * padded_height * padded_width;
        const int8_t* input_channel = input_centered_buffer + input_offset;
        int8_t* output_channel = padded_buffer + output_offset;
        std::memset(output_channel, 0, padded_height * padded_width * sizeof(int8_t));
        for (int64_t h = 0; h < height; ++h) {
          int64_t h_pad = h + pads[0];
          int8_t* output_row = output_channel + h_pad * padded_width + pads[1];
          const int8_t* input_row = input_channel + h * width;
          std::memcpy(output_row, input_row, width * sizeof(int8_t));
        }
      }
    };

    if (tp != nullptr && batch_size > 1) {
      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, pad_func);
    } else {
      for (int64_t n = 0; n < batch_size; ++n) {
        pad_func(n);
      }
    }
    im2col_input = padded_buffer;
  } else {
    im2col_input = input_centered_buffer;
  }
  auto end_padding = std::chrono::high_resolution_clock::now();
  std::cout << "padding time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end_padding - start_padding).count()
            << " microseconds" << std::endl;

  const int64_t output_h = (padded_height - kernel_h) / stride_h + 1;
  const int64_t output_w = (padded_width - kernel_w) / stride_h + 1;

  auto im2col_start = std::chrono::high_resolution_clock::now();

  if (kernel_h == 7 && kernel_w == 7) {
    im2col_7x7(im2col_input, output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else if (kernel_h == 3 && kernel_w == 3) {
    im2col_3x3(im2col_input, output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else if (kernel_h == 1 && kernel_w == 1) {
    im2col_1x1(im2col_input, output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else {
    im2col_generic(im2col_input, output, batch_size, channels,
                   padded_height, padded_width, kernel_h, kernel_w,
                   stride_h, output_h, output_w, tp);
  }

  auto im2col_end = std::chrono::high_resolution_clock::now();
  std::cout << "\nIm2col time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(im2col_end - im2col_start).count()
            << " microseconds" << std::endl;
}

void gemm_i8_after_im2col(
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
    bool fused_relu) {
  for (int64_t b = 0; b < batch_size; ++b) {
    const int8_t* batch_input = im2col_output + b * K * patches_per_image;
    int8_t* batch_output = output + b * OC * patches_per_image;
    for (int64_t oc = 0; oc < OC; ++oc) {
      const int8_t* weight_row = weights + oc * K;
      for (int64_t n = 0; n < patches_per_image; ++n) {
        int32_t acc = 0;
        for (int64_t k = 0; k < K; ++k) {
          acc += static_cast<int32_t>(weight_row[k]) *
                 static_cast<int32_t>(batch_input[k * patches_per_image + n]);
        }

        if (has_bias) {
          acc += bias[oc];
        }

        int64_t scaled_acc = static_cast<int64_t>(acc) * M_fixed;
        int32_t sum = static_cast<int32_t>((scaled_acc + 0x4000) >> 15);

        if (fused_relu) {
          sum = std::max(sum, 0);
        }

        sum += output_zero_point;
        sum = std::clamp(sum, -128, 127);
        batch_output[oc * patches_per_image + n] = static_cast<int8_t>(sum);
      }
    }
  }
}