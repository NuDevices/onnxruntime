#include "core/providers/nudgev/operators/nudgev_conv.h"
#include "core/framework/op_kernel_context_internal.h"
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>
#include <functional>

namespace onnxruntime {
namespace nudgev {

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
void im2col_3x3_optimized(
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
  const int64_t pad_t = 0;
  const int64_t pad_l = 0;
  const int64_t pad_b = 0;
  const int64_t pad_r = 0;

  const int64_t patches_per_image = output_h * output_w;
  const int64_t kernel_size = 3 * 3;
  const int64_t K = channels * kernel_size;

  if (stride_h == 1) {
    auto process_batch = [&](ptrdiff_t batch_idx) {
      const int64_t b = static_cast<int64_t>(batch_idx);
      const int8_t* batch_data_im = im2col_input + b * channels * padded_height * padded_width;
      int8_t* batch_data_col = output + b * K * patches_per_image;

      for (int64_t c = 0; c < channels; ++c) {
        const int8_t* channel_data = batch_data_im + c * padded_height * padded_width;

        for (int64_t kh = 0; kh < 3; ++kh) {
          for (int64_t kw = 0; kw < 3; ++kw) {
            const int64_t k_index = c * kernel_size + kh * 3 + kw;
            int8_t* col_ptr = batch_data_col + k_index * patches_per_image;

            for (int64_t h = 0; h < output_h; ++h) {
              const int8_t* row_ptr = channel_data + (h + kh) * padded_width + kw;
              std::memcpy(col_ptr + h * output_w, row_ptr, output_w * sizeof(int8_t));
            }
          }
        }
      }
    };

    if (tp != nullptr && batch_size > 1) {
      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, batch_size,
          [&](std::ptrdiff_t batch_idx) {
            process_batch(batch_idx);
          });
    } else {
      for (int64_t b = 0; b < batch_size; ++b) {
        process_batch(b);
      }
    }

    return;
  }

  constexpr int64_t TILE_H = 64;
  constexpr int64_t TILE_W = 64;

  auto process_tile = [&](int64_t b, int64_t h_start, int64_t w_start) {
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

            if (h_idx + 1 < h_end) {
              _mm_prefetch(
                  reinterpret_cast<const char*>(channel_input +
                                                (h_idx * stride_h + kh + stride_h) * padded_width),
                  _MM_HINT_T0);
            }

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
  };

  if (tp != nullptr) {
    const int64_t num_h_tiles = (output_h + TILE_H - 1) / TILE_H;
    const int64_t num_w_tiles = (output_w + TILE_W - 1) / TILE_W;
    const int64_t total_tiles = batch_size * num_h_tiles * num_w_tiles;

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
        tp, total_tiles,
        [&](std::ptrdiff_t work_index) {
          const int64_t b = static_cast<int64_t>(work_index) / (num_h_tiles * num_w_tiles);
          const int64_t tile_idx = static_cast<int64_t>(work_index) % (num_h_tiles * num_w_tiles);
          const int64_t tile_h = tile_idx / num_w_tiles;
          const int64_t tile_w = tile_idx % num_w_tiles;

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

  const int64_t TILE_H = 128 / (kernel_h * kernel_w / 9 + 1);
  const int64_t TILE_W = 128 / (kernel_h * kernel_w / 9 + 1);

  auto process_tile = [=](int64_t b, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);
    const int8_t* batch_input = im2col_input + b * channels * padded_height * padded_width;

    for (int64_t c = 0; c < channels; ++c) {
      const int8_t* channel_input = batch_input + c * padded_height * padded_width;

      for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
          const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
          const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

          for (int64_t h_idx = h_start; h_idx < h_end; ++h_idx) {
            const int64_t h_input = h_idx * stride_h + kh;
            const int8_t* input_row = channel_input + h_input * padded_width + w_start * stride_h + kw;
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

void im2col(
    const int8_t* input,
    int8_t** output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    const std::vector<int64_t>& pads,
    int8_t* padded_buffer,
    onnxruntime::concurrency::ThreadPool* tp) {
  if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 &&
      pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
    *output = const_cast<int8_t*>(input);
    return;
  }
  if (kernel_h == 1 && kernel_w == 1 && stride_h == 1) {
    int64_t padded_height = height + pads[0] + pads[2];
    int64_t padded_width = width + pads[1] + pads[3];
    if (tp != nullptr && batch_size > 1) {
      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](int64_t b) {
        for (int64_t c = 0; c < channels; ++c) {
          const int64_t input_offset = (b * channels + c) * height * width;
          const int64_t output_offset = (b * channels + c) * padded_height * padded_width;
          int8_t* output_center = padded_buffer + output_offset +
                                  pads[0] * padded_width + pads[1];
          const int8_t* input_start = input + input_offset;
          constexpr int vec_size = 32;
          for (int64_t h = 0; h < height; ++h) {
            const int8_t* input_row = input_start + h * width;
            int8_t* output_row = output_center + h * padded_width;
            int64_t width_vec_count = width / vec_size;
            for (int64_t w = 0; w < width_vec_count * vec_size; w += vec_size) {
              __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_row + w));
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_row + w), v);
            }
            for (int64_t w = width_vec_count * vec_size; w < width; ++w) {
              output_row[w] = input_row[w];
            }
          }
        }
      });
    } else {
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
          const int64_t input_offset = (b * channels + c) * height * width;
          const int64_t output_offset = (b * channels + c) * padded_height * padded_width;

          int8_t* output_center = padded_buffer + output_offset +
                                  pads[0] * padded_width + pads[1];
          const int8_t* input_start = input + input_offset;

          constexpr int vec_size = 32;
          for (int64_t h = 0; h < height; ++h) {
            const int8_t* input_row = input_start + h * width;
            int8_t* output_row = output_center + h * padded_width;

            int64_t width_vec_count = width / vec_size;

            for (int64_t w = 0; w < width_vec_count * vec_size; w += vec_size) {
              __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_row + w));
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_row + w), v);
            }

            for (int64_t w = width_vec_count * vec_size; w < width; ++w) {
              output_row[w] = input_row[w];
            }
          }
        }
      }
    }
    *output = padded_buffer;
    return;
  }

  int64_t padded_height = height + pads[0] + pads[2];
  int64_t padded_width = width + pads[1] + pads[3];
  const int8_t* im2col_input = nullptr;

  if (pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
    constexpr int vec_size = 32;
    const int64_t total_elements = batch_size * channels * height * width;
    if (reinterpret_cast<uintptr_t>(input) % vec_size != 0) {
      if (tp != nullptr && batch_size > 1) {
        onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](int64_t b) {
          const int64_t batch_elements = channels * height * width;
          const int8_t* batch_input = input + b * batch_elements;
          int8_t* batch_padded = padded_buffer + b * batch_elements;

          for (ptrdiff_t i = 0; i + vec_size <= batch_elements; i += vec_size) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(batch_input + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(batch_padded + i), v);
          }
        });
      } else {
        for (ptrdiff_t i = 0; i + vec_size <= total_elements; i += vec_size) {
          __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
          _mm256_store_si256(reinterpret_cast<__m256i*>(padded_buffer + i), v);
        }
      }
      im2col_input = padded_buffer;
    } else {
      im2col_input = input;
    }
  } else {
    if (tp != nullptr && batch_size > 1) {
      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, [=](int64_t b) {
        for (int64_t c = 0; c < channels; ++c) {
          const int64_t input_offset = (b * channels + c) * height * width;
          const int64_t output_offset = (b * channels + c) * padded_height * padded_width;

          int8_t* output_center = padded_buffer + output_offset +
                                  pads[0] * padded_width + pads[1];
          const int8_t* input_start = input + input_offset;

          constexpr int vec_size = 32;
          for (int64_t h = 0; h < height; ++h) {
            const int8_t* input_row = input_start + h * width;
            int8_t* output_row = output_center + h * padded_width;

            int64_t width_vec_count = width / vec_size;

            for (int64_t w = 0; w < width_vec_count * vec_size; w += vec_size) {
              __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_row + w));
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_row + w), v);
            }

            for (int64_t w = width_vec_count * vec_size; w < width; ++w) {
              output_row[w] = input_row[w];
            }
          }
        }
      });
    } else {
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
          const int64_t input_offset = (b * channels + c) * height * width;
          const int64_t output_offset = (b * channels + c) * padded_height * padded_width;

          int8_t* output_center = padded_buffer + output_offset +
                                  pads[0] * padded_width + pads[1];
          const int8_t* input_start = input + input_offset;

          constexpr int vec_size = 32;
          for (int64_t h = 0; h < height; ++h) {
            const int8_t* input_row = input_start + h * width;
            int8_t* output_row = output_center + h * padded_width;

            int64_t width_vec_count = width / vec_size;

            for (int64_t w = 0; w < width_vec_count * vec_size; w += vec_size) {
              __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_row + w));
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_row + w), v);
            }

            for (int64_t w = width_vec_count * vec_size; w < width; ++w) {
              output_row[w] = input_row[w];
            }
          }
        }
      }
    }
    im2col_input = padded_buffer;
  }

  const int64_t output_h = (padded_height - kernel_h) / stride_h + 1;
  const int64_t output_w = (padded_width - kernel_w) / stride_h + 1;

  if (kernel_h == 7 && kernel_w == 7) {
    im2col_7x7(im2col_input, *output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else if (kernel_h == 3 && kernel_w == 3) {
    im2col_3x3_optimized(im2col_input, *output, batch_size, channels,
                         padded_height, padded_width, stride_h,
                         output_h, output_w, tp);
  } else if (kernel_h == 1 && kernel_w == 1 && stride_h == 1) {
    *output = const_cast<int8_t*>(im2col_input);
    return;
  } else if (kernel_h == 1 && kernel_w == 1) {
    im2col_1x1(im2col_input, *output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else {
    im2col_generic(im2col_input, *output, batch_size, channels,
                   padded_height, padded_width, kernel_h, kernel_w,
                   stride_h, output_h, output_w, tp);
  }
}

void gemm_i8_after_im2col_xzp(
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
    int8_t input_zero_point,
    bool fused_relu,
    onnxruntime::concurrency::ThreadPool* tp) {
  constexpr int vector_size = 16;

  std::vector<int32_t> zp_compensation(OC);
  for (int oc = 0; oc < OC; ++oc) {
    int32_t weight_sum = 0;
    const int8_t* weight_row = weights + oc * K;
    for (int k = 0; k < K; ++k) {
      weight_sum += static_cast<int32_t>(weight_row[k]);
    }
    zp_compensation[oc] = weight_sum * static_cast<int32_t>(input_zero_point);
  }

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

        __m256i zp_comp_vec = _mm256_set1_epi32(-zp_compensation[oc]);
        acc_lo = _mm256_add_epi32(acc_lo, zp_comp_vec);
        acc_hi = _mm256_add_epi32(acc_hi, zp_comp_vec);

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

        acc -= zp_compensation[oc];

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
Status ComputeNudgeVConv(ConvQuantParams* conv_params, OrtKernelContext* context) {
  auto start = std::chrono::high_resolution_clock::now();

  if (conv_params->weight_shape.empty()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "weight_shape is empty");
  }
  if (conv_params->strides.empty()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "strides is empty");
  }
  if (conv_params->pads.empty()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "pads is empty");
  }

  auto* ctx_internal = reinterpret_cast<OpKernelContextInternal*>(context);
  concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

  const Tensor* input = ctx_internal->Input<Tensor>(0);
  if (!input) {
    return Status(common::ONNXRUNTIME, common::FAIL, "input tensor is null");
  }

  const auto& input_shape = input->Shape();
  const int64_t actual_batch_size = input_shape[0];
  const int64_t input_channels = input_shape[1];
  const int64_t input_height = input_shape[2];
  const int64_t input_width = input_shape[3];

  if (conv_params->dynamic_batch && actual_batch_size != conv_params->batch_size) {
    const int64_t new_N = actual_batch_size * conv_params->output_height * conv_params->output_width;
  }

  const int64_t output_channels = conv_params->weight_shape[0];
  const int64_t kernel_height = conv_params->weight_shape[2];
  const int64_t kernel_width = conv_params->weight_shape[3];
  TensorShape output_shape({actual_batch_size, output_channels,
                            conv_params->output_height, conv_params->output_width});

  Tensor* Y = ctx_internal->Output(0, output_shape);
  if (!Y) {
    return Status(common::ONNXRUNTIME, common::FAIL, "failed to create output tensor");
  }
  if (Y->DataType() != DataTypeImpl::GetType<int8_t>()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Output type must be int8");
  }

  const auto* input_data = input->Data<int8_t>();
  auto* output_data = Y->MutableData<int8_t>();
  int8_t* im2row_ptr = conv_params->im2row_buffer.data();

  // std::cout << "Input data address: " << static_cast<const void*>(input_data) << std::endl;
  // std::cout << "Output data address: " << static_cast<void*>(output_data) << std::endl;

  onnxruntime::nudgev::im2col(
      input_data,
      &im2row_ptr,
      actual_batch_size,
      input_channels,
      input_height,
      input_width,
      kernel_height,
      kernel_width,
      conv_params->strides[0],
      conv_params->pads,
      conv_params->padded_buffer.data(),
      tp);

  auto end_im2col = std::chrono::high_resolution_clock::now();
  auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
  auto start_gemm = std::chrono::high_resolution_clock::now();
  const int64_t patches_per_image = conv_params->output_height * conv_params->output_width;
  onnxruntime::nudgev::gemm_i8_after_im2col_xzp(
      conv_params->weights.data(),
      im2row_ptr,
      output_data,
      output_channels,
      conv_params->K,
      patches_per_image,
      actual_batch_size,
      conv_params->has_bias ? conv_params->bias.data() : nullptr,
      conv_params->has_bias,
      conv_params->M_fixed,
      conv_params->output_zp,
      conv_params->input_zp,
      conv_params->fused_relu,
      tp);

  auto end_gemm = std::chrono::high_resolution_clock::now();
  auto duration_gemm = std::chrono::duration_cast<std::chrono::microseconds>(end_gemm - start_gemm);
  std::cout << "Conv Op - Im2col execution time: " << duration_im2col.count() << " microseconds" << std::endl;
  std::cout << "Conv Op - Total Gemm execution time: " << duration_gemm.count() << " microseconds" << std::endl;

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime