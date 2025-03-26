#include "core/providers/nudgev/operators/nudgev_conv_accelerated.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/nudgev/mock/mock_accelerator_compute.h"
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "core/platform/threadpool.h"
#include <immintrin.h>
#include <functional>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <future>

namespace onnxruntime {
namespace nudgev {

void apply_padding(
  const int8_t* input,
  int8_t* padded_buffer,
  int64_t batch_size,
  int64_t channels,
  int64_t height,
  int64_t width,
  const std::vector<int64_t>& pads,
  int8_t input_zp,
  onnxruntime::concurrency::ThreadPool* tp) {
const int64_t padded_height = height + pads[0] + pads[2];
const int64_t padded_width = width + pads[1] + pads[3];
const int64_t padded_size = batch_size * channels * padded_height * padded_width;
std::fill_n(padded_buffer, padded_size, input_zp);

if (tp != nullptr) {
    const int64_t total_work_items = batch_size * channels;

    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, total_work_items,
                                                                [=](int64_t work_idx) {
                                                                  const int64_t b = work_idx / channels;
                                                                  const int64_t c = work_idx % channels;

                                                                  const int64_t input_offset = (b * channels + c) * height * width;
                                                                  const int64_t output_offset = (b * channels + c) * padded_height * padded_width;
                                                                  int8_t* output_center = padded_buffer + output_offset + pads[0] * padded_width + pads[1];
                                                                  const int8_t* input_start = input + input_offset;

                                                                  for (int64_t h = 0; h < height; ++h) {
                                                                    const int8_t* input_row = input_start + h * width;
                                                                    int8_t* output_row = output_center + h * padded_width;

                                                                    std::memcpy(output_row, input_row, width * sizeof(int8_t));
                                                                  }
                                                                });
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t c = 0; c < channels; ++c) {
        const int64_t input_offset = (b * channels + c) * height * width;
        const int64_t output_offset = (b * channels + c) * padded_height * padded_width;
        int8_t* output_center = padded_buffer + output_offset + pads[0] * padded_width + pads[1];
        const int8_t* input_start = input + input_offset;

        for (int64_t h = 0; h < height; ++h) {
          const int8_t* input_row = input_start + h * width;
          int8_t* output_row = output_center + h * padded_width;

          std::memcpy(output_row, input_row, width * sizeof(int8_t));
        }
      }
    }
  }
}

void acc_im2col_1x1(
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

void acc_im2col_3x3(
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
  const int64_t patches_per_image = output_h * output_w;
  const int64_t kernel_size = 3 * 3;
  const int64_t K = channels * kernel_size;

  if (stride_h == 1) {
    if (tp != nullptr) {
      const int64_t total_work_items = batch_size * channels;

      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, total_work_items, [=](int64_t work_idx) {
            const int64_t b = work_idx / channels;
            const int64_t c = work_idx % channels;
            const int8_t* channel_data = im2col_input + b * channels * padded_height * padded_width +
                                         c * padded_height * padded_width;
            int8_t* batch_data_col = output + b * K * patches_per_image;
            for (int64_t kh = 0; kh < 3; ++kh) {
              for (int64_t kw = 0; kw < 3; ++kw) {
                const int64_t k_index = c * kernel_size + kh * 3 + kw;
                int8_t* col_ptr = batch_data_col + k_index * patches_per_image;
                if (kh < 2 || kw < 2) {
                  const int8_t* next_kernel_data = nullptr;
                  if (kw < 2) {
                    next_kernel_data = channel_data + (kh * padded_width + (kw + 1));
                  } else {
                    next_kernel_data = channel_data + ((kh + 1) * padded_width);
                  }
                  _mm_prefetch(reinterpret_cast<const char*>(next_kernel_data), _MM_HINT_T0);
                }

                for (int64_t h = 0; h < output_h; ++h) {
                  const int8_t* row_ptr = channel_data + (h + kh) * padded_width + kw;
                  int8_t* dest_ptr = col_ptr + h * output_w;
                  if (h + 1 < output_h) {
                    _mm_prefetch(
                        reinterpret_cast<const char*>(channel_data + (h + kh + 1) * padded_width + kw),
                        _MM_HINT_T0);
                  }

                  std::memcpy(dest_ptr, row_ptr, output_w * sizeof(int8_t));
                }
              }
            }
          });
    } else {
      for (int64_t b = 0; b < batch_size; ++b) {
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
      }
    }
    return;
  }

  constexpr int64_t TILE_H = 64;
  constexpr int64_t TILE_W = 64;

  auto process_channel_tile = [=](int64_t b, int64_t c, int64_t h_start, int64_t w_start) {
    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);
    const int8_t* channel_input = im2col_input + b * channels * padded_height * padded_width +
                                  c * padded_height * padded_width;

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
                reinterpret_cast<const char*>(channel_input + (h_idx * stride_h + kh + stride_h) * padded_width),
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
  };

  if (tp != nullptr) {
    const int64_t num_h_tiles = (output_h + TILE_H - 1) / TILE_H;
    const int64_t num_w_tiles = (output_w + TILE_W - 1) / TILE_W;
    const bool paralellize_channels = (channels > 8) && (batch_size * num_h_tiles * num_w_tiles < 32);

    if (paralellize_channels) {
      const int64_t total_work = batch_size * channels * num_h_tiles * num_w_tiles;

      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, total_work, [=](int64_t work_idx) {
            const int64_t bc_part = work_idx / (num_h_tiles * num_w_tiles);
            const int64_t b = bc_part / channels;
            const int64_t c = bc_part % channels;
            const int64_t tile_idx = work_idx % (num_h_tiles * num_w_tiles);
            const int64_t h_tile = tile_idx / num_w_tiles;
            const int64_t w_tile = tile_idx % num_w_tiles;

            process_channel_tile(b, c, h_tile * TILE_H, w_tile * TILE_W);
          });
    } else {
      const int64_t total_tiles = batch_size * num_h_tiles * num_w_tiles;

      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, total_tiles, [=](int64_t work_index) {
            const int64_t b = work_index / (num_h_tiles * num_w_tiles);
            const int64_t tile_idx = work_index % (num_h_tiles * num_w_tiles);
            const int64_t h_tile = tile_idx / num_w_tiles;
            const int64_t w_tile = tile_idx % num_w_tiles;

            for (int64_t c = 0; c < channels; ++c) {
              process_channel_tile(b, c, h_tile * TILE_H, w_tile * TILE_W);
            }
          });
    }
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t c = 0; c < channels; ++c) {
        for (int64_t h = 0; h < output_h; h += TILE_H) {
          for (int64_t w = 0; w < output_w; w += TILE_W) {
            process_channel_tile(b, c, h, w);
          }
        }
      }
    }
  }
}

void im2col_3x3_stride1_with_padding_avx2(
  const int8_t* __restrict__ input,
  int8_t* __restrict__ output,
  int64_t batch_size,
  int64_t channels,
  int64_t height,
  int64_t width,
  const std::vector<int64_t>& pads,
  int8_t pad_value,
  onnxruntime::concurrency::ThreadPool* tp = nullptr) {
  if (input == nullptr || output == nullptr || batch_size <= 0 || channels <= 0 ||
      height <= 0 || width <= 0 || pads.size() != 4) {
    return;
  }

  const int kernel_h = 3;
  const int kernel_w = 3;
  const int kernel_size = kernel_h * kernel_w;
  const int64_t padded_height = height + pads[0] + pads[2];
  const int64_t padded_width = width + pads[1] + pads[3];
  const int64_t output_h = (padded_height - kernel_h) + 1;
  const int64_t output_w = (padded_width - kernel_w) + 1;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  auto process_batch = [&](int64_t b) {
    if (b >= batch_size) return;

    for (int64_t c = 0; c < channels; ++c) {
      const int64_t input_channel_offset = (b * channels + c) * height * width;

      for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
          const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
          const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

          const int64_t common_w_left_padding_end = std::max((int64_t)0, pads[1] - kw);
          const int64_t common_input_offset_w = kw - pads[1];

          for (int64_t h = 0; h < output_h; ++h) {
            const int64_t h_input = h + kh - pads[0];
            int8_t* output_ptr = output + output_k_offset + h * output_w;

            if (h_input < 0 || h_input >= height) {
              memset(output_ptr, pad_value, output_w);
            } else {
              const int8_t* input_row = input + input_channel_offset + h_input * width;
              const int64_t w_right_padding_start = std::min(output_w, width + pads[1] - kw);

              if (common_w_left_padding_end > 0) {
                memset(output_ptr, pad_value, common_w_left_padding_end);
              }

              if (common_w_left_padding_end < w_right_padding_start) {
                const int8_t* input_ptr = input_row + common_w_left_padding_end + common_input_offset_w;
                const int64_t central_length = w_right_padding_start - common_w_left_padding_end;

                if (common_w_left_padding_end + common_input_offset_w >= 0 &&
                    common_w_left_padding_end + common_input_offset_w + central_length <= width) {
                  memcpy(output_ptr + common_w_left_padding_end, input_ptr, central_length);
                } else {
                  for (int64_t w = common_w_left_padding_end; w < w_right_padding_start; ++w) {
                    const int64_t w_input = w + common_input_offset_w;
                    if (w_input >= 0 && w_input < width) {
                      output_ptr[w] = input_row[w_input];
                    } else {
                      output_ptr[w] = pad_value;
                    }
                  }
                }
              }

              if (w_right_padding_start < output_w) {
                memset(output_ptr + w_right_padding_start, pad_value, output_w - w_right_padding_start);
              }

              if (h + 1 < output_h) {
                const int64_t next_h_input = h + 1 + kh - pads[0];
                if (next_h_input >= 0 && next_h_input < height) {
                  const int8_t* next_input_row = input + input_channel_offset + next_h_input * width;
                  __builtin_prefetch(next_input_row, 0, 1);
                }
              }
            }
          }
        }
      }
    }
  };

  if (tp != nullptr && batch_size > 1) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, batch_size, process_batch);
  } else {
    for (int64_t b = 0; b < batch_size; ++b) {
      process_batch(b);
    }
  }
}

void im2col_3x3_stride2_with_padding_avx2(
  const int8_t* __restrict__ input,
  int8_t* __restrict__ output,
  int64_t batch_size,
  int64_t channels,
  int64_t height,
  int64_t width,
  const std::vector<int64_t>& pads,
  int8_t pad_value,
  onnxruntime::concurrency::ThreadPool* tp = nullptr) {
  const int kernel_h = 3;
  const int kernel_w = 3;
  const int kernel_size = kernel_h * kernel_w;
  const int64_t padded_height = height + pads[0] + pads[2];
  const int64_t padded_width = width + pads[1] + pads[3];
  const int64_t output_h = (padded_height - kernel_h) / 2 + 1;
  const int64_t output_w = (padded_width - kernel_w) / 2 + 1;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  __m256i pad_vector = _mm256_set1_epi8(pad_value);
  __m128i pad_vector_128 = _mm_set1_epi8(pad_value);
  const __m128i shuffle_mask_128 = _mm_setr_epi8(
      0, 2, 4, 6, 8, 10, 12, 14,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

  auto process_kernel_position = [&](int64_t work_idx) {
    const int64_t bckh = work_idx / kernel_w;
    const int64_t kw = work_idx % kernel_w;
    const int64_t bck = bckh / kernel_h;
    const int64_t kh = bckh % kernel_h;
    const int64_t b = bck / channels;
    const int64_t c = bck % channels;

    if (b >= batch_size || c >= channels || kh >= kernel_h || kw >= kernel_w)
      return;

    const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
    const int64_t output_k_offset = (b * K + k_index) * patches_per_image;
    const int64_t input_offset = (b * channels + c) * height * width;

    for (int64_t h = 0; h < output_h; ++h) {
      const int64_t h_input = h * 2 + kh - pads[0];
      int8_t* output_row = output + output_k_offset + h * output_w;

      if (h_input < 0 || h_input >= height) {
        int64_t w = 0;
        for (; w + 32 <= output_w; w += 32) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          _mm256_storeu_si256((__m256i*)(output_row + w + 16), pad_vector);
        }
        if (w + 16 <= output_w) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          w += 16;
        }
        for (; w < output_w; ++w) {
          output_row[w] = pad_value;
        }
      } else {
        const int8_t* input_row = input + input_offset + h_input * width;

        if (h + 1 < output_h) {
          const int64_t next_h_input = (h + 1) * 2 + kh - pads[0];
          if (next_h_input >= 0 && next_h_input < height) {
            __builtin_prefetch(input + input_offset + next_h_input * width, 0, 0);
          }
        }

        const int64_t w_left_padding_end = std::max((int64_t)0, (pads[1] - kw + 1) / 2);
        const int64_t w_right_padding_start = std::min(output_w, (width + pads[1] - kw + 1) / 2);

        int64_t w = 0;
        for (; w + 16 <= w_left_padding_end; w += 16) {
          _mm_storeu_si128((__m128i*)(output_row + w), pad_vector_128);
        }
        for (; w < w_left_padding_end; ++w) {
          output_row[w] = pad_value;
        }

        const int64_t central_start = w_left_padding_end;
        const int64_t central_end = w_right_padding_start;
        const int64_t central_size = central_end - central_start;
        int64_t w_vec = 0;

        for (; w_vec + 16 <= central_size; w_vec += 16) {
          const int8_t* input_ptr = input_row + (central_start + w_vec) * 2 + kw - pads[1];
          int8_t* output_ptr = output_row + central_start + w_vec;

          __m128i in1 = _mm_loadu_si128((const __m128i*)(input_ptr));
          __m128i in2 = _mm_loadu_si128((const __m128i*)(input_ptr + 16));

          __m128i out1 = _mm_shuffle_epi8(in1, shuffle_mask_128);
          __m128i out2 = _mm_shuffle_epi8(in2, shuffle_mask_128);

          __m128i shifted_out2 = _mm_slli_si128(out2, 8);
          __m128i combined = _mm_or_si128(out1, shifted_out2);

          _mm_storeu_si128((__m128i*)output_ptr, combined);
        }

        for (int64_t w_rem = central_start + w_vec; w_rem < central_end; ++w_rem) {
          const int64_t w_input = w_rem * 2 + kw - pads[1];
          output_row[w_rem] = input_row[w_input];
        }

        w = central_end;
        for (; w + 16 <= output_w; w += 16) {
          _mm_storeu_si128((__m128i*)(output_row + w), pad_vector_128);
        }
        for (; w < output_w; ++w) {
          output_row[w] = pad_value;
        }
      }
    }
  };

  const int64_t total_work = batch_size * channels * kernel_h * kernel_w;

  if (tp != nullptr) {
    onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(tp, total_work, process_kernel_position);
  } else {
    for (int64_t work_idx = 0; work_idx < total_work; ++work_idx) {
      process_kernel_position(work_idx);
    }
  }
}

void acc_im2col_7x7(
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

void acc_im2col_generic(
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

void acc_im2col(
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
  int8_t input_zp,
  onnxruntime::concurrency::ThreadPool* tp) {

  if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 &&
      pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
    *output = const_cast<int8_t*>(input);
    return;
  }

  if (kernel_h == 3 && kernel_w == 3) {
    if (stride_h == 1) {
      im2col_3x3_stride1_with_padding_avx2(
          input,
          *output,
          batch_size,
          channels,
          height,
          width,
          pads,
          input_zp,
          tp);
      return;
    }

    if (stride_h == 2) {
      im2col_3x3_stride2_with_padding_avx2(
          input,
          *output,
          batch_size,
          channels,
          height,
          width,
          pads,
          input_zp,
          tp);
      return;
    }
  }

  int64_t padded_height = height + pads[0] + pads[2];
  int64_t padded_width = width + pads[1] + pads[3];
  const int8_t* im2col_input = nullptr;

  if (pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
    im2col_input = input;
  } else {
    apply_padding(input, padded_buffer, batch_size, channels, height, width, pads, input_zp, tp);
    im2col_input = padded_buffer;
  }

  const int64_t output_h = (padded_height - kernel_h) / stride_h + 1;
  const int64_t output_w = (padded_width - kernel_w) / stride_h + 1;

  if (kernel_h == 7 && kernel_w == 7) {
    acc_im2col_7x7(im2col_input, *output, batch_size, channels,
              padded_height, padded_width, stride_h,
              output_h, output_w, tp);
  } else if (kernel_h == 3 && kernel_w == 3) {
    im2col_3x3_optimized(im2col_input, *output, batch_size, channels,
              padded_height, padded_width, stride_h,
              output_h, output_w, tp);
  } else if (kernel_h == 1 && kernel_w == 1) {
    acc_im2col_1x1(im2col_input, *output, batch_size, channels,
              padded_height, padded_width, stride_h,
              output_h, output_w, tp);
  } else {
    acc_im2col_generic(im2col_input, *output, batch_size, channels,
                  padded_height, padded_width, kernel_h, kernel_w,
                  stride_h, output_h, output_w, tp);
  }
}

void gemm_with_accelerator(
  const int8_t* im2col_output,
  int8_t* output,
  int64_t OC,
  int64_t K,
  int64_t patches_per_image,
  int64_t batch_size,
  int32_t M_fixed,
  int8_t output_zero_point,
  int8_t input_zero_point,
  int64_t k_blocks,
  int64_t oc_blocks,
  onnxruntime::concurrency::ThreadPool* tp) {
ORT_UNUSED_PARAMETER(tp);

const int64_t block_size = 32;
const int16_t multiplier = static_cast<int16_t>(M_fixed);
const bool debug_mode = false;

auto start_time = std::chrono::high_resolution_clock::now();

int device_write_fd = open("/dev/xdma0_h2c_0", O_WRONLY);
int device_read_fd = open("/dev/xdma0_c2h_0", O_RDONLY);

if (device_write_fd < 0 || device_read_fd < 0) {
  std::cerr << "Error: failed to open accelerator device files. Write FD: "
            << device_write_fd << ", Read FD: " << device_read_fd
            << ", errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
  return;
}

int64_t J = (patches_per_image + block_size - 1) / block_size;
std::atomic<bool> should_stop(false);
std::atomic<size_t> packets_sent(0);
std::atomic<size_t> packets_received(0);
std::mutex output_mutex;

mock::MatrixMultiplicationHeader header;
header.xzp = static_cast<uint8_t>(input_zero_point);
header.yzp = static_cast<uint8_t>(output_zero_point);
header.m_lsb = multiplier & 0xFF;
header.m_msb = (multiplier >> 8) & 0xFF;
header.n_lsb = K & 0xFF;
header.n_msb = (K >> 8) & 0xFF;
header.k = oc_blocks;
header.j_lsb = J & 0xFF;
header.j_msb = (J >> 8) & 0xFF;
header.batch_size = batch_size;
header.reserved1 = 0;
header.reserved2 = 0;

size_t block_data_size = K * block_size * sizeof(int8_t);
size_t total_input_size = batch_size * J * block_data_size;
std::vector<int8_t> reorganized_input(total_input_size, 0);

for (int64_t b = 0; b < batch_size; b++) {
  for (int64_t j = 0; j < J; j++) {
    size_t block_offset = (b * J + j) * block_data_size;
    int64_t valid_patches = std::min(block_size, patches_per_image - j * block_size);
    for (int64_t k = 0; k < K; k++) {
      size_t dst_row_offset = block_offset + k * block_size;
      size_t src_row_offset = b * K * patches_per_image + k * patches_per_image + j * block_size;
      if (src_row_offset + valid_patches > static_cast<size_t>(batch_size * K * patches_per_image) ||
          dst_row_offset + valid_patches > reorganized_input.size()) {
        std::cerr << "  ERROR: Invalid indices during input reorganization:" << std::endl;
        std::cerr << "    src_row_offset: " << src_row_offset
                  << ", dst_row_offset: " << dst_row_offset
                  << ", valid_patches: " << valid_patches
                  << ", input size: " << batch_size * K * patches_per_image
                  << ", buffer size: " << reorganized_input.size() << std::endl;
        continue;
      }
      std::memcpy(
          reorganized_input.data() + dst_row_offset,
          im2col_output + src_row_offset,
          valid_patches * sizeof(int8_t));
      if (valid_patches < block_size) {
        std::memset(
            reorganized_input.data() + dst_row_offset + valid_patches,
            0,
            (block_size - valid_patches) * sizeof(int8_t));
      }
    }
  }
}

auto receiver_thread = std::thread([&]() {
  int64_t received_batch;
  ssize_t read_result = 0;
  int read_attempts = 0;
  const int max_read_attempts = 100;

  while (read_attempts < max_read_attempts && !should_stop.load()) {
    read_result = read(device_read_fd, &received_batch, sizeof(received_batch));

    if (read_result == sizeof(received_batch)) {
      break;
    }

    if (read_result < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        read_attempts++;
        continue;
      }

      std::cerr << "Error: failed to read batch index from results. Errno: "
                << errno << " (" << strerror(errno) << ")" << std::endl;
      should_stop.store(true);
      return;
    }

    std::cerr << "Error: incomplete read of batch index. Read " << read_result
              << " of " << sizeof(received_batch) << " bytes." << std::endl;
    read_attempts++;
  }

  if (read_attempts >= max_read_attempts) {
    std::cerr << "Error: timeout waiting for batch index from accelerator" << std::endl;
    should_stop.store(true);
    return;
  }

  uint32_t result_size;
  read_result = read(device_read_fd, &result_size, sizeof(result_size));

  if (read_result != sizeof(result_size)) {
    std::cerr << "Error: failed to read result size. Read " << read_result
              << " of " << sizeof(result_size) << " bytes. Errno: "
              << errno << " (" << strerror(errno) << ")" << std::endl;
    should_stop.store(true);
    return;
  }

  size_t expected_size = static_cast<size_t>(J * batch_size * oc_blocks * block_size * block_size);
  if (result_size != expected_size) {
    std::cerr << "Warning: Received result size (" << result_size
              << ") differs from expected size (" << expected_size << ")" << std::endl;
  }

  std::vector<int8_t> result_data(result_size);

  ssize_t bytes_read = 0;
  size_t total_bytes_read = 0;

  while (total_bytes_read < result_size && !should_stop.load()) {
    bytes_read = read(device_read_fd,
                      result_data.data() + total_bytes_read,
                      result_size - total_bytes_read);

    if (bytes_read <= 0) {
      if (bytes_read < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      std::cerr << "Error: failed to read result data. Read " << total_bytes_read
                << " of " << result_size << " bytes. Errno: " << errno
                << " (" << strerror(errno) << ")" << std::endl;
      should_stop.store(true);
      return;
    }

    total_bytes_read += bytes_read;

    if (debug_mode && total_bytes_read % (1024 * 1024) == 0) {
      std::cout << "  DEBUG: Read progress: " << total_bytes_read << "/" << result_size << " bytes" << std::endl;
    }
  }

  if (total_bytes_read != result_size) {
    std::cerr << "Error: incomplete read of result data. Read " << total_bytes_read
              << " of " << result_size << " bytes" << std::endl;
    should_stop.store(true);
    return;
  }

  if (debug_mode) {
    std::cout << "  DEBUG: Successfully read " << total_bytes_read << " bytes from accelerator" << std::endl;
  }

  {
    std::lock_guard<std::mutex> lock(output_mutex);

    std::memset(output, 0, batch_size * patches_per_image * OC * sizeof(int8_t));
    int64_t max_blocks = result_size / (oc_blocks * block_size * block_size);

    for (int64_t b = 0; b < batch_size; b++) {
      for (int64_t j = 0; j < J; j++) {
        int64_t patch_base = j * block_size;
        int64_t valid_patches = std::min(block_size, patches_per_image - patch_base);
        if (valid_patches <= 0 || patch_base >= patches_per_image) continue;

        int64_t block_idx = b * J + j;
        if (block_idx >= max_blocks) {
          std::cerr << "  WARNING: Skipping block out of bounds: " << block_idx
                    << " (max: " << max_blocks << ")" << std::endl;
          continue;
        }

        for (int64_t ocb = 0; ocb < oc_blocks; ocb++) {
          int64_t oc_base = ocb * block_size;
          int64_t valid_oc = std::min(block_size, OC - oc_base);

          size_t block_offset = (block_idx * oc_blocks + ocb) * block_size * block_size;
          if (block_offset + block_size * block_size > result_data.size()) {
            std::cerr << "  WARNING: Skipping block offset out of bounds: " << block_offset
                      << " (max: " << result_data.size() << ")" << std::endl;
            continue;
          }

          for (int64_t p_local = 0; p_local < valid_patches; p_local++) {
            int64_t p_global = patch_base + p_local;
            for (int64_t oc_local = 0; oc_local < valid_oc; oc_local++) {
              int64_t oc_global = oc_base + oc_local;
              size_t src_idx = block_offset + oc_local * block_size + p_local;
              size_t dst_idx = (b * OC + oc_global) * patches_per_image + p_global;
              output[dst_idx] = result_data[src_idx];
            }
          }
        }
      }
    }
  }

  packets_received++;

  if (debug_mode) {
    std::cout << "  DEBUG: Output reorganization complete" << std::endl;
  }
});

auto sender_thread = std::thread([&]() {
  if (write(device_write_fd, &header, sizeof(header)) != sizeof(header)) {
    std::cerr << "Error: failed to write header to accelerator. Errno: "
              << errno << " (" << strerror(errno) << ")" << std::endl;
    should_stop.store(true);
    return;
  }

  int64_t all_batches = 0;
  if (write(device_write_fd, &all_batches, sizeof(all_batches)) != sizeof(all_batches)) {
    std::cerr << "Error: failed to write batch index to accelerator. Errno: "
              << errno << " (" << strerror(errno) << ")" << std::endl;
    should_stop.store(true);
    return;
  }

  uint32_t data_size = static_cast<uint32_t>(reorganized_input.size());
  if (write(device_write_fd, &data_size, sizeof(data_size)) != sizeof(data_size)) {
    std::cerr << "Error: failed to write data size to accelerator. Errno: "
              << errno << " (" << strerror(errno) << ")" << std::endl;
    should_stop.store(true);
    return;
  }

  ssize_t bytes_written = 0;
  size_t total_bytes_written = 0;
  int write_attempts = 0;
  const int max_write_attempts = 100;

  while (total_bytes_written < data_size && write_attempts < max_write_attempts && !should_stop.load()) {
    bytes_written = write(device_write_fd,
                         reorganized_input.data() + total_bytes_written,
                         data_size - total_bytes_written);

    if (bytes_written <= 0) {
      if (bytes_written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        write_attempts++;
        continue;
      }

      std::cerr << "Error: failed to write data to accelerator. Wrote " << total_bytes_written
                << " of " << data_size << " bytes. Errno: " << errno
                << " (" << strerror(errno) << ")" << std::endl;
      should_stop.store(true);
      return;
    }

    total_bytes_written += bytes_written;

    if (debug_mode && total_bytes_written % (1024 * 1024) == 0) {
      std::cout << "  DEBUG: Write progress: " << total_bytes_written << "/" << data_size << " bytes" << std::endl;
    }
  }

  if (total_bytes_written != data_size) {
    std::cerr << "Error: incomplete data write: " << total_bytes_written << " of " << data_size
              << " bytes after " << write_attempts << " attempts" << std::endl;
    should_stop.store(true);
    return;
  }

  if (debug_mode) {
    std::cout << "  DEBUG: Successfully sent " << total_bytes_written << " bytes to accelerator" << std::endl;
  }

  fsync(device_write_fd);
  packets_sent++;
});

const int max_wait_time_ms = 30000;
int wait_time_ms = 0;
const int check_interval_ms = 100;

while (wait_time_ms < max_wait_time_ms) {
  if (packets_received.load() > 0 && packets_sent.load() > 0) {
    break;
  }

  if (should_stop.load()) {
    break;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(check_interval_ms));
  wait_time_ms += check_interval_ms;
}

if (wait_time_ms >= max_wait_time_ms) {
  std::cerr << "Warning: timeout waiting for send/receive operations to complete" << std::endl;
  should_stop.store(true);
}

if (sender_thread.joinable()) {
  sender_thread.join();
}

if (receiver_thread.joinable()) {
  receiver_thread.join();
}

close(device_write_fd);
close(device_read_fd);

auto end_time = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

if (debug_mode) {
  std::cout << "  DEBUG: GEMM with accelerator completed in " << duration.count() << " microseconds" << std::endl;
}
}

Status ComputeNudgeVConvWithAccelerator(ConvQuantParams* conv_params, OrtKernelContext* context) {
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
  int8_t* im2col_ptr = conv_params->im2col_buffer.data();

  if (conv_params->output_height == 0 || conv_params->output_width == 0 ||
      (conv_params->dynamic_batch && actual_batch_size != conv_params->batch_size)) {
    const int64_t kernel_height = conv_params->weight_shape[2];
    const int64_t kernel_width = conv_params->weight_shape[3];
    const int64_t padded_height = input_height + conv_params->pads[0] + conv_params->pads[2];
    const int64_t padded_width = input_width + conv_params->pads[1] + conv_params->pads[3];

    const int64_t output_height = (padded_height - kernel_height) / conv_params->strides[0] + 1;
    const int64_t output_width = (padded_width - kernel_width) / conv_params->strides[0] + 1;
    conv_params->output_height = output_height;
    conv_params->output_width = output_width;
    const int64_t K = input_channels * kernel_height * kernel_width;
    conv_params->K = K;

    if (conv_params->dynamic_batch && actual_batch_size != conv_params->batch_size) {
      const int64_t new_N = actual_batch_size * output_height * output_width;
      conv_params->im2col_buffer.resize(static_cast<size_t>(new_N * K));
      im2col_ptr = conv_params->im2col_buffer.data();
      conv_params->padded_buffer.resize(static_cast<size_t>(actual_batch_size * input_channels * padded_height * padded_width));
      std::fill(conv_params->padded_buffer.begin(), conv_params->padded_buffer.end(), conv_params->input_zp);
      conv_params->batch_size = actual_batch_size;
    }
  }

  const int64_t required_im2col_size = actual_batch_size * conv_params->K * conv_params->output_height * conv_params->output_width;
  if (static_cast<int64_t>(conv_params->im2col_buffer.size()) != required_im2col_size) {
    conv_params->im2col_buffer.resize(static_cast<size_t>(required_im2col_size));
    im2col_ptr = conv_params->im2col_buffer.data();

    const int64_t padded_height = input_height + conv_params->pads[0] + conv_params->pads[2];
    const int64_t padded_width = input_width + conv_params->pads[1] + conv_params->pads[3];

    const int64_t required_padded_size = actual_batch_size * input_channels * padded_height * padded_width;
    if (static_cast<int64_t>(conv_params->padded_buffer.size()) != required_padded_size) {
      conv_params->padded_buffer.resize(static_cast<size_t>(required_padded_size));
      std::fill(conv_params->padded_buffer.begin(), conv_params->padded_buffer.end(), conv_params->input_zp);
    }
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
  auto im2col_start = std::chrono::high_resolution_clock::now();

  onnxruntime::nudgev::acc_im2col(
      input_data,
      &im2col_ptr,
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

  auto im2col_end = std::chrono::high_resolution_clock::now();
  auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(im2col_end - im2col_start);
  std::cout << "Conv Op - Im2col execution time: " << duration_im2col.count() << " microseconds" << std::endl;
  auto gemm_start = std::chrono::high_resolution_clock::now();

  if (conv_params->k_blocks == 0 || conv_params->oc_blocks == 0) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Weights have not been loaded into accelerator memory");
  }

  const int64_t patches_per_image = conv_params->output_height * conv_params->output_width;
  onnxruntime::nudgev::gemm_with_accelerator(
      im2col_ptr,
      output_data,
      output_channels,
      conv_params->K,
      patches_per_image,
      actual_batch_size,
      conv_params->M_fixed,
      conv_params->output_zp,
      conv_params->input_zp,
      conv_params->k_blocks,
      conv_params->oc_blocks,
      tp);

  auto gemm_end = std::chrono::high_resolution_clock::now();
  auto duration_gemm = std::chrono::duration_cast<std::chrono::microseconds>(gemm_end - gemm_start);
  std::cout << "Conv Op - Accelerator GEMM execution time: " << duration_gemm.count() << " microseconds" << std::endl;

  return Status::OK();
}

}  // namespace nudgev
}  // namespace onnxruntime
