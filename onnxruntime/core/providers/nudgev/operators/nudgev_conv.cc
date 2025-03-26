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
#include <omp.h>
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

void debug_im2col_3x3(
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
  std::cout << "DEBUG: im2col_3x3 started" << std::endl;
  std::cout << "DEBUG: input=" << (void*)im2col_input << ", output=" << (void*)output << std::endl;
  std::cout << "DEBUG: batch_size=" << batch_size << ", channels=" << channels << std::endl;
  std::cout << "DEBUG: padded_height=" << padded_height << ", padded_width=" << padded_width << std::endl;
  std::cout << "DEBUG: stride_h=" << stride_h << ", output_h=" << output_h << ", output_w=" << output_w << std::endl;

  const int64_t patches_per_image = output_h * output_w;
  const int64_t kernel_size = 3 * 3;
  const int64_t K = channels * kernel_size;

  std::cout << "DEBUG: patches_per_image=" << patches_per_image << ", K=" << K << std::endl;

  if (stride_h == 1) {
    std::cout << "DEBUG: stride_h == 1 branch" << std::endl;
    if (tp != nullptr) {
      std::cout << "DEBUG: parallel execution with ThreadPool" << std::endl;
      const int64_t total_work_items = batch_size * channels;
      std::cout << "DEBUG: total_work_items=" << total_work_items << std::endl;

      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, total_work_items, [=](int64_t work_idx) {
            std::cout << "DEBUG: processing work_idx=" << work_idx << std::endl;
            const int64_t b = work_idx / channels;
            const int64_t c = work_idx % channels;
            std::cout << "DEBUG: b=" << b << ", c=" << c << std::endl;

            const int8_t* channel_data = im2col_input + b * channels * padded_height * padded_width +
                                         c * padded_height * padded_width;
            int8_t* batch_data_col = output + b * K * patches_per_image;

            std::cout << "DEBUG: channel_data=" << (void*)channel_data << ", batch_data_col=" << (void*)batch_data_col << std::endl;

            for (int64_t kh = 0; kh < 3; ++kh) {
              for (int64_t kw = 0; kw < 3; ++kw) {
                const int64_t k_index = c * kernel_size + kh * 3 + kw;
                int8_t* col_ptr = batch_data_col + k_index * patches_per_image;
                std::cout << "DEBUG: kh=" << kh << ", kw=" << kw << ", k_index=" << k_index << ", col_ptr=" << (void*)col_ptr << std::endl;

                if (kh < 2 || kw < 2) {
                  const int8_t* next_kernel_data = nullptr;
                  if (kw < 2) {
                    next_kernel_data = channel_data + (kh * padded_width + (kw + 1));
                  } else {
                    next_kernel_data = channel_data + ((kh + 1) * padded_width);
                  }
                  std::cout << "DEBUG: prefetch next_kernel_data=" << (void*)next_kernel_data << std::endl;
                  _mm_prefetch(reinterpret_cast<const char*>(next_kernel_data), _MM_HINT_T0);
                }

                for (int64_t h = 0; h < output_h; ++h) {
                  const int8_t* row_ptr = channel_data + (h + kh) * padded_width + kw;
                  int8_t* dest_ptr = col_ptr + h * output_w;
                  std::cout << "DEBUG: h=" << h << ", row_ptr=" << (void*)row_ptr << ", dest_ptr=" << (void*)dest_ptr << std::endl;

                  if (h + 1 < output_h) {
                    const int8_t* next_row = channel_data + (h + kh + 1) * padded_width + kw;
                    std::cout << "DEBUG: prefetch next_row=" << (void*)next_row << std::endl;
                    _mm_prefetch(
                        reinterpret_cast<const char*>(next_row),
                        _MM_HINT_T0);
                  }

                  std::cout << "DEBUG: memcpy size=" << (output_w * sizeof(int8_t)) << std::endl;
                  std::memcpy(dest_ptr, row_ptr, output_w * sizeof(int8_t));
                }
              }
            }
          });
    } else {
      std::cout << "DEBUG: sequential execution without ThreadPool" << std::endl;
    }
    return;
  }

  std::cout << "DEBUG: stride_h != 1 branch" << std::endl;
  constexpr int64_t TILE_H = 64;
  constexpr int64_t TILE_W = 64;

  auto process_channel_tile = [=](int64_t b, int64_t c, int64_t h_start, int64_t w_start) {
    std::cout << "DEBUG: process_channel_tile b=" << b << ", c=" << c << ", h_start=" << h_start << ", w_start=" << w_start << std::endl;

    const int64_t h_end = std::min(h_start + TILE_H, output_h);
    const int64_t w_end = std::min(w_start + TILE_W, output_w);
    const int8_t* channel_input = im2col_input + b * channels * padded_height * padded_width +
                                  c * padded_height * padded_width;

    for (int64_t kh = 0; kh < 3; ++kh) {
      for (int64_t kw = 0; kw < 3; ++kw) {
        const int64_t k_index = c * kernel_size + kh * 3 + kw;
        const int64_t output_k_offset = (b * K + k_index) * patches_per_image;
        std::cout << "DEBUG: tile kh=" << kh << ", kw=" << kw << ", k_index=" << k_index << ", output_k_offset=" << output_k_offset << std::endl;

        for (int64_t h_idx = h_start; h_idx < h_end; ++h_idx) {
          const int64_t h_input = h_idx * stride_h + kh;
          const int8_t* input_row = channel_input + h_input * padded_width + w_start * stride_h + kw;
          int8_t* output_base = output + output_k_offset + h_idx * output_w + w_start;
          const int64_t w_count = w_end - w_start;

          std::cout << "DEBUG: tile h_idx=" << h_idx << ", h_input=" << h_input << ", input_row=" << (void*)input_row
                    << ", output_base=" << (void*)output_base << ", w_count=" << w_count << std::endl;

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
    std::cout << "DEBUG: tiled parallel execution" << std::endl;
    const int64_t num_h_tiles = (output_h + TILE_H - 1) / TILE_H;
    const int64_t num_w_tiles = (output_w + TILE_W - 1) / TILE_W;
    const bool paralellize_channels = (channels > 8) && (batch_size * num_h_tiles * num_w_tiles < 32);

    std::cout << "DEBUG: num_h_tiles=" << num_h_tiles << ", num_w_tiles=" << num_w_tiles
              << ", paralellize_channels=" << paralellize_channels << std::endl;

    if (paralellize_channels) {
      std::cout << "DEBUG: parallelizing channels" << std::endl;
      const int64_t total_work = batch_size * channels * num_h_tiles * num_w_tiles;
      std::cout << "DEBUG: total_work=" << total_work << std::endl;

      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, total_work, [=](int64_t work_idx) {
            std::cout << "DEBUG: tile work_idx=" << work_idx << std::endl;
            const int64_t bc_part = work_idx / (num_h_tiles * num_w_tiles);
            const int64_t b = bc_part / channels;
            const int64_t c = bc_part % channels;
            const int64_t tile_idx = work_idx % (num_h_tiles * num_w_tiles);
            const int64_t h_tile = tile_idx / num_w_tiles;
            const int64_t w_tile = tile_idx % num_w_tiles;

            std::cout << "DEBUG: tile b=" << b << ", c=" << c << ", h_tile=" << h_tile << ", w_tile=" << w_tile << std::endl;

            process_channel_tile(b, c, h_tile * TILE_H, w_tile * TILE_W);
          });
    } else {
      std::cout << "DEBUG: parallelizing tiles" << std::endl;
      const int64_t total_tiles = batch_size * num_h_tiles * num_w_tiles;
      std::cout << "DEBUG: total_tiles=" << total_tiles << std::endl;

      onnxruntime::concurrency::ThreadPool::TrySimpleParallelFor(
          tp, total_tiles, [=](int64_t work_index) {
            std::cout << "DEBUG: tile batch work_index=" << work_index << std::endl;
            const int64_t b = work_index / (num_h_tiles * num_w_tiles);
            const int64_t tile_idx = work_index % (num_h_tiles * num_w_tiles);
            const int64_t h_tile = tile_idx / num_w_tiles;
            const int64_t w_tile = tile_idx % num_w_tiles;

            std::cout << "DEBUG: tile batch b=" << b << ", h_tile=" << h_tile << ", w_tile=" << w_tile << std::endl;

            for (int64_t c = 0; c < channels; ++c) {
              std::cout << "DEBUG: tile batch c=" << c << std::endl;
              process_channel_tile(b, c, h_tile * TILE_H, w_tile * TILE_W);
            }
          });
    }
  } else {
    std::cout << "DEBUG: sequential tiled execution" << std::endl;
  }

  std::cout << "DEBUG: im2col_3x3 completed" << std::endl;
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

// Implementazione im2col 1x1 con padding on-the-fly per stride 1
void im2col_1x1_stride1_with_padding_avx2(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    const std::vector<int64_t>& pads,
    int8_t input_zp,
    onnxruntime::concurrency::ThreadPool* tp = nullptr) {
  const int kernel_h = 1;
  const int kernel_w = 1;
  const int kernel_size = kernel_h * kernel_w;
  const int64_t padded_height = height + pads[0] + pads[2];
  const int64_t padded_width = width + pads[1] + pads[3];
  const int64_t output_h = padded_height - kernel_h + 1;  // stride = 1
  const int64_t output_w = padded_width - kernel_w + 1;   // stride = 1
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  // Vettori per il padding utilizzati nelle istruzioni SIMD
  __m256i pad_vector = _mm256_set1_epi8(input_zp);
  __m128i pad_vector_128 = _mm_set1_epi8(input_zp);

  // Parallelizziamo su batch e canali
  auto process_bc = [&](std::ptrdiff_t bc_idx) {
    const int64_t b = bc_idx / channels;
    const int64_t c = bc_idx % channels;

    // Per 1x1 kernel, abbiamo solo una posizione (kh=0, kw=0)
    const int64_t kh = 0;
    const int64_t kw = 0;
    const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
    const int64_t output_k_offset = (b * K + k_index) * patches_per_image;
    const int64_t input_offset = (b * channels + c) * height * width;

    for (int64_t h = 0; h < output_h; ++h) {
      const int64_t h_input = h + kh - pads[0];  // stride = 1
      int8_t* output_row = output + output_k_offset + h * output_w;

      if (h_input < 0 || h_input >= height) {
        // Se la riga è fuori dai limiti, riempiamo con input_zp
        int64_t w = 0;
        // Utilizzo di AVX2 per riempire 32 elementi alla volta quando possibile
        for (; w + 32 <= output_w; w += 32) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          _mm256_storeu_si256((__m256i*)(output_row + w + 16), pad_vector);
        }
        if (w + 16 <= output_w) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          w += 16;
        }
        for (; w < output_w; ++w) {
          output_row[w] = input_zp;
        }
      } else {
        const int8_t* input_row = input + input_offset + h_input * width;

        // Calcolo limiti di padding orizzontale
        const int64_t w_start = std::max<int64_t>(0, pads[1] - kw);
        const int64_t w_end = std::min(output_w, width - kw + pads[1]);

        // Padding a sinistra
        int64_t w = 0;
        for (; w + 32 <= w_start; w += 32) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          _mm256_storeu_si256((__m256i*)(output_row + w + 16), pad_vector);
        }
        if (w + 16 <= w_start) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          w += 16;
        }
        for (; w < w_start; ++w) {
          output_row[w] = input_zp;
        }

        // Regione centrale con dati effettivi - possiamo usare memcpy per stride=1
        if (w_end > w) {
          std::memcpy(output_row + w, input_row + w - w_start, (w_end - w) * sizeof(int8_t));
        }
        w = w_end;

        // Padding a destra
        for (; w + 32 <= output_w; w += 32) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          _mm256_storeu_si256((__m256i*)(output_row + w + 16), pad_vector);
        }
        if (w + 16 <= output_w) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          w += 16;
        }
        for (; w < output_w; ++w) {
          output_row[w] = input_zp;
        }
      }
    }
  };

  const int64_t total_bc = batch_size * channels;

  if (tp != nullptr) {
    // Utilizziamo parallelismo sui canali
    std::ptrdiff_t num_batches = std::min<std::ptrdiff_t>(total_bc, 16);
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        tp, total_bc, process_bc, num_batches);
  } else {
    // Esecuzione sequenziale
    for (int64_t bc_idx = 0; bc_idx < total_bc; ++bc_idx) {
      process_bc(bc_idx);
    }
  }
}

// Implementazione im2col 1x1 con padding on-the-fly per stride 2
void im2col_1x1_stride2_with_padding_avx2(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    const std::vector<int64_t>& pads,
    int8_t input_zp,
    onnxruntime::concurrency::ThreadPool* tp = nullptr) {
  const int kernel_h = 1;
  const int kernel_w = 1;
  const int kernel_size = kernel_h * kernel_w;
  const int64_t padded_height = height + pads[0] + pads[2];
  const int64_t padded_width = width + pads[1] + pads[3];
  const int64_t output_h = (padded_height - kernel_h) / 2 + 1;  // stride = 2
  const int64_t output_w = (padded_width - kernel_w) / 2 + 1;   // stride = 2
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  // Vettori per il padding utilizzati nelle istruzioni SIMD
  __m256i pad_vector = _mm256_set1_epi8(input_zp);
  __m128i pad_vector_128 = _mm_set1_epi8(input_zp);

  // Parallelizziamo su batch e canali
  auto process_bc = [&](std::ptrdiff_t bc_idx) {
    const int64_t b = bc_idx / channels;
    const int64_t c = bc_idx % channels;

    // Per 1x1 kernel, abbiamo solo una posizione (kh=0, kw=0)
    const int64_t kh = 0;
    const int64_t kw = 0;
    const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
    const int64_t output_k_offset = (b * K + k_index) * patches_per_image;
    const int64_t input_offset = (b * channels + c) * height * width;

    for (int64_t h = 0; h < output_h; ++h) {
      const int64_t h_input = h * 2 + kh - pads[0];  // stride = 2
      int8_t* output_row = output + output_k_offset + h * output_w;

      if (h_input < 0 || h_input >= height) {
        // Se la riga è fuori dai limiti, riempiamo con input_zp
        int64_t w = 0;
        // Utilizzo di AVX2 per riempire 32 elementi alla volta quando possibile
        for (; w + 32 <= output_w; w += 32) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          _mm256_storeu_si256((__m256i*)(output_row + w + 16), pad_vector);
        }
        if (w + 16 <= output_w) {
          _mm256_storeu_si256((__m256i*)(output_row + w), pad_vector);
          w += 16;
        }
        for (; w < output_w; ++w) {
          output_row[w] = input_zp;
        }
      } else {
        const int8_t* input_row = input + input_offset + h_input * width;

        // Prefetching per migliorare le performance
        if (h + 1 < output_h) {
          const int64_t next_h_input = (h + 1) * 2 + kh - pads[0];
          if (next_h_input >= 0 && next_h_input < height) {
            __builtin_prefetch(input + input_offset + next_h_input * width, 0, 0);
          }
        }

        // Calcolo limiti di padding per stride=2
        const int64_t w_left_padding_end = std::max<int64_t>(0, (pads[1] - kw + 1) / 2);
        const int64_t w_right_padding_start = std::min(output_w, (width + pads[1] - kw + 1) / 2);

        // Padding a sinistra
        int64_t w = 0;
        for (; w + 16 <= w_left_padding_end; w += 16) {
          _mm_storeu_si128((__m128i*)(output_row + w), pad_vector_128);
        }
        for (; w < w_left_padding_end; ++w) {
          output_row[w] = input_zp;
        }

        // Regione centrale con dati effettivi - dobbiamo fare strided access
        for (; w < w_right_padding_start; ++w) {
          const int64_t w_input = w * 2 + kw - pads[1];
          output_row[w] = (w_input >= 0 && w_input < width) ? input_row[w_input] : input_zp;
        }

        // Padding a destra
        for (; w + 16 <= output_w; w += 16) {
          _mm_storeu_si128((__m128i*)(output_row + w), pad_vector_128);
        }
        for (; w < output_w; ++w) {
          output_row[w] = input_zp;
        }
      }
    }
  };

  const int64_t total_bc = batch_size * channels;

  if (tp != nullptr) {
    // Utilizziamo parallelismo sui canali
    std::ptrdiff_t num_batches = std::min<std::ptrdiff_t>(total_bc, 16);
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        tp, total_bc, process_bc, num_batches);
  } else {
    // Esecuzione sequenziale
    for (int64_t bc_idx = 0; bc_idx < total_bc; ++bc_idx) {
      process_bc(bc_idx);
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
  // Validazione base dei parametri
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

  // Ripensamento completo del parallelismo: parallelizziamo per batch
  auto process_batch = [&](int64_t b) {
    if (b >= batch_size) return;

    // Elabora ogni canale
    for (int64_t c = 0; c < channels; ++c) {
      const int64_t input_channel_offset = (b * channels + c) * height * width;

      // Elabora ogni posizione del kernel
      for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
          const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
          const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

          // Calcola una volta per tutte i limiti di padding orizzontale
          const int64_t common_w_left_padding_end = std::max((int64_t)0, pads[1] - kw);
          const int64_t common_input_offset_w = kw - pads[1];

          // Ottimizzazione: prefetch della prossima riga
          for (int64_t h = 0; h < output_h; ++h) {
            const int64_t h_input = h + kh - pads[0];
            int8_t* output_ptr = output + output_k_offset + h * output_w;

            if (h_input < 0 || h_input >= height) {
              // Intera riga è padding - usiamo un approccio ottimizzato per il padding
              memset(output_ptr, pad_value, output_w);
            } else {
              const int8_t* input_row = input + input_channel_offset + h_input * width;
              const int64_t w_right_padding_start = std::min(output_w, width + pads[1] - kw);

              // Left padding
              if (common_w_left_padding_end > 0) {
                memset(output_ptr, pad_value, common_w_left_padding_end);
              }

              // Central region - memcpy quando possibile, elemento per elemento per casi limite
              if (common_w_left_padding_end < w_right_padding_start) {
                // Regione centrale (dato effettivo)
                const int8_t* input_ptr = input_row + common_w_left_padding_end + common_input_offset_w;
                const int64_t central_length = w_right_padding_start - common_w_left_padding_end;

                // Verifica se gli indici sono sicuri
                if (common_w_left_padding_end + common_input_offset_w >= 0 &&
                    common_w_left_padding_end + common_input_offset_w + central_length <= width) {
                  // Sicuro per memcpy - molto più veloce delle operazioni elemento per elemento
                  memcpy(output_ptr + common_w_left_padding_end, input_ptr, central_length);
                } else {
                  // Non è sicuro per memcpy - copia elemento per elemento con controlli di sicurezza
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

              // Right padding
              if (w_right_padding_start < output_w) {
                memset(output_ptr + w_right_padding_start, pad_value, output_w - w_right_padding_start);
              }

              // Prefetch della prossima riga se non siamo all'ultima
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

  // Parallelize by batch using ThreadPool or process sequentially
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
  // Precompute shuffle mask for stride 2
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
        // Vectorized padding
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

        // Left padding
        int64_t w = 0;
        for (; w + 16 <= w_left_padding_end; w += 16) {
          _mm_storeu_si128((__m128i*)(output_row + w), pad_vector_128);
        }
        for (; w < w_left_padding_end; ++w) {
          output_row[w] = pad_value;
        }

        // Central region with vectorized stride 2 access
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

        // Remaining elements in central region
        for (int64_t w_rem = central_start + w_vec; w_rem < central_end; ++w_rem) {
          const int64_t w_input = w_rem * 2 + kw - pads[1];
          output_row[w_rem] = input_row[w_input];
        }

        // Right padding
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

void im2col_7x7_stride1_with_padding_avx2(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    const std::vector<int64_t>& pads,
    int8_t input_zp,
    onnxruntime::concurrency::ThreadPool* tp = nullptr) {
  const int kernel_h = 7;
  const int kernel_w = 7;
  const int kernel_size = kernel_h * kernel_w;
  const int64_t padded_height = height + pads[0] + pads[2];
  const int64_t padded_width = width + pads[1] + pads[3];
  const int64_t output_h = padded_height - kernel_h + 1;
  const int64_t output_w = padded_width - kernel_w + 1;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  __m256i pad_vector = _mm256_set1_epi8(input_zp);
  constexpr int64_t TILE_H = 32;
  constexpr int64_t TILE_W = 32;

  auto process_bc = [&](std::ptrdiff_t bc_idx) {
    const int64_t b = bc_idx / channels;
    const int64_t c = bc_idx % channels;
    const int64_t input_offset = (b * channels + c) * height * width;

    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
        const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

        for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
          const int64_t h_end = std::min(h_start + TILE_H, output_h);

          for (int64_t h = h_start; h < h_end; ++h) {
            const int64_t h_input = h + kh - pads[0];
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
                output_row[w] = input_zp;
              }
            } else {
              const int8_t* input_row = input + input_offset + h_input * width;

              if (h + 1 < h_end) {
                const int64_t next_h_input = (h + 1) + kh - pads[0];
                if (next_h_input >= 0 && next_h_input < height) {
                  __builtin_prefetch(input + input_offset + next_h_input * width, 0, 0);
                }
              }

              for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
                const int64_t w_end = std::min(w_start + TILE_W, output_w);

                for (int64_t w = w_start; w < w_end; ++w) {
                  const int64_t w_input = w + kw - pads[1];
                  output_row[w] = (w_input >= 0 && w_input < width) ? input_row[w_input] : input_zp;
                }
              }
            }
          }
        }
      }
    }
  };

  const int64_t total_bc = batch_size * channels;

  if (tp != nullptr) {
    std::ptrdiff_t num_batches = std::min<std::ptrdiff_t>(total_bc, 16);
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        tp, total_bc, process_bc, num_batches);
  } else {
    for (int64_t bc_idx = 0; bc_idx < total_bc; ++bc_idx) {
      process_bc(bc_idx);
    }
  }
}

void im2col_7x7_stride2_with_padding_avx2(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    const std::vector<int64_t>& pads,
    int8_t input_zp,
    onnxruntime::concurrency::ThreadPool* tp = nullptr) {
  const int kernel_h = 7;
  const int kernel_w = 7;
  const int kernel_size = kernel_h * kernel_w;
  const int64_t padded_height = height + pads[0] + pads[2];
  const int64_t padded_width = width + pads[1] + pads[3];
  const int64_t output_h = (padded_height - kernel_h) / 2 + 1;
  const int64_t output_w = (padded_width - kernel_w) / 2 + 1;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t K = channels * kernel_size;

  __m256i pad_vector = _mm256_set1_epi8(input_zp);
  constexpr int64_t TILE_H = 32;
  constexpr int64_t TILE_W = 32;

  auto process_bc = [&](std::ptrdiff_t bc_idx) {
    const int64_t b = bc_idx / channels;
    const int64_t c = bc_idx % channels;
    const int64_t input_offset = (b * channels + c) * height * width;

    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        const int64_t k_index = c * kernel_size + kh * kernel_w + kw;
        const int64_t output_k_offset = (b * K + k_index) * patches_per_image;

        for (int64_t h_start = 0; h_start < output_h; h_start += TILE_H) {
          const int64_t h_end = std::min(h_start + TILE_H, output_h);

          for (int64_t h = h_start; h < h_end; ++h) {
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
                output_row[w] = input_zp;
              }
            } else {
              const int8_t* input_row = input + input_offset + h_input * width;

              if (h + 1 < h_end) {
                const int64_t next_h_input = (h + 1) * 2 + kh - pads[0];
                if (next_h_input >= 0 && next_h_input < height) {
                  __builtin_prefetch(input + input_offset + next_h_input * width, 0, 0);
                }
              }

              for (int64_t w_start = 0; w_start < output_w; w_start += TILE_W) {
                const int64_t w_end = std::min(w_start + TILE_W, output_w);

                for (int64_t w = w_start; w < w_end; ++w) {
                  const int64_t w_input = w * 2 + kw - pads[1];
                  output_row[w] = (w_input >= 0 && w_input < width) ? input_row[w_input] : input_zp;
                }
              }
            }
          }
        }
      }
    }
  };

  const int64_t total_bc = batch_size * channels;

  if (tp != nullptr) {
    std::ptrdiff_t num_batches = std::min<std::ptrdiff_t>(total_bc, 16);
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(
        tp, total_bc, process_bc, num_batches);
  } else {
    for (int64_t bc_idx = 0; bc_idx < total_bc; ++bc_idx) {
      process_bc(bc_idx);
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
    int8_t input_zp,
    onnxruntime::concurrency::ThreadPool* tp) {
  auto start = std::chrono::high_resolution_clock::now();

  if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 &&
      pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
    *output = const_cast<int8_t*>(input);
    return;
  }
  /*
  if (kernel_h == 1 && kernel_w == 1) {
    if (stride_h == 1) {
      std::cout << "Im2col kernel 1 stride 1 " << std::endl;
      im2col_1x1_stride1_with_padding_avx2(
          input,
          *output,
          batch_size,
          channels,
          height,
          width,
          pads,
          input_zp,
          tp);

      auto end_im2col = std::chrono::high_resolution_clock::now();
      auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
      return;
    } else if (stride_h == 2) {
      std::cout << "Im2col kernel 1 stride 2 " << std::endl;
      im2col_1x1_stride2_with_padding_avx2(
          input,
          *output,
          batch_size,
          channels,
          height,
          width,
          pads,
          input_zp,
          tp);
      auto end_im2col = std::chrono::high_resolution_clock::now();
      auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
      // std::cout << "Conv Op - Im2col 3x3 stride 2 execution time: " << duration_im2col.count() << " microseconds" << std::endl;
      return;
    }
  }
  */

  if (kernel_h == 3 && kernel_w == 3) {
    if (stride_h == 1) {
      std::cout << "Im2col kernel 3 stride 1 " << std::endl;
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

      auto end_im2col = std::chrono::high_resolution_clock::now();
      auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
      return;
    }

    if (stride_h == 2) {
      std::cout << "Im2col kernel 3 stride 2 " << std::endl;
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
      auto end_im2col = std::chrono::high_resolution_clock::now();
      auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
      // std::cout << "Conv Op - Im2col 3x3 stride 2 execution time: " << duration_im2col.count() << " microseconds" << std::endl;
      return;
    }
  }

  /*
    if (kernel_h == 7 && kernel_w == 7) {
      if (stride_h == 1) {
        im2col_7x7_stride1_with_padding_avx2(
            input,
            *output,
            batch_size,
            channels,
            height,
            width,
            pads,
            input_zp,
            tp);

        auto end_im2col = std::chrono::high_resolution_clock::now();
        auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
        return;
      } else if (stride_h == 2) {
        im2col_7x7_stride2_with_padding_avx2(
            input,
            *output,
            batch_size,
            channels,
            height,
            width,
            pads,
            input_zp,
            tp);
        auto end_im2col = std::chrono::high_resolution_clock::now();
        auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
        // std::cout << "Conv Op - Im2col 3x3 stride 2 execution time: " << duration_im2col.count() << " microseconds" << std::endl;
        return;
      }
    }
    */
  int64_t padded_height = height + pads[0] + pads[2];
  int64_t padded_width = width + pads[1] + pads[3];
  const int8_t* im2col_input = nullptr;

  if (pads[0] == 0 && pads[1] == 0 && pads[2] == 0 && pads[3] == 0) {
    im2col_input = input;
  } else {
    apply_padding(input, padded_buffer, batch_size, channels, height, width, pads, input_zp, tp);
    im2col_input = padded_buffer;
  }

  auto end_padding = std::chrono::high_resolution_clock::now();
  const int64_t output_h = (padded_height - kernel_h) / stride_h + 1;
  const int64_t output_w = (padded_width - kernel_w) / stride_h + 1;

  if (kernel_h == 7 && kernel_w == 7) {
    im2col_7x7(im2col_input, *output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else if (kernel_h == 3 && kernel_w == 3) {
    im2col_3x3(im2col_input, *output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else if (kernel_h == 1 && kernel_w == 1) {
    im2col_1x1(im2col_input, *output, batch_size, channels,
               padded_height, padded_width, stride_h,
               output_h, output_w, tp);
  } else {
    im2col_generic(im2col_input, *output, batch_size, channels,
                   padded_height, padded_width, kernel_h, kernel_w,
                   stride_h, output_h, output_w, tp);
  }
  auto end_im2col = std::chrono::high_resolution_clock::now();
  auto duration_padding = std::chrono::duration_cast<std::chrono::microseconds>(end_padding - start);
  auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
  // std::cout << "Conv Op - Total Padding execution time: " << duration_padding.count() << " microseconds" << std::endl;
  // std::cout << "Conv Op - Im2col execution time: " << duration_im2col.count() << " microseconds" << std::endl;
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
      conv_params->im2col_buffer.resize(new_N * K);
      im2col_ptr = conv_params->im2col_buffer.data();
      conv_params->padded_buffer.resize(actual_batch_size * input_channels * padded_height * padded_width);
      std::fill(conv_params->padded_buffer.begin(), conv_params->padded_buffer.end(), conv_params->input_zp);
      conv_params->batch_size = actual_batch_size;
    }
  }

  const int64_t required_im2col_size = actual_batch_size * conv_params->K * conv_params->output_height * conv_params->output_width;
  if (conv_params->im2col_buffer.size() != required_im2col_size) {
    conv_params->im2col_buffer.resize(required_im2col_size);
    im2col_ptr = conv_params->im2col_buffer.data();

    const int64_t padded_height = input_height + conv_params->pads[0] + conv_params->pads[2];
    const int64_t padded_width = input_width + conv_params->pads[1] + conv_params->pads[3];

    const int64_t required_padded_size = actual_batch_size * input_channels * padded_height * padded_width;
    if (conv_params->padded_buffer.size() != required_padded_size) {
      conv_params->padded_buffer.resize(required_padded_size);
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

  onnxruntime::nudgev::im2col(
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
      conv_params->input_zp,
      tp);

  auto end_im2col = std::chrono::high_resolution_clock::now();
  auto duration_im2col = std::chrono::duration_cast<std::chrono::microseconds>(end_im2col - start);
  auto start_gemm = std::chrono::high_resolution_clock::now();

  const int64_t patches_per_image = conv_params->output_height * conv_params->output_width;
  onnxruntime::nudgev::gemm_i8_after_im2col_xzp(
      conv_params->weights.data(),
      im2col_ptr,
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
  // std::cout << "Conv Op - Total Gemm execution time: " << duration_gemm.count() << " microseconds" << std::endl;

  return Status::OK();
}
}  // namespace nudgev
}  // namespace onnxruntime