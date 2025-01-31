#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <stdexcept>
#include <chrono>

template <int KH, int KW>
void im2row_kconst(
    const int32_t* input,
    int32_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t stride) {
  const int64_t output_h = (height - KH) / stride + 1;
  const int64_t output_w = (width - KW) / stride + 1;
  const int64_t patches_per_image = output_h * output_w;
  const int64_t patch_size = KH * KW;

  for (int64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int32_t* batch_input = input + batch_idx * channels * height * width;
    int32_t* batch_output = output + batch_idx * channels * patches_per_image * patch_size;

    for (int64_t h = 0; h < output_h; h++) {
      const int64_t h_offset = h * stride;

      for (int64_t w = 0; w < output_w; w++) {
        const int64_t w_offset = w * stride;
        const int64_t window_idx = (h * output_w + w) * (channels * patch_size);

        for (int64_t c = 0; c < channels; c++) {
          const int32_t* channel_input = batch_input + c * height * width;
          const int64_t patch_offset = window_idx + (c * patch_size);

          for (int64_t kh = 0; kh < KH; kh++) {
            const int64_t input_row = (h_offset + kh) * width + w_offset;
            const int64_t output_row = patch_offset + kh * KW;
            std::memcpy(
                batch_output + output_row,
                channel_input + input_row,
                KW * sizeof(int32_t));
          }
        }
      }
    }
  }
}

void im2row(
    const int8_t* input,
    int32_t* output,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    const std::vector<int64_t>& pads,
    int8_t x_zero_point) {
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<int32_t> input_centered(batch_size * channels * height * width);
  for (size_t i = 0; i < input_centered.size(); ++i) {
    input_centered[i] = static_cast<int32_t>(input[i]) - x_zero_point;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Conversion time: " << duration.count() << " microseconds" << std::endl;

  std::vector<int32_t> input_padded;
  int64_t padded_height = height;
  int64_t padded_width = width;

  if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
    padded_height = height + pads[0] + pads[2];
    padded_width = width + pads[1] + pads[3];
    input_padded.resize(batch_size * channels * padded_height * padded_width, 0);

    for (int64_t n = 0; n < batch_size; ++n) {
      for (int64_t c = 0; c < channels; ++c) {
        for (int64_t h = 0; h < height; ++h) {
          const int64_t h_pad = h + pads[0];
          for (int64_t w = 0; w < width; ++w) {
            const int64_t w_pad = w + pads[1];
            const int64_t in_idx = ((n * channels + c) * height + h) * width + w;
            const int64_t out_idx = ((n * channels + c) * padded_height + h_pad) * padded_width + w_pad;
            input_padded[out_idx] = input_centered[in_idx];
          }
        }
      }
    }
  } else {
    input_padded = std::move(input_centered);
  }

  if (kernel_h == 1 && kernel_w == 1) {
    im2row_kconst<1, 1>(input_padded.data(), output, batch_size, channels, padded_height, padded_width, stride_h);
  } else if (kernel_h == 3 && kernel_w == 3) {
    im2row_kconst<3, 3>(input_padded.data(), output, batch_size, channels, padded_height, padded_width, stride_h);
  } else if (kernel_h == 7 && kernel_w == 7) {
    im2row_kconst<7, 7>(input_padded.data(), output, batch_size, channels, padded_height, padded_width, stride_h);
  } else {
    throw std::runtime_error("Unsupported kernel size");
  }
}