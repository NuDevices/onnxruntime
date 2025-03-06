#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <string>
#include <algorithm>

namespace onnxruntime {
namespace nudgev {

// FPGA Matrix Multiplication Header Structure (16 bytes total)
struct FpgaMatrixMultHeader {
  uint8_t xzp;              // X zero point (input zero point)
  uint8_t yzp;              // Y zero point (weight zero point)
  uint16_t m;               // Multiplier for scaling
  uint16_t n;               // Number of vectors in tile
  uint8_t k;                // Number of weight tiles
  uint8_t reserved1;        // Reserved for future use
  uint8_t batch_size;       // Size of batch
  uint32_t weights_offset;  // Offset to weights data
  uint8_t reserved2[3];     // Reserved for future use
};

// Function to prepare data packet for FPGA
std::vector<uint8_t> prepare_fpga_data_packet(
    const int8_t* input_data,
    const int8_t* weights_data,
    int64_t batch_size,
    int64_t n_dimension,
    int64_t k_dimension,
    int8_t input_zp,
    int8_t weight_zp,
    uint16_t multiplier,
    size_t input_size,
    size_t weights_size) {
  // Calculate total packet size
  const size_t header_size = sizeof(FpgaMatrixMultHeader);
  const size_t total_size = header_size + input_size + weights_size;

  // Create data packet buffer
  std::vector<uint8_t> packet(total_size);

  // Setup header
  FpgaMatrixMultHeader header;
  header.xzp = static_cast<uint8_t>(input_zp);
  header.yzp = static_cast<uint8_t>(weight_zp);
  header.m = multiplier;
  header.n = static_cast<uint16_t>(n_dimension);
  header.k = static_cast<uint8_t>(k_dimension > 255 ? 255 : k_dimension);  // Clamp to valid range
  header.reserved1 = 0;
  header.batch_size = static_cast<uint8_t>(batch_size > 255 ? 255 : batch_size);  // Clamp to valid range
  header.weights_offset = header_size + input_size;
  memset(header.reserved2, 0, sizeof(header.reserved2));

  // Copy header to packet
  memcpy(packet.data(), &header, header_size);

  // Copy input data
  memcpy(packet.data() + header_size, input_data, input_size);

  // Copy weights data
  memcpy(packet.data() + header.weights_offset, weights_data, weights_size);

  return packet;
}

// Function to split data into tiles for FPGA processing
std::vector<std::vector<uint8_t>> prepare_fpga_data_tiles(
    const int8_t* input_data,
    const int8_t* weights_data,
    int64_t batch_size,
    int64_t n_dimension,  // Output channels
    int64_t k_dimension,  // Input channels * kernel_h * kernel_w
    int8_t input_zp,
    int8_t weight_zp,
    uint16_t multiplier,
    int64_t max_tile_size = 65536) {
  // Constants for tiling
  const int64_t MAX_N_PER_TILE = 256;
  const int64_t MAX_K_PER_TILE = 255;  // Limited by header field size

  std::vector<std::vector<uint8_t>> tiles;

  // Calculate how many tiles we need
  const int64_t n_tiles = (n_dimension + MAX_N_PER_TILE - 1) / MAX_N_PER_TILE;
  const int64_t k_tiles = (k_dimension + MAX_K_PER_TILE - 1) / MAX_K_PER_TILE;

  for (int64_t n_tile = 0; n_tile < n_tiles; n_tile++) {
    // Calculate N dimension for this tile
    const int64_t n_start = n_tile * MAX_N_PER_TILE;
    const int64_t n_count = std::min(MAX_N_PER_TILE, n_dimension - n_start);

    for (int64_t k_tile = 0; k_tile < k_tiles; k_tile++) {
      // Calculate K dimension for this tile
      const int64_t k_start = k_tile * MAX_K_PER_TILE;
      const int64_t k_count = std::min(MAX_K_PER_TILE, k_dimension - k_start);

      // Calculate sizes for this tile
      const size_t input_tile_size = batch_size * k_count * sizeof(int8_t);
      const size_t weights_tile_size = n_count * k_count * sizeof(int8_t);

      // Extract the relevant portions of input and weights
      std::vector<int8_t> tiled_input(batch_size * k_count);
      std::vector<int8_t> tiled_weights(n_count * k_count);

      // Copy input data for this tile
      for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t k = 0; k < k_count; k++) {
          tiled_input[b * k_count + k] =
              input_data[b * k_dimension + (k_start + k)];
        }
      }

      // Copy weights data for this tile
      for (int64_t n = 0; n < n_count; n++) {
        for (int64_t k = 0; k < k_count; k++) {
          tiled_weights[n * k_count + k] =
              weights_data[(n_start + n) * k_dimension + (k_start + k)];
        }
      }

      // Create packet for this tile
      auto tile_packet = prepare_fpga_data_packet(
          tiled_input.data(),
          tiled_weights.data(),
          batch_size,
          n_count,
          k_count,
          input_zp,
          weight_zp,
          multiplier,
          input_tile_size,
          weights_tile_size);

      tiles.push_back(std::move(tile_packet));
    }
  }

  return tiles;
}

}  // namespace nudgev
}  // namespace onnxruntime