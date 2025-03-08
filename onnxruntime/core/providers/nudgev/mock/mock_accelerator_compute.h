#pragma once

#include "core/providers/nudgev/mock/mock_accelerator_memory.h"
#include "core/platform/threadpool.h"
#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <thread>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <atomic>
#include <unistd.h>
#include <sys/select.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>

namespace onnxruntime {
namespace nudgev {
namespace mock {

#pragma pack(push, 1)
struct MatrixMultiplicationHeader {
  uint8_t xzp;
  uint8_t yzp;
  uint8_t m_lsb;
  uint8_t m_msb;
  uint8_t n_lsb;
  uint8_t n_msb;
  uint8_t k;
  uint8_t j_lsb;
  uint8_t j_msb;
  uint8_t batch_size;
  uint8_t reserved1;
  uint8_t reserved2;
};
#pragma pack(pop)

class AcceleratorHardwareSimulator {
 public:
  static AcceleratorHardwareSimulator& getInstance() {
    static AcceleratorHardwareSimulator instance;
    return instance;
  }

  void initialize(int read_fd, int write_fd, bool verbose = false) {
    read_fd_ = read_fd;
    write_fd_ = write_fd;
    verbose_ = verbose;
    should_stop_ = false;
    processing_thread_ = std::thread(&AcceleratorHardwareSimulator::processPackets, this);

    if (verbose_) {
      std::cout << "Hardware simulator initialized with FDs: read=" << read_fd_
                << ", write=" << write_fd_ << std::endl;
    }
  }

  void shutdown() {
    should_stop_ = true;
    if (processing_thread_.joinable())
      processing_thread_.join();
  }

 private:
  AcceleratorHardwareSimulator() : read_fd_(-1), write_fd_(-1), verbose_(false), should_stop_(false) {}
  ~AcceleratorHardwareSimulator() { shutdown(); }

  void processPackets() {
    if (read_fd_ < 0 || write_fd_ < 0) {
      std::cerr << "Error: invalid file descriptors" << std::endl;
      return;
    }

    const bool debug_mode = verbose_;
    if (debug_mode) {
      std::cout << "Hardware simulator started processing packets" << std::endl;
      std::cout << "File descriptors: read=" << read_fd_ << ", write=" << write_fd_ << std::endl;
    }

    while (!should_stop_) {
      fd_set read_fds;
      struct timeval tv;
      tv.tv_sec = 0;
      tv.tv_usec = 10000;

      FD_ZERO(&read_fds);
      FD_SET(read_fd_, &read_fds);

      int ready = select(read_fd_ + 1, &read_fds, NULL, NULL, &tv);
      if (ready <= 0) {
        if (ready < 0 && errno != EINTR) {
          std::cerr << "Error in select(): " << strerror(errno) << std::endl;
        }
        continue;
      }

      MatrixMultiplicationHeader header;
      ssize_t bytes_read = read(read_fd_, &header, sizeof(header));
      if (bytes_read != sizeof(header)) {
        if (bytes_read < 0) {
          std::cerr << "Error: failed to read packet header from host: " << strerror(errno) << std::endl;
        } else if (bytes_read == 0) {
          std::cerr << "Error: unexpected EOF while reading packet header" << std::endl;
        } else {
          std::cerr << "Error: incomplete packet header read (got " << bytes_read << " bytes)" << std::endl;
        }
        continue;
      }

      int64_t batch_idx;
      bytes_read = read(read_fd_, &batch_idx, sizeof(batch_idx));
      if (bytes_read != sizeof(batch_idx)) {
        std::cerr << "Error: failed to read batch index from host: " << strerror(errno) << std::endl;
        continue;
      }

      uint32_t data_size;
      bytes_read = read(read_fd_, &data_size, sizeof(data_size));
      if (bytes_read != sizeof(data_size)) {
        std::cerr << "Error: failed to read packet data size from host: " << strerror(errno) << std::endl;
        continue;
      }

      if (data_size == 0 || data_size > 100 * 1024 * 1024) {
        std::cerr << "Error: invalid data size: " << data_size << " bytes" << std::endl;
        continue;
      }

      if (debug_mode) {
        std::cout << "Accelerator received data packet:" << std::endl;
        std::cout << "  Header info: XZP=" << static_cast<int>(static_cast<int8_t>(header.xzp))
                  << ", YZP=" << static_cast<int>(static_cast<int8_t>(header.yzp))
                  << ", multiplier=" << ((header.m_msb << 8) | header.m_lsb) << std::endl;
        std::cout << "  Batch=" << batch_idx
                  << ", patches=" << ((header.n_msb << 8) | header.n_lsb)
                  << ", oc_blocks=" << (int)header.k
                  << ", j_blocks=" << (int)header.j_lsb << std::endl;
        std::cout << "  Batch size=" << (int)header.batch_size
                  << ", Reserved=" << (int)header.reserved1 << "," << (int)header.reserved2 << std::endl;
      }

      std::vector<int8_t> input_data(data_size);
      size_t bytes_received = 0;
      int read_attempts = 0;
      const int max_attempts = 10;

      while (bytes_received < data_size && read_attempts < max_attempts) {
        ssize_t result = read(read_fd_, input_data.data() + bytes_received, data_size - bytes_received);

        if (result <= 0) {
          if (result < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
              read_attempts++;
              continue;
            }

            std::cerr << "Error: failed to read packet data from host. Read " << bytes_received
                      << " of " << data_size << " bytes. Errno: " << errno
                      << " (" << strerror(errno) << ")" << std::endl;
          } else {
            std::cerr << "Error: unexpected EOF while reading packet data" << std::endl;
          }
          break;
        }

        bytes_received += result;
        if (debug_mode && bytes_received % (1024 * 1024) == 0) {
          std::cout << "    Read progress: " << bytes_received << "/" << data_size << " bytes" << std::endl;
        }
      }

      if (bytes_received != data_size) {
        std::cerr << "Error: incomplete read. Got " << bytes_received << " of " << data_size
                  << " bytes after " << read_attempts << " attempts" << std::endl;
        continue;
      }

      if (debug_mode) {
        std::cout << "  Successfully read all " << data_size << " bytes" << std::endl;
      }

      size_t weight_addr, bias_addr;
      if (read(read_fd_, &weight_addr, sizeof(weight_addr)) != sizeof(weight_addr) ||
          read(read_fd_, &bias_addr, sizeof(bias_addr)) != sizeof(bias_addr)) {
        std::cerr << "Error: failed to read addresses from host: " << strerror(errno) << std::endl;
        continue;
      }

      bool fused_relu;
      if (read(read_fd_, &fused_relu, sizeof(fused_relu)) != sizeof(fused_relu)) {
        std::cerr << "Error: failed to read fused_relu flag from host: " << strerror(errno) << std::endl;
        continue;
      }

      if (debug_mode) {
        std::cout << "  Addresses: weights=0x" << std::hex << weight_addr
                  << ", bias=0x" << bias_addr << std::dec << std::endl;
        std::cout << "  Fused ReLU: " << (fused_relu ? "yes" : "no") << std::endl;
      }

      uint16_t K_dim = (header.n_msb << 8) | header.n_lsb;
      uint8_t oc_blocks = header.k;
      uint16_t j_blocks = (header.j_msb << 8) | header.j_lsb;
      uint8_t batch_size = header.batch_size;
      const int64_t block_size = 32;
      int64_t patches = j_blocks * block_size;
      int16_t multiplier = (header.m_msb << 8) | header.m_lsb;
      int8_t input_zp = static_cast<int8_t>(header.xzp);
      int8_t output_zp = static_cast<int8_t>(header.yzp);

      if (K_dim == 0 || oc_blocks == 0 || j_blocks == 0) {
        std::cerr << "Error: invalid dimensions in header" << std::endl;
        continue;
      }

      if (debug_mode) {
        std::cout << "  Header info: XZP=" << static_cast<int>(input_zp)
                  << ", YZP=" << static_cast<int>(output_zp)
                  << ", multiplier=" << multiplier << std::endl;
        std::cout << "  Batch=" << batch_idx
                  << ", N (features)=" << K_dim
                  << ", oc_blocks=" << (int)oc_blocks
                  << ", j_blocks=" << j_blocks << std::endl;
        std::cout << "  Batch size=" << (int)batch_size
                  << ", Reserved=" << (int)header.reserved1 << "," << (int)header.reserved2 << std::endl;
      }

      auto& mem_manager = AcceleratorMemoryManager::getInstance();
      int64_t total_blocks = j_blocks * batch_size;
      int64_t OC = oc_blocks * block_size;
      int64_t k_blocks = (K_dim + block_size - 1) / block_size;

      if (debug_mode) {
        std::cout << "  Received data size: " << data_size << " bytes" << std::endl;
        std::cout << "  Total_blocks: " << total_blocks << " (batch_size=" << (int)batch_size
                  << ", j_blocks=" << j_blocks << ")" << std::endl;
        std::cout << "  K from the header: " << K_dim << std::endl;
        std::cout << "  K Blocks Required: " << k_blocks << " (rounded to blocks of " << block_size << ")" << std::endl;
      }

      if (debug_mode) {
        std::cout << "  Processing " << j_blocks << " blocks per batch, " << batch_size << " batches" << std::endl;
        std::cout << "  K dimension: " << K_dim << ", OC dimension: " << OC << std::endl;
        std::cout << "  OC blocks: " << (int)oc_blocks << ", total blocks: " << total_blocks << std::endl;
      }

      // ============================================================================
      // PHASE 1: Reconstruct the complete input matrix from blocks (K×patches)
      // ============================================================================

      if (debug_mode) {
        std::cout << " PHASE 1: Input matrix reconstruction " << K_dim << "×" << patches << std::endl;
      }

      std::vector<int8_t> input_matrix(K_dim * patches, 0);
      for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t j = 0; j < j_blocks; j++) {
          int64_t patch_base = j * block_size;
          if (patch_base >= patches) continue;
          int64_t valid_patches = std::min(block_size, patches - patch_base);
          int64_t block_idx = b * j_blocks + j;
          size_t block_offset = block_idx * K_dim * block_size;
          for (int64_t k = 0; k < K_dim; k++) {
            const int8_t* src_ptr = input_data.data() + block_offset + k * block_size;
            int8_t* dst_ptr = input_matrix.data() + k * patches + patch_base;
            std::memcpy(dst_ptr, src_ptr, valid_patches * sizeof(int8_t));
          }
        }
      }

      // ============================================================================
      // PHASE 2: Transpose the input matrix (K×patches -> patches×K)
      // ============================================================================

      if (debug_mode) {
        std::cout << " PHASE 2: Transpose input matrix" << K_dim << "×" << patches
                  << " -> " << patches << "×" << K_dim << std::endl;
      }

      std::vector<int8_t> input_transposed(patches * K_dim, 0);
      for (int64_t k = 0; k < K_dim; k++) {
        for (int64_t p = 0; p < patches; p++) {
          size_t src_idx = k * patches + p;
          size_t dst_idx = p * K_dim + k;
          input_transposed[dst_idx] = input_matrix[src_idx];
        }
      }
      input_matrix.clear();
      input_matrix.shrink_to_fit();

      // ============================================================================
      // PHASE 3: Loading and weight reconstruction (K×OC)
      // ============================================================================

      if (debug_mode) {
        std::cout << "  PHASE 3: Loading and weight reconstruction " << K_dim << "×" << OC << std::endl;
        std::cout << "  Weight address in memory: 0x" << std::hex << weight_addr << std::dec << std::endl;
        std::cout << "  Blocks to read: " << k_blocks << "×" << oc_blocks << " (blocks from " << block_size << "×" << block_size << ")" << std::endl;
      }
      size_t total_weight_size = k_blocks * oc_blocks * block_size * block_size;
      std::vector<int8_t> blocked_weights(total_weight_size, 0);
      if (!mem_manager.read_from_memory(weight_addr, blocked_weights.data(), total_weight_size)) {
        std::cerr << "Error: failed to read weights from memory at address 0x"
                  << std::hex << weight_addr << std::dec
                  << ", size " << total_weight_size << " bytes" << std::endl;
        continue;
      }

      if (debug_mode) {
        std::cout << " Weights successfully read from memory" << std::endl;
      }
      std::vector<int8_t> weight_matrix(K_dim * OC, 0);

      for (int64_t kb = 0; kb < k_blocks; kb++) {
        for (int64_t ocb = 0; ocb < oc_blocks; ocb++) {
          int64_t k_base = kb * block_size;
          int64_t oc_base = ocb * block_size;
          int64_t block_idx = kb * oc_blocks + ocb;
          size_t block_offset = block_idx * block_size * block_size;
          int64_t valid_k = std::min(block_size, K_dim - k_base);
          int64_t valid_oc = std::min(block_size, OC - oc_base);
          for (int64_t k_local = 0; k_local < valid_k; k_local++) {
            for (int64_t oc_local = 0; oc_local < valid_oc; oc_local++) {
              size_t src_idx = block_offset + k_local * block_size + oc_local;
              int64_t k_global = k_base + k_local;
              int64_t oc_global = oc_base + oc_local;
              size_t dst_idx = k_global * OC + oc_global;

              if (dst_idx < weight_matrix.size() && src_idx < blocked_weights.size()) {
                weight_matrix[dst_idx] = blocked_weights[src_idx];
              } else if (debug_mode) {
                std::cerr << " Warning: index out of weight limits - src:" << src_idx
                          << " (max: " << blocked_weights.size() << "), dst: " << dst_idx
                          << " (max: " << weight_matrix.size() << ")" << std::endl;
              }
            }
          }
        }
      }
      blocked_weights.clear();
      blocked_weights.shrink_to_fit();

      // ============================================================================
      // PHASE 4: Reading biases (if any)
      // ============================================================================

      std::vector<int32_t> bias;
      if (bias_addr != 0) {
        if (debug_mode) {
          std::cout << "  PHASE 4: Bias reading" << std::endl;
          std::cout << "  Bias address in memory: 0x" << std::hex << bias_addr << std::dec << std::endl;
          std::cout << "  Blocks to read: " << oc_blocks << " (blocks of " << block_size << " elements)" << std::endl;
        }
        size_t bias_block_size = oc_blocks * block_size;
        std::vector<int32_t> blocked_bias(bias_block_size, 0);

        if (!mem_manager.read_from_memory(bias_addr, blocked_bias.data(), bias_block_size * sizeof(int32_t))) {
          std::cerr << "Error: failed to read bias from memory at address 0x"
                    << std::hex << bias_addr << std::dec
                    << ", size " << bias_block_size * sizeof(int32_t) << " bytes" << std::endl;
        } else {
          if (debug_mode) {
            std::cout << " Bias successfully read from memory" << std::endl;
          }

          bias.resize(OC, 0);
          for (int64_t ocb = 0; ocb < oc_blocks; ocb++) {
            int64_t oc_base = ocb * block_size;
            int64_t valid_oc = std::min(block_size, OC - oc_base);
            for (int64_t oc = 0; oc < valid_oc; oc++) {
              int64_t oc_global = oc_base + oc;
              size_t src_idx = ocb * block_size + oc;
              if (oc_global < OC && src_idx < blocked_bias.size()) {
                bias[oc_global] = blocked_bias[src_idx];
              } else if (debug_mode) {
                std::cerr << " Warning: index out of bounds for bias - src: " << src_idx
                          << " (max: " << blocked_bias.size() << "), dst: " << oc_global
                          << " (max: " << OC << ")" << std::endl;
              }
            }
          }
        }
      }

      // ============================================================================
      // PHASE 5: Execute GEMM (patches×K) × (K×OC) -> (patches×OC)
      // ============================================================================

      if (debug_mode) {
        std::cout << "  FASE 5: Gemm Execution ("
                  << patches << "×" << K_dim << ") × ("
                  << K_dim << "×" << OC << ") -> ("
                  << patches << "×" << OC << ")" << std::endl;
      }

      auto processing_start = std::chrono::high_resolution_clock::now();
      std::vector<int8_t> output_transposed(patches * OC, 0);
      for (int64_t p = 0; p < patches; p++) {
        for (int64_t oc = 0; oc < OC; oc++) {
          int32_t acc = 0;
          for (int64_t k = 0; k < K_dim; k++) {
            int32_t input_val = static_cast<int32_t>(input_transposed[p * K_dim + k]) - input_zp;
            int32_t weight_val = static_cast<int32_t>(weight_matrix[k * OC + oc]);
            acc += input_val * weight_val;
          }
          if (!bias.empty() && oc < static_cast<int64_t>(bias.size())) {
            acc += bias[oc];
          }
          int32_t scaled = static_cast<int32_t>((static_cast<int64_t>(acc) * multiplier + 0x4000) >> 15);
          if (fused_relu && scaled < 0) {
            scaled = 0;
          }
          scaled += output_zp;
          scaled = std::min(127, std::max(-128, scaled));
          output_transposed[p * OC + oc] = static_cast<int8_t>(scaled);
        }
      }
      input_transposed.clear();
      input_transposed.shrink_to_fit();
      weight_matrix.clear();
      weight_matrix.shrink_to_fit();

      auto processing_mid = std::chrono::high_resolution_clock::now();
      auto gemm_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           processing_mid - processing_start)
                           .count();

      if (debug_mode) {
        std::cout << "  GEMM completed in " << gemm_time << " microseconds" << std::endl;
      }

      // ============================================================================
      // PHASE 6: Output Transposition (patches×OC -> OC×patches)
      // ============================================================================

      if (debug_mode) {
        std::cout << "  PHASE 6: Output Transposition "
                  << patches << "×" << OC << " -> "
                  << OC << "×" << patches << std::endl;
      }
      std::vector<int8_t> output_matrix(OC * patches);
      for (int64_t p = 0; p < patches; p++) {
        for (int64_t oc = 0; oc < OC; oc++) {
          size_t src_idx = p * OC + oc;
          size_t dst_idx = oc * patches + p;
          output_matrix[dst_idx] = output_transposed[src_idx];
        }
      }
      output_transposed.clear();
      output_transposed.shrink_to_fit();

      // ============================================================================
      // PHASE 7: Creating output blocks for sending (OC×patches -> blocks)
      // ============================================================================

      if (debug_mode) {
        std::cout << "  PHASE 7: Creating output blocks" << std::endl;
      }

      size_t output_block_size = block_size * block_size;
      size_t total_output_size = total_blocks * oc_blocks * output_block_size;
      std::vector<int8_t> blocked_output(total_output_size, 0);
      std::fill(blocked_output.begin(), blocked_output.end(), 0);
      int64_t max_j_blocks = std::min(static_cast<int64_t>(j_blocks), (patches + block_size - 1) / block_size);
      for (int64_t b = 0; b < batch_size; b++) {
        for (int64_t j = 0; j < max_j_blocks; j++) {
          int64_t patch_base = j * block_size;
          if (patch_base >= patches) continue;
          int64_t valid_patches = std::min(block_size, patches - patch_base);
          for (int64_t ocb = 0; ocb < oc_blocks; ocb++) {
            int64_t oc_base = ocb * block_size;
            int64_t valid_oc = std::min(block_size, OC - oc_base);
            int64_t block_idx = (b * j_blocks + j) * oc_blocks + ocb;
            if (block_idx * output_block_size >= blocked_output.size()) {
              continue;
            }
            size_t block_offset = block_idx * output_block_size;
            for (int64_t oc_local = 0; oc_local < valid_oc; oc_local++) {
              int64_t oc_global = oc_base + oc_local;
              for (int64_t p_local = 0; p_local < valid_patches; p_local++) {
                int64_t p_global = patch_base + p_local;
                size_t src_idx = oc_global * patches + p_global;
                size_t dst_idx = block_offset + oc_local * block_size + p_local;
                if (dst_idx < blocked_output.size() && src_idx < output_matrix.size()) {
                  blocked_output[dst_idx] = output_matrix[src_idx];
                }
              }
            }
          }
        }
      }

      auto processing_end = std::chrono::high_resolution_clock::now();
      auto total_processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                       processing_end - processing_start)
                                       .count();

      if (debug_mode) {
        std::cout << "  Processing completed in " << total_processing_time << " microseconds" << std::endl;
      }

      // ============================================================================
      // PHASE 8: Sending results to host
      // ============================================================================

      if (debug_mode) {
        std::cout << "  PHASE 8: Sending results to host" << std::endl;
      }
      ssize_t bytes_sent = write(write_fd_, &batch_idx, sizeof(batch_idx));
      if (bytes_sent != sizeof(batch_idx)) {
        std::cerr << "Error: failed to write batch index to host, errno="
                  << errno << " (" << strerror(errno) << ")" << std::endl;
        continue;
      }
      uint32_t result_size = blocked_output.size();
      bytes_sent = write(write_fd_, &result_size, sizeof(result_size));
      if (bytes_sent != sizeof(result_size)) {
        std::cerr << "Error: failed to write result size to host, errno="
                  << errno << " (" << strerror(errno) << ")" << std::endl;
        continue;
      }
      size_t bytes_written = 0;
      int write_attempts = 0;
      const int max_write_attempts = 10;
      while (bytes_written < result_size && write_attempts < max_write_attempts) {
        ssize_t result = write(write_fd_, blocked_output.data() + bytes_written, result_size - bytes_written);
        if (result <= 0) {
          if (result < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
              std::this_thread::sleep_for(std::chrono::milliseconds(10));
              write_attempts++;
              continue;
            }
            std::cerr << "Error: failed to write output data to host, errno="
                      << errno << " (" << strerror(errno) << ")" << std::endl;
          } else {
            std::cerr << "Error: unexpected condition writing output data" << std::endl;
          }
          break;
        }

        bytes_written += result;
        if (debug_mode && bytes_written % (1024 * 1024) == 0) {
          std::cout << "    Write progress: " << bytes_written << "/" << result_size << " bytes" << std::endl;
        }
      }

      if (bytes_written != result_size) {
        std::cerr << "Error: incomplete write. Sent " << bytes_written << " of " << result_size
                  << " bytes after " << write_attempts << " attempts" << std::endl;
        continue;
      }

      if (debug_mode) {
        std::cout << "  Successfully sent all " << result_size << " bytes back to host" << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  int read_fd_;
  int write_fd_;
  bool verbose_;
  std::atomic<bool> should_stop_;
  std::thread processing_thread_;
};

inline MatrixMultiplicationHeader create_header(
    int8_t input_zp,
    int8_t output_zp,
    int16_t multiplier,
    uint16_t n_vectors,
    uint8_t oc_blocks,
    uint16_t j_tiles,
    uint8_t batch_size) {
  MatrixMultiplicationHeader header;
  header.xzp = static_cast<uint8_t>(input_zp);
  header.yzp = static_cast<uint8_t>(output_zp);
  header.m_lsb = multiplier & 0xFF;
  header.m_msb = (multiplier >> 8) & 0xFF;
  header.n_lsb = n_vectors & 0xFF;
  header.n_msb = (n_vectors >> 8) & 0xFF;
  header.k = oc_blocks;
  header.j_lsb = j_tiles & 0xFF;
  header.j_msb = (j_tiles >> 8) & 0xFF;
  header.batch_size = batch_size;
  header.reserved1 = 0;
  header.reserved2 = 0;
  return header;
}

}  // namespace mock
}  // namespace nudgev
}  // namespace onnxruntime