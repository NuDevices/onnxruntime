#pragma once

#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <cstdint>
#include <iomanip>

namespace onnxruntime {
namespace nudgev {
namespace mock {

class AcceleratorMemory {
 public:
  AcceleratorMemory(size_t memory_size = 1024 * 1024 * 1024) {
    memory_.resize(memory_size, 0);
    std::cout << "Initialized accelerator DDR3 memory: " << memory_size << " bytes" << std::endl;
  }
  bool write_data(size_t address, const void* data, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (address % 8 != 0) {
      std::cerr << "Error: address not aligned to 8 bytes: 0x"
                << std::hex << address << std::dec << std::endl;
      return false;
    }
    if (address + size > memory_.size()) {
      std::cerr << "Error: attempt to write beyond memory limits" << std::endl;
      return false;
    }
    size_t burst_size = 64;
    const uint8_t* src_ptr = static_cast<const uint8_t*>(data);

    for (size_t offset = 0; offset < size; offset += burst_size) {
      size_t copy_size = std::min(burst_size, size - offset);
      std::memcpy(memory_.data() + address + offset, src_ptr + offset, copy_size);
      simulate_ddr3_latency();
    }

    return true;
  }
  bool read_data(size_t address, void* buffer, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (address % 8 != 0) {
      std::cerr << "Error: address not aligned to 8 bytes: 0x"
                << std::hex << address << std::dec << std::endl;
      return false;
    }
    if (address + size > memory_.size()) {
      std::cerr << "Error: attempt to read beyond memory limits" << std::endl;
      return false;
    }
    size_t burst_size = 64;
    uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);

    for (size_t offset = 0; offset < size; offset += burst_size) {
      size_t copy_size = std::min(burst_size, size - offset);
      simulate_ddr3_latency();

      std::memcpy(dst_ptr + offset, memory_.data() + address + offset, copy_size);
    }

    return true;
  }
  size_t size() const {
    return memory_.size();
  }

 private:
  std::vector<uint8_t> memory_;
  std::mutex mutex_;
  void simulate_ddr3_latency() {
    // std::this_thread::sleep_for(std::chrono::nanoseconds(10));
  }
};

class AcceleratorMemoryManager {
 public:
  static AcceleratorMemoryManager& getInstance() {
    static AcceleratorMemoryManager instance;
    return instance;
  }
  void initialize(size_t memory_size = 1024 * 1024 * 1024,
                  size_t bias_offset = 100 * 1024 * 1024) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!memory_) {
      memory_ = std::make_unique<AcceleratorMemory>(memory_size);
      bias_memory_offset_ = bias_offset;
      next_weights_address_ = 0;
      next_bias_address_ = bias_memory_offset_;
      std::cout << "DDR3 memory initialized with:" << std::endl
                << "  - Total size: " << memory_size << " bytes" << std::endl
                << "  - Bias offset: 0x" << std::hex << bias_offset
                << std::dec << " (" << bias_offset << " bytes)" << std::endl;
    }
  }

  size_t allocate_weights_memory(size_t size, size_t alignment = 128) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!memory_) {
      std::cerr << "Error: memory not initialized" << std::endl;
      return 0;
    }
    size_t aligned_address = (next_weights_address_ + alignment - 1) & ~(alignment - 1);
    if (aligned_address + size > bias_memory_offset_) {
      std::cerr << "Error: not enough memory for weights before bias offset" << std::endl;
      return 0;
    }

    next_weights_address_ = aligned_address + size;
    return aligned_address;
  }
  size_t allocate_bias_memory(size_t size, size_t alignment = 128) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!memory_) {
      std::cerr << "Error: memory not initialized" << std::endl;
      return 0;
    }
    size_t aligned_address = (next_bias_address_ + alignment - 1) & ~(alignment - 1);
    if (aligned_address + size > memory_->size()) {
      std::cerr << "Error: not enough memory for bias" << std::endl;
      return 0;
    }

    next_bias_address_ = aligned_address + size;
    return aligned_address;
  }
  bool write_to_memory(size_t address, const void* data, size_t size) {
    if (!memory_) {
      std::cerr << "Error: memory not initialized" << std::endl;
      return false;
    }

    return memory_->write_data(address, data, size);
  }
  bool read_from_memory(size_t address, void* buffer, size_t size) {
    if (!memory_) {
      std::cerr << "Error: memory not initialized" << std::endl;
      return false;
    }

    return memory_->read_data(address, buffer, size);
  }
  size_t get_bias_memory_offset() const {
    return bias_memory_offset_;
  }
  bool read_weight_block(size_t base_address, int64_t k_block, int64_t oc_block,
                         int64_t block_size, int8_t* buffer) {
    if (!memory_) {
      std::cerr << "Error: memory not initialized" << std::endl;
      return false;
    }
    size_t block_idx = k_block * oc_block;
    size_t block_address = base_address + block_idx * block_size * block_size;
    return memory_->read_data(block_address, buffer, block_size * block_size);
  }
  bool read_bias_block(size_t base_address, int64_t oc_block, int64_t block_size, int32_t* buffer) {
    if (!memory_) {
      std::cerr << "Error: memory not initialized" << std::endl;
      return false;
    }
    size_t block_address = base_address + oc_block * block_size * sizeof(int32_t);
    return memory_->read_data(block_address, buffer, block_size * sizeof(int32_t));
  }

 private:
  std::unique_ptr<AcceleratorMemory> memory_;
  std::mutex mutex_;
  size_t bias_memory_offset_ = 0;
  size_t next_weights_address_ = 0;
  size_t next_bias_address_ = 0;
  AcceleratorMemoryManager() {}
};

}  // namespace mock
}  // namespace nudgev
}  // namespace onnxruntime