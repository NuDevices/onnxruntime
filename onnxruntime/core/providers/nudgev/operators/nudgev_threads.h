#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>
#include <map>
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace nudgev {

// Structure for FPGA data packet with identification
struct FpgaDataPacket {
  std::vector<uint8_t> data;
  size_t packet_id;
  uint64_t destination_offset;
};

// Structure for FPGA result data
struct FpgaResultPacket {
  std::vector<int8_t> data;
  size_t packet_id;
};

// Thread-safe queue for packets
template <typename T>
class ThreadSafeQueue {
 public:
  void push(T item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    condition_.notify_one();
  }

  bool pop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !queue_.empty() || done_; });

    if (queue_.empty()) {
      return false;
    }

    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void setDone() {
    std::lock_guard<std::mutex> lock(mutex_);
    done_ = true;
    condition_.notify_all();
  }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool done_ = false;
};

// Enhanced FPGA Interface with threaded send/receive
class ThreadedFpgaInterface {
 public:
  // Singleton pattern
  static ThreadedFpgaInterface& getInstance() {
    static ThreadedFpgaInterface instance;
    return instance;
  }

  // Initialize FPGA connection
  bool initialize(const std::string& device_path = "/dev/fpga_accel0",
                  uint64_t base_addr = 0x0,
                  size_t mem_size = 0x1000000) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) {
      return true;
    }

    fd_ = open(device_path.c_str(), O_RDWR | O_SYNC);
    if (fd_ < 0) {
      std::cerr << "Failed to open FPGA device: " << device_path << std::endl;
      return false;
    }

    fpga_mem_ = mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, base_addr);
    if (fpga_mem_ == MAP_FAILED) {
      std::cerr << "Failed to map FPGA memory" << std::endl;
      close(fd_);
      fd_ = -1;
      return false;
    }

    mem_size_ = mem_size;
    initialized_ = true;
    return true;
  }

  // Cleanup resources
  ~ThreadedFpgaInterface() {
    stopThreads();

    if (initialized_) {
      munmap(fpga_mem_, mem_size_);
      close(fd_);
    }
  }

  // Start sender and receiver threads
  bool startThreads(concurrency::ThreadPool* tp) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_ || threads_running_) {
      return false;
    }

    threadpool_ = tp;
    stop_threads_ = false;
    threads_running_ = true;

    // Start the sender thread
    threadpool_->Schedule([this]() {
      senderThread();
    });

    // Start the receiver thread
    threadpool_->Schedule([this]() {
      receiverThread();
    });

    return true;
  }

  // Stop sender and receiver threads
  void stopThreads() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!threads_running_) {
      return;
    }

    stop_threads_ = true;
    send_queue_.setDone();
    threads_running_ = false;

    // Wake up any waiting threads
    std::lock_guard<std::mutex> receiver_lock(receiver_mutex_);
    receiver_condition_.notify_all();

    std::lock_guard<std::mutex> result_lock(result_mutex_);
    result_condition_.notify_all();
  }

  // Queue a data packet for sending to FPGA
  void queueDataPacket(std::vector<uint8_t> data, size_t packet_id, uint64_t offset) {
    FpgaDataPacket packet;
    packet.data = std::move(data);
    packet.packet_id = packet_id;
    packet.destination_offset = offset;
    send_queue_.push(std::move(packet));
  }

  // Get a result packet from FPGA (blocking)
  bool getResultPacket(FpgaResultPacket& result, uint32_t timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(result_mutex_);

    // Wait for the result to be available
    bool has_result = result_condition_.wait_for(lock,
                                                 std::chrono::milliseconds(timeout_ms),
                                                 [this, &result]() {
                                                   auto it = result_map_.find(next_result_id_);
                                                   if (it != result_map_.end()) {
                                                     result = std::move(it->second);
                                                     result_map_.erase(it);
                                                     next_result_id_++;
                                                     return true;
                                                   }
                                                   return false;
                                                 });

    return has_result;
  }

  // Start FPGA computation for a specific packet
  bool startComputation(size_t packet_id, uint32_t command = 0x1) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_) {
      return false;
    }

    // Write to command register (assuming command register is at offset 0)
    // Include packet ID in the upper bits
    uint32_t cmd_with_id = (packet_id << 16) | (command & 0xFFFF);
    *static_cast<volatile uint32_t*>(fpga_mem_) = cmd_with_id;
    return true;
  }

 private:
  // Private constructor for singleton pattern
  ThreadedFpgaInterface()
      : fd_(-1),
        fpga_mem_(nullptr),
        mem_size_(0),
        initialized_(false),
        threads_running_(false),
        stop_threads_(false),
        next_result_id_(0) {}

  // Prevent copy and assignment
  ThreadedFpgaInterface(const ThreadedFpgaInterface&) = delete;
  ThreadedFpgaInterface& operator=(const ThreadedFpgaInterface&) = delete;

  // Thread function for sending data to FPGA
  void senderThread() {
    while (!stop_threads_) {
      FpgaDataPacket packet;
      if (send_queue_.pop(packet)) {
        // Send data to FPGA
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (initialized_) {
            memcpy(static_cast<char*>(fpga_mem_) + packet.destination_offset,
                   packet.data.data(), packet.data.size());

            // Start computation for this packet
            startComputation(packet.packet_id);
          }
        }

        // Notify receiver thread that data has been sent
        {
          std::lock_guard<std::mutex> lock(receiver_mutex_);
          pending_packets_.push_back(packet.packet_id);
          receiver_condition_.notify_one();
        }
      }
    }
  }

  // Thread function for receiving results from FPGA
  void receiverThread() {
    while (!stop_threads_) {
      // Wait for notification that data has been sent
      size_t packet_id;
      {
        std::unique_lock<std::mutex> lock(receiver_mutex_);
        receiver_condition_.wait(lock, [this] {
          return !pending_packets_.empty() || stop_threads_;
        });

        if (stop_threads_ || pending_packets_.empty()) {
          continue;
        }

        packet_id = pending_packets_.front();
        pending_packets_.erase(pending_packets_.begin());
      }

      // Wait for computation to complete
      bool completed = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
          // Read status register (assuming status register is at offset 4)
          volatile uint32_t* status_reg = static_cast<volatile uint32_t*>(fpga_mem_) + 1;

          // Simple polling with timeout
          const uint32_t COMPLETION_BIT = 0x1;
          const uint32_t MAX_POLLS = 1000;
          const uint32_t POLL_INTERVAL_US = 100;

          uint32_t poll_count = 0;
          while ((*status_reg & COMPLETION_BIT) == 0 && poll_count < MAX_POLLS) {
            usleep(POLL_INTERVAL_US);
            poll_count++;
          }

          completed = (poll_count < MAX_POLLS);
        }
      }

      if (completed) {
        // Read results from FPGA
        FpgaResultPacket result;
        result.packet_id = packet_id;

        // Calculate result buffer location based on packet ID
        uint64_t result_offset = 0x10000 + (packet_id * 0x1000);  // Example calculation

        // Determine result size (could be read from FPGA registers)
        const size_t result_size = 1024;  // Example size

        result.data.resize(result_size);

        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (initialized_) {
            memcpy(result.data.data(),
                   static_cast<char*>(fpga_mem_) + result_offset,
                   result_size);
          }
        }

        // Store the result
        {
          std::lock_guard<std::mutex> lock(result_mutex_);
          result_map_[packet_id] = std::move(result);
          result_condition_.notify_all();
        }
      }
    }
  }

  int fd_;            // File descriptor for FPGA device
  void* fpga_mem_;    // Mapped memory for FPGA communication
  size_t mem_size_;   // Size of mapped memory
  bool initialized_;  // Initialization flag
  std::mutex mutex_;  // Mutex for thread safety

  concurrency::ThreadPool* threadpool_;  // Thread pool for worker threads
  bool threads_running_;                 // Flag indicating if threads are running
  bool stop_threads_;                    // Flag to stop threads

  ThreadSafeQueue<FpgaDataPacket> send_queue_;  // Queue for data to send

  std::mutex receiver_mutex_;
  std::condition_variable receiver_condition_;
  std::vector<size_t> pending_packets_;  // Packets waiting for results

  std::mutex result_mutex_;
  std::condition_variable result_condition_;
  std::map<size_t, FpgaResultPacket> result_map_;  // Results received from FPGA
  size_t next_result_id_;                          // Next result ID to be processed
};

// Function to process FPGA operations with separate send/receive threads
bool process_with_fpga_threads(
    const std::vector<std::vector<uint8_t>>& data_packets,
    int8_t* output_data,
    int64_t batch_size,
    int64_t output_channels,
    int64_t output_height,
    int64_t output_width,
    concurrency::ThreadPool* tp) {
  // Get FPGA interface instance
  auto& fpga = ThreadedFpgaInterface::getInstance();

  // Initialize FPGA if not already initialized
  if (!fpga.initialize()) {
    std::cerr << "Failed to initialize FPGA" << std::endl;
    return false;
  }

  // Start sender and receiver threads
  if (!fpga.startThreads(tp)) {
    std::cerr << "Failed to start FPGA threads" << std::endl;
    return false;
  }

  // Calculate memory layout for FPGA
  constexpr uint64_t CONTROL_REGS_SIZE = 0x1000;
  uint64_t input_offset = CONTROL_REGS_SIZE;
  uint64_t current_offset = input_offset;

  // Queue all data packets for sending
  for (size_t i = 0; i < data_packets.size(); i++) {
    const auto& packet = data_packets[i];

    fpga.queueDataPacket(packet, i, current_offset);
    current_offset += packet.size();
  }

  // Wait for all results
  bool success = true;
  const int64_t total_elements = batch_size * output_channels * output_height * output_width;
  std::vector<int8_t> result_buffer(total_elements);

  for (size_t i = 0; i < data_packets.size(); i++) {
    FpgaResultPacket result;
    if (!fpga.getResultPacket(result)) {
      std::cerr << "Failed to get result for packet " << i << std::endl;
      success = false;
      break;
    }

    // Copy result data to the output buffer
    // The exact copying logic depends on how the FPGA organizes its output
    // This is just an example

    // Calculate the position in the output buffer for this packet's results
    size_t position_in_output = i * (total_elements / data_packets.size());
    size_t elements_in_packet = std::min(
        static_cast<size_t>(total_elements / data_packets.size()),
        result.data.size());

    // Copy the result data to the output buffer
    memcpy(output_data + position_in_output, result.data.data(), elements_in_packet);
  }

  // Stop the threads after processing
  fpga.stopThreads();

  return success;
}

}  // namespace nudgev
}  // namespace onnxruntime