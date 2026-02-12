#include <sched.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

// Pin thread to a specific CPU core
void PinThreadToCore(int core_id) {
  pid_t tid = syscall(SYS_gettid);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  int ret = syscall(SYS_sched_setaffinity, tid, sizeof(cpu_set_t), &cpuset);
  if (ret)
    sprintf(stderr, "Failed to pin thread %d to core %d\n", tid, core_id);
  // if (ret == 0) {
  //   std::cout << "Thread " << tid << " pinned to CPU core " << core_id
  //             << std::endl;
  // } else {
  //   std::cerr << "Failed to pin thread " << tid << " to core " << core_id
  //             << std::endl;
  // }
}

// Worker thread function
void MemoryCopyWorker(int core_id, size_t buffer_size, int duration_seconds,
                      std::atomic<uint64_t>* total_bytes,
                      std::atomic<uint64_t>* total_ops) {
  // Pin this thread to the specified core
  PinThreadToCore(core_id);

  // Allocate buffers
  void* src = malloc(buffer_size);
  void* dst = malloc(buffer_size);

  if (!src || !dst) {
    std::cerr << "Failed to allocate buffers in worker thread" << std::endl;
    return;
  }

  memset(src, 0xAB, buffer_size);

  uint64_t local_bytes = 0;
  uint64_t local_ops = 0;

  auto start = std::chrono::high_resolution_clock::now();

  while (true) {
    memcpy(dst, src, buffer_size);
    local_bytes += buffer_size;
    local_ops++;

    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

    if (elapsed >= duration_seconds) {
      break;
    }
  }

  // Update global counters
  *total_bytes += local_bytes;
  *total_ops += local_ops;

  free(src);
  free(dst);
}

int main(int argc, char* argv[]) {
  std::cout << "=== Multi-threaded Pin Thread Performance Test ==="
            << std::endl;

  int duration_seconds = 10;
  size_t buffer_size_bytes = 1024 * 1024;  // Default 1 MB
  std::vector<int> cores = {3, 5, 7};      // Default cores to test

  // Parse arguments
  if (argc > 1) {
    duration_seconds = std::atoi(argv[1]);
  }
  if (argc > 2) {
    buffer_size_bytes = std::atoll(argv[2]);
  }
  if (argc > 3) {
    // Parse cores from comma-separated list
    cores.clear();
    std::string cores_str = argv[3];
    size_t pos = 0;
    while (pos < cores_str.length()) {
      size_t comma = cores_str.find(',', pos);
      if (comma == std::string::npos) comma = cores_str.length();
      cores.push_back(std::stoi(cores_str.substr(pos, comma - pos)));
      pos = comma + 1;
    }
  }

  std::cout << "Duration: " << duration_seconds << " seconds" << std::endl;
  std::cout << "Buffer size: " << buffer_size_bytes << " bytes ("
            << (double)buffer_size_bytes / (1024 * 1024) << " MB)" << std::endl;
  std::cout << "CPU cores to pin: ";
  for (int core : cores) {
    std::cout << core << " ";
  }
  std::cout << std::endl << std::endl;

  std::atomic<uint64_t> total_bytes(0);
  std::atomic<uint64_t> total_ops(0);

  // Create worker threads
  std::vector<std::thread> threads;
  std::cout << "Creating " << cores.size() << " worker threads..." << std::endl;

  for (int core : cores) {
    threads.emplace_back(MemoryCopyWorker, core, buffer_size_bytes,
                         duration_seconds, &total_bytes, &total_ops);
  }

  std::cout << std::string(70, '=') << std::endl;
  std::cout << "Starting memory copy on cores: ";
  for (int core : cores) std::cout << core << " ";
  std::cout << std::endl;
  std::cout << "Monitor with: htop or top" << std::endl;
  std::cout << std::string(70, '=') << std::endl << std::endl;

  // Wait for all threads to finish
  for (auto& t : threads) {
    t.join();
  }

  std::cout << std::endl << std::string(70, '=') << std::endl;
  std::cout << "=== Final Statistics ===" << std::endl;

  uint64_t total_b = total_bytes.load();
  uint64_t total_o = total_ops.load();
  double total_time_s = duration_seconds;
  double throughput_gbs = (total_b / total_time_s) / 1e9;
  double ops_per_sec = total_o / total_time_s;

  std::cout << "Total threads: " << cores.size() << std::endl;
  std::cout << "Total bytes copied: " << total_b << " bytes" << std::endl;
  std::cout << "Total operations: " << total_o << std::endl;
  std::cout << "Total time: " << std::fixed << std::setprecision(2)
            << total_time_s << " seconds" << std::endl;
  std::cout << "Overall throughput: " << std::setprecision(3) << throughput_gbs
            << " GB/s" << std::endl;
  std::cout << "Operations/sec: " << std::setprecision(0) << ops_per_sec
            << std::endl;

  std::cout << std::endl << "Test completed!" << std::endl;
  std::cout << "Expected: Cores " << cores[0];
  for (size_t i = 1; i < cores.size(); i++) {
    std::cout << ", " << cores[i];
  }
  std::cout << " should show ~100% CPU usage in htop/top" << std::endl;

  return 0;
}
