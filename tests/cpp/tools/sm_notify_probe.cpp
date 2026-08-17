#include <cstdio>
#include <cstdlib>

#include "scheduler/sm_notify_source.h"

// Test helper: attach to a shm segment via the real ShmSmTargetSource consumer
// and print Read() for (gpu_id, partition_idx). The Python shm-contract test
// uses this to prove a foreign-language producer is byte-compatible with the
// C++ consumer's SmNotifyHeader/SmNotifySlot layout and seqlock protocol.
//
// Usage: sm_notify_probe <shm_name> <gpu_id> <partition_idx>
int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <shm_name> <gpu_id> <partition_idx>\n",
                 argv[0]);
    return 2;
  }
  ShmSmTargetSource src({argv[1], std::atoi(argv[2]), std::atoi(argv[3]),
                         /*require=*/false});
  std::printf("%d\n", src.Read());
  return 0;
}
