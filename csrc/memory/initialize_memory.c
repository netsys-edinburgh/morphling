

#include "initialize_memory.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define LOG_FILE \
  "/home/eren/DeviceEmulator/csrc/intercept/logs/shared_memory.log"
#define INITIALIZATION_FLAG 0xDEADBEEF

shared_memory_t* shared_mem_ptr = NULL;  // Central shared memory pointer

void log_message(const char* message) {
  FILE* log_file = fopen(LOG_FILE, "a");
  if (log_file) {
    time_t now = time(NULL);
    struct tm* local = localtime(&now);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", local);
    fprintf(log_file, "[%s] %s\n", timestamp, message);
    fclose(log_file);
  } else {
    perror("Failed to open log file");
  }
}

void init_task_queue(task_queue_t* task_queue) {
  atomic_store(&task_queue->head, 0);
  atomic_store(&task_queue->tail, 0);
  atomic_store(&task_queue->size, 0);

  log_message("Task queue initialized. Head, tail, and size set to 0.");
}

shared_memory_t* initialize_shared_memory(size_t shm_size) {
  if (shared_mem_ptr != NULL) {
    // Shared memory is already initialized
    return shared_mem_ptr;
  }

  char log_buffer[256];

  int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, 0666);
  if (fd == -1) {
    perror("shm_open");
    log_message("Failed to open shared memory.");
    return NULL;
  }

  if (ftruncate(fd, shm_size) == -1) {
    perror("ftruncate");
    log_message("Failed to set the size of the shared memory.");
    close(fd);
    return NULL;
  }

  shared_mem_ptr =
      mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (shared_mem_ptr == MAP_FAILED) {
    perror("mmap");
    snprintf(log_buffer, sizeof(log_buffer),
             "Failed to map shared memory. Size: %zu bytes.", shm_size);
    log_message(log_buffer);
    close(fd);
    return NULL;
  }

  close(fd);

  if (shared_mem_ptr->task_queue.initialization_flag != INITIALIZATION_FLAG) {
    init_task_queue(&shared_mem_ptr->task_queue);
    shared_mem_ptr->task_queue.initialization_flag = INITIALIZATION_FLAG;
    pthread_mutex_init(&shared_mem_ptr->offset_mutex, NULL);
    shared_mem_ptr->current_offset = sizeof(shared_memory_t);
    snprintf(log_buffer, sizeof(log_buffer),
             "Shared memory and task queue initialized. Size: %zu bytes.",
             shm_size);
    log_message(log_buffer);
  } else {
    snprintf(log_buffer, sizeof(log_buffer),
             "Shared memory already initialized. Size: %zu bytes.", shm_size);
    log_message(log_buffer);
  }

  return shared_mem_ptr;
}

void destroy_shared_memory(shared_memory_t* shared_mem_ptr, size_t shm_size) {
  if (munmap(shared_mem_ptr, shm_size) == -1) {
    perror("munmap");
    log_message("Failed to unmap shared memory.");
  } else {
    log_message("Shared memory unmapped successfully.");
  }

  if (shm_unlink(SHM_NAME) == -1) {
    perror("shm_unlink");
    log_message("Failed to unlink shared memory.");
  } else {
    log_message("Shared memory unlinked successfully.");
  }
}
