#ifndef INITIALIZATION_MEMORY_H
#define INITIALIZATION_MEMORY_H

#include <stdatomic.h>
#include <stdint.h>
#include <pthread.h>
#include <stddef.h>

#define MAX_TASKS 1024
#define QUEUE_SIZE 1024
#define INITIALIZATION_FLAG 0xDEADBEEF
#define SHM_NAME "/sgemm_shm"

typedef struct {
    atomic_int head;
    atomic_int tail;
    atomic_int size;
    atomic_int initialization_flag;
    int queue[QUEUE_SIZE];
} task_queue_t;

typedef struct {
    pthread_mutex_t offset_mutex; // To protect offset updates
    size_t current_offset;        // Global offset for all processes
    task_queue_t task_queue;
    // Define meta_data_t structure here
    struct {
        atomic_int flag;
        size_t offset;
    } meta_data[MAX_TASKS];
    // Other shared memory structures...
} shared_memory_t;

extern shared_memory_t *shared_mem_ptr;
//define shared_memory_t

typedef struct {
    atomic_int flag;
    size_t offset;
} task_meta_data_t;




void log_message(const char *message);
int dequeue_task(task_queue_t *task_queue);
void init_task_queue(task_queue_t *task_queue);
shared_memory_t *initialize_shared_memory(size_t shm_size);
void destroy_shared_memory(shared_memory_t *shared_mem_ptr, size_t shm_size);
size_t get_shared_memory_size();

#endif 
