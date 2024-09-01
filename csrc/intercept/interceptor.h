#ifndef INTERCEPTOR_H
#define INTERCEPTOR_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <sys/sysinfo.h>
#include <stdint.h>
#include <time.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include "../memory/initialize_memory.h"
#include <stdbool.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define SHM_NAME "/sgemm_shm"
#define LOG_DIR "/home/eren/DeviceEmulator/csrc/intercept/logs/"
#define EMPTY 0
#define WRITING 1
#define WRITTEN 2
#define READING 3
#define EXECUTING 4
#define COMPLETE 5
#define MAX_TASKS 1024
#define QUEUE_SIZE 1024

typedef struct {
    char transa;
    char transb;
    int m;
    int n;
    int k;
    float alpha;
    const float *a;
    int lda;
    const float *b;
    int ldb;
    float beta;
    float *c;
    int ldc;
} sgemm_args_t;

typedef void (*sgemm_type)(const char*, const char*, const int*, const int*, const int*,
                           const float*, const float*, const int*, const float*, const int*,
                           const float*, float*, const int*);

typedef void (*sgemm_batch_type)(const char* transa_array, const char* transb_array,
                                 const int* m_array, const int* n_array, const int* k_array,
                                 const float* alpha_array, const float* a_array[], const int* lda_array,
                                 const float* b_array[], const int* ldb_array,
                                 const float* beta_array, float* c_array[], const int* ldc_array,
                                 const int* group_count, const int* group_size);


extern FILE *log_file;   
extern pthread_mutex_t shm_mutex; 
extern sgemm_type orig_sgemm;  

// Function declarations
void initialize_logging();
void get_timestamp(char* buffer, size_t size);
void log_message(const char *message);
void lock_memory();
void unlock_memory();
size_t get_shared_memory_size();
int enqueue_task(task_queue_t *task_queue, int task_index);
void sgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* lda, const float* b, const int* ldb,
            const float* beta, float* c, const int* ldc);
void process_task_from_queue(shared_memory_t *shared_mem_ptr);

#endif 
