#include "worker.h"

#include "memory/initialize_memory.h"

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      return "Unknown CUBLAS error";
  }
}

void print_first_10_values(const char* matrix_name, const float* matrix,
                           int size) {
  char message[1024];
  snprintf(message, sizeof(message),
           "First 10 values of matrix %s (at %p): ", matrix_name,
           (void*)matrix);

  for (int i = 0; i < 10 && i < size; i++) {
    char value_str[32];
    snprintf(value_str, sizeof(value_str), "%f ", matrix[i]);
    strncat(message, value_str, sizeof(message) - strlen(message) - 1);
  }

  log_message(message);
}

void log_device_pointer(const char* matrix_name, const float* device_ptr) {
  char message[256];
  snprintf(message, sizeof(message), "Device pointer for %s: %p", matrix_name,
           (void*)device_ptr);
  log_message(message);
}

char* get_timestamp() {
  time_t now = time(NULL);
  struct tm* t = localtime(&now);
  static char timestamp[64];
  strftime(timestamp, sizeof(timestamp) - 1, "%Y-%m-%d %H:%M:%S", t);
  return timestamp;
}

int dequeue_task(task_queue_t* task_queue) {
  int current_size = atomic_load(&task_queue->size);
  if (current_size == 0) {
    return -1;
  }

  int head = atomic_load(&task_queue->head);
  int task_index = task_queue->queue[head];

  if (task_index < 0 || task_index >= MAX_TASKS) {
    printf(
        "[%s] Invalid task index %d retrieved from queue. Head: %d, Size: %d\n",
        get_timestamp(), task_index, head, current_size);
    return -1;
  }

  atomic_store(&task_queue->head, (head + 1) % QUEUE_SIZE);
  atomic_fetch_sub(&task_queue->size, 1);

  return task_index;
}

void log_message(const char* message) {
  printf("[%s] %s\n", get_timestamp(), message);
}

size_t get_shared_memory_size() {
  struct sysinfo info;
  sysinfo(&info);
  return info.totalram / 2;
}

void real_execute(sgemm_args_t* task, size_t offset, int task_index) {
  log_message("Starting SGEMM Task with cuBLAS.");

  size_t freeMem, totalMem;
  char log_buffer[1024];

  // log offset and memory information
  snprintf(log_buffer, sizeof(log_buffer), "Offset: %zu", offset);
  log_message(log_buffer);
  cudaMemGetInfo(&freeMem, &totalMem);
  snprintf(log_buffer, sizeof(log_buffer),
           "GPU Memory: Free = %zu bytes, Total = %zu bytes", freeMem,
           totalMem);
  log_message(log_buffer);

  // log SGEMM parameters
  snprintf(log_buffer, sizeof(log_buffer),
           "SGEMM Parameters - Task Index: %d, transa=%c, transb=%c, m=%d, "
           "n=%d, k=%d, alpha=%f, beta=%f, lda=%d, ldb=%d, ldc=%d",
           task_index, task->transa, task->transb, task->m, task->n, task->k,
           task->alpha, task->beta, task->lda, task->ldb, task->ldc);
  log_message(log_buffer);

  memset(task->c, 0, task->m * task->n * sizeof(float));

  cublasHandle_t handle;
  CHECK_CUBLAS_ERROR(cublasCreate(&handle));

  CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  log_message("cuBLAS handle created successfully.");

  // allocate device memory for matrices A, B, and C
  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  size_t size_A = task->lda * task->k * sizeof(float);
  size_t size_B = task->ldb * task->n * sizeof(float);
  size_t size_C = task->ldc * task->n * sizeof(float);

  log_message("Allocating device memory for matrices.");

  if (cudaMalloc((void**)&d_A, size_A) != cudaSuccess) {
    log_message("Failed to allocate device memory for Matrix A.");
    goto cleanup;
  }
  snprintf(log_buffer, sizeof(log_buffer), "Device pointer for A: %p",
           (void*)d_A);
  log_message(log_buffer);

  if (cudaMalloc((void**)&d_B, size_B) != cudaSuccess) {
    log_message("Failed to allocate device memory for Matrix B.");
    goto cleanup;
  }
  snprintf(log_buffer, sizeof(log_buffer), "Device pointer for B: %p",
           (void*)d_B);
  log_message(log_buffer);

  if (cudaMalloc((void**)&d_C, size_C) != cudaSuccess) {
    log_message("Failed to allocate device memory for Matrix C.");
    goto cleanup;
  }
  snprintf(log_buffer, sizeof(log_buffer), "Device pointer for C: %p",
           (void*)d_C);
  log_message(log_buffer);

  // copy matrices A, B, and C to device memory
  log_message("Copying matrices A, B, and C to device memory.");

  if (cudaMemcpy(d_A, task->a, size_A, cudaMemcpyHostToDevice) != cudaSuccess) {
    log_message("Failed to copy Matrix A to device memory.");
    goto cleanup;
  }
  log_message("Matrix A copied to device memory successfully.");

  if (cudaMemcpy(d_B, task->b, size_B, cudaMemcpyHostToDevice) != cudaSuccess) {
    log_message("Failed to copy Matrix B to device memory.");
    goto cleanup;
  }
  log_message("Matrix B copied to device memory successfully.");

  cudaMemGetInfo(&freeMem, &totalMem);
  snprintf(
      log_buffer, sizeof(log_buffer),
      "GPU Memory after copying matrices: Free = %zu bytes, Total = %zu bytes",
      freeMem, totalMem);
  log_message(log_buffer);

  if (cudaMemcpy(d_C, task->c, size_C, cudaMemcpyHostToDevice) != cudaSuccess) {
    log_message("Failed to copy Matrix C to device memory.");
    goto cleanup;
  }
  log_message("Matrix C copied to device memory successfully.");

  // Set cuBLAS operation modes
  cublasOperation_t transa =
      (task->transa == 'N' || task->transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t transb =
      (task->transb == 'N' || task->transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  // perform matrix multiplication
  log_message("Performing matrix multiplication with cuBLAS.");

  CHECK_CUBLAS_ERROR(cublasSgemm_v2(handle, transa, transb, task->m, task->n,
                                    task->k, &task->alpha, d_A, task->lda, d_B,
                                    task->ldb, &task->beta, d_C, task->ldc));
  log_message("Matrix multiplication completed successfully.");

  // copy result back to host memory and log first 10 values
  log_message("Copying result Matrix C back to host memory.");
  CHECK_CUDA_ERROR(cudaMemcpy(task->c, d_C, size_C, cudaMemcpyDeviceToHost));

  print_first_10_values("Matrix C after computation", task->c,
                        task->m * task->n);

  atomic_store(&shared_mem_ptr->meta_data[task_index].flag, COMPLETE);

  log_message("Successfully copied result Matrix C back to host memory.");

cleanup:
  if (d_A) cudaFree(d_A);
  if (d_B) cudaFree(d_B);
  if (d_C) cudaFree(d_C);
  cublasDestroy(handle);
  cudaDeviceSynchronize();
  log_message("SGEMM Task completed with cuBLAS.");
}

void* gpu_process(void* arg) {
  log_message("GPU process started.");

  size_t shm_size = get_shared_memory_size();
  log_message("Shared memory size calculated.");

  shared_memory_t* shared_mem_ptr = initialize_shared_memory(shm_size);
  if (!shared_mem_ptr) {
    perror("Failed to initialize shared memory");
    log_message("Failed to initialize shared memory.");
    exit(EXIT_FAILURE);
  }
  log_message("Shared memory initialized successfully.");

  while (1) {
    int task_index = dequeue_task(&shared_mem_ptr->task_queue);
    if (task_index != -1) {
      atomic_store(&shared_mem_ptr->meta_data[task_index].flag, READING);
      log_message("Task dequeued successfully.");

      char log_buffer[128];
      snprintf(log_buffer, sizeof(log_buffer), "Processing task at index %d.",
               task_index);
      log_message(log_buffer);

      if (task_index < 0 || task_index >= MAX_TASKS) {
        snprintf(log_buffer, sizeof(log_buffer),
                 "Invalid task index %d detected. Skipping task.", task_index);
        log_message(log_buffer);
        continue;
      }

      size_t offset = shared_mem_ptr->meta_data[task_index].offset;

      sgemm_args_t* task_ptr = (sgemm_args_t*)((char*)shared_mem_ptr + offset);

      snprintf(log_buffer, sizeof(log_buffer), "Task pointer address: %p",
               (void*)task_ptr);
      log_message(log_buffer);

      size_t task_offset = sizeof(sgemm_args_t);
      task_ptr->a = (float*)((char*)task_ptr + task_offset);
      task_ptr->b = (float*)((char*)task_ptr->a +
                             task_ptr->lda * task_ptr->k * sizeof(float));
      task_ptr->c = (float*)((char*)task_ptr->b +
                             task_ptr->ldb * task_ptr->n * sizeof(float));

      if ((char*)task_ptr->a < (char*)shared_mem_ptr ||
          (char*)task_ptr->a + task_ptr->lda * task_ptr->k * sizeof(float) >
              (char*)shared_mem_ptr + shm_size ||
          (char*)task_ptr->b < (char*)shared_mem_ptr ||
          (char*)task_ptr->b + task_ptr->ldb * task_ptr->n * sizeof(float) >
              (char*)shared_mem_ptr + shm_size ||
          (char*)task_ptr->c < (char*)shared_mem_ptr ||
          (char*)task_ptr->c + task_ptr->ldc * task_ptr->n * sizeof(float) >
              (char*)shared_mem_ptr + shm_size) {
        log_message(
            "Error: Matrix pointer is outside of expected shared memory range. "
            "Exiting execution.");
        continue;
      }

      atomic_store(&shared_mem_ptr->meta_data[task_index].flag, EXECUTING);
      real_execute(task_ptr, offset, task_index);

      log_message("Task execution COMPLETED");
    } else {
      usleep(POLLING_INTERVAL);
    }
  }
  cudaDeviceSynchronize();

  destroy_shared_memory(shared_mem_ptr, shm_size);

  log_message("Shared memory destroyed and GPU process exiting.");
  return NULL;
}

int main() {
  log_message("Main process started.");

  FILE* log_file = fopen(LOG_FILE, "a");
  if (log_file == NULL) {
    perror("Failed to open log file");
    log_message("Failed to open log file.");
    return 1;
  }
  log_message("Log file opened successfully.");

  dup2(fileno(log_file), STDOUT_FILENO);
  dup2(fileno(log_file), STDERR_FILENO);

  log_message("Log redirection complete.");

  pthread_t gpu_thread;

  log_message("Creating GPU process thread.");
  if (pthread_create(&gpu_thread, NULL, gpu_process, NULL) != 0) {
    perror("Failed to create GPU process thread");
    log_message("Failed to create GPU process thread.");
    return 1;
  }
  log_message("GPU process thread created successfully.");

  if (pthread_join(gpu_thread, NULL) != 0) {
    perror("Failed to join GPU process thread");
    log_message("Failed to join GPU process thread.");
    return 1;
  }
  log_message("GPU process thread joined successfully.");

  fclose(log_file);
  log_message("Log file closed.");

  log_message("Main process exiting.");
  return 0;
}
