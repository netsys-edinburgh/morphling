#pragma once
#include <cstdint>

#define SHM_NAME "/emulator_shm"
#define ALIGNMENT 512

#define SHM_SIZE (100L * 1024L * 1024L * 1024L)
#define MAX_TASKS 100

struct __attribute__((__packed__)) GemmArgs {
  char transa;  // operation op(A) that is non- or (conj.) transpose.
  char transb;  // operation op(B) that is non- or (conj.) transpose.
  int m;        // number of rows of matrix op(A) and C.
  int n;        // number of columns of matrix op(B) and C.
  int k;        // number of columns of op(A) and rows of op(B).
  void* alpha;  // host or device <type> scalar used for multiplication.
  void* a;  // device <type> array of dimensions lda x k with lda>=max(1,m) if
            // transa == CUBLAS_OP_N and lda x m with lda>=max(1,k) otherwise.
  int lda;  // leading dimension of two-dimensional array used to store the
            // matrix A.
  void* b;  // device <type> array of dimension ldb x n with ldb>=max(1,k) if
            // transb == CUBLAS_OP_N and ldb x k with ldb>=max(1,n) otherwise.
  int ldb;  // leading dimension of two-dimensional array used to store matrix
            // B.
  void* beta;  // host or device <type> scalar used for multiplication. If
               // beta==0, C does not have to be a valid input.
  void* c;     // device in/out <type> array of dimensions ldc x n with
               // ldc>=max(1,m).
  int ldc;     // leading dimension of a two-dimensional array used to store the
               // matrix C.
};

typedef struct {
  unsigned char flag;
  uint8_t type;
  uint64_t offset;
  uint64_t size;
} meta_struct_t;

template <typename T>
struct DoNothingDeleter {
  void operator()(T* ptr) const {}
};
