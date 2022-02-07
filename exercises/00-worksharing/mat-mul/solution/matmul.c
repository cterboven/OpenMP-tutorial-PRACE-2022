//===-- matmul.c - Different implementations of matrix multiplies -*- C -*-===//
//
// Part of the LOMP Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define DUMP_MATRIX 0

void matmul_seq(double * C, double * A, double * B, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

void matmul_par(double * C, double * A, double * B, size_t n) {
#pragma omp parallel for shared(A,B,C) firstprivate(n) \
                         schedule(static) // collapse(2)
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

void init_mat(double * C, double * A, double * B, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] = 0.0;
      A[i * n + j] = 0.5;
      B[i * n + j] = 0.25;
    }
  }
}

void dump_mat(double * mtx, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      printf("%f ", mtx[i * n + j]);
    }
    printf("\n");
  }
}

double sum_mat(double * mtx, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      sum += mtx[i * n + j];
    }
  }
  return sum;
}

int main(int argc, char * argv[]) {
  double ts, te;
  double t_seq;

  double * C;
  double * A;
  double * B;

  // If number of arguments is not 1, print help
  if (argc != 2) {
    printf("%s: matrix_size\n", argv[0]);
    return EXIT_FAILURE;
  } 
  const int  n = atoi(argv[1]); // matrix size

  C = (double *)malloc(sizeof(*C) * n * n);
  A = (double *)malloc(sizeof(*A) * n * n);
  B = (double *)malloc(sizeof(*B) * n * n);

  init_mat(C, A, B, n);
  ts = omp_get_wtime();
  matmul_seq(C, A, B, n);
  te = omp_get_wtime();
#if DUMP_MATRIX
  dump_mat(C, n);
#endif
  t_seq = te - ts;
  printf("Sum of matrix (serial):   %f, wall time %lf, speed-up %.2lf\n",
         sum_mat(C, n), (te - ts), t_seq / (te - ts));

  init_mat(C, A, B, n);
  ts = omp_get_wtime();
  matmul_par(C, A, B, n);
  te = omp_get_wtime();
#if DUMP_MATRIX
  dump_mat(C, n);
#endif
  printf("Sum of matrix (parallel): %f, wall time %lf, speed-up %.2lf\n",
         sum_mat(C, n), (te - ts), t_seq / (te - ts));

  return EXIT_SUCCESS;
}
