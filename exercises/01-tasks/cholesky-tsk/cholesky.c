#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include <sys/time.h>
#include <sys/times.h>

#include <math.h>
#include <mkl.h>

#include "omp.h"

#if !defined(_OPENMP)
    int omp_get_max_threads() { return 1; }
    int omp_get_num_threads() { return 1; }
#endif

void cholesky(int ts, int nt, double* Ah[nt][nt])
{
#ifdef VERBOSE
	printf("> Computing Cholesky Factorization: indirect blocked matrix...\n");
#endif

   // TODO: Parallel region is required in order to run in parallel.
   // TODO: Kernel invocations could be task candidates.
   // TODO: Add scheduler restrictions: taskwait, taskgroup, etc.
   for (int k = 0; k < nt; k++) {
      // Diagonal Block factorization: using LAPACK
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', ts, Ah[k][k], ts);

      // Triangular systems
      for (int i = k + 1; i < nt; i++) {
         cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, ts, ts, 1.0, Ah[k][k], ts, Ah[k][i], ts);
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ts, ts, ts, -1.0, Ah[k][i], ts, Ah[k][j], ts, 1.0, Ah[j][i], ts);
         }
         cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, ts, ts, -1.0, Ah[k][i], ts, 1.0, Ah[i][i], ts);
      }
   }  

#ifdef VERBOSE
	printf("> ...end of Cholesky Factorization.\n");
#endif
}

float get_time()
{
	static double gtod_ref_time_sec = 0.0;

	struct timeval tv;
	gettimeofday(&tv, NULL);

	// If this is the first invocation of through dclock(), then initialize the
	// "reference time" global variable to the seconds field of the tv struct.
	if (gtod_ref_time_sec == 0.0) gtod_ref_time_sec = (double) tv.tv_sec;

	// Normalize the seconds field of the tv struct so that it is relative to the
	// "reference time" that was recorded during the first invocation of dclock().
	const double norm_sec = (double) tv.tv_sec - gtod_ref_time_sec;

	// Compute the number of seconds since the reference time.
	const double t = norm_sec + tv.tv_usec * 1.0e-6;

	return (float) t;
}

// Robust Check the factorization of the matrix A2
// Using directly Fortran services: dlacpy_, dtrmm, dlange_
static int check_factorization(int N, double *A1, double *A2, int LDA, char uplo, double eps)
{
#ifdef VERBOSE
	printf("> Checking the Cholesky Factorization... \n");
#endif
	char NORM = 'I', ALL = 'A', UP = 'U', LO = 'L', TR = 'T', NU = 'N', RI = 'R';

	double *Residual = (double *) malloc(N*N*sizeof(double));
	double *L1       = (double *) malloc(N*N*sizeof(double));
	double *L2       = (double *) malloc(N*N*sizeof(double));
	double *work     = (double *) malloc(N*sizeof(double));

	memset((void*)L1, 0, N*N*sizeof(double));
	memset((void*)L2, 0, N*N*sizeof(double));

	double alpha= 1.0;

	dlacpy_(&ALL, &N, &N, A1, &LDA, Residual, &N);

	/* Dealing with L'L or U'U  */
	if (uplo == 'U'){
		dlacpy_(&UP, &N, &N, A2, &LDA, L1, &N);
		dlacpy_(&UP, &N, &N, A2, &LDA, L2, &N);
		dtrmm(&LO, &uplo, &TR, &NU, &N, &N, &alpha, L1, &N, L2, &N);
	} else{
		dlacpy_(&LO, &N, &N, A2, &LDA, L1, &N);
		dlacpy_(&LO, &N, &N, A2, &LDA, L2, &N);
		dtrmm(&RI, &LO, &TR, &NU, &N, &N, &alpha, L1, &N, L2, &N);
	}

	/* Compute the Residual || A -L'L|| */
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];

	double Rnorm = dlange_(&NORM, &N, &N, Residual, &N, work);
	double Anorm = dlange_(&NORM, &N, &N, A1, &N, work);

#ifdef VERBOSE
	printf("> - ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n",Rnorm / (Anorm*N*eps));
#endif

	const int info_factorization = isnan(Rnorm/(Anorm*N*eps)) || isinf(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 60.0);

#ifdef VERBOSE
	if ( info_factorization) printf("> - Factorization is suspicious!\n");
	else printf("> - Factorization is CORRECT!\n");
#endif

	free(Residual); free(L1); free(L2); free(work);

	return info_factorization;
}

void initialize_matrix(const int n, const int ts, double *matrix)
{
#ifdef VERBOSE
	printf("> Initializing matrix with random values...\n");
#endif

	int ISEED[4] = {0,0,0,1};
	int intONE=1;

	for (int i = 0; i < n*n; i+=n) {
		dlarnv_(&intONE, &ISEED[0], &n, &matrix[i]);
	}

	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++) {
			matrix[j*n+i] = matrix[j*n+i] + matrix[i*n+j];
			matrix[i*n+j] = matrix[j*n+i];
		}
	}

   // Diagonal values
	for (int i = 0; i < n; i++) {
		matrix[i*n+i] += (double) n;
   }
}

void gather_block(const int N, const int ts, double *Alin, double *A)
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++) {
			A[i*ts + j] = Alin[i*N + j];
		}
}

void scatter_block(const int N, const int ts, double *A, double *Alin)
{
	for (int i = 0; i < ts; i++)
		for (int j = 0; j < ts; j++) {
			Alin[i*N + j] = A[i*ts + j];
		}
}

void convert_to_blocks(const int ts, const int DIM, const int N, double Alin[N][N], double *A[DIM][DIM])
{
#ifdef VERBOSE
	printf("> Converting linear matrix to blocks...\n");
#endif
   for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++) {
         gather_block ( N, ts, &Alin[i*ts][j*ts], A[i][j]);
      }
}

void convert_to_linear(const int ts, const int DIM, const int N, double *A[DIM][DIM], double Alin[N][N])
{
#ifdef VERBOSE
	printf("> Converting blocked matrix to linear...\n");
#endif
   for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++) {
         scatter_block ( N, ts, A[i][j], (double *) &Alin[i*ts][j*ts]);
      }
}

int main(int argc, char* argv[])
{
   char *result[3] = {"n/a","pass","FAIL"};
	const double eps = pow(2.0, -53);

   // If number of arguments is not 4, print help
   if ( argc != 4) {
      printf( "%s: matrix_size block_size check[0|1]?\n", argv[0] );
      exit( -1 );
   }

   const int  n = atoi(argv[1]); // matrix size
   const int ts = atoi(argv[2]); // tile size
   int check    = atoi(argv[3]); // check result?

   // Compute number of tiles
   const int nt = n / ts;
   assert((nt*ts) == n); // tile size should divide size

   // Allocate matrix
   double * const matrix = (double *) malloc(n * n * sizeof(double));
   assert(matrix != NULL);

   // Initialize matrix
   initialize_matrix(n, ts, matrix);

   // Allocate original matrix, and duplicate it, for debugging purposes
   double * const original_matrix = (double *) malloc(n * n * sizeof(double));
   assert(original_matrix != NULL);

   // Save a copy of the original matrix
   for (int i = 0; i < n * n; i++ ) {
      original_matrix[i] = matrix[i];
   }

   // Set version description: Indirect blocked matrix
   const char *version = "I-Blocked matrix";

   // Allocate blocked matrix
   double *Ah[nt][nt];
   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         Ah[i][j] = malloc(ts * ts * sizeof(double));
         assert(Ah[i][j] != NULL);
      }
   }

   // ---------------------------------------
   // Convert, compute (time), and re-convert
   // ---------------------------------------
   convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);
   const float tref = get_time();
   cholesky(ts, nt, (double* (*)[nt]) Ah);
   const float time = get_time() - tref;
   convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);

   // Free blocked matrix
   for (int i = 0; i < nt; i++) {
      for (int j = 0; j < nt; j++) {
         assert(Ah[i][j] != NULL);
         free(Ah[i][j]);
      }
   }

   // Check result, if requested
   if ( check ) {
      const char uplo = 'L';
      if ( check_factorization( n, original_matrix, matrix, n, uplo, eps) ) check++;
   }

   // Free original matrix, not needed anymore
   free(original_matrix);

   // Compute GFLOPs (Not verified)
   float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

   // Print results
#ifdef VERBOSE
   printf( "\n" );
   printf( "============ CHOLESKY RESULTS ============\n" );
   printf( "  test                  %s\n", argv[0]);
   printf( "  version               %s\n", version);
   printf( "  matrix size:          %dx%d\n", n, n);
   printf( "  tile size:            %dx%d\n", ts, ts);
   printf( "  number of threads:    %d\n", omp_get_max_threads());
   printf( "  time (s):             %f\n", time);
   printf( "  performance (gflops): %f\n", gflops);
   printf( "  check:                %s\n", result[check]);
   printf( "==========================================\n" );
#else
   printf("test, %s, version, %s, n, %d, ts, %d, num_threads, %d, gflops, %f, time, %f, check, %s\n", argv[0], version, n, ts, omp_get_max_threads(), gflops, time, result[check]);
#endif

   // Free matrix
   free(matrix);

   return 0;
}

