//radix-2 dit

#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#define MAX_NUMBER 8


typedef struct {
    double re;
    double im;
} complex;


void init(unsigned size, complex *a);
void save_array(unsigned size, complex *a, char* name);
void FFT_parallel(complex *A, complex *a, complex *W, unsigned N, unsigned stride, complex *D, int p);
void FFT(complex *A, complex *a, complex *W, unsigned N, unsigned stride, complex *D);
void Compute_twiddles(unsigned size, complex *W);


void init(unsigned size, complex *a) {
  for(int i = 0; i < size; i++){
    a[i].re = rand() % MAX_NUMBER;
    a[i].im = rand() % MAX_NUMBER;
  }
}

void save_array(unsigned size, complex *a, char* name) {
  FILE* ff = fopen(name,"w");
  for(int i = 0; i < size; i++){
    fprintf(ff,"%f%+fj", a[i].re, a[i].im);
    fprintf(ff,"\n");
  }
  fclose(ff);
}

void FFT_parallel(complex *A, complex *a, complex *W, unsigned N, unsigned stride, complex *D, int p) {
  complex *B, *C, Aux, W_i;
  unsigned n;

  if (p == 1)
  {
    FFT(A, a, W, N, stride, D);
  }
  else {
#pragma omp parallel
    {
      if (N == 1) {
        A[0].re = a[0].re;
        A[0].im = a[0].im;
      } else {
#pragma omp single nowait
        {
          n = (N >> 1);   // N = N div 2
#pragma omp task
          FFT_parallel(D, a, W, n, stride << 1, A, p / 2);
#pragma omp task
          FFT_parallel(D + n, a + stride, W, n, stride << 1, A + n, p - p / 2);

#pragma omp taskwait
          {
            B = D;
            C = D + n;

#pragma omp parallel for default(none) private(Aux, W_i) shared(stride, n, A, B, C, W)
            for (int i = 0; i <= n - 1; i++) {
              W_i = *(W + i * stride);
              Aux.re = W_i.re * C[i].re - W_i.im * C[i].im;
              Aux.im = W_i.re * C[i].im + W_i.im * C[i].re;

              A[i].re = B[i].re + Aux.re;
              A[i].im = B[i].im + Aux.im;
              A[i + n].re = B[i].re - Aux.re;
              A[i + n].im = B[i].im - Aux.im;
            }
          }
        }
      }
    }
  }
}

void FFT(complex *A, complex *a, complex *W, unsigned N, unsigned stride, complex *D) {
  complex *B, *C, Aux, W_i;
  unsigned n;


  if (N == 1)
  {
    A[0].re = a[0].re;
    A[0].im = a[0].im;
  }
  else
  {
    n = (N >> 1);   /* N = N div 2 */

    FFT(D, a, W, n, stride << 1, A);
    FFT(D + n, a + stride, W, n, stride << 1, A + n);

    B = D;
    C = D + n;

    for (int i = 0; i <= n - 1; i++) {
      W_i = *(W + i * stride);
      Aux.re = W_i.re * C[i].re - W_i.im * C[i].im;
      Aux.im = W_i.re * C[i].im + W_i.im * C[i].re;

      A[i].re = B[i].re + Aux.re;
      A[i].im = B[i].im + Aux.im;
      A[i + n].re = B[i].re - Aux.re;
      A[i + n].im = B[i].im - Aux.im;
    }
  }
}

void Compute_twiddles(unsigned size, complex *W) {
  complex Omega;

  double phi = M_PI / (double) size;
  Omega.re = cos(phi);
  Omega.im = sin(phi);
  W[0].re = 1.0;
  W[0].im = 0.0;
  for(int i = 1; i < size; i++) {
    W[i].re = W[i-1].re * Omega.re - W[i-1].im * Omega.im;
    W[i].im = W[i-1].re * Omega.im + W[i-1].im * Omega.re;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3)
  {
    printf ("usage: %s N p\n", argv[0]);
    return EXIT_FAILURE;
  }
  char* end_ptr = NULL;
  int N = (int) strtol (argv[1], &end_ptr, 10);
  int p = (int) strtol (argv[2], &end_ptr, 10);
  int processors = omp_get_num_procs ();
  if (p > processors)
  {
    printf ("Warning: %d threads requested, will run_omp on %d processors available\n", p, processors);
    omp_set_num_threads (p);
  }
  int max_threads = omp_get_max_threads ();
  if (p > max_threads)
  {
    printf ("Error: Cannot use %d threads, only %d threads available\n", p, max_threads);
    return EXIT_FAILURE;
  }

  complex *a, *A, *W, *D;
  a = (complex*) malloc(N * sizeof(complex));
  A = (complex*) malloc(N * sizeof(complex));
  D = (complex*) malloc(N * sizeof(complex));
  W = (complex*) malloc((N>>1) * sizeof(complex));
  if((a==NULL) || (A==NULL) || (D==NULL) || (W==NULL)) {
    printf("Not enough memory initializing arrays\n");
    EXIT_FAILURE;
  }

  init(N, a);
  save_array(N, a, "signal.txt");

  Compute_twiddles(N >> 1, W);

  double start = omp_get_wtime ();
  FFT_parallel(A, a, W, N, 1, D, p);
  double end = omp_get_wtime ();

  save_array(N, A, "result.txt");

  printf ("%d\t%d\t%g\n", N, p, end - start);

  free(W);
  free(D);
  free(A);
  free(a);

  return EXIT_SUCCESS;
}
