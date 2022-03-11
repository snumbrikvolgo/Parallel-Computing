#include <stdio.h>
#include <stdlib.h>
#include "sys/resource.h"
#include "omp.h"

#include "matrix.h"

double calculate(const struct rusage* b, const struct rusage* a)
{
    if (b == NULL || a == NULL)
    {
        return 0.0;
    }
    else
    {
        return ((((a->ru_utime.tv_sec * 1000000 + a->ru_utime.tv_usec) - (b->ru_utime.tv_sec * 1000000 + b->ru_utime.tv_usec)) +
                 ((a->ru_stime.tv_sec * 1000000 + a->ru_stime.tv_usec) - (b->ru_stime.tv_sec * 1000000 + b->ru_stime.tv_usec))) / 1000000.0);
    }
}

void all() {
  FILE * pFile;
  if((pFile=fopen("results_all.csv", "a+"))==NULL)
  {
    printf("suck\n");
  }
  struct rusage before, after;

  unsigned int N = 32;
  while (N <= 2048*2) {
    unsigned int runs = 1;
    double naive_mult_time = .0;
    double winograd_mult_time = .0;
    double strassen_mult_time = .0;

    matrix *m1 = alloc_matrix();
    matrix *m2 = alloc_matrix();
    matrix *naive_res = alloc_matrix();
    matrix *winograd_res = alloc_matrix();
    matrix *strassen_res = alloc_matrix();

    random_matrix(N, N, m1);
    random_matrix(N, N, m2);

    for (int i = 0; i < runs; i++) {
      getrusage(RUSAGE_SELF, &before);
      naive_mult(m1, m2, naive_res);
      getrusage(RUSAGE_SELF, &after);
      naive_mult_time += calculate(&before, &after);
    }
    naive_mult_time /= runs;


    for (int i = 0; i < runs; i++) {
      getrusage(RUSAGE_SELF, &before);
      winograd_mult(m1, m2, winograd_res);
      getrusage(RUSAGE_SELF, &after);
      winograd_mult_time += calculate(&before, &after);
    }
    winograd_mult_time /= runs;


    for (int i = 0; i < runs; i++) {
      getrusage(RUSAGE_SELF, &before);
      strassen_mult(m1, m2, strassen_res, 100, 32);
      getrusage(RUSAGE_SELF, &after);
      strassen_mult_time += calculate(&before, &after);
    }
    strassen_mult_time /= runs;
    //N \t naive \t winograd \t strassen
    fprintf(pFile, "%d\t%.6f\t%.6f\t%.6f\n", N, naive_mult_time, winograd_mult_time, strassen_mult_time);
    printf("%d\t%.6f\t%.6f\t%.6f\n", N, naive_mult_time, winograd_mult_time, strassen_mult_time);
    free_matrix(m1);
    free_matrix(m2);
    free_matrix(naive_res);
    free_matrix(winograd_res);
    free_matrix(strassen_res);

    N *= 2;
  }
  fclose (pFile);
}


void only_strassen() {

  FILE * pFile;
  if((pFile=fopen("results.csv", "a+"))==NULL)
  {
    printf("suck\n");
  }

  struct rusage before, after;

  unsigned int N = 32;

  while (N <= 2048*2) {
    unsigned int runs = 1;
    double strassen_mult_time = .0;

    matrix *m1 = alloc_matrix();
    matrix *m2 = alloc_matrix();
    matrix *strassen_res = alloc_matrix();

    random_matrix(N, N, m1);
    random_matrix(N, N, m2);

    for (int i = 0; i < runs; i++) {
      getrusage(RUSAGE_SELF, &before);
      strassen_mult(m1, m2, strassen_res, 100, 32);
      getrusage(RUSAGE_SELF, &after);
      strassen_mult_time += calculate(&before, &after);
    }
    strassen_mult_time /= runs;
      //N \t p \t strassen
    fprintf(pFile, "%d\t%.6f\n", N, strassen_mult_time);
    //fprintf(pFile, "hui");
    free_matrix(m1);
    free_matrix(m2);

    free_matrix(strassen_res);

    N *= 2;
  }

  fclose (pFile);
}

int main(int argc, const char * argv[])
{
  //only_strassen();
  all();
  return EXIT_SUCCESS;
}

