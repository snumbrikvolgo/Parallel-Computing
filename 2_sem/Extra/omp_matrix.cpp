#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <iostream>

int main(int argc, char** argv) 
{
    int i,j,k;
    omp_set_num_threads(omp_get_num_procs());

    int N = atoi(argv[1]);
    // std::cout << N << "\n";

    int** A = (int**) malloc((N) * sizeof(int*)); 
    for (i = 0; i < N; i++) 
          A[i] = (int*)malloc(N * sizeof(int));
    int** B = (int**) malloc((N) * sizeof(int*)); 
    for (i = 0; i < N; i++) 
          B[i] = (int*)malloc(N * sizeof(int));
    int** C = (int**) malloc((N) * sizeof(int*)); 
    for (i = 0; i < N; i++) 
          C[i] = (int*)malloc(N * sizeof(int));

    for (i= 0; i< N; i++)
        for (j= 0; j< N; j++)
	{
            A[i][j] = i + j;
            B[i][j] = i - j;
	}

    double time1 = omp_get_wtime();
    #pragma omp parallel for private(i,j,k) shared(A,B,C)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double time2 = omp_get_wtime();
    double parallel = time2-time1;
    // std::cout << "Parallel computation equals " << time2-time1 << " in s\n"; 

    time1 = omp_get_wtime();
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    time2 = omp_get_wtime();
    double usual = time2-time1;
    // std::cout << "Usual computation equals " << time2-time1 << " in s\n"; 
    std::cout << N << ","<< parallel<<"," << usual << "\n";
   
    /*for (i= 0; i< N; i++)
    {
        for (j= 0; j< N; j++)
        {
            printf("%d\t",C[i][j]);
        }
        printf("\n");
    }*/

    free(A);
    free(B);
    free(C);

    return 0;
}