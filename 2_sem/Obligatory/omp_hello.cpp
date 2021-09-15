#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
  
int main(int argc, char* argv[])
{
    int nthreads, tid;
  
    #pragma omp parallel private(nthreads, tid)
    {
        tid = omp_get_thread_num();
        printf("Hello world from thread = %d\n", tid);

    }

    return 0;
}