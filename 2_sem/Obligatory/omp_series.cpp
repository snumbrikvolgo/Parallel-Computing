#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
  
int main(int argc, char* argv[])
{
    int nthreads, tid;
    if (argc < 2){
        std::cerr << "Number N should be typed\n";
        return -1;
    }

    int N = atoi(argv[1]);
    double sum = 0.0;
    int i;
    #pragma omp parallel for private(i,tid) schedule(guided) reduction(+:sum)
    for (i = 1; i <= N; i++)
    {
        // tid = omp_get_thread_num();
        // std::cout << tid << "," << 1.0 / i << std::endl;
        sum += 1.0 / i;
    }
    
    std::cout << "Sum of " << N << " elements of harmonic series" << std::endl << 
        "sum = " << std::fixed << std::setprecision(15) << sum << std::endl;

    return 0;
}