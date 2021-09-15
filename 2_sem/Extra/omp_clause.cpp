#include <iostream>
#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv)
{
    int i, my_id;
    omp_set_num_threads(4); 
    
    std::cout << "\nusual\n\n";

    #pragma omp parallel for private(i)//нет балансировки
    for (i = 0; i < 65; i++)
    {
        printf("Thread_num (%d) -- %02d\n", omp_get_thread_num(), i);
    }
    std::cout << "\nstatic, chunk 1\n\n";

    #pragma omp parallel for private(i) schedule(static, 1) //статическая балансировка, 
    for (i = 0; i < 65; i++)
    {
        printf("Thread_num (%d) -- %02d\n", omp_get_thread_num(), i);
    }

    std::cout << "\nstatic, chunk 4\n\n";

    #pragma omp parallel for private(i) schedule(static, 4) //статическая балансировка, 
    for (i = 0; i < 65; i++)
    {
        printf("Thread_num (%d) -- %02d\n", omp_get_thread_num(), i);
    }

    std::cout << "\ndynamic, chunk 1\n\n";

    #pragma omp parallel for private(i) schedule(dynamic, 1) //динамическая балансировка, 
    for (i = 0; i < 65; i++)
    {
        printf("Thread_num (%d) -- %02d\n", omp_get_thread_num(), i);
    }

    std::cout << "\ndynamic, chunk 4\n\n";

    #pragma omp parallel for private(i) schedule(dynamic, 4) //динамическая балансировка, 
    for (i = 0; i < 65; i++)
    {
        printf("Thread_num (%d) -- %02d\n", omp_get_thread_num(), i);
    }


    std::cout << "\nguided, chunk 1\n\n";

    #pragma omp parallel for private(i) schedule(guided, 1) //гайдуд балансировка, 
    for (i = 0; i < 65; i++)
    {
        printf("Thread_num (%d) -- %02d\n", omp_get_thread_num(), i);
    }

    std::cout << "\nguided, chunk 4\n\n";

    #pragma omp parallel for private(i) schedule(guided, 4) //гайдуд балансировка, 
    for (i = 0; i < 65; i++)
    {
        printf("Thread_num (%d) -- %02d\n", omp_get_thread_num(), i);
    }

    return 0;
}