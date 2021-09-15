
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
 
int a = 12345; //создаем глобальную переменную, риватная для мастера
#pragma omp threadprivate(a)
 
int main(int argc, char* argv[])
{
    a = 123; //мастер-тред меняет глобальную на 123, но остальные не замечают этого

    #pragma omp parallel 
    {
        #pragma omp master
        {   //мастер меняет переменную на 67890
            printf("[First parallel region] Master thread changes the value of a = %d to 67890.\n", a);
            a = 67890;
        }
 
        #pragma omp barrier
        //остальные не заметили изменения
        printf("[First parallel region] Thread %d: a = %d.\n", omp_get_thread_num(), a);
    }
 
    #pragma omp parallel copyin(a) //копируем значение а у мастер-треда в значение а у других потоков
    {
        #pragma omp master
        {
            printf("[First parallel region] Master thread changes the value of a to 55555.\n");
            a = 55555;
        }
        printf("[Second parallel region] Thread %d: a = %d.\n", omp_get_thread_num(), a);
    }
 
    return 0;
}
