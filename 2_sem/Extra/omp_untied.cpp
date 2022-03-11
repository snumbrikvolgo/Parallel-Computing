#include <iostream>
#include <omp.h>
#include <thread>

int dummy = 0;
#pragma omp threadprivate(dummy)


void foo() {
    auto  threadNum = omp_get_thread_num();
    std::printf("Foo! I'm thread %d, i = %d\n", threadNum, ++dummy);
}

void bar() {
    auto  threadNum = omp_get_thread_num();
    std::printf("Bar! I'm thread %d, i = %d\n", threadNum, ++dummy);
}

int main(int const argc, char const *argv[])
{
    omp_set_num_threads(4);

    #pragma omp parallel
    {   
        #pragma omp task untied
        {
            foo();

            #pragma omp taskyield

            bar();

        }
    }
}
